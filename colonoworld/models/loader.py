# SPDX-License-Identifier: CC-BY-NC-4.0
"""Model construction, weight loading and single-chunk generation.

:func:`load_model` builds the Cosmos3-Nano video model with the ColonoWorld conditioning
(:mod:`.backbone`), loads the Cosmos base weights and the ColonoWorld checkpoint, and returns a
:class:`ColonoWorldModel`.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colonoworld.models.camera import plucker_rays
from colonoworld.models.checkpoint import canonical_name, load_trained, read_state
from colonoworld.models.conditions import (
    MASK_CHANNELS,
    global_flags,
    palette_class_masks,
)
from colonoworld.models.network import ConditionState


# ----------------------------------------------------------------------------- framework setup
def _framework_root() -> Path:
    import cosmos_framework

    return Path(cosmos_framework.__file__).resolve().parents[1]


def _set_backend_flags(cudnn_benchmark: bool, deterministic: bool = False) -> None:
    """TF32 off; cuDNN autotuning on by default. ``deterministic`` selects deterministic cuDNN
    kernels without autotuning and a fixed cuBLAS workspace, at some speed cost."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def build_backbone_model(cfg: Mapping[str, Any]) -> Any:
    """Instantiate the ColonoWorld Cosmos model (randomly initialised weights) on the GPU."""
    import hydra
    import omegaconf

    framework = omegaconf.OmegaConf.create(dict(cfg["framework"]))
    framework.parallelism.enable_inference_mode = bool(cfg["runtime"].get("bf16_weights", True))
    node = omegaconf.OmegaConf.create(
        {"_target_": "colonoworld.models.backbone.build_backbone", "_recursive_": False}
    )
    node.config = framework
    # The framework reads some configuration files by path relative to its source root.
    with contextlib.chdir(_framework_root()):
        return hydra.utils.instantiate(node)


@torch.no_grad()
def load_base_weights(model: Any, path: str | Path) -> int:
    """Load Cosmos base weights from a DCP directory or safetensors file(s) into ``model.net``.

    Parameters that ColonoWorld trains (LoRA, camera and condition modules) are skipped.
    Returns the number of loaded tensors.
    """
    from colonoworld.models.checkpoint import is_trained

    path = Path(path)
    targets = {}
    for name, tensor in list(model.net.named_parameters()) + list(model.net.named_buffers()):
        key = canonical_name("net." + name)
        if not is_trained(key):
            targets[key] = tensor
    if any(path.glob("*.distcp")):
        import torch.distributed.checkpoint as dcp

        reader = dcp.FileSystemReader(str(path))
        available = set(reader.read_metadata().state_dict_metadata)
        state = {k: v for k, v in targets.items() if k in available}
        dcp.load(state, storage_reader=reader, no_dist=True)
        return len(state)
    from safetensors.torch import load_file

    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    count = 0
    for file in files:
        for key, value in load_file(str(file)).items():
            key = key if key.startswith("net.") else "net." + key
            if key in targets:
                targets[key].copy_(value.to(targets[key]))
                count += 1
    return count


@contextlib.contextmanager
def seeded_rng(seed: int):
    """Seed Python, NumPy and torch (CPU and every CUDA device) inside the scope, then restore."""
    import random

    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


# ----------------------------------------------------------------------------- prompts
def build_prompts(caption: str, frames: int, fps: int, height: int, width: int) -> tuple[str, str]:
    """Positive and negative prompts in the Cosmos image-to-video format.

    The duration and resolution sentences of the Cosmos defaults are appended to the caption;
    the negative prompt is the Cosmos default negative prompt with the same sentences.
    """
    defaults = _framework_root() / "cosmos_framework/inference/defaults"
    args = json.loads((defaults / "image2video/sample_args.json").read_text())
    negative = json.dumps(json.loads((defaults / args["negative_prompt_file"]).read_text()))

    def fmt(text: str, inverse: bool = False) -> str:
        pre = "inverse_" if inverse else ""
        text = text.strip()
        duration, resolution = args[pre + "duration_template"], args[pre + "resolution_template"]
        if duration is not None and (frames > 1 or inverse):
            text = text.rstrip(".") + ". " + duration.format(duration=frames / fps, fps=fps)
        text = text.strip()
        if resolution is not None:
            text = text.rstrip(".") + ". " + resolution.format(height=height, width=width)
        return text

    mode = args["negative_metadata_mode"]
    if args.get("negative_prompt_keep_metadata") and mode == "none":
        mode = "same"
    if mode in ("same", "inverse"):
        negative = fmt(negative, mode == "inverse").lstrip(".").strip()
    return fmt(caption), negative


# ----------------------------------------------------------------------------- model wrapper
class ColonoWorldModel:
    """Single-chunk ColonoWorld generator (first frame + camera + tube [+ mask] -> video)."""

    def __init__(self, backbone: Any, cfg: Mapping[str, Any], device: str = "cuda"):
        self.backbone = backbone
        self.cfg = cfg
        self.device = device
        video = cfg["video"]
        self.frames, self.height, self.width = (
            int(video["frames"]),
            int(video["height"]),
            int(video["width"]),
        )
        self.fps = int(video["fps"])

    # -- VAE
    @torch.no_grad()
    def encode_video(self, video: np.ndarray) -> torch.Tensor:
        """VAE-encode a ``uint8`` (T, H, W, 3) video to a bf16 latent (C, T', H/16, W/16)."""
        frames = torch.from_numpy(np.ascontiguousarray(video)).permute(3, 0, 1, 2)
        with torch.autocast("cuda", enabled=False):
            x = frames.to(self.device, non_blocking=True)[None].to(torch.float32) / 127.5 - 1.0
            latent = self.backbone.encode(x).contiguous().float()
        latent = latent[:, :, :, : self.height // 16, : self.width // 16]
        return latent[0].to(torch.bfloat16)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> np.ndarray:
        """Decode a latent (C, T', h, w) to a ``uint8`` (T, H, W, 3) video."""
        pixels = self.backbone.decode(latent)
        pixels = pixels[0] if pixels.ndim == 5 else pixels
        out = (((pixels.float().clamp(-1, 1) + 1) / 2) * 255).round().to(torch.uint8)
        return out.permute(1, 2, 3, 0).cpu().numpy()

    # -- inputs
    def camera_rays(
        self,
        cameras: np.ndarray,
        intrinsics: Mapping[str, Any],
        field_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pluecker rays (T', 6, H/16, W/16) for camera-to-world poses; cells outside the
        optional latent-grid ``field_mask`` are zeroed."""
        rays, valid = plucker_rays(cameras, intrinsics, (self.height, self.width))
        keep = valid if field_mask is None else valid & np.asarray(field_mask, bool)
        rays[:, :, ~keep] = 0
        return rays

    def _batch(self, frames: np.ndarray, caption: str) -> dict[str, Any]:
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        known = len(frames)
        if (known - 1) % 4 or not 1 <= known < self.frames:
            raise ValueError("context frames must number 4n+1 and be shorter than the chunk")
        video = torch.full((3, self.frames, self.height, self.width), 128, dtype=torch.uint8)
        video[:, :known] = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2)
        positive, negative = build_prompts(caption, self.frames, self.fps, self.height, self.width)
        key = self.backbone.input_caption_key
        size = torch.tensor([[self.height, self.width, self.height, self.width]])
        batch = {
            "video": [video.to(self.device)],
            "image_size": [size.to(self.device)],
            "conditioning_fps": [torch.tensor(self.fps).to(self.device)],
            "video_num_frames": [torch.tensor(self.frames).to(self.device)],
            "sequence_plan": [
                SequencePlan(
                    has_text=True,
                    has_vision=True,
                    has_action=False,
                    condition_frame_indexes_vision=list(range(1 + (known - 1) // 4)),
                )
            ],
            key: [positive],
            "neg_" + key: [negative],
        }
        return batch

    # -- generation
    @torch.no_grad()
    def generate(
        self,
        first_frame: np.ndarray,
        cameras: np.ndarray | None,
        tube_video: np.ndarray,
        cond_mask: np.ndarray | None = None,
        nbi: bool | np.ndarray | None = False,
        caption: str | None = None,
        tube_cfg: float | None = None,
        mask_cfg: float | None = None,
        guidance: float | None = None,
        num_steps: int | None = None,
        seed: int = 0,
        context_frames: np.ndarray | None = None,
        *,
        intrinsics: Mapping[str, Any] | None = None,
        field_mask: np.ndarray | None = None,
        rays: np.ndarray | None = None,
        segment_ids: Sequence[int] | np.ndarray | None = None,
        mask_choice: Mapping[str, str | None] | None = None,
        mask_classes: Mapping[str, np.ndarray] | None = None,
        flags: np.ndarray | None = None,
        return_latents: bool = False,
        rng_seed: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, torch.Tensor]:
        """Generate one chunk.

        Args:
            first_frame: ``uint8`` (H, W, 3) frame the chunk starts from.
            cameras: camera-to-world poses (T, 4, 4) of the chunk (ignored when ``rays`` is given).
            tube_video: rendered tube ``uint8`` (T, H, W, 3) along the same cameras.
            cond_mask: optional palette mask video ``uint8`` (T, H, W, 3).
            nbi: special light: ``True`` (all frames), a per-frame bool array, or ``False``/``None``
                (not specified).
            caption: text prompt (default: the generic colonoscopy caption).
            tube_cfg, mask_cfg, guidance, num_steps: sampling settings (defaults from the config).
            seed: noise seed.
            context_frames: optional ``uint8`` (K, H, W, 3) known prefix, K = 4n+1, replacing
                ``first_frame``.
            intrinsics: KB4 calibration for ``cameras`` (see :func:`.camera.plucker_rays`).
            field_mask: optional (H/16, W/16) boolean aperture applied to the rays.
            rays: precomputed rays (T', 6, H/16, W/16) instead of ``cameras``.
            segment_ids: per-latent-frame colon segment ids (0..5, 6 = unknown).
            mask_choice: per mask channel ``"mask"`` / ``"empty"`` / ``None`` for the global
                flags; by default every channel found in ``cond_mask`` is ``"mask"`` and the others
                ``"empty"``.
            mask_classes: boolean (T, H, W) masks per channel for the global flags (default:
                recovered from the palette).
            flags: explicit global flags (T', 13) overriding ``nbi`` / ``mask_choice``.
            return_latents: also return the final latent.
            rng_seed: seed of the global RNGs (Python, NumPy, torch CPU and CUDA) during the call;
                defaults to ``seed``. The RNG states are restored afterwards.

        Returns:
            ``uint8`` (T, H, W, 3) video, frame 0 being the decoded first frame.
        """
        with seeded_rng(int(seed) if rng_seed is None else int(rng_seed)):
            s = self.cfg["sampling"]
            tube_cfg = float(s["tube_cfg"] if tube_cfg is None else tube_cfg)
            mask_cfg = float(s["mask_cfg"] if mask_cfg is None else mask_cfg)
            guidance = float(s["guidance"] if guidance is None else guidance)
            num_steps = int(s["num_steps"] if num_steps is None else num_steps)
            caption = s["caption"] if caption is None else caption
            frames = context_frames if context_frames is not None else np.asarray(first_frame)[None]
            if tube_video.shape != (self.frames, self.height, self.width, 3):
                raise ValueError(
                    f"tube video must be ({self.frames}, {self.height}, {self.width}, 3)"
                )
            if rays is None:
                if cameras is None or intrinsics is None:
                    raise ValueError("pass cameras with intrinsics, or precomputed rays")
                rays = self.camera_rays(cameras, intrinsics, field_mask)
            num_latent = 1 + (self.frames - 1) // 4

            tube = self.encode_video(tube_video)
            mask = None
            if cond_mask is not None:
                mask = self.encode_video(cond_mask)
            if flags is None:
                choice = mask_choice
                classes = dict(mask_classes or {})
                if cond_mask is not None and not classes:
                    classes = palette_class_masks(cond_mask)
                if cond_mask is not None and choice is None:
                    choice = {c: ("mask" if c in classes else "empty") for c in MASK_CHANNELS}
                flags = global_flags(
                    self.frames, nbi=None if nbi is False else nbi, choice=choice, masks=classes
                )
            state = ConditionState(
                rays=[torch.from_numpy(np.ascontiguousarray(rays)).to(self.device)],
                tube=tube,
                mask=mask,
                flags=(
                    None
                    if flags is None
                    else torch.as_tensor(flags, dtype=torch.float32).to(self.device)
                ),
                segment_ids=(
                    None
                    if segment_ids is None
                    else torch.as_tensor(np.asarray(segment_ids), dtype=torch.long)
                ),
            )
            if state.flags is not None and state.flags.shape != (num_latent, 13):
                raise ValueError(f"global flags must be ({num_latent}, 13)")
            batch = self._batch(np.asarray(frames), caption)
            model = self.backbone
            model.tube_cfg, model.mask_cfg = tube_cfg, mask_cfg
            with model.condition_scope(state):
                result = model.generate_samples_from_batch(
                    batch,
                    seed=[int(seed)],
                    num_steps=num_steps,
                    guidance=guidance,
                    shift=float(s["shift"]),
                    has_negative_prompt=True,
                    use_batched_cfg=False,
                    normalize_cfg=False,
                    guidance_interval=None,
                    upsample_task=None,
                )
            latent = result["vision"][0]
            video = self.decode_latent(latent)
            return (video, latent) if return_latents else video


def load_model(cfg: Mapping[str, Any], device: str = "cuda") -> ColonoWorldModel:
    """Build the model from ``configs/model.yaml`` and load the base and ColonoWorld weights."""
    if device != "cuda":
        raise ValueError("ColonoWorld inference runs on CUDA")
    runtime = cfg["runtime"]
    _set_backend_flags(
        bool(runtime.get("cudnn_benchmark", True)), bool(runtime.get("deterministic", False))
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    backbone = build_backbone_model(cfg)
    weights = cfg["weights"]
    load_base_weights(backbone, weights["base"])
    use = weights.get("use", "ema")
    load_trained(backbone, read_state(weights["checkpoint"], use))
    backbone.eval()
    return ColonoWorldModel(backbone, cfg, device)
