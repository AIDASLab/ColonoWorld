# SPDX-License-Identifier: CC-BY-NC-4.0
"""Joint-conditioning fine-tuning loop.

One update = ``world_size x microbatches`` draws (8 in the released recipe), each draw a single
video window. Only the LoRA adapters of the generation expert and the ColonoWorld condition
modules are trained; the Cosmos backbone, the camera adapter and the VAE stay frozen.

Per draw the trainer

1. builds the CPU sample (:class:`~colonoworld.training.data.JointDataset`),
2. VAE-encodes the augmented tube video and, when present, the condition-mask palette video,
3. draws the tube/camera condition dropout (full .85 / null tube .10 / null tube and camera .05),
4. runs the flow-matching loss of the Cosmos model under these conditions, with the irrigation
   loss weights and the corrupted conditioning-frame latent applied inside the framework loss.

Optimisation: AdamW (betas .9/.999, eps 1e-8, no weight decay), learning rates 1e-4 for the
condition tokens, 1e-3 for the mask-stream input projection and the global-flag adapter, 2e-5 for
LoRA, 100-update linear warmup, global gradient-norm clipping at 1, EMA decay .99.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colonoworld.training import data as D

TRAINABLE = ("lora_", "tube_stream.", "mask_stream.", "global_adapter.")
PROJECTORS = ("mask_stream.input.weight", "global_adapter.weight")


@dataclass
class OptimConfig:
    lr_tokens: float = 1e-4
    lr_projectors: float = 1e-3
    lr_lora: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_updates: int = 100
    clip_norm: float = 1.0
    ema_decay: float = 0.99


@dataclass
class RunConfig:
    updates: int = 750
    start_update: int = 0
    microbatches: int = 1
    checkpoint_every: int = 250
    iteration_offset: int = 1200
    output_dir: str = "outputs/joint"
    init_checkpoint: str | None = None
    optim: OptimConfig = field(default_factory=OptimConfig)


# --------------------------------------------------------------------------------------------
# Parameters, optimizer, EMA
# --------------------------------------------------------------------------------------------


def _local(p: torch.Tensor) -> torch.Tensor:
    return p.to_local() if hasattr(p, "to_local") else p


def freeze(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Enable gradients only for LoRA and the condition modules; return the trainables."""
    train = {}
    for name, p in model.named_parameters():
        p.requires_grad_(any(key in name for key in TRAINABLE))
        if p.requires_grad:
            train[name] = p
    if not train:
        raise ValueError("no trainable parameters found")
    return train


def param_group(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if name.endswith(PROJECTORS):
        return "projectors"
    return "tokens"


def make_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    """AdamW with three learning-rate groups (tokens, projectors x10, LoRA)."""
    train = freeze(model)
    lrs = dict(tokens=cfg.lr_tokens, projectors=cfg.lr_projectors, lora=cfg.lr_lora)
    groups = []
    for kind, lr in lrs.items():
        names = [n for n in train if param_group(n) == kind]
        if names:
            groups.append(
                dict(params=[train[n] for n in names], param_names=names, lr=lr, initial_lr=lr)
            )
    return torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0, foreach=False)


class EMA:
    """Float32 exponential moving average of the trainable parameters (local shards)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        self.values = {
            n: _local(p).detach().float().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.values:
                self.values[n].lerp_(_local(p).detach().float(), 1.0 - self.decay)
        self.updates += 1


def set_learning_rate(opt: torch.optim.Optimizer, update: int, start: int, warmup: int) -> None:
    scale = min((update - start) / float(warmup), 1.0) if warmup > 0 else 1.0
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * scale


def clip_gradients(params: list[torch.nn.Parameter], max_norm: float) -> float:
    """Global gradient norm over all ranks; gradients scaled by ``min(1, max_norm/(norm+1e-6))``."""
    import torch.distributed as dist

    device = params[0].device
    squared = sum(
        (_local(p.grad).detach().float().square().sum() for p in params),
        torch.zeros((), device=device),
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(squared)
    norm = math.sqrt(float(squared))
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    with torch.no_grad():
        for p in params:
            _local(p.grad).mul_(min(1.0, max_norm / (norm + 1e-6)))
    return norm


# --------------------------------------------------------------------------------------------
# One draw through the Cosmos model
# --------------------------------------------------------------------------------------------


def encode_video(backbone: Any, pixels: np.ndarray) -> torch.Tensor:
    """VAE latent [C,t,h,w] of a uint8 [T,H,W,3] video."""
    x = torch.from_numpy(np.ascontiguousarray(pixels)).permute(3, 0, 1, 2)
    device = next(backbone.parameters()).device
    with torch.no_grad():
        return backbone.encode(x[None].to(device=device, dtype=torch.float32) / 127.5 - 1.0)[0]


def framework_batch(sample: D.TrainingDraw, tokenizer: Any) -> dict[str, Any]:
    """Cosmos data batch of one draw (video conditioned on its first frame)."""
    from cosmos_framework.data.generator.sequence_packing import SequencePlan

    n, h, w, _ = sample.video.shape
    return dict(
        video=[torch.from_numpy(sample.video).permute(3, 0, 1, 2)],
        image_size=[torch.tensor([[h, w, h, w]])],
        ai_caption=[sample.caption],
        text_token_ids=torch.tensor(
            tokenizer.tokenize_text(sample.caption, is_video=False, use_system_prompt=False),
            dtype=torch.long,
        )[None],
        conditioning_fps=[torch.tensor(D.FPS)],
        video_num_frames=[torch.tensor(n)],
        sequence_plan=[
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=False,
                condition_frame_indexes_vision=[0],
            )
        ],
    )


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


@contextmanager
def weighted_loss(weights: torch.Tensor | None) -> Iterator[list[float]]:
    """Multiply the per-element flow-matching error of the target video by ``weights``.

    Implemented as ``pred <- target + sqrt(W) (pred - target)`` inside the framework's
    ``compute_flow_matching_loss``, so time weighting and normalisation are unchanged.
    """
    calls: list[float] = []
    if weights is None:
        yield calls
        return
    modules = [
        m
        for n, m in list(sys.modules.items())
        if n.startswith("cosmos_framework.model.generator")
        and m is not None
        and hasattr(m, "compute_flow_matching_loss")
    ]
    saved = [(m, m.compute_flow_matching_loss) for m in modules]

    def wrap(original):
        def wrapped(*args, **kwargs):
            pred = kwargs["pred"] if "pred" in kwargs else args[0]
            target = kwargs["target"] if "target" in kwargs else args[1]
            root = weights.reshape((1,) * (pred[0].dim() - 3) + tuple(weights.shape[-3:]))
            root = root.to(device=pred[0].device, dtype=pred[0].dtype).sqrt()
            new = [target[0] + root * (pred[0] - target[0])] + list(pred[1:])
            if "pred" in kwargs:
                kwargs = dict(kwargs, pred=new)
            else:
                args = (new,) + tuple(args[1:])
            calls.append(float(weights.mean()))
            return original(*args, **kwargs)

        return wrapped

    try:
        for m, f in saved:
            m.compute_flow_matching_loss = wrap(f)
        yield calls
    finally:
        for m, f in saved:
            m.compute_flow_matching_loss = f
    if len(calls) != 1:
        raise RuntimeError("the loss weight must be applied exactly once per draw")


@contextmanager
def corrupted_first_latent(backbone: Any, frame: torch.Tensor | None, params: dict[str, Any]):
    """Replace the latent of the conditioning frame of the target by a corrupted one."""
    from colonoworld.training.augmentation import conditioning_latent

    if frame is None or params["mode"] == "clean":
        yield
        return
    name = "_encode_vision_x0_tokens"
    original = getattr(backbone, name)

    @torch.no_grad()
    def encode(*args, **kwargs):
        clean = original(*args, **kwargs)
        z0 = conditioning_latent(frame, backbone.encode, backbone.decode, params)
        result = clean[0].clone()
        result[:, :, :1].copy_(z0.to(result))
        return [result]

    setattr(backbone, name, encode)
    try:
        yield
    finally:
        delattr(backbone, name)


@contextmanager
def linked_unused_streams(backbone: Any):
    """Keep absent optional streams (mask stream, global-flag adapter) in the autograd graph
    with an exact zero term, so every rank produces gradients for the same parameters."""
    net = backbone.net
    original = net._encode_vision

    def encode_vision(packed_seq, packed_sequence, target_dtype):
        shapes = original(packed_seq, packed_sequence, target_dtype)
        if torch.is_grad_enabled():
            state = net.colonoworld_state
            unused = [
                m
                for m, absent in (
                    (net.mask_stream, state.mask is None),
                    (net.global_adapter, state.flags is None),
                )
                if absent
            ]
            zero = None
            for module in unused:
                for p in module.parameters():
                    if p.requires_grad:
                        term = p.reshape(-1)[:1].sum() * 0
                        zero = term if zero is None else zero + term
            if zero is not None:
                packed_sequence[:1] = packed_sequence[:1] + zero.to(packed_sequence.dtype)
        return shapes

    net._encode_vision = encode_vision
    try:
        yield
    finally:
        del net._encode_vision


def draw_loss(
    backbone: Any,
    tokenizer: Any,
    sample: D.TrainingDraw,
    kind: str,
    iteration: int,
    noise_seed: int = 202610060,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Flow-matching loss of one draw under its condition-dropout ``kind``.

    The flow-matching time and noise are drawn from RNGs seeded with ``noise_seed + iteration``.
    """
    from colonoworld.models.loader import seeded_rng
    from colonoworld.models.network import ConditionState

    device = next(backbone.parameters()).device
    tube = None
    if kind == "full" and sample.tube_latent:
        tube = torch.load(sample.tube_latent, map_location=device).float()
    elif kind == "full":
        tube = encode_video(backbone, sample.tube)
    mask = encode_video(backbone, sample.mask_video) if sample.mask_video is not None else None
    rays = torch.from_numpy(sample.rays).to(device)
    if kind == "null_both":
        rays = torch.zeros_like(rays)
    t = rays.shape[0]
    weights = None
    if sample.loss_weights is not None:
        weights = D.loss_weight_latent(sample.loss_weights, (0, t, *rays.shape[-2:])).to(device)
    state = ConditionState(
        rays=[rays],
        tube=tube,
        mask=mask,
        flags=(
            None
            if sample.global_flags is None
            else torch.from_numpy(sample.global_flags).to(device)
        ),
        segment_ids=(
            torch.tensor(sample.segment_ids, dtype=torch.long, device=device)
            if kind == "full"
            else None
        ),
    )
    batch = _to_device(framework_batch(sample, tokenizer), device)
    frame = batch["video"][0][:, :1] if sample.handoff["mode"] != "clean" else None
    with (
        backbone.condition_scope(state),
        linked_unused_streams(backbone),
        weighted_loss(weights),
        corrupted_first_latent(
            backbone, None if frame is None else frame.to(device), sample.handoff
        ),
        seeded_rng(noise_seed + iteration),
    ):
        _, loss = backbone.training_step(batch, iteration)
    info = dict(id=sample.window_id, stratum=sample.stratum, kind=kind, choice=sample.choice)
    return loss, info


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def save_checkpoint(
    path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, ema: EMA, update: int
) -> None:
    """Trainable raw and EMA weights plus optimizer state (rank-local shards)."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if rank == 0 else path.with_suffix(f".rank{rank:03d}.pt")
    state = dict(
        update=update,
        raw={
            n: _local(p).detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        },
        ema={n: v.cpu() for n, v in ema.values.items()},
        ema_decay=ema.decay,
        ema_updates=ema.updates,
        optimizer=opt.state_dict(),
    )
    tmp = target.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(target)


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, opt: torch.optim.Optimizer | None, ema: EMA
) -> int:
    """Load raw weights (and optimizer state when given) and the EMA; returns the update."""
    from colonoworld.models.checkpoint import canonical_name

    state = torch.load(path, map_location="cpu", weights_only=False)
    params = {canonical_name(n): p for n, p in model.named_parameters()}
    names = {canonical_name(n): n for n in ema.values}
    with torch.no_grad():
        for n, v in state["raw"].items():
            if canonical_name(n) in params:
                p = _local(params[canonical_name(n)])
                p.copy_(v.to(p))
    for n, v in state["ema"].items():
        if canonical_name(n) in names:
            ema.values[names[canonical_name(n)]].copy_(v)
    ema.updates = int(state.get("ema_updates", state.get("update", 0)))
    if opt is not None and "optimizer" in state:
        opt.load_state_dict(state["optimizer"])
    return int(state.get("update", 0))


def _copy_full_(param: torch.Tensor, value: torch.Tensor) -> None:
    """Copy a full logical value into a plain or FSDP-sharded (DTensor) parameter."""
    value = value.to(device=param.device, dtype=param.dtype)
    if hasattr(param, "device_mesh"):
        from torch.distributed.tensor import distribute_tensor

        value = distribute_tensor(value, param.device_mesh, param.placements)
    param.copy_(value)


@torch.no_grad()
def _reset_condition_tokens(self: Any, seed: int = 0) -> None:
    """Initialisation of the condition-token streams that also works on sharded parameters."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.nn.init.zeros_(self.input.weight)
    for param in (self.time.weight, self.height.weight, self.width.weight, self.type_embedding):
        _copy_full_(param, torch.randn(param.shape, generator=generator) * 0.01)
    if self.segment_embedding is not None:
        torch.nn.init.zeros_(self.segment_embedding.weight)


def load_training_model(model_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Cosmos model with fp32 master weights and activation checkpointing, base weights loaded,
    and the framework text tokenizer."""
    import copy

    import hydra
    import omegaconf

    from colonoworld.models.loader import (
        _set_backend_flags,
        build_backbone_model,
        load_base_weights,
    )

    cfg = copy.deepcopy(model_cfg)
    runtime = cfg.get("runtime", {})
    _set_backend_flags(
        bool(runtime.get("cudnn_benchmark", True)), bool(runtime.get("deterministic", False))
    )
    cfg.setdefault("runtime", {})["bf16_weights"] = False
    cfg["framework"]["activation_checkpointing"]["mode"] = "full"
    from colonoworld.models.conditions import ConditionTokens

    reset = ConditionTokens.reset_parameters
    ConditionTokens.reset_parameters = _reset_condition_tokens
    try:
        backbone = build_backbone_model(cfg).to("cuda")
    finally:
        ConditionTokens.reset_parameters = reset
    # Training uses the plain conditional denoiser; tube/mask guidance is a sampling-time option.
    from types import MethodType

    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    backbone.denoise = MethodType(OmniMoTModel.denoise, backbone)
    backbone.on_train_start(torch.preserve_format)
    load_base_weights(backbone, cfg["weights"]["base"])
    tokenizer = hydra.utils.instantiate(
        omegaconf.OmegaConf.create(cfg["framework"]["vlm_config"]["tokenizer"])
    )
    return backbone, tokenizer


# --------------------------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------------------------


def train(cfg: dict[str, Any]) -> None:
    """Run joint-conditioning training (``torchrun --nproc_per_node=N scripts/train.py``)."""
    import torch.distributed as dist

    run = cfg["run"]
    if "WORLD_SIZE" in __import__("os").environ and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(int(__import__("os").environ.get("LOCAL_RANK", 0)))

    model_cfg = cfg["model"]
    if isinstance(model_cfg, str):
        from colonoworld.utils.config import load_config

        model_cfg = load_config(model_cfg)
    backbone, tokenizer = load_training_model(model_cfg)
    opt = make_optimizer(backbone, OptimConfig(**cfg.get("optim", {})))
    ema = EMA(backbone, cfg.get("optim", {}).get("ema_decay", 0.99))
    start = int(run.get("start_update", 0))
    if run.get("init_checkpoint"):
        start = load_checkpoint(run["init_checkpoint"], backbone, opt, ema)

    windows = D.read_manifest(cfg["data"]["manifest"])
    D.mark_top_motion(windows, cfg["data"].get("top_motion_fraction", 0.3))
    sampler_cfg = D.SamplerConfig(**cfg.get("sampler", {}))
    aug = cfg.get("augmentation", {})
    dataset = D.JointDataset(
        windows,
        sampler_cfg,
        augment=aug.get("enabled", True),
        static_schedule=str(aug.get("static_schedule", "0.10")),
        start_update=start,
        handoff=aug.get("handoff_corruption", True),
        loss_weight=cfg.get("loss", {}).get("irrigation_weight", 3.0),
        loss_dilate_px=cfg.get("loss", {}).get("irrigation_dilate_px", 25),
    )
    micro = int(run.get("microbatches", 1))
    end = start + int(run["updates"])
    out = Path(run["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    warmup = int(cfg.get("optim", {}).get("warmup_updates", 100))
    clip = float(cfg.get("optim", {}).get("clip_norm", 1.0))
    offset = int(run.get("iteration_offset", 0))
    log = (out / ("updates.jsonl" if rank == 0 else f"updates.rank{rank:03d}.jsonl")).open("a")
    started = time.monotonic()
    for update in range(start + 1, end + 1):
        set_learning_rate(opt, update, start, warmup)
        opt.zero_grad(set_to_none=True)
        backbone.train()
        kind = D.dropout_kind(sampler_cfg.dropout_seed, update, 0)
        losses, infos = [], []
        for m in range(micro):
            draw = rank + m * world
            sample = dataset(update, draw)
            iteration = offset + (update - 1) * world * micro + draw
            loss, info = draw_loss(
                backbone, tokenizer, sample, kind, iteration, int(run.get("noise_seed", 202610060))
            )
            (loss / micro).backward()
            losses.append(float(loss.detach()))
            infos.append(info)
        params = [p for p in backbone.parameters() if p.requires_grad and p.grad is not None]
        norm = clip_gradients(params, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(backbone)
        record = dict(
            update=update,
            loss=sum(losses) / micro,
            grad_norm=norm,
            draws=infos,
            seconds=time.monotonic() - started,
        )
        log.write(json.dumps(record) + "\n")
        log.flush()
        if update % int(run.get("checkpoint_every", 250)) == 0 or update == end:
            save_checkpoint(out / f"checkpoint_{update:06d}.pt", backbone, opt, ema, update)
    log.close()
