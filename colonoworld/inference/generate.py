# SPDX-License-Identifier: CC-BY-NC-4.0
"""Single-chunk generation: one 189-frame chunk from a first frame, cameras and a tube.

Inputs of a chunk (all at the generated resolution, 24 fps, T = 4n+1 frames):

* ``first_frame``: RGB image (``.png``/``.jpg``) or ``uint8`` ``.npy`` (H, W, 3);
* ``cameras``: ``.npy`` camera-to-world poses (T, 4, 4) and ``intrinsics`` (KB4 calibration JSON);
* ``tube``: either a rendered tube video (``.mp4`` or ``uint8`` ``.npy``) or a saved
  :class:`colonoworld.tube.TubeModel` (``.npz``), rendered here along the cameras;
* optional ``mask``: palette mask video (``.mp4`` / ``.npy``) or a ``.npz`` of boolean class masks;
* optional ``nbi`` (special light on every frame) and ``segments`` (per-frame colon segment names).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.utils.io import save_npz, write_video


@dataclass
class ChunkInputs:
    """Arrays for one chunk; see :meth:`colonoworld.models.ColonoWorldModel.generate`."""

    first_frame: np.ndarray
    cameras: np.ndarray
    intrinsics: dict[str, Any]
    tube_video: np.ndarray
    cond_mask: np.ndarray | None = None
    mask_classes: dict[str, np.ndarray] | None = None
    nbi: bool = False
    segment_ids: np.ndarray | None = None
    field_mask: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _read_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    from colonoworld.utils.io import read_video

    return read_video(path)[0]


def read_image(path: str | Path) -> np.ndarray:
    """RGB ``uint8`` image from ``.npy`` or an image file."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_chunk_inputs(spec: Mapping[str, Any], size: tuple[int, int] = (832, 1104)) -> ChunkInputs:
    """Read the files named in ``spec`` (the ``inputs`` section of ``configs/inference.yaml``)."""
    from colonoworld.models.conditions import segment_ids_from_labels

    cameras = np.load(spec["cameras"])
    intrinsics = json.loads(Path(spec["intrinsics"]).read_text())
    tube_path = Path(spec["tube"])
    if tube_path.suffix == ".npz":
        from colonoworld.tube import TubeModel
        from colonoworld.tube.render import render_tube_video

        tube_video = render_tube_video(TubeModel.load(tube_path), cameras, intrinsics, size)
    else:
        tube_video = _read_frames(tube_path)
    cond_mask, classes = None, None
    if spec.get("mask"):
        mask_path = Path(spec["mask"])
        if mask_path.suffix == ".npz":
            from colonoworld.conditions.palette import MASK_CLASSES, render_palette_video

            with np.load(mask_path) as data:
                classes = {c: data[c].astype(bool) for c in MASK_CLASSES if c in data.files}
            cond_mask = render_palette_video(classes)
        else:
            cond_mask = _read_frames(mask_path)
    segment_ids = None
    if spec.get("segments"):
        labels = spec["segments"]
        if isinstance(labels, str):
            labels = [labels] * len(cameras)
        segment_ids = segment_ids_from_labels(labels)
    field_mask = np.load(spec["field_mask"]) if spec.get("field_mask") else None
    return ChunkInputs(
        first_frame=read_image(spec["first_frame"]),
        cameras=cameras,
        intrinsics=intrinsics,
        tube_video=tube_video,
        cond_mask=cond_mask,
        mask_classes=classes,
        nbi=bool(spec.get("nbi", False)),
        segment_ids=segment_ids,
        field_mask=field_mask,
    )


def generate_chunk(
    model: Any, chunk: ChunkInputs, sampling: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Generate one chunk with ``model`` (:class:`colonoworld.models.ColonoWorldModel`)."""
    s = dict(sampling or {})
    return model.generate(
        chunk.first_frame,
        chunk.cameras,
        chunk.tube_video,
        cond_mask=chunk.cond_mask,
        nbi=chunk.nbi,
        caption=s.get("caption"),
        tube_cfg=s.get("tube_cfg"),
        mask_cfg=s.get("mask_cfg"),
        guidance=s.get("guidance"),
        num_steps=s.get("num_steps"),
        seed=int(s.get("seed", 0)),
        intrinsics=chunk.intrinsics,
        field_mask=chunk.field_mask,
        segment_ids=chunk.segment_ids,
        mask_classes=chunk.mask_classes,
    )


def run(cfg: Mapping[str, Any]) -> Path:
    """Load the model, generate the chunk described by ``cfg`` and write the outputs.

    Writes ``<output.dir>/generated.mp4`` (and ``tube.mp4`` / ``mask.mp4`` previews when
    ``output.previews`` is set) and returns the output directory.
    """
    from colonoworld.models import load_model

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    video_cfg = cfg["video"]
    chunk = load_chunk_inputs(cfg["inputs"], (int(video_cfg["height"]), int(video_cfg["width"])))
    model = load_model(cfg)
    video = generate_chunk(model, chunk, cfg.get("sampling"))
    fps = float(video_cfg["fps"])
    write_video(out / "generated.mp4", video, fps=fps)
    if cfg["output"].get("previews", True):
        write_video(out / "tube.mp4", chunk.tube_video, fps=fps)
        if chunk.cond_mask is not None:
            write_video(out / "mask.mp4", chunk.cond_mask, fps=fps)
    if cfg["output"].get("save_frames", False):
        save_npz(out / "generated.npz", video=video)
    return out
