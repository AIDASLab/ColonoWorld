# SPDX-License-Identifier: CC-BY-NC-4.0
"""Video and array I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_video(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Read a video as ``uint8`` RGB frames ``(T, H, W, 3)`` and return it with its frame rate."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), float(fps)


def write_video(path: str | Path, frames: np.ndarray, fps: float = 30.0) -> None:
    """Write ``uint8`` RGB frames ``(T, H, W, 3)`` to an H.264 mp4 (falls back to mp4v)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            break
    for frame in frames:
        writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays to a compressed ``.npz``, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
