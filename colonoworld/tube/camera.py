# SPDX-License-Identifier: CC-BY-NC-4.0
"""Fisheye camera model used for tube rendering and camera conditioning.

The endoscope is modelled as a Kannala-Brandt (KB4) fisheye lens on the native sensor, followed by a
fixed affine map from native pixels to the 1104x832 clip canvas. A calibration is a dict with:

- ``K_native`` (3x3) and ``D`` (4 KB4 coefficients) of the native sensor,
- ``native_to_clip`` (3x3 affine) and ``K_clip = native_to_clip @ K_native``,
- ``clip_size`` ``[W, H]`` of the canvas, and optionally ``field_radius_px`` (native pixels): rays
  farther than this from the principal point are outside the lens field.

Camera poses are OpenCV camera-to-world matrices (x right, y down, z forward).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

Calibration = dict[str, Any]


@functools.lru_cache(maxsize=16)
def theta_limit(coefficients: tuple[float, ...]) -> float:
    """Largest incidence angle for which the KB4 radius is still monotone."""
    k = np.asarray(coefficients)
    roots = np.polynomial.polynomial.polyroots(
        [1, 0, 3 * k[0], 0, 5 * k[1], 0, 7 * k[2], 0, 9 * k[3]]
    )
    turning = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < np.pi / 2]
    return min([np.pi / 2 - 1e-7] + turning) - 1e-9


def kb4_radius(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    """KB4 distorted radius ``r(theta)`` on the normalised image plane."""
    return theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)


def rays_from_clip(uv: np.ndarray, cam: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Unit viewing rays (camera frame) for clip-canvas pixel coordinates.

    Args:
        uv: ``(N, 2)`` pixel coordinates on the clip canvas (pixel centres are integers).
        cam: calibration dict.

    Returns:
        ``(rays, valid)``: ``(N, 3)`` unit rays (zero where invalid) and an ``(N,)`` validity mask.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(cam["native_to_clip"], dtype=np.float64)
    k = np.asarray(cam["K_native"], dtype=np.float64)
    d = np.asarray(cam["D"], dtype=np.float64)
    if a.shape != (3, 3) or k.shape != (3, 3) or d.shape != (4,):
        raise ValueError("expected a 3x3 affine, a 3x3 K and 4 KB4 coefficients")
    if not np.allclose(a @ k, np.asarray(cam["K_clip"]), atol=1e-5):
        raise ValueError("K_clip must equal native_to_clip @ K_native")
    native = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(a).T
    native = native[:, :2] / native[:, 2:3]
    xy = np.c_[native, np.ones(len(native))] @ np.linalg.inv(k).T
    xy = xy[:, :2] / xy[:, 2:3]
    rd = np.linalg.norm(xy, axis=1)
    hi = np.full(len(rd), theta_limit(tuple(d)))
    lo = np.zeros(len(rd))
    valid = np.isfinite(rd) & (rd < kb4_radius(hi, d))
    if "field_radius_px" in cam:
        valid &= np.linalg.norm(native - k[:2, 2], axis=1) <= cam["field_radius_px"]
    for _ in range(48):  # bisection for theta with kb4_radius(theta) = rd
        mid = (lo + hi) / 2
        less = kb4_radius(mid, d) < rd
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    theta = (lo + hi) / 2
    scale = np.divide(np.sin(theta), rd, out=np.ones_like(rd), where=rd > 1e-12)
    rays = np.c_[xy * scale[:, None], np.cos(theta)]
    rays[~valid] = 0
    return rays, valid


def pixel_grid_rays(cam: Calibration, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rays at the pixel centres of a ``width x height`` grid laid over the clip canvas.

    Returns ``(rays (height, width, 3), valid (height, width))``.
    """
    clip_w, clip_h = cam.get("clip_size", (1104, 832))
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    uv = np.stack([(xx + 0.5) * clip_w / width - 0.5, (yy + 0.5) * clip_h / height - 0.5], -1)
    d, ok = rays_from_clip(uv.reshape(-1, 2), cam)
    return d.reshape(height, width, 3), ok.reshape(height, width)


def load_calibration(path: str | Path) -> Calibration:
    """Load a calibration JSON (either the dict itself or ``{"camera": {...}}``)."""
    data = json.loads(Path(path).read_text())
    return data.get("camera", data) if isinstance(data, dict) else data


def load_camera_path(path: str | Path, key: str = "relative_c2w") -> np.ndarray:
    """Load ``(T, 4, 4)`` camera-to-world poses from an npz written by the pose estimator.

    Poses are expected relative to the first frame (``poses[0] = I``) with translation in units of
    the first frame's median scene depth.
    """
    with np.load(path, allow_pickle=False) as z:
        return z[key].astype(np.float64)


def relative_poses(poses: np.ndarray) -> np.ndarray:
    """Express poses relative to the first one: ``inv(P[0]) @ P[t]``."""
    poses = np.asarray(poses, dtype=np.float64)
    return np.linalg.inv(poses[0])[None] @ poses
