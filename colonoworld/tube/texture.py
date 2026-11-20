# SPDX-License-Identifier: CC-BY-NC-4.0
"""Procedural mucosa texture for tube rendering.

The texture is fully synthetic (no real frames): seeded multi-scale Gaussian noise modulates a
neutral mucosa colour, and faint curvilinear vessels are drawn as random walks. Every component
wraps around the image borders, so the texture tiles seamlessly on the wall chart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MUCOSA_SEED = 20261024
BASE_COLOUR = (0.56, 0.235, 0.185)  # linear RGB
VESSEL_TINT = (0.15, 0.40, 0.35)  # per-channel darkening along vessels


def make_mucosa_texture(seed: int = MUCOSA_SEED, size: int = 1024) -> np.ndarray:
    """Generate the ``(size, size, 3)`` uint8 RGB (gamma 1/2.2) mucosa texture."""
    import cv2
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    n = size
    bands = []
    for sigma in (3.0, 9.0, 27.0, 80.0):
        a = gaussian_filter(rng.normal(size=(n, n)).astype(np.float32), sigma, mode="wrap")
        bands.append(a / max(float(a.std()), 1e-8))
    pattern = 0.13 * bands[0] + 0.16 * bands[1] + 0.13 * bands[2] + 0.06 * bands[3]
    vessels = np.zeros((n, n), np.float32)
    for _ in range(85):
        start = rng.uniform(0, n, 2)
        angle = rng.uniform(-np.pi, np.pi)
        pts = [start.copy()]
        for _ in range(int(rng.integers(8, 32))):
            angle += rng.normal(0, 0.22)
            pts.append(pts[-1] + rng.uniform(3, 8) * np.array([np.cos(angle), np.sin(angle)]))
        pts = np.asarray(pts)
        for dx in (-n, 0, n):
            for dy in (-n, 0, n):
                cv2.polylines(
                    vessels,
                    [np.rint(pts + [dx, dy]).astype(np.int32)],
                    False,
                    float(rng.uniform(0.35, 0.65)),
                    int(rng.integers(1, 3)),
                    cv2.LINE_AA,
                )
    vessels = gaussian_filter(vessels, 1.2, mode="wrap")
    base = np.array(BASE_COLOUR, np.float32)
    linear = base[None, None] * np.clip(1 + pattern[:, :, None], 0.48, 1.55)
    linear *= 1 - vessels[:, :, None] * np.array(VESSEL_TINT)[None, None]
    return np.rint(255 * np.clip(linear, 0, 1) ** (1 / 2.2)).astype(np.uint8)


def save_mucosa_texture(path: str | Path, seed: int = MUCOSA_SEED, size: int = 1024) -> None:
    """Write the texture as a PNG."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(make_mucosa_texture(seed, size), cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")
