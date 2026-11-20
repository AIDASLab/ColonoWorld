# SPDX-License-Identifier: CC-BY-NC-4.0
"""Camera conditioning: Pluecker rays, the input ray adapter and the per-block ray encoders.

Rays are computed on the VAE latent grid (16 px cells) with a Kannala-Brandt (KB4) fisheye model.
Each ray is ``[m, d]`` with ``d`` the world-space unit direction and ``m = o x d`` its moment,
relative to the first camera of the clip (OpenCV axes, no scale normalisation). Latent frame ``i``
takes the pose of output frame ``0`` for ``i = 0`` and frame ``4 i`` otherwise.

The camera signal enters the network in two places:

* the input adapter adds ``proj(rays)`` to every target video token after the vision projection;
* each of the 36 transformer blocks adds ``encoder_i(rays)`` to the post-norm generation tokens
  right before self-attention. Latent frame 0 (the given first frame) receives no block signal.

Both additions go through :func:`bounded_delta`, which rescales the camera term so that its RMS
never exceeds 10 % of the receiving tokens' RMS.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch
from torch import nn

MOMENT_DIVISOR = 5.1
BOUND = 0.1
_CHUNK = 1024


# ----------------------------------------------------------------------------- geometry
@functools.lru_cache(maxsize=16)
def _theta_limit(coefficients: tuple[float, ...]) -> float:
    k = np.asarray(coefficients)
    roots = np.polynomial.polynomial.polyroots(
        [1, 0, 3 * k[0], 0, 5 * k[1], 0, 7 * k[2], 0, 9 * k[3]]
    )
    turning = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < np.pi / 2]
    return min([np.pi / 2 - 1e-7] + turning) - 1e-9


def _kb4_radius(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    return theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)


def unproject_kb4(uv: np.ndarray, calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Unit camera rays for clip pixel coordinates ``uv`` (N, 2).

    ``calibration`` holds ``K_native`` (3x3), KB4 coefficients ``D`` (4,), the affine
    ``native_to_clip`` (3x3) from native sensor pixels to clip pixels, and optionally
    ``field_radius_px`` (circular field of view in native pixels).

    Returns:
        ``(rays, valid)``: rays (N, 3), zero where invalid, and the boolean validity (N,).
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    affine = np.asarray(calibration["native_to_clip"], dtype=np.float64)
    k_mat = np.asarray(calibration["K_native"], dtype=np.float64)
    dist = np.asarray(calibration["D"], dtype=np.float64)
    if affine.shape != (3, 3) or k_mat.shape != (3, 3) or dist.shape != (4,):
        raise ValueError("calibration needs native_to_clip (3x3), K_native (3x3) and D (4,)")
    native = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(affine).T
    native = native[:, :2] / native[:, 2:3]
    xy = np.c_[native, np.ones(len(native))] @ np.linalg.inv(k_mat).T
    xy = xy[:, :2] / xy[:, 2:3]
    rd = np.linalg.norm(xy, axis=1)
    hi = np.full(len(rd), _theta_limit(tuple(dist)))
    lo = np.zeros(len(rd))
    valid = np.isfinite(rd) & (rd < _kb4_radius(hi, dist))
    if "field_radius_px" in calibration:
        valid &= np.linalg.norm(native - k_mat[:2, 2], axis=1) <= calibration["field_radius_px"]
    for _ in range(48):  # bisection for theta with radius(theta) = rd
        mid = (lo + hi) / 2
        less = _kb4_radius(mid, dist) < rd
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    theta = (lo + hi) / 2
    scale = np.divide(np.sin(theta), rd, out=np.ones_like(rd), where=rd > 1e-12)
    rays = np.c_[xy * scale[:, None], np.cos(theta)]
    rays[~valid] = 0
    return rays, valid


def latent_time_indices(num_frames: int) -> np.ndarray:
    """Output frames whose poses represent each latent frame: ``0, 4, 8, ...``."""
    if num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError("the causal video VAE needs 4n+1 frames")
    return np.r_[0, np.arange(4, num_frames, 4)].astype(np.int64)


def plucker_rays(
    c2w: np.ndarray,
    calibration: dict[str, Any],
    image_size: tuple[int, int] = (832, 1104),
    stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Pluecker rays on the latent grid.

    Args:
        c2w: camera-to-world poses (T, 4, 4), T = 4n+1; translation in any consistent unit.
        calibration: KB4 calibration (see :func:`unproject_kb4`) with ``clip_size``,
            the (width, height) of the clip the calibration refers to.
        image_size: generated frame size (height, width).
        stride: VAE spatial stride.

    Returns:
        ``(rays, valid)``: float32 rays (T_latent, 6, H/stride, W/stride), channels ``[m, d]``,
        and the boolean (H/stride, W/stride) map of cells with a valid ray.
    """
    poses = np.asarray(c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("c2w must be finite (T, 4, 4)")
    rot = poses[:, :3, :3]
    if not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-5) or not np.allclose(
        np.linalg.det(rot), 1, atol=1e-5
    ):
        raise ValueError("c2w rotations must be orthonormal")
    height, width = map(int, image_size)
    clip_w, clip_h = map(int, calibration["clip_size"])
    if height % stride or width % stride:
        raise ValueError("image size must be a multiple of the VAE stride")
    gh, gw = height // stride, width // stride
    yy, xx = np.mgrid[:gh, :gw]
    u = (xx + 0.5) * stride - 0.5
    v = (yy + 0.5) * stride - 0.5
    uv = np.stack([(u + 0.5) * clip_w / width - 0.5, (v + 0.5) * clip_h / height - 0.5], -1)
    directions, valid = unproject_kb4(uv.reshape(-1, 2), calibration)
    valid &= ((u < width - 0.5) & (v < height - 0.5)).ravel()
    directions[~valid] = 0
    relative = np.linalg.inv(poses[0])[None] @ poses
    selected = latent_time_indices(len(poses))
    world = np.einsum("tij,pj->tpi", relative[selected, :3, :3], directions)
    moment = np.cross(relative[selected, :3, 3][:, None, :], world)
    rays = np.concatenate([moment, world], axis=-1).reshape(len(selected), gh, gw, 6)
    rays = rays.transpose(0, 3, 1, 2).astype(np.float32)
    if not np.isfinite(rays).all() or not valid.any():
        raise ValueError("no valid finite camera rays")
    return rays, valid.reshape(gh, gw)


# ----------------------------------------------------------------------------- bounded addition
class _BoundedDelta(torch.autograd.Function):
    """``0.1 * r * z / sqrt(r^2 + mean(z^2))`` per token, ``r`` = RMS of the receiving token."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, receiving: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if z.shape != receiving.shape or z.ndim != 2:
            raise ValueError("expected two token matrices of equal shape")
        parts, rs, denoms = [], [], []
        for a, h in zip(z.split(_CHUNK), receiving.detach().split(_CHUNK), strict=True):
            a, h = a.float(), h.float()
            r = h.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            denom = (r.square() + a.square().mean(-1, keepdim=True)).sqrt()
            parts.append((BOUND * r * a / denom).to(z.dtype))
            rs.append(r)
            denoms.append(denom)
        ctx.save_for_backward(z, torch.cat(rs), torch.cat(denoms))
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        z, r, denom = ctx.saved_tensors
        parts = []
        for a, g, rr, dd in zip(
            z.split(_CHUNK), grad.split(_CHUNK), r.split(_CHUNK), denom.split(_CHUNK), strict=True
        ):
            a, g = a.float(), g.float()
            dz = BOUND * rr / dd * (g - a * (g * a).mean(-1, keepdim=True) / dd.square())
            parts.append(dz.to(z.dtype))
        return torch.cat(parts), None


def bounded_delta(z: torch.Tensor, receiving: torch.Tensor) -> torch.Tensor:
    """Camera addition for ``receiving`` tokens, bounded to 10 % of their RMS."""
    return _BoundedDelta.apply(z, receiving)


# ----------------------------------------------------------------------------- modules
class PluckerEmbedder(nn.Module):
    """Linear map from 2x2 patches of 6-channel rays to the transformer width (zero initialised)."""

    def __init__(self, hidden_size: int = 4096, patch: int = 2):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(6 * patch * patch, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)

    def patchify(self, rays: torch.Tensor) -> torch.Tensor:
        """(T, 6, H, W) rays -> (T * H/p * W/p, 6 p^2) in video patch order (t, y, x, py, px, c)."""
        if rays.ndim == 5 and rays.shape[0] == 1:
            rays = rays[0]
        if rays.ndim != 4 or rays.shape[1] != 6:
            raise ValueError("expected rays of shape (T, 6, H, W)")
        rays = torch.cat((rays[:, :3] / MOMENT_DIVISOR, rays[:, 3:]), dim=1)
        t, c, h, w = rays.shape
        p = self.patch
        if h % p or w % p:
            raise ValueError("ray grid must align with the patch grid")
        rays = rays.permute(1, 0, 2, 3).reshape(c, t, h // p, p, w // p, p)
        return rays.permute(1, 2, 4, 3, 5, 0).reshape(-1, p * p * c)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        return self.proj(self.patchify(rays))


class PerBlockEncoders(nn.ModuleList):
    """One zero-initialised ``Linear(24, width)`` ray encoder per transformer block."""

    def __init__(self, num_blocks: int = 36, width: int = 4096, patch: int = 2):
        super().__init__(
            [nn.Linear(6 * patch * patch, width, bias=False) for _ in range(num_blocks)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for encoder in self:
            nn.init.zeros_(encoder.weight)


def pad_rays(rays: torch.Tensor, shape: tuple[int, int, int], patch: int) -> torch.Tensor:
    """Crop rays to the latent grid ``shape`` = (t, h, w) and zero-pad to the patch grid."""
    if rays.ndim == 5 and rays.shape[0] == 1:
        rays = rays[0]
    t, h, w = map(int, shape)
    if rays.shape[0] != t or rays.shape[2] < h or rays.shape[3] < w:
        raise ValueError(f"camera rays {tuple(rays.shape)} do not cover the video latent {shape}")
    return torch.nn.functional.pad(rays[:, :, :h, :w], (0, (-w) % patch, 0, (-h) % patch))


def block_payload(
    adapter: PluckerEmbedder, rays: torch.Tensor, shape: tuple[int, int, int], indexes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchified rays for the per-block encoders, without latent frame 0.

    Returns:
        ``(sequence_indexes, ray_patches)`` of the target tokens that receive a block signal.
    """
    t, h, w = map(int, shape)
    p = adapter.patch
    tokens = adapter.patchify(pad_rays(rays, shape, p)).detach()
    spatial = ((h + p - 1) // p) * ((w + p - 1) // p)
    keep = torch.arange(len(tokens), device=tokens.device) >= spatial
    if len(tokens) != len(indexes):
        raise ValueError("camera and video token counts differ")
    return indexes.to(tokens.device)[keep].detach().clone(), tokens[keep].clone()
