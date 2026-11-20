# SPDX-License-Identifier: CC-BY-NC-4.0
"""Signed-distance reconstruction and lumen carve of a training window (seed steps 2 and 3).

The fused wall evidence of :func:`colonoworld.tube.seed_depth.collect_evidence` is turned into one
static signed-distance field (SDF, negative inside the lumen) on a voxel grid in units of the
reference radius ``R``:

1. **Block fields**: every 8th window frame, wall depths within 8 frames vote a signed screened
   field; lumen rays and a camera clearance ball (0.18 R) are hard free space; a harmonic
   (Jacobi) solve fills the unobserved volume.
2. **Joint field**: the median of the block fields initialises one jointly screened solve over
   all observations.
3. **Regularised field**: heat-kernel smoothing at 0.28 R with fine detail kept where at least
   three keyframes agree; hard free space projected back; only the camera's connected free
   volume is kept.
4. **Envelope**: support-aware refinement (fine detail needs three views within 0.12 R) and a
   smooth outward union with the free-space envelope and the camera clearance.
5. **Lumen carve** (:func:`carve_lumen`): one elliptical frustum fitted to the largest dark
   component of every frame (robust world bearing axis, common section at 3 R) is united with
   the field, and a completion shell (7.4 R balls around all cameras) closes far cavities.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from colonoworld.tube.build_from_video import _gray, specular_mask
from colonoworld.tube.camera import Calibration, rays_from_clip
from colonoworld.tube.seed_depth import (
    DEPTH_H,
    DEPTH_W,
    SMALL,
    decode_small,
    pixel_rays,
    project_to_clip,
    small_field,
)

CAMERA_BALL = 0.18  # camera clearance radius (R)
COMPLETION_R = 7.4  # far completion shell around every camera (R)


# ----------------------------------------------------------------------------- grid helpers
def grid_points(shape: Any, lo: np.ndarray, step: float) -> np.ndarray:
    """Voxel centres ``(N, 3)`` float32 of a grid."""
    return (np.indices(shape, dtype=np.float32).reshape(3, -1).T * step + lo).astype(np.float32)


def project_grid(x: np.ndarray, e: dict, k_depth: np.ndarray, radius: float) -> tuple:
    """View-axis depth and depth-grid pixels of points ``x`` seen from observation ``e``."""
    q = e["Q"]
    xc = (x - q[:3, 3] / radius) @ q[:3, :3]
    z = xc[:, 2]
    u = np.rint(np.clip(xc[:, 0] / np.maximum(z, 1e-8) * k_depth[0, 0] + k_depth[0, 2], -1e4, 1e4))
    v = np.rint(np.clip(xc[:, 1] / np.maximum(z, 1e-8) * k_depth[1, 1] + k_depth[1, 2], -1e4, 1e4))
    u, v = u.astype(np.int32), v.astype(np.int32)
    m = (z > 0.02) & (u >= 0) & (u < DEPTH_W) & (v >= 0) & (v < DEPTH_H)
    ii = np.flatnonzero(m)
    return z, ii, u[ii], v[ii]


def lumen_free(x: np.ndarray, e: dict, radius: float, cam: Calibration) -> np.ndarray:
    """Points inside the dark-lumen rays of ``e`` (fisheye camera, cue grid) before its cutoff."""
    q = e["Q"]
    xc = (x - q[:3, 3] / radius) @ q[:3, :3]
    z = xc[:, 2]
    ii = np.flatnonzero((z > 0.01) & (z < e["cutoff"]))
    out = np.zeros(len(x), bool)
    if not len(ii):
        return out
    uv, ok = project_to_clip(xc[ii], cam)
    u = np.rint(np.clip((uv[:, 0] + 0.5) / 4 - 0.5, -10000, 10000)).astype(int)
    v = np.rint(np.clip((uv[:, 1] + 0.5) / 4 - 0.5, -10000, 10000)).astype(int)
    good = ok & (u >= 0) & (u < SMALL[0]) & (v >= 0) & (v < SMALL[1])
    ids = np.flatnonzero(good)
    out[ii[ids]] = e["small_lumen"][v[ids], u[ids]]
    return out


def harmonic_solve(
    f: np.ndarray, num: np.ndarray, screen: np.ndarray, free: np.ndarray, iterations: int
) -> np.ndarray:
    """Screened Jacobi iterations ``f = (sum of 6 neighbours + num) / (6 + screen)`` (float32).

    Grid faces are fixed to +1 (wall), ``free`` voxels to -1. Neighbour sums are accumulated in
    double precision in a fixed order.
    """
    src = np.ascontiguousarray(f, np.float32)
    inner = (slice(1, -1),) * 3
    num_i = num[inner].astype(np.float32)
    den = (np.float32(6) + screen[inner]).astype(np.float32)
    free_i = free[inner]
    for _ in range(iterations):
        a = src.astype(np.float64)
        s = (
            a[:-2, 1:-1, 1:-1]
            + a[1:-1, :-2, 1:-1]
            + a[1:-1, 1:-1, :-2]
            + a[1:-1, 1:-1, 2:]
            + a[1:-1, 2:, 1:-1]
            + a[2:, 1:-1, 1:-1]
        ).astype(np.float32)
        new = np.ones_like(src)
        new[inner] = np.where(free_i, np.float32(-1), (s + num_i) / den)
        src = new
    return src


def _signed_distance(inside: np.ndarray, step: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return (distance_transform_edt(~inside) - distance_transform_edt(inside)) * step


def _wall_votes(x, ev, num, screen, free, c):
    """Screened wall votes of each observation (Gaussian in the depth residual, 0.16 R)."""
    for e in ev:
        z, ii, u, v = project_grid(x, e, c["Kd"], c["R"])
        valid = e["valid"][v, u]
        ids = ii[valid]
        delta = z[ids] - e["z"][v[valid], u[valid]]
        near = np.abs(delta) < 0.32
        w = 12 * np.exp(-0.5 * (delta[near] / 0.16) ** 2).astype(np.float32)
        num[ids[near]] += w * np.clip(delta[near] / 0.16, -1, 1)
        screen[ids[near]] += w
        free[ids[delta < -0.10]] = True
        free |= lumen_free(x, e, c["R"], c["cam"])


def _camera_sdf(c: dict, x: np.ndarray, shape: Any, step: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    cams = c["P"][:, :3, 3] / c["R"]
    dist = cKDTree(cams).query(x, workers=6)[0].reshape(tuple(shape))
    return dist - CAMERA_BALL - 1.5 * step


def _set_faces(f: np.ndarray, value: float) -> None:
    f[[0, -1], :, :] = value
    f[:, [0, -1], :] = value
    f[:, :, [0, -1]] = value


# ----------------------------------------------------------------------------- 1. block fields
def block_fields(c: dict[str, Any]) -> dict[str, Any]:
    """Per-block closed fields (every 8th frame); returned as float16 like their storage."""
    from scipy.ndimage import gaussian_filter

    r, ev = c["R"], c["ev"]
    cams = c["P"][:, :3, 3] / r
    clouds = [e["points"] for e in ev] + [cams] + c.get("domain_points", [])
    rays = pixel_rays(c["Kd"])
    for e in ev:
        rr = rays[::4, ::4][e["lumen"][::4, ::4]]
        if len(rr):
            clouds.append((rr * e["cutoff"]) @ e["Q"][:3, :3].T + e["Q"][:3, 3] / r)
    pts = np.concatenate(clouds)
    lo = pts.min(0) - 1.2
    hi = pts.max(0) + 1.2
    step = max(0.055, float(np.max(hi - lo)) / 165)
    shape = np.ceil((hi - lo) / step).astype(int) + 1
    x = grid_points(shape, lo, step)
    camera_sdf = _camera_sdf(c, x, shape, step).astype(np.float32)
    centres = np.unique(np.r_[np.arange(0, len(c["P"]), 8), len(c["P"]) - 1]).astype(int)
    fields = []
    for t in centres:
        j = int(c["video_indices"][t])
        ee = [e for e in ev if abs(e["j"] - j) <= 8] or [min(ev, key=lambda e: abs(e["j"] - j))]
        num = np.zeros(len(x), np.float32)
        screen = np.zeros(len(x), np.float32)
        free = np.zeros(len(x), bool)
        for e in ee:
            z, ii, u, v = project_grid(x, e, c["Kd"], r)
            valid = e["valid"][v, u]
            ids = ii[valid]
            delta = z[ids] - e["z"][v[valid], u[valid]]
            near = np.abs(delta) < 0.32
            weight = np.exp(-0.5 * (delta[near] / 0.16) ** 2).astype(np.float32) * 12
            num[ids[near]] += weight * np.clip(delta[near] / 0.16, -1, 1)
            screen[ids[near]] += weight
            free[ids[delta < -0.10]] = True
            free |= lumen_free(x, e, r, c["cam"])
        for e in c["times"]:
            if e["accepted"] and abs(e["j"] - j) <= 8:
                free |= lumen_free(x, e, r, c["cam"])
        free = free.reshape(tuple(shape)) | (camera_sdf < 0)
        num = num.reshape(tuple(shape))
        screen = screen.reshape(tuple(shape))
        f = ((gaussian_filter(num, 3) + 0.03) / (gaussian_filter(screen, 3) + 0.1)).astype(
            np.float32
        )
        f = np.clip(f, -1, 1)
        f[free] = -1
        f = harmonic_solve(f, num, screen, free, 320)
        sdf = _signed_distance(f < 0, step)
        narrow = np.abs(f) < 0.5
        sdf[narrow] = f[narrow] * step * 2
        fields.append(np.minimum(sdf, camera_sdf).astype(np.float32))
    return dict(fields=np.array(fields, dtype=np.float16), lo=lo, step=step, shape=shape)


# ----------------------------------------------------------------------------- 2. joint field
def joint_field(c: dict[str, Any], blocks: dict[str, Any]) -> np.ndarray:
    """One jointly screened static field initialised by the median block field."""
    lo, step, shape = blocks["lo"], float(blocks["step"]), blocks["shape"].astype(int)
    x = grid_points(shape, lo, step)
    prior = np.median(blocks["fields"].astype(np.float32), axis=0)
    prior = np.clip(prior / (2 * step), -1, 1).astype(np.float32)
    num = (0.08 * prior).reshape(-1)
    screen = np.full(len(x), 0.08, np.float32)
    free = np.zeros(len(x), bool)
    _wall_votes(x, c["ev"], num, screen, free, c)
    for e in c["times"]:
        if e["accepted"]:
            free |= lumen_free(x, e, c["R"], c["cam"])
    cam_sdf = _camera_sdf(c, x, shape, step)
    free = free.reshape(tuple(shape)) | (cam_sdf < 0)
    num = num.reshape(tuple(shape))
    screen = screen.reshape(tuple(shape))
    f = prior.copy()
    f[free] = -1
    f = harmonic_solve(f, num, screen, free, 480)
    sdf = _signed_distance(f < 0, step)
    narrow = np.abs(f) < 0.5
    sdf[narrow] = f[narrow] * step * 2
    # Union of all free-space intervals with a two-voxel safety shell.
    fs = _signed_distance(free, step) - 2 * step
    sdf = np.minimum(np.minimum(sdf, fs), cam_sdf).astype(np.float32)
    _set_faces(sdf, step)
    return sdf


def _keep_camera_component(f: np.ndarray, cameras: np.ndarray, lo, step) -> None:
    """Keep only the free volume connected to the cameras; fill enclosed wall bubbles."""
    from scipy.ndimage import binary_fill_holes, label

    labs, _ = label(f < 0)
    idx = np.rint((cameras - lo) / step).astype(int)
    cam_label = np.unique(labs[tuple(idx.T)])
    if len(cam_label) != 1 or cam_label[0] == 0:
        raise RuntimeError("cameras are not inside one connected free volume")
    keep = labs == cam_label[0]
    filled = binary_fill_holes(keep)
    f[~filled] = np.maximum(f[~filled], step * 0.01)
    f[filled & ~keep] = -step


# ----------------------------------------------------------------------------- 3. regularised
def regularized_field(
    c: dict[str, Any], sdf: np.ndarray, lo: np.ndarray, step: float
) -> dict[str, Any]:
    """Heat-kernel regularised field; returns ``sdf``, ``hard_free``, ``points``, ``view_ids``."""
    from scipy.ndimage import gaussian_filter

    shape = np.array(sdf.shape)
    x = grid_points(shape, lo, step)
    counts = np.zeros(len(x), np.uint16)
    free = np.zeros(len(x), bool)
    for e in c["ev"]:
        z, ii, u, v = project_grid(x, e, c["Kd"], c["R"])
        valid = e["valid"][v, u]
        ids = ii[valid]
        delta = z[ids] - e["z"][v[valid], u[valid]]
        counts[ids[np.abs(delta) < 0.10]] += 1  # one vote per view within 0.10 R
        free |= lumen_free(x, e, c["R"], c["cam"])
    for e in c["times"]:
        if e["accepted"]:
            free |= lumen_free(x, e, c["R"], c["cam"])
    cameras = c["P"][:, :3, 3] / c["R"]
    cam_sdf = _camera_sdf(c, x, shape, step)
    free = free.reshape(tuple(shape)) | (cam_sdf < 0)
    counts = counts.reshape(tuple(shape))
    proper = _signed_distance(sdf < 0, step)
    coarse = gaussian_filter(proper, 0.28 / step)
    fine = gaussian_filter(proper, 0.12 / step)
    consensus = np.clip((gaussian_filter(counts.astype(np.float32), 0.075 / step) - 2) / 2, 0, 1)
    regular = (coarse + 0.35 * consensus * (fine - coarse)).astype(np.float32)
    fs = _signed_distance(free, step) - 2 * step
    need = np.maximum(regular - fs, 0) * (fs < 0)
    regular -= gaussian_filter(need, 0.15 / step)
    regular = np.minimum(np.minimum(regular, fs), cam_sdf)
    _keep_camera_component(regular, cameras, lo, step)
    _set_faces(regular, step)
    points = np.concatenate([e["points"] for e in c["ev"]])
    view_ids = np.concatenate([np.full(len(e["points"]), e["j"]) for e in c["ev"]])
    return dict(sdf=regular, hard_free=free, points=points, view_ids=view_ids)


# ----------------------------------------------------------------------------- 4. envelope
def smooth_min(a: np.ndarray, b: np.ndarray, k: float = 0.20) -> np.ndarray:
    """Polynomial smooth minimum (union) of two fields."""
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def envelope_field(
    reg: dict[str, Any], lo: np.ndarray, step: float, poses: np.ndarray, radius: float
) -> np.ndarray:
    """Support-aware static refinement of the regularised field."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    sdf, free, obs, view_ids = reg["sdf"], reg["hard_free"], reg["points"], reg["view_ids"]
    grid = np.indices(sdf.shape, dtype=np.float32).reshape(3, -1).T * step + lo
    near = np.flatnonzero(np.abs(sdf.ravel()) < 0.75)
    counts = np.zeros(sdf.size, np.uint16)
    for key in np.unique(view_ids):
        dd = cKDTree(obs[view_ids == key]).query(grid[near], workers=6, distance_upper_bound=0.12)[
            0
        ]
        counts[near[dd <= 0.12]] += 1
    counts = counts.reshape(sdf.shape)
    # Fine residual where three keyframes support the wall; weak evidence low-passed at 0.26 R.
    coarse = gaussian_filter(sdf, 0.26 / step)
    consensus = np.clip((gaussian_filter(counts.astype(np.float32), 0.06 / step) - 2) / 1.5, 0, 1)
    fine = gaussian_filter(sdf, 0.035 / step)
    regular = (coarse + 0.95 * consensus * (fine - coarse)).astype(np.float32)
    # Smooth outward union with the free-space envelope, offset by its worst boundary incursion.
    fs = _signed_distance(free, step) - 2 * step
    fs_blur = gaussian_filter(fs, 0.22 / step)
    offset = float(max(0, fs_blur[fs <= 0].max()) + 0.015)
    regular = smooth_min(regular, fs_blur - offset)
    cameras = poses[:, :3, 3] / radius
    cam_sdf = (cKDTree(cameras).query(grid, workers=6)[0] - CAMERA_BALL - 1.5 * step).reshape(
        sdf.shape
    )
    regular = smooth_min(regular, cam_sdf, 0.12).astype(np.float32)
    _keep_camera_component(regular, cameras, lo, step)
    _set_faces(regular, step)
    return regular


def reconstruct_sdf(c: dict[str, Any]) -> dict[str, Any]:
    """Steps 1-4: the static window SDF ``sdf`` on the grid ``lo``, ``step`` (units of R)."""
    blocks = block_fields(c)
    lo, step = blocks["lo"], float(blocks["step"])
    joint = joint_field(c, blocks)
    reg = regularized_field(c, joint, lo, step)
    sdf = envelope_field(reg, lo, step, c["P"], c["R"])
    return dict(sdf=sdf, lo=lo, step=step, blocks=blocks, joint=joint, regular=reg)


# ----------------------------------------------------------------------------- 5. lumen carve
def largest_dark_masks(
    video: str, video_indices: np.ndarray, cam: Calibration, field: np.ndarray
) -> np.ndarray:
    """Largest-area dark component (2 % darkest radially normalised pixels) of every frame."""
    import cv2

    small, _ = decode_small(video, video_indices)
    mask = small_field(field)
    yy, xx = np.mgrid[: SMALL[1], : SMALL[0]]
    k = np.asarray(cam["K_clip"])
    cx, cy = (k[0, 2] + 0.5) / 4 - 0.5, (k[1, 2] + 0.5) / 4 - 0.5
    r = np.hypot(xx - cx, yy - cy)
    rmax = r[mask].max()
    bins = np.minimum((r / rmax * 12).astype(int), 11)
    inner = mask & (r < 0.8 * rmax)
    out = []
    for j in video_indices:
        frame = small[int(j)]
        g8 = np.clip(_gray(frame), 0, 255).astype(np.uint8)
        hole = (specular_mask(frame) | ~mask).astype(np.uint8)
        g = cv2.inpaint(g8, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)
        blur = cv2.GaussianBlur(g, (0, 0), 9)
        prof = np.array(
            [
                np.median(blur[mask & (bins == b)]) if (mask & (bins == b)).sum() > 20 else np.nan
                for b in range(12)
            ]
        )
        prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
        norm = blur / np.maximum(prof[bins], 1.0)
        dark = (inner & (norm <= np.percentile(norm[inner], 2))).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        m = np.zeros_like(dark)
        if n > 1:
            m = (lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        out.append(m)
    return np.array(out)


def _mask_moments(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex, boundary-smoothed masks and temporally filtered moments (x, y, area, covariance)."""
    import cv2
    from scipy.ndimage import gaussian_filter1d, median_filter

    cleaned, mom = [], []
    for mask in raw:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        out = np.zeros_like(mask, np.uint8)
        if n > 1:
            a = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
            a = (cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.4) > 0.40).astype(np.uint8)
            yy, xx = np.nonzero(a)
            if len(xx) >= 5:
                cv2.fillConvexPoly(out, cv2.convexHull(np.c_[xx, yy].astype(np.int32)), 1)
        yy, xx = np.nonzero(out)
        if len(xx) >= 5:
            xy = np.c_[xx, yy]
            mom.append([*xy.mean(0), float(out.sum()), *np.cov(xy.T, bias=True).reshape(-1)])
        else:
            mom.append([np.nan] * 7)
        cleaned.append(out)
    mom = np.array(mom)
    times = np.arange(len(mom))
    ok = np.isfinite(mom).all(1)
    if not ok.any():
        raise RuntimeError("no lumen component in the window")
    filled = np.stack([np.interp(times, times[ok], v[ok]) for v in mom.T], axis=1)
    smooth = gaussian_filter1d(
        median_filter(filled, size=(7, 1), mode="nearest"), 2, axis=0, mode="nearest"
    )
    smooth[:, 2] = np.exp(
        gaussian_filter1d(median_filter(np.log(filled[:, 2]), 7, mode="nearest"), 2, mode="nearest")
    )
    return np.array(cleaned), smooth


def fit_lumen_frustum(
    masks: np.ndarray, poses: np.ndarray, radius: float, cam: Calibration
) -> dict[str, Any]:
    """One static elliptical frustum (units of R) from the per-frame largest dark components."""
    from scipy.optimize import least_squares

    clean, sm = _mask_moments(masks)
    o = poses[:, :3, 3] / radius
    rays, _ = rays_from_clip((sm[:, :2] + 0.5) * 4 - 0.5, cam)
    d = np.einsum("tij,tj->ti", poses[:, :3, :3], rays)
    n = np.median(d, axis=0)
    n /= np.linalg.norm(n)
    # Robust bearing-only focal point, weakly regularised toward 3 R ahead.
    q = np.median(o + 3 * d, axis=0)

    def resid(cc):
        a = cc - o
        perp = a - (a * d).sum(1)[:, None] * d
        return np.r_[perp.ravel(), 0.12 * (cc - q)]

    focal = least_squares(resid, q, loss="soft_l1", f_scale=0.16).x
    z0 = float(np.median(o @ n))
    origin = focal - n * ((focal @ n) - z0)
    u = np.cross(n, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    err = np.linalg.norm(np.cross(focal - o, d), axis=1)
    w = 1 / np.maximum(1, err / 0.22)
    # Mask moments on a common world section 3 R ahead.
    section = z0 + 3.0
    cent, covs = [], []
    for t, ma in enumerate(clean):
        yy, xx = np.nonzero(ma)
        rr, kk = rays_from_clip((np.c_[xx, yy] + 0.5) * 4 - 0.5, cam)
        dd = rr @ poses[t, :3, :3].T
        den = dd @ n
        good = kk & (den > 0.10)
        pts = o[t] + dd[good] * ((section - o[t] @ n) / den[good])[:, None]
        xy = np.c_[(pts - origin) @ u, (pts - origin) @ v]
        if len(xy) < 5:
            cent.append([np.nan] * 2)
            covs.append(np.eye(2) * np.nan)
            continue
        cent.append(xy.mean(0))
        covs.append(np.cov(xy.T, bias=True))
    cent, covs = np.array(cent), np.array(covs)
    valid = np.isfinite(cent).all(1)
    ww = w[valid]
    center = least_squares(
        lambda a: ((cent[valid] - a) * np.sqrt(ww)[:, None]).ravel(),
        np.nanmedian(cent, axis=0).ravel(),
        loss="soft_l1",
        f_scale=0.08,
    ).x
    ev, qm = np.linalg.eigh(np.nanmedian(covs, axis=0))
    semi = 2 * np.sqrt(np.maximum(ev, 1e-5))
    origin = origin + u * center[0] + v * center[1]
    axis_u = u * qm[0, 0] + v * qm[1, 0]
    axis_v = u * qm[0, 1] + v * qm[1, 1]
    return dict(
        origin=origin,
        n=n,
        u=axis_u,
        v=axis_v,
        slope=semi / 3.0,
        floor=0.035,
        z0=z0,
        start=float(np.min(o @ n) - 0.8),
        end=float(np.max(o @ n) + 9.4),
    )


def carve_lumen(
    sdf: np.ndarray, lo0: np.ndarray, step: float, poses: np.ndarray, radius: float, par: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Unite the field with the lumen frustum and close far cavities. Returns ``(sdf, lo)``."""
    o = poses[:, :3, 3] / radius
    origin, n, u, v, slope, z0 = (par[k] for k in ("origin", "n", "u", "v", "slope", "z0"))
    end_c = origin + n * (par["end"] - z0)
    rmax = par["floor"] + max(slope) * (par["end"] - z0)
    start_c = origin + n * (par["start"] - z0)
    extra = np.array([end_c - rmax, end_c + rmax, start_c - 0.5, start_c + 0.5])
    lo = np.minimum(lo0, extra.min(0) - 3 * step)
    lo = lo0 - np.ceil((lo0 - lo) / step) * step
    hi = np.maximum(lo0 + (np.array(sdf.shape) - 1) * step, extra.max(0) + 3 * step)
    shape = np.ceil((hi - lo) / step).astype(int) + 1
    out = np.full(tuple(shape), step, np.float32)
    pad = np.rint((lo0 - lo) / step).astype(int)
    out[tuple(slice(int(a), int(a + b)) for a, b in zip(pad, sdf.shape, strict=True))] = sdf
    yy, zz = np.mgrid[: shape[1], : shape[2]]
    for i in range(shape[0]):
        x = np.c_[np.full(yy.size, i), yy.ravel(), zz.ravel()] * step + lo
        delta = x - origin
        z = delta @ n
        aa = par["floor"] + slope[0] * np.maximum(z, 0)
        bb = par["floor"] + slope[1] * np.maximum(z, 0)
        radial = (np.sqrt(((delta @ u) / aa) ** 2 + ((delta @ v) / bb) ** 2) - 1) * np.minimum(
            aa, bb
        )
        cone = np.maximum.reduce([radial, par["start"] - (x @ n), (x @ n) - par["end"]])
        completion = np.max(np.linalg.norm(x[:, None] - o[None], axis=2), axis=1) - COMPLETION_R
        completion = np.maximum(completion, np.linalg.norm(x - o[-1], axis=1) - COMPLETION_R)
        base = np.maximum(out[i].ravel(), completion)
        out[i] = smooth_min(base, cone, 0.08).reshape(out[i].shape)
    _set_faces(out, step)
    return out, lo


def closed_extract(f: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Watertight zero level set (marching cubes; tiny level offsets avoid coincident vertices)."""
    import trimesh
    from skimage.measure import marching_cubes

    for level in (0.0, step * 1e-4, -step * 1e-4, step * 1e-3, -step * 1e-3):
        v, fc, _, _ = marching_cubes(f, level, spacing=(step,) * 3, allow_degenerate=False)
        mesh = trimesh.Trimesh(v, fc, process=True, validate=True)
        if mesh.is_watertight:
            return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    raise RuntimeError("no watertight level set")


def carve_mesh(
    sdf: np.ndarray, lo: np.ndarray, step: float, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh of the carved field in pose units."""
    v, f = closed_extract(sdf, step)
    return (v + lo).astype(np.float32) * radius, f
