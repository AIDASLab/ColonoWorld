# SPDX-License-Identifier: CC-BY-NC-4.0
"""Lumen fit of a training-window tube.

A training window starts from a *seed* tube: a profile sweep (see :mod:`colonoworld.tube.wall_fit`)
fitted to the window's depth, rendered along the window's camera path. The lumen fit reshapes the
far end of that sweep so that the dark lumen seen in the real video and the deep opening of the
tube agree in presence, position and size, while the tube keeps following the real camera motion:

1. **Parametric family.** A 15-number vector deforms the seed beyond the front-most camera: the
   distance to a neck, its log radius, a lateral target offset, a bend (angle, slope, length), a
   body blend width, a near-radius gain, an elliptic throat anisotropy, a stationary shoulder bump
   and a static dilation about the camera centroid. Every candidate is re-inflated so that a
   0.36 R ball around every camera stays inside.
2. **Spatial search.** Candidates are scored on every second frame by the agreement of the real
   dark-lumen mask with the tube's deep-opening mask (presence mismatch, centroid distance, area
   ratio, opening margins). The search combines a fixed grid, random samples, differential
   evolution and a multi-scale local walk.
3. **Constrained selection.** A diverse bank of the best candidates is re-scored at several body
   scales with a sparse render: camera-following (optical-flow agreement with the real video)
   must not drop below the seed tube's, and the mean brightness must lie inside the training range.
   The seed tube itself (open side wall, scaled) is always a fallback candidate.
4. **Final check.** Up to three finalists are rendered at full resolution; the first one that
   passes every guard (single lumen, no wall crossing, clearance, brightness, camera following)
   with a spatial PASS is kept.

All tube videos here use the training renderer with exposure gain 1 (light radius = R).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.camera import Calibration, rays_from_clip
from colonoworld.tube.model import TubeModel
from colonoworld.tube.render import TubeScene, load_texture, write_tube_video
from colonoworld.tube.synthetic_colon import surface_distance
from colonoworld.tube.wall_fit import (
    Params,
    arc_length,
    centreline,
    ellipse_radius,
    fold_factor,
    interp_profile,
    sweep_mesh,
)

LUMA = np.array([0.299, 0.587, 0.114])
GRID_W, GRID_H = 160, 120  # analysis grid (clip / 6.9)
_YY, _XX = np.mgrid[:GRID_H, :GRID_W]
INNER = np.hypot((_XX - 79.5) / 80, (_YY - 59.5) / 60) < 0.85
FLOW_PARAMS = (0.5, 3, 21, 3, 5, 1.2, 0)  # Farneback
ACTIVE_FLOW = 0.1  # px per 4 frames: a block is moving
STATIC_FLOW = 0.2  # px per 4 frames: a frame pair is static

# Search space of the 15-number deformation vector (see :func:`construct`).
LOW = np.array(
    [
        0.12,
        np.log(0.07),
        -2.5,
        -2.5,
        -np.pi,
        0.05,
        0.05,
        0.02,
        1.0,
        -0.8,
        -0.8,
        -2.5,
        -2.5,
        0.0,
        0.65,
    ]
)
HIGH = np.array(
    [6.0, np.log(2.5), 2.5, 2.5, np.pi, 10.0, 3.0, 2.5, 2.4, 0.8, 0.8, 2.5, 2.5, 2.0, 4.0]
)
STEPS = np.array([0.8, 0.4, 0.3, 0.3, 0.8, 1.4, 0.4, 0.4, 0.12, 0.20, 0.20, 0.5, 0.5, 0.35, 0.18])


# ----------------------------------------------------------------------------- frames
def sample_times(n_frames: int, stride: int) -> np.ndarray:
    """Frames ``0, stride, 2 stride, ...`` plus the last frame."""
    return np.unique(np.r_[np.arange(0, n_frames, stride), n_frames - 1]).astype(int)


def decode_frames(
    path: str | Path, indices: Any, size: tuple[int, int] = (GRID_W, GRID_H), area: bool = True
) -> np.ndarray:
    """Decode the given frame indices (one decoder thread) and resize them to ``size``.

    ``area=True`` uses area interpolation (motion analysis), otherwise bilinear (lumen masks).
    """
    import av
    import cv2

    wanted = [int(i) for i in indices]
    keys = set(wanted)
    interp = cv2.INTER_AREA if area else cv2.INTER_LINEAR
    found: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            if i in keys:
                found[i] = cv2.resize(
                    frame.to_ndarray(format="rgb24", threads=1), size, interpolation=interp
                )
            if i >= max(keys):
                break
    if set(found) != keys:
        raise ValueError(f"missing frames in {path}")
    return np.array([found[i] for i in wanted])


def decoded_mean_luma(path: str | Path, field: np.ndarray) -> float:
    """Mean luma inside ``field`` over all decoded frames of a video."""
    import av

    values = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        for frame in container.decode(video=0):
            img = frame.reformat(format="rgb24", threads=1).to_ndarray()
            values.append(float(img[field].mean(axis=0) @ LUMA))
    return float(np.mean(values))


def load_window_frames(
    real_video: str | Path, video_indices: list[int], seed_tube_video: str | Path
) -> dict[str, np.ndarray]:
    """Real and seed-tube frames used by :func:`fit_training_tube`.

    Returns ``real_spatial`` (every 2nd window frame, bilinear 160x120), ``real_motion`` and
    ``seed_motion`` (every 4th window frame, area 160x120).
    """
    n = len(video_indices)
    ts, mt = sample_times(n, 2), sample_times(n, 4)
    ids = np.asarray(video_indices)
    return dict(
        real_spatial=decode_frames(real_video, ids[ts], area=False),
        real_motion=decode_frames(real_video, ids[mt]),
        seed_motion=decode_frames(seed_tube_video, mt),
    )


# ----------------------------------------------------------------------------- lumen agreement
def real_lumen_mask(rgb: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, bool]:
    """Darkest connected region (< 0.45 x median luma) inside the inner field, and its presence."""
    import cv2

    gray = rgb.astype(float) @ LUMA
    threshold = max(1.0, float(np.median(gray[field])) * 0.45)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ((gray < threshold) & field & INNER).astype(np.uint8), connectivity=8
    )
    eligible = [
        (float(gray[labels == i].mean()), i)
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= 15
    ]
    mask = labels == min(eligible)[1] if eligible else np.zeros_like(field)
    fraction = float(mask.sum() / field.sum())
    return mask, bool(eligible) and 0.005 <= fraction <= 0.60


def tube_opening_mask(depth: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, bool, int, float]:
    """Largest region deeper than twice the median depth (area > 1 % of the field)."""
    import cv2

    valid = field & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("no valid tube depth")
    median = float(np.median(depth[valid]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (valid & (depth > 2 * median)).astype(np.uint8), connectivity=8
    )
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.01 * field.sum()]
    best = max(ids, key=lambda i: (stats[i, cv2.CC_STAT_AREA], -i)) if ids else None
    mask = labels == best if best is not None else np.zeros_like(field)
    return mask, best is not None, len(ids), float(1 - valid.sum() / field.sum())


def _centre(mask: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(mask)
    return np.array([xx.mean(), yy.mean()]) if len(xx) else None


def _native_point(point: np.ndarray, cam: Calibration) -> np.ndarray:
    cw, ch = cam["clip_size"]
    clip_point = (point + 0.5) * np.array([cw / GRID_W, ch / GRID_H]) - 0.5
    affine = np.asarray(cam["native_to_clip"], dtype=float)
    return np.linalg.solve(affine[:2, :2], clip_point - affine[:2, 2])


def frame_agreement(
    rgb: np.ndarray, depth: np.ndarray, field_depth: np.ndarray, field: np.ndarray, cam: Calibration
) -> dict[str, Any]:
    """Real-lumen vs tube-opening agreement on one frame (160x120 grid, depth at stride 8)."""
    import cv2

    rm, rp = real_lumen_mask(rgb, field)
    tm_d, tp, deep, missing = tube_opening_mask(depth, field_depth)
    tm = cv2.resize(tm_d.astype(np.uint8), (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST)
    tm = tm.astype(bool) & field
    rc, tc = _centre(rm), _centre(tm)
    paired = bool(rp and tp and rc is not None and tc is not None)
    distance = ratio = None
    if paired:
        offset = _native_point(rc, cam) - _native_point(tc, cam)
        distance = float(np.linalg.norm(offset) / cam["field_radius_px"])
        ratio = float(tm.sum() / rm.sum())
    return dict(
        real_present=rp,
        tube_present=tp,
        presence_mismatch=bool(rp != tp),
        both_present=paired,
        centroid_distance_R=distance,
        area_ratio=ratio,
        tube_deep_components=deep,
        tube_missing_depth_fraction=missing,
    )


def _quantile(rows: list[dict], key: str, q: float) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return float(np.quantile(values, q)) if values else None


def window_agreement(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Window summary; PASS when presence agrees on >= 80 % of frames and, with >= 10 paired
    frames, the median centroid distance is at most 0.25 field radii."""
    out = dict(
        both_present_frames=sum(f["both_present"] for f in frames),
        presence_mismatch_fraction=float(np.mean([f["presence_mismatch"] for f in frames])),
        median_centroid_distance_R=_quantile(frames, "centroid_distance_R", 0.5),
        median_area_ratio=_quantile(frames, "area_ratio", 0.5),
    )
    out["centroid_required"] = out["both_present_frames"] >= 10
    fail = out["centroid_required"] and out["median_centroid_distance_R"] > 0.25
    out["status"] = "PASS" if not fail and out["presence_mismatch_fraction"] <= 0.20 else "FAIL"
    return out


def agreement_loss(s: dict[str, Any], frames: list[dict[str, Any]]) -> float:
    """Spatial objective (lower is better)."""
    mis = s["presence_mismatch_fraction"]
    cent = s["median_centroid_distance_R"] or 0
    area = s["median_area_ratio"]
    err = abs(np.log(area)) if area is not None and area > 0 else 0
    paired = s["both_present_frames"] >= 10
    return (
        12 * max(0, mis - 0.2)
        + 10 * max(0, cent - 0.25) * paired
        + 3 * mis
        + 0.8 * min(cent, 1) * paired
        + 0.12 * min(err, 4)
        + 12 * sum(f["tube_deep_components"] > 1 for f in frames) / len(frames)
        + 30 * max(0, max(f["tube_missing_depth_fraction"] for f in frames) - 0.003)
    )


def margin_loss(depth: np.ndarray, field: np.ndarray, frames: list[dict[str, Any]]) -> float:
    """Pushes the tube opening area to the safe side of the presence thresholds."""
    import cv2

    values = []
    for d, f in zip(depth, frames, strict=True):
        valid = field & np.isfinite(d) & (d > 0)
        med = np.median(d[valid])
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            (valid & (d > 2 * med)).astype(np.uint8), 8
        )
        area = float(stats[1:, cv2.CC_STAT_AREA].max() / field.sum()) if len(stats) > 1 else 0
        target = 0.013 if f["real_present"] else 0.006
        residual = max(0, target - area) if f["real_present"] else max(0, area - target)
        values.append(min(residual / 0.01, 3))
    return 0.8 * float(np.mean(values))


# ----------------------------------------------------------------------------- camera following
def _motion_field(real: list[np.ndarray], cam: Calibration) -> np.ndarray:
    import cv2

    _, valid = rays_from_clip(
        np.stack(
            [
                (_XX + 0.5) * cam["clip_size"][0] / GRID_W - 0.5,
                (_YY + 0.5) * cam["clip_size"][1] / GRID_H - 0.5,
            ],
            -1,
        ).reshape(-1, 2),
        cam,
    )
    mx = np.max([f.max(-1) for f in (real[0], real[len(real) // 2], real[-1])], 0)
    dark = (mx < 14).astype(np.uint8)
    _, lab = cv2.connectedComponents(dark, connectivity=4)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    border = np.isin(lab, list(edge)) & (dark > 0)
    border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = valid.reshape(GRID_H, GRID_W) & ~border
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _photometric_mask(rgb: np.ndarray, field: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    spec = (hsv[..., 2] > 0.95) & (hsv[..., 1] < 0.15)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bad = spec | (gray <= np.percentile(gray[field], 2))
    return field & ~(cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-4 or np.std(b[ok]) < 1e-4:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _similarity(flow: np.ndarray, mask: np.ndarray, cam: Calibration) -> np.ndarray:
    """Least-squares translation, radial expansion and rotation of a flow field."""
    k = np.asarray(cam["K_clip"])
    cw, ch = cam["clip_size"]
    x = ((_XX + 0.5) - (k[0, 2] + 0.5) * GRID_W / cw) / 80
    y = ((_YY + 0.5) - (k[1, 2] + 0.5) * GRID_H / ch) / 80
    x, y = x[mask], y[mask]
    u, v = flow[mask].T
    one, zero = np.ones(len(x)), np.zeros(len(x))
    design = np.concatenate([np.stack([one, zero, x, -y], 1), np.stack([zero, one, y, x], 1)])
    return np.linalg.lstsq(design, np.r_[u, v], rcond=None)[0]


def camera_following(real: list[np.ndarray], tube: list[np.ndarray], cam: Calibration) -> dict:
    """Optical-flow agreement between real and tube frame sequences (160x120, every 4th frame).

    ``block_cos``: mean cosine of median flow over 8x8 blocks where either moves;
    ``radial_r``: correlation of the radial (dolly) components; ``magnitude_ratio``: median
    tube/real block flow ratio; plus static fractions and flow percentiles.
    """
    import cv2

    field = _motion_field(real, cam)
    if field.mean() <= 0.2:
        raise ValueError("insufficient field")
    masks = [[_photometric_mask(f, field) for f in seq] for seq in (real, tube)]
    grays = [[cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq] for seq in (real, tube)]
    cosines, ratios, real_coeff, tube_coeff, mags, coverage = [], [], [], [], [], []
    yy, xx = np.mgrid[:GRID_H, :GRID_W].astype(np.float32)
    for t in range(len(real) - 1):
        flows = [cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, *FLOW_PARAMS) for g in grays]
        mask = field.copy()
        for k, f in enumerate(flows):
            target = cv2.remap(
                masks[k][t + 1].astype(np.uint8),
                xx + f[..., 0],
                yy + f[..., 1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask &= masks[k][t] & (target > 0)
        coverage.append(mask.sum() / field.sum())
        if mask.sum() <= 500:
            raise ValueError("insufficient flow support")
        mags.append([float(np.linalg.norm(f[mask], axis=1).mean()) for f in flows])
        real_coeff.append(_similarity(flows[0], mask, cam))
        tube_coeff.append(_similarity(flows[1], mask, cam))
        for by in range(8):
            for bx in range(8):
                sl = np.s_[by * 15 : (by + 1) * 15, bx * 20 : (bx + 1) * 20]
                m = mask[sl]
                if m.sum() < 60:
                    continue
                a, b = (np.median(f[sl][m], axis=0) for f in flows)
                ma, mb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if max(ma, mb) >= ACTIVE_FLOW:
                    c = float(np.dot(a, b) / (ma * mb)) if min(ma, mb) >= ACTIVE_FLOW else 0.0
                    cosines.append(float(np.clip(c, -1, 1)))
                if ma >= ACTIVE_FLOW:
                    ratios.append(mb / ma)
    rc, tc, m = np.asarray(real_coeff), np.asarray(tube_coeff), np.asarray(mags)
    out = dict(
        block_cos=float(np.mean(cosines)) if cosines else float("nan"),
        magnitude_ratio=float(np.median(ratios)) if ratios else float("nan"),
        valid_fraction_min=float(np.min(coverage)),
        real_p90_flow=float(np.percentile(m[:, 0], 90)),
        tube_p90_flow=float(np.percentile(m[:, 1], 90)),
        both_static_fraction=float(np.mean((m < STATIC_FLOW).all(1))),
        radial_r=_pearson(rc[:, 2], tc[:, 2]),
    )
    out["balanced"] = following_ok(out)
    return out


def following_ok(r: dict[str, Any]) -> bool:
    """Balanced camera-following criterion (static, or moving with agreeing direction and scale)."""
    static = (
        r["both_static_fraction"] >= 0.8 and max(r["real_p90_flow"], r["tube_p90_flow"]) <= 0.35
    )
    moving = r["block_cos"] >= 0.35 and r["radial_r"] >= 0.40 and 0.4 <= r["magnitude_ratio"] <= 2.5
    return bool(r["valid_fraction_min"] >= 0.4 and (static or moving))


def cosine_ok(new: dict, old: dict, tolerance: float) -> bool:
    """Block cosine not below the reference minus ``tolerance`` (static pairs count as equal)."""
    a, b = new["block_cos"], old["block_cos"]
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
        return a >= b - tolerance

    def static(m):
        return (
            m["both_static_fraction"] >= 0.8 and max(m["real_p90_flow"], m["tube_p90_flow"]) <= 0.35
        )

    undefined = (a is None or not np.isfinite(a)) and (b is None or not np.isfinite(b))
    return undefined and static(old) and static(new)


# ----------------------------------------------------------------------------- geometry
def lumen_mesh(
    p: Params, ds: float = 0.065, n_phi: int = 72
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Open side wall of a lumen-fit sweep (chart from the unscaled rings)."""
    lo, hi = p["z_R"][0], p["z_R"][-1]
    zz = np.linspace(lo, hi, max(100, int((hi - lo) / ds) + 1))
    ph = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    q = interp_profile(p, zz)
    rr = ellipse_radius(q[:, None, :], ph[None]) * fold_factor(p, zz, ph)
    b = np.asarray(p["basis"])
    c = centreline(p, zz)
    v = c[:, None] + rr[:, :, None] * (
        np.cos(ph)[None, :, None] * b[0] + np.sin(ph)[None, :, None] * b[1]
    )
    v = v.reshape(-1, 3)
    ns = len(zz)
    ii = np.arange((ns - 1) * n_phi).reshape(-1, n_phi)
    jj = np.roll(ii, -1, axis=1)
    faces = np.stack(
        [np.stack([ii, jj, ii + n_phi], -1), np.stack([jj, jj + n_phi, ii + n_phi], -1)], -2
    )
    edges = np.linalg.norm(
        np.roll(v.reshape(ns, n_phi, 3), -1, axis=1) - v.reshape(ns, n_phi, 3), axis=2
    )
    circ = edges.sum(1)
    sv = np.repeat(arc_length(p, zz), n_phi)
    tv = (np.c_[np.zeros(ns), np.cumsum(edges, axis=1)[:, :-1]] / circ[:, None]).ravel()
    fv = np.repeat(circ / 2, n_phi)
    chart = (sv.astype(np.float32), tv.astype(np.float32), fv.astype(np.float32))
    return (v * p["R_pose"]).astype(np.float32), faces.reshape(-1, 3).astype(np.int32), chart


def tube_mesh(p: Params) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Mesh of a candidate: the seed side wall for the seed fallback, else the lumen-fit sweep."""
    if p.get("seed_side_wall"):
        v, f, chart, _ = sweep_mesh(p)
        return v, f, chart
    return lumen_mesh(p)


def cameras_inside(p: Params, points: np.ndarray) -> np.ndarray:
    """Whether each point lies strictly inside the folded sweep."""
    b = np.asarray(p["basis"])
    d = np.asarray(points) / p["R_pose"] - np.asarray(p["origin_R"])
    z = d @ b[2]
    xy = d @ b[:2].T
    zc = np.clip(z, p["z_R"][0], p["z_R"][-1])
    q = interp_profile(p, zc)
    v = xy - q[:, :2]
    ph = np.arctan2(v[:, 1], v[:, 0])
    rad = ellipse_radius(q, ph)
    f = p["fold"]
    s = arc_length(p, zc)
    mx = np.zeros(len(z))
    smax = float(arc_length(p, np.array([p["z_R"][-1]]))[0])
    crests = np.arange(
        -f["spacing_R"] + f["phase"] * f["spacing_R"], smax + f["spacing_R"], f["spacing_R"]
    )
    for k, sj in enumerate(crests):
        for sec in [k % 3] if k % 3 else [k % 3, (k + 1) % 3]:
            dp = np.angle(np.exp(1j * (ph - (f["taenia_rad"] + np.pi / 3 + sec * 2 * np.pi / 3))))
            v0 = np.clip(abs(dp) / (f["span_rad"] / 2), 0, 1)
            a = np.cos(np.pi * v0 / 2) ** 2 * (v0 < 1)
            g = np.exp(-0.5 * ((s - sj - 0.08 * np.cos(dp)) / f["width_R"]) ** 2)
            mx = np.maximum(mx, f["depth_frac"] * g * a)
    return (z > p["z_R"][0]) & (z < p["z_R"][-1]) & (np.linalg.norm(v, axis=1) < rad * (1 - mx))


def setup_context(
    p0: Params, poses: np.ndarray, real: np.ndarray, field: np.ndarray, cam: Calibration
) -> dict[str, Any]:
    """Seed frame, camera positions and per-frame real-lumen viewing directions (sweep frame)."""
    b = np.asarray(p0["basis"])
    origin = np.asarray(p0["origin_R"])
    r = p0["R_pose"]
    xyz = poses[:, :3, 3] / r - origin
    zc = xyz @ b[2]
    masks = [real_lumen_mask(im, field) for im in real]
    good = np.array([m[1] for m in masks])
    uv = np.array([_centre(m[0]) if m[1] else [79.5, 59.5] for m in masks])
    cw, ch = cam["clip_size"]
    rays, _ = rays_from_clip((uv + 0.5) * [cw / GRID_W, ch / GRID_H] - 0.5, cam)
    directions = np.einsum("tij,tj->ti", poses[:, :3, :3], rays) @ b.T
    areas = np.array([m[0].sum() / field.sum() if m[1] else 0 for m in masks])
    return dict(
        B=b,
        origin=origin,
        R=r,
        camera=xyz @ b.T,
        zcammax=float(zc.max()),
        zcammin=float(zc.min()),
        directions=directions,
        good=good,
        areas=areas,
    )


def _contain_ball(q, z, cameras, fold_depth, ball):
    ax = z[:, None] - cameras[None, :, 2]
    lateral = np.linalg.norm(q[:, None, :2] - cameras[None, :, :2], axis=2)
    required = np.max(
        np.where(np.abs(ax) < ball, np.sqrt(np.maximum(0, ball**2 - ax**2)) + lateral, 0), axis=1
    )
    minor = np.exp(q[:, 2] - np.hypot(q[:, 3], q[:, 4])) * (1 - fold_depth)
    q[:, 2] += np.log(np.maximum(1.0, required / np.maximum(minor, 1e-8)))


def construct(base: Params, ctx: dict[str, Any], x: np.ndarray) -> Params:
    """Seed sweep deformed by the 15-number vector ``x``.

    ``x``: neck distance beyond the front-most camera, log neck radius, target dx/dy, bend angle,
    bend slope, taper length, body blend width, near-radius gain, throat anisotropy (2), shoulder
    bump offset (2), shoulder position, static dilation factor.
    """
    x = np.asarray(x, float)
    p = copy.deepcopy(base)
    zmax = ctx["zcammax"]
    dist, logr, dx, dy, angle, slope, taper, blend, near_scale = x[:9]
    neck = zmax + dist
    lo = min(base["z_R"][0], ctx["zcammin"] - 2)
    hi = max(neck + 7, base["z_R"][-1])
    cams = ctx.get("camera_all", ctx["camera"])
    cz = cams[:, 2]
    safety = np.linspace(
        cz.min() - 0.38, cz.max() + 0.38, max(20, int((np.ptp(cz) + 0.76) / 0.025) + 1)
    )
    z = np.unique(np.r_[np.linspace(lo, hi, 64), safety, zmax, neck, neck + taper])
    q = interp_profile(base, np.clip(z, base["z_R"][0], base["z_R"][-1]))
    q[:, 2] += np.log(near_scale)
    at_neck = interp_profile(base, np.array([np.clip(neck, base["z_R"][0], base["z_R"][-1])]))[0]
    cam, dd = ctx["camera"], ctx["directions"]
    good = ctx["good"] & (dd[:, 2] > 0.1)
    if good.any():
        points = cam[good, :2] + (neck - cam[good, 2, None]) * dd[good, :2] / dd[good, 2, None]
        target = np.median(points, axis=0)
    else:
        target = at_neck[:2].copy()
    target += np.array([dx, dy])
    # The observed body stays near the seed; the front deforms toward the neck.
    u = np.clip((z - (zmax - blend)) / (dist + blend), 0, 1)
    w = u * u * (3 - 2 * u)
    q[:, :2] = q[:, :2] * (1 - w[:, None]) + target * w[:, None]
    q[:, 2] = q[:, 2] * (1 - w) + logr * w
    q[:, 3:] *= 1 - w[:, None]
    u = np.maximum(z - neck, 0)
    q[:, :2] += (u * u / (u + taper))[:, None] * slope * np.array([np.cos(angle), np.sin(angle)])
    q[:, 2] += np.log1p(0.08 * np.maximum(z - neck, 0))  # tail widens past the throat
    # Static envelope containing a 0.34 R ball around every camera.
    axial = z[:, None] - cz[None]
    ball = np.sqrt(np.maximum(0, 0.34**2 - axial**2))
    lateral = np.linalg.norm(q[:, None, :2] - cams[None, :, :2], axis=2)
    required = np.max(np.where(np.abs(axial) < 0.34, ball + lateral, 0), axis=1)
    guaranteed = np.exp(q[:, 2] - np.hypot(q[:, 3], q[:, 4])) * (1 - p["fold"]["depth_frac"])
    q[:, 2] += np.log(np.maximum(1.0, required / np.maximum(guaranteed, 1e-8)))
    p.update(z_R=z.tolist(), profiles=q.tolist())
    p.pop("vector", None)
    # Throat anisotropy and a stationary shoulder bump, then the 0.36 R ball envelope again.
    q = np.array(p["profiles"])
    z = np.array(p["z_R"])
    u = np.clip((z - (zmax - x[7])) / (x[0] + x[7]), 0, 1)
    w = u * u * (3 - 2 * u)
    q[:, 3] += x[9] * w
    q[:, 4] += x[10] * w
    bump = np.exp(-0.5 * ((z - (neck + x[13])) / max(0.20, x[6])) ** 2)
    q[:, :2] += bump[:, None] * x[11:13]
    _contain_ball(q, z, ctx["camera_all"], p["fold"]["depth_frac"], 0.36)
    # Static dilation about the camera centroid (poses and reference radius unchanged).
    factor = x[14]
    center = ctx["world_camera_mean_R"]
    p["origin_R"] = (center + factor * (np.array(p["origin_R"]) - center)).tolist()
    p["z_R"] = (factor * z).tolist()
    q[:, :2] *= factor
    q[:, 2] += np.log(factor)
    p["profiles"] = q.tolist()
    p["fold"]["spacing_R"] *= factor
    p["fold"]["width_R"] *= factor
    p["lumen_fit"] = dict(vector=x.tolist(), neck_z_R=float(neck))
    p["static_dilation"] = dict(factor=float(factor), center_R=center.tolist())
    return p


def seed_side_wall(p0: Params, ctx: dict[str, Any], factor: float) -> Params:
    """The seed sweep as an open side wall, dilated by ``factor`` about the camera centroid."""
    p = copy.deepcopy(p0)
    p["seed_side_wall"] = True
    center = ctx["world_camera_mean_R"]
    q = np.array(p["profiles"])
    p["origin_R"] = (center + factor * (np.array(p["origin_R"]) - center)).tolist()
    p["z_R"] = (factor * np.array(p["z_R"])).tolist()
    q[:, :2] *= factor
    q[:, 2] += np.log(factor)
    p["profiles"] = q.tolist()
    p["fold"]["spacing_R"] *= factor
    p["fold"]["width_R"] *= factor
    p["static_dilation"] = dict(factor=factor, center_R=center.tolist())
    return p


# ----------------------------------------------------------------------------- rendering
def _mesh_tube(vertices, faces, chart) -> TubeModel:
    return TubeModel(
        centerline=np.zeros((0, 3)),
        frames=np.zeros((0, 3, 3)),
        radii=np.zeros((0, 0)),
        angles=np.zeros(0),
        arclength=np.zeros(0),
        vertices=vertices,
        faces=faces,
        chart=np.stack([np.asarray(c, dtype=np.float32) for c in chart]),
        meta=dict(kind="lumen_fit"),
    )


def _grid_scene(tube, cam, uv, size, field, texture=None) -> TubeScene:
    """A :class:`TubeScene` sampling rays at clip coordinates ``uv`` on a ``size`` grid."""
    scene = TubeScene(tube, cam, np.ones((1, 1), bool), texture, size=(1, 1))
    rays, ok = rays_from_clip(uv, cam)
    scene.rays, scene.ok, scene.size, scene.field = rays.astype(np.float32), ok, size, field
    return scene


def probe_depth(
    tube: TubeModel, cam: Calibration, field: np.ndarray, poses: np.ndarray
) -> np.ndarray:
    """Ray depth at every 8th clip pixel (float16; NaN outside the field or without a hit)."""
    yy, xx = np.mgrid[:832:8, :1104:8]
    scene = _grid_scene(
        tube, cam, np.c_[xx.ravel(), yy.ravel()], xx.shape, field[::8, ::8].astype(bool)
    )
    return np.asarray([scene.depth(pose).astype(np.float16) for pose in poses])


def sparse_frames(
    tube, cam, field, poses, light_radius, texture
) -> tuple[list[np.ndarray], np.ndarray]:
    """Shaded frames on the 160x120 analysis grid (and the grid field mask)."""
    import cv2

    yy, xx = np.meshgrid(
        (np.arange(GRID_H) + 0.5) * 832 / GRID_H - 0.5,
        (np.arange(GRID_W) + 0.5) * 1104 / GRID_W - 0.5,
        indexing="ij",
    )
    small = cv2.resize(field.astype(np.uint8), (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST)
    scene = _grid_scene(
        tube, cam, np.c_[xx.ravel(), yy.ravel()], (GRID_H, GRID_W), small.astype(bool), texture
    )
    return [scene.frame(pose, light_radius)[0] for pose in poses], scene.field


def opening_counts(depth: np.ndarray, field: np.ndarray) -> list[dict[str, Any]]:
    """Per frame: deep openings (> 2x median depth, > 1 % of the field) and wall crossing."""
    import cv2

    fs = field[::8, ::8] if field.shape == (832, 1104) else field
    out = []
    medwin = float(np.nanmedian(np.where(fs, depth, np.nan)))
    for d in depth.astype(np.float32):
        valid = fs & np.isfinite(d)
        med = float(np.median(d[valid]))
        _, _, st, _ = cv2.connectedComponentsWithStats((valid & (d > 2 * med)).astype(np.uint8), 8)
        area = st[1:, cv2.CC_STAT_AREA]
        miss = float((fs & ~np.isfinite(d)).sum() / fs.sum())
        near = float((valid & (d < 0.1 * medwin)).sum() / fs.sum())
        out.append(
            dict(
                opening_count=int((area > 0.01 * fs.sum()).sum()),
                wall_cross=bool(miss > 0.5 or near > 0.5),
            )
        )
    return out


# ----------------------------------------------------------------------------- fit
def fit_training_tube(
    seed_params: Params,
    poses: np.ndarray,
    radius: float,
    field_mask: np.ndarray,
    seed_depth: np.ndarray,
    frames: dict[str, np.ndarray],
    cam: Calibration,
    seed: int,
    cfg: dict[str, Any],
    work_dir: str | Path,
) -> tuple[Params, TubeModel, dict[str, Any]]:
    """Lumen fit of one training window.

    Args:
        seed_params: seed profile-sweep parameters of the window.
        poses: ``(T, 4, 4)`` camera-to-world poses of the window (pose units).
        radius: reference radius of the seed tube (pose units); sets the light radius and the
            clearance unit.
        field_mask: ``(832, 1104)`` bool field of the seed tube.
        seed_depth: ``(T, 104, 138)`` depth of the seed tube render (stride 8).
        frames: output of :func:`load_window_frames`.
        cam: fisheye calibration.
        seed: RNG seed of the search (base seed + window index).
        cfg: the ``training_fit`` section of ``configs/tube.yaml`` plus ``texture``.
        work_dir: directory for the finalist renders (``attempt<k>/anchor.mp4``).

    Returns:
        ``(params, tube, info)`` of the selected finalist; ``info`` holds the baseline and finalist
        scores, the screen and the selected attempt index.
    """
    from scipy.optimize import differential_evolution

    texture = load_texture(str(cfg["texture"]))
    radius = float(radius)
    field = field_mask.astype(bool)
    n = len(poses)
    ts, mt = sample_times(n, 2), sample_times(n, 4)
    real, real_motion = frames["real_spatial"], frames["real_motion"]
    fs = field[::8, ::8]
    small_field = _small_field(fs)
    luma_lo, luma_hi = cfg["luma_bounds"]

    # Seed tube reference scores.
    base_frames = [
        frame_agreement(real[k], seed_depth[t].astype(np.float32), fs, small_field, cam)
        for k, t in enumerate(ts)
    ]
    base = dict(
        spatial=window_agreement(base_frames),
        motion=camera_following(list(real_motion), list(frames["seed_motion"]), cam),
    )

    p0 = seed_params
    ctx = setup_context(p0, poses[ts], real, small_field, cam)
    ctx["camera_all"] = (poses[:, :3, 3] / radius - np.array(p0["origin_R"])) @ np.array(
        p0["basis"]
    ).T
    ctx["world_camera_mean_R"] = poses[:, :3, 3].mean(0) / radius
    ctx["zcammin"] = float(ctx["camera_all"][:, 2].min())
    ctx["zcammax"] = float(ctx["camera_all"][:, 2].max())
    rng = np.random.default_rng(seed)
    cache: dict[tuple, tuple] = {}
    best: list = [None]

    def agreement_of(depth_ts):
        rows = [
            frame_agreement(real[k], d.astype(np.float32), fs, small_field, cam)
            for k, d in enumerate(depth_ts)
        ]
        return rows, window_agreement(rows)

    def evaluate(x):
        x = np.clip(np.asarray(x), LOW, HIGH)
        key = tuple(np.round(x, 5))
        if key in cache:
            return
        p = construct(p0, ctx, x)
        if not cameras_inside(p, poses[:, :3, 3]).all():
            return
        tube = _mesh_tube(*tube_mesh(p))
        depth = probe_depth(tube, cam, field, poses[ts])
        rows, s = agreement_of(depth)
        value = (
            agreement_loss(s, rows)
            + margin_loss(depth, fs, rows)
            + 0.8 * (x[8] - 1) ** 2
            + 0.06 * x[7]
        )
        cache[key] = (value, x.copy(), s)
        if best[0] is None or value < best[0][0]:
            best[0] = (value, x.copy())

    good = ctx["good"]
    rad = np.clip(np.sqrt(np.median(ctx["areas"][good])) * 4, 0.10, 1.5) if good.any() else 0.25
    x0 = np.array(
        [1.6, np.log(rad), 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
    )
    for dist in [0.3, 0.8, 1.6, 3.2, 5.5]:
        for angle in [0, np.pi / 2, np.pi, -np.pi / 2]:
            for factor in [1.0, 1.6]:
                x = x0.copy()
                x[0], x[4], x[8] = dist, angle, factor
                evaluate(x)
    for _ in range(cfg["random_samples"]):
        x = x0.copy()
        x[0] = rng.uniform(0.15, 5.5)
        x[1] = rng.uniform(np.log(0.08), np.log(2))
        x[2:4] = rng.normal(0, 0.4, 2)
        x[4] = rng.uniform(-np.pi, np.pi)
        x[5] = rng.uniform(0.1, 8)
        x[6:8] = rng.uniform(0.05, 1.8, 2)
        x[8] = rng.uniform(1, 1.8)
        x[9:11] = rng.normal(0, 0.2, 2)
        x[11:13] = rng.normal(0, 0.4, 2)
        evaluate(x)
    if best[0] is None:
        raise ValueError("no candidate keeps every camera inside")

    def objective(v):
        x = np.r_[v, 1.0]
        evaluate(x)
        key = tuple(np.round(np.clip(x, LOW, HIGH), 5))
        return cache[key][0] if key in cache else 10000.0

    differential_evolution(
        objective,
        list(zip(LOW[:14], HIGH[:14], strict=True)),
        seed=seed,
        popsize=6,
        maxiter=cfg["evolution_generations"],
        polish=False,
        workers=1,
        updating="immediate",
        tol=0,
        atol=0,
    )
    for scale in [1.0, 0.5, 0.25, 0.12]:
        for _ in range(cfg["local_steps"]):
            pool = sorted(cache.values(), key=lambda v: v[0])[:12]
            parent = best[0][1] if rng.random() < 0.7 else pool[rng.integers(len(pool))][1]
            x = parent.copy()
            idx = rng.choice(14, size=rng.integers(1, 5), replace=False)
            x[idx] += rng.normal(size=len(idx)) * STEPS[idx] * scale
            x[4] = (x[4] + np.pi) % (2 * np.pi) - np.pi
            evaluate(x)

    # Diverse bank of the best candidates plus the two least-deformed ones.
    ordered = sorted(cache.values(), key=lambda v: v[0])
    bank = []
    for v in ordered:
        if (
            not bank
            or min(np.linalg.norm((v[1][:14] - u[1][:14]) / STEPS[:14]) for u in bank) > 0.9
        ):
            bank.append(v)
        if len(bank) == 10:
            break
    bank += sorted(
        cache.values(),
        key=lambda v: np.linalg.norm(v[1][2:4]) + abs(v[1][8] - 1) + np.linalg.norm(v[1][9:13]),
    )[:2]

    def screen_one(p, label, x):
        if not cameras_inside(p, poses[:, :3, 3]).all():
            return None, True
        v, f, chart = tube_mesh(p)
        tube = _mesh_tube(v, f, chart)
        depth = probe_depth(tube, cam, field, poses)
        if any(c["opening_count"] > 1 or c["wall_cross"] for c in opening_counts(depth, field)):
            return None, True
        if (
            label == "seed"
            and surface_distance(v, f, poses[:, :3, 3]).min() / radius < cfg["min_clearance"]
        ):
            return None, True
        rows, s = agreement_of(depth[ts])
        imgs, gfield = sparse_frames(tube, cam, field, poses[mt], radius, texture)
        lum = float(np.mean([im[gfield].mean(axis=0) @ LUMA for im in imgs]))
        motion = camera_following(list(real_motion), imgs, cam)
        ok_motion = int(motion["balanced"]) >= int(base["motion"]["balanced"]) and cosine_ok(
            motion, base["motion"], cfg["screen_cos_tolerance"]
        )
        ok_luma = luma_lo + 1 <= lum <= luma_hi - 1
        row = dict(
            bank=label,
            x=x,
            parameters=p if label == "seed" else None,
            spatial=s,
            loss=agreement_loss(s, rows) + margin_loss(depth[ts], fs, rows),
            luma_preview=lum,
            motion_preview=motion,
            motion_eligible=ok_motion,
            luma_eligible=ok_luma,
        )
        return row, not (ok_motion and ok_luma)

    screen = []
    for bi, (_, x, _) in enumerate(bank):
        for factor in cfg["body_scales"]:
            xx = x.copy()
            xx[14] = factor
            row, again = screen_one(construct(p0, ctx, xx), bi, xx)
            if row is not None:
                screen.append(row)
            if not again:
                break
    for factor in cfg["body_scales"]:
        row, again = screen_one(seed_side_wall(p0, ctx, factor), "seed", None)
        if row is not None:
            screen.append(row)
        if not again:
            break
    if not screen:
        raise ValueError("empty candidate bank")

    def block_cos(m):
        c = m["block_cos"]
        return c if c is not None and np.isfinite(c) else 0

    def preview_rank(a):
        return (
            not a["motion_eligible"],
            not a["luma_eligible"],
            a["spatial"]["status"] != "PASS",
            a["loss"],
            -block_cos(a["motion_preview"]),
        )

    screen_order = list(screen)  # evaluation order, before ranking
    screen.sort(key=preview_rank)
    finalists = [screen[0]]
    pure_spatial = min(
        screen, key=lambda a: (not a["luma_eligible"], a["spatial"]["status"] != "PASS", a["loss"])
    )
    seed_rows = [a for a in screen if a["bank"] == "seed"]
    for a in [pure_spatial] + ([min(seed_rows, key=preview_rank)] if seed_rows else []) + screen:
        if all(a is not b for b in finalists):
            finalists.append(a)
        if len(finalists) == 3:
            break

    attempts = []
    for ai, a in enumerate(finalists):
        p = a["parameters"] or construct(p0, ctx, a["x"])
        v, f, chart = tube_mesh(p)
        if surface_distance(v, f, poses[:, :3, 3]).min() / radius < cfg["min_clearance"]:
            continue
        record = _final_check(
            p,
            radius,
            (v, f, chart),
            poses,
            field,
            real,
            real_motion,
            cam,
            base,
            texture,
            cfg,
            Path(work_dir) / f"attempt{ai}",
        )
        eligible = (
            all(record["guards"].values())
            and record["motion_delta_balanced"] >= 0
            and cosine_ok(record["motion"], base["motion"], cfg["final_cos_tolerance"])
        )
        attempts.append(
            dict(index=ai, eligible=eligible, record=record, params=p, mesh=(v, f, chart))
        )
        if eligible and record["spatial"]["status"] == "PASS":
            break
    if not attempts:
        raise ValueError("no finalist keeps the camera clearance")

    def exact_rank(a):
        r = a["record"]
        s = r["spatial"]
        return (
            not all(r["guards"].values()),
            r["motion_delta_balanced"] < 0,
            not cosine_ok(r["motion"], base["motion"], cfg["final_cos_tolerance"]),
            s["status"] != "PASS",
            agreement_loss(s, [dict(tube_deep_components=0, tube_missing_depth_fraction=0)]),
            -block_cos(r["motion"]),
        )

    chosen = min(attempts, key=exact_rank)
    params = chosen["params"]
    v, f, chart = chosen["mesh"]
    tube = _mesh_tube(v, f, chart)
    tube.meta.update(kind="training_lumen_fit")
    info = dict(
        baseline=base,
        screen=[{k: a[k] for k in a if k != "parameters"} for a in screen_order],
        attempts=[
            dict(index=a["index"], eligible=a["eligible"], record=a["record"]) for a in attempts
        ],
        selected=chosen["index"],
        candidates_evaluated=len(cache),
    )
    return params, tube, info


def _small_field(fs: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(
        fs.astype(np.uint8), (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _final_check(
    p, radius, mesh, poses, field, real, real_motion, cam, base, texture, cfg, dst
) -> dict:
    """Full-resolution render of a finalist, encoded and decoded like a training video."""
    v, f, chart = mesh
    tube = _mesh_tube(v, f, chart)
    scene = TubeScene(tube, cam, field, texture)
    frames, depth = [], []
    for pose in poses:
        img, dep = scene.frame(pose, radius)
        frames.append(img)
        depth.append(dep[::8, ::8].astype(np.float16))
    dst.mkdir(parents=True, exist_ok=True)
    video = dst / "anchor.mp4"
    write_tube_video(video, np.asarray(frames))
    depth = np.asarray(depth).astype(np.float32)
    n = len(poses)
    ts, mt = sample_times(n, 2), sample_times(n, 4)
    fs = field[::8, ::8]
    small_field = _small_field(fs)
    rows = [frame_agreement(real[k], depth[t], fs, small_field, cam) for k, t in enumerate(ts)]
    spatial = window_agreement(rows)
    motion = camera_following(list(real_motion), list(decode_frames(video, mt)), cam)
    counts = opening_counts(depth, field)
    clear = surface_distance(v, f, poses[:, :3, 3]) / radius
    luma = decoded_mean_luma(video, field)
    lo, hi = cfg["luma_bounds"]
    guards = dict(
        max_one_lumen=all(c["opening_count"] <= 1 for c in counts),
        zero_wall_cross=not any(c["wall_cross"] for c in counts),
        clearance=bool(clear.min() >= cfg["min_clearance"]),
        luma=bool(lo <= luma <= hi),
    )
    return dict(
        spatial=spatial,
        motion=motion,
        motion_delta_balanced=int(motion["balanced"]) - int(base["motion"]["balanced"]),
        guards=guards,
        mean_luma=luma,
        clearance_min=float(clear.min()),
        video=str(video),
        depth=depth.astype(np.float16),
    )
