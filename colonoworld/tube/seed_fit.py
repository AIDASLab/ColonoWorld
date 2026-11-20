# SPDX-License-Identifier: CC-BY-NC-4.0
"""Seed sweep of a training window: 24-knot profile fit and native-rate re-render.

The seed is the starting point of :func:`colonoworld.tube.fit_training.fit_training_tube`. Two
routes produce it:

- **SDF route** (:func:`seed_from_sdf`): the carved window SDF (:mod:`colonoworld.tube.seed_sdf`) is
  sampled along 24 axial stations of the lumen frustum's axis; each station gets the robust
  ellipse of its first radial exits, refined against the signed distance, and haustral folds
  are read from radial deficits (or drawn from the training fold patterns). The sweep is
  inflated so a 0.18 R ball around every camera fits inside. When a projection of the sweep
  into a window camera shows no single opening beyond 8 R (or several deep regions), a
  low-order sweep (linear radius, straight or half-bent axis) is refitted to the SDF and the
  candidate with the fewest failing frames is kept.
- **Direct route** (:func:`seed_from_keyframes`): when no SDF is available, the ellipse
  profiles are fitted directly to the aligned keyframe wall points (``R = 0.3`` pose units),
  inflated for a 0.36 R camera ball, and enlarged in steps (x1.15 ... x2) until the
  projections show one opening, little lumen blockage (<= 20 % of the dark lumen nearer than
  3 R) and no wall crossing.

:func:`render_seed` renders the seed at the native frame rate (consecutive decoded frames from
the window's first frame) with the training-tube shader; its frames and stride-8 depth are the
``seed_video`` / ``seed_depth`` inputs of the lumen fit.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from colonoworld.tube.camera import Calibration, rays_from_clip
from colonoworld.tube.model import TubeModel
from colonoworld.tube.wall_fit import (
    Params,
    arc_length,
    centreline,
    ellipse_radius,
    interp_profile,
    sweep_mesh,
)

KNOTS = 24
PHI = np.linspace(-np.pi, np.pi, 72, endpoint=False)
FOLD_KEYS = ("spacing_R", "depth_frac", "width_R", "phase", "taenia_rad", "span_rad")
DIRECT_RADIUS = 0.3  # pose units; reference radius of the direct route
RADIUS_STEPS = (1.0, 1.15, 1.35, 1.6, 2.0)


# ----------------------------------------------------------------------------- geometry
def sdf_sampler(sdf: np.ndarray, lo: np.ndarray, step: float) -> Callable[[np.ndarray], np.ndarray]:
    """Trilinear SDF lookup (``+3`` outside the grid) of points in units of R."""
    from scipy.ndimage import map_coordinates

    def ev(x: np.ndarray) -> np.ndarray:
        return map_coordinates(
            sdf, ((np.asarray(x) - lo) / step).T, order=1, mode="constant", cval=3.0
        )

    return ev


def closed_sweep(p: Params) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], dict]:
    """Sweep mesh closed by two end fans (pose units) and its wall chart."""
    v, f, (sv, tv, fv), rings = sweep_mesh(p)
    n_phi = len(rings["phi"])
    ns = len(rings["z"])
    c = rings["C"]
    caps = (np.vstack([c[0], c[-1]]) * p["R_pose"]).astype(np.float32)
    a = np.arange(n_phi)
    b = np.roll(a, -1)
    faces = np.vstack(
        [
            f,
            np.c_[np.full(n_phi, ns * n_phi), b, a],
            np.c_[np.full(n_phi, ns * n_phi + 1), (ns - 1) * n_phi + a, (ns - 1) * n_phi + b],
        ]
    ).astype(np.int32)
    chart = (
        np.r_[sv, sv[[0, -1]]].astype(np.float32),
        np.r_[tv, 0.0, 0.0].astype(np.float32),
        np.r_[fv, fv[[0, -1]]].astype(np.float32),
    )
    return np.vstack([v, caps]), faces, chart, rings


def _fold_scale(p: Params, z: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Fold depth at points with axial station ``z`` and angle ``ph`` (one value per point)."""
    f = p["fold"]
    s = arc_length(p, z)
    smax = float(arc_length(p, np.array([p["z_R"][-1]]))[0])
    out = np.zeros(len(z))
    space = f["spacing_R"]
    for k, sj in enumerate(np.arange(-space + f["phase"] * space, smax + space, space)):
        for sec in [k % 3] if k % 3 else [k % 3, (k + 1) % 3]:
            dp = np.angle(np.exp(1j * (ph - (f["taenia_rad"] + np.pi / 3 + sec * 2 * np.pi / 3))))
            v0 = np.clip(abs(dp) / (f["span_rad"] / 2), 0, 1)
            a = np.cos(np.pi * v0 / 2) ** 2 * (v0 < 1)
            g = np.exp(-0.5 * ((s - sj - 0.08 * np.cos(dp)) / f["width_R"]) ** 2)
            out = np.maximum(out, f["depth_frac"] * g * a)
    return out


def inside_sweep(p: Params, points: np.ndarray) -> np.ndarray:
    """Whether points (pose units) lie strictly inside the folded sweep."""
    b = np.asarray(p["basis"])
    d = np.asarray(points) / p["R_pose"] - np.asarray(p["origin_R"])
    z = d @ b[2]
    zc = np.clip(z, p["z_R"][0], p["z_R"][-1])
    q = interp_profile(p, zc)
    v = d @ b[:2].T - q[:, :2]
    ph = np.arctan2(v[:, 1], v[:, 0])
    rad = ellipse_radius(q, ph) * (1 - _fold_scale(p, zc, ph))
    return (z > p["z_R"][0]) & (z < p["z_R"][-1]) & (np.linalg.norm(v, axis=1) < rad)


def contain_cameras(p: Params, poses: np.ndarray, ball: float) -> tuple[Params, float]:
    """Inflate the sweep so a ball of ``ball`` R around every camera fits inside."""
    p = copy.deepcopy(p)
    b = np.asarray(p["basis"])
    d = poses[:, :3, 3] / p["R_pose"] - np.asarray(p["origin_R"])
    z = d @ b[2]
    q = interp_profile(p, z)
    dd = d @ b[:2].T - q[:, :2]
    rho = np.linalg.norm(dd, axis=1)
    ph = np.arctan2(dd[:, 1], dd[:, 0])
    depth = p["fold"]["depth_frac"]
    factor = max(1.0, float(np.max((rho + ball) / (ellipse_radius(q, ph) * (1 - depth)))))
    p["profiles"] = np.asarray(p["profiles"]).copy()
    p["profiles"][:, 2] += np.log(factor)
    return p, factor


def draw_fold(key: str, fold_prior: list[dict[str, float]]) -> dict[str, float]:
    """Fold pattern drawn (seeded by the window key) from the training fold patterns."""
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    fold = dict(fold_prior[int(rng.integers(len(fold_prior)))])
    fold["phase"] = float(rng.uniform())
    return fold


def _as_lists(p: Params) -> Params:
    """Parameters with plain-list arrays (the stored form)."""
    out = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {
                kk: (float(vv) if isinstance(vv, np.floating) else vv) for kk, vv in v.items()
            }
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- projections
def probe_depth(
    vertices: np.ndarray, faces: np.ndarray, poses: np.ndarray, cam: Calibration, field: np.ndarray
) -> np.ndarray:
    """``(T, 104, 138)`` float16 ray distance at every 8th pixel (NaN: no hit or outside field)."""
    import drjit as dr
    import mitsuba as mi

    mi.set_variant("llvm_ad_rgb")
    dr.set_thread_count(1)
    mesh = mi.Mesh("seed_probe", len(vertices), len(faces))
    prm = mi.traverse(mesh)
    prm["vertex_positions"] = mi.Float(np.asarray(vertices).reshape(-1))
    prm["faces"] = mi.UInt32(np.asarray(faces).astype(np.uint32).reshape(-1))
    prm.update()
    scene = mi.load_dict({"type": "scene", "wall": mesh})
    yy, xx = np.mgrid[:832:8, :1104:8]
    ray0, valid = rays_from_clip(np.c_[xx.ravel(), yy.ravel()], cam)
    ray0 = ray0.astype(np.float32)
    out = []
    for pose in poses:
        d = ray0 @ pose[:3, :3].T.astype(np.float32)
        o = pose[:3, 3]
        ray = mi.Ray3f(
            mi.Point3f(*[np.full(len(d), x, np.float32) for x in o]),
            mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]),
        )
        t = np.array(scene.ray_intersect_preliminary(ray).t, np.float32)
        t[~valid | ~np.isfinite(t)] = np.nan
        out.append(t.reshape(104, 138).astype(np.float16))
    depth = np.asarray(out)
    depth[:, ~np.asarray(field)[::8, ::8].astype(bool)] = np.nan
    return depth


def opening_counts(depth: np.ndarray, field: np.ndarray, radius: float) -> list[dict[str, Any]]:
    """Per frame: connected regions beyond 8 R, deep regions (> 2 x median depth, > 1 % of the
    field) and wall crossing (more than half the field missing or nearer than 0.1 x median)."""
    import cv2

    fs = field[::8, ::8] if field.shape == (832, 1104) else field
    medwin = float(np.nanmedian(np.where(fs, depth, np.nan)))
    out = []
    for d in depth.astype(np.float32):
        valid = fs & np.isfinite(d)
        med = float(np.median(d[valid]))
        _, _, st, _ = cv2.connectedComponentsWithStats((valid & (d > 2 * med)).astype(np.uint8), 8)
        area = st[1:, cv2.CC_STAT_AREA]
        miss = float((fs & ~np.isfinite(d)).sum() / fs.sum())
        near = float((valid & (d < 0.1 * medwin)).sum() / fs.sum())
        n8 = cv2.connectedComponents((valid & (d >= 8 * radius)).astype(np.uint8), 8)[0] - 1
        out.append(
            dict(
                deep=int((area > 0.01 * fs.sum()).sum()),
                openings_8R=int(n8),
                wall_cross=bool(miss > 0.5 or near > 0.5),
            )
        )
    return out


def opening_rank(counts: list[dict[str, Any]]) -> tuple[int, int]:
    """``(frames without exactly one 8 R opening, frames with several deep regions)``."""
    return sum(c["openings_8R"] != 1 for c in counts), sum(c["deep"] > 1 for c in counts)


# ----------------------------------------------------------------------------- mesh distance
def _closest_triangle_distance(tris: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance from ``p`` to each triangle (Ericson's closest-point regions)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp, cp = p - b, p - c
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    v = vb / np.maximum(den, 1e-25)
    w = vc / np.maximum(den, 1e-25)
    out = a + v[:, None] * ab + w[:, None] * ac
    m = (d1 <= 0) & (d2 <= 0)
    out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    out[m] = c[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, 1e-25)
    out[m] = (a + t[:, None] * ab)[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, 1e-25)
    out[m] = (a + t[:, None] * ac)[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-25)
    out[m] = (b + t[:, None] * (c - b))[m]
    return np.linalg.norm(out - p, axis=1)


def surface_distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact distance from each point to a triangle mesh."""
    from scipy.spatial import cKDTree

    tri = vertices[faces].astype(float)
    cent = tri.mean(1)
    rad = np.linalg.norm(tri - cent[:, None], axis=2).max(1)
    rmax = rad.max()
    tree = cKDTree(cent)
    upper = cKDTree(vertices).query(points, workers=1)[0]
    out = []
    for p, u in zip(points, upper, strict=True):
        idx = np.asarray(tree.query_ball_point(p, float(u + rmax + 1e-7)), int)
        idx = idx[np.linalg.norm(cent[idx] - p, axis=1) - rad[idx] <= u + 1e-7]
        out.append(float(_closest_triangle_distance(tri[idx], p).min()))
    return np.array(out)


def mesh_to_carve_median(
    p: Params, poses: np.ndarray, ref_vertices: np.ndarray, ref_faces: np.ndarray
) -> float:
    """Median distance (units of R) from 500 sweep vertices near the camera path (from 0.75 R
    behind the rear-most camera to 4 R ahead of the front-most one) to the carved mesh."""
    radius = p["R_pose"]
    v, _, _, _ = closed_sweep(p)
    b = np.asarray(p["basis"])
    origin = np.asarray(p["origin_R"])
    zc = (poses[:, :3, 3] / radius - origin) @ b[2]
    zv = (v / radius - origin) @ b[2]
    use = np.where((zv >= zc.min() - 0.75) & (zv <= zc.max() + 4.0))[0]
    sel = use[np.linspace(0, len(use) - 1, min(500, len(use))).astype(int)]
    return float(np.median(surface_distance(ref_vertices, ref_faces, v[sel]) / radius))


# ----------------------------------------------------------------------------- SDF route
def fit_profile(
    ev: Callable[[np.ndarray], np.ndarray], poses: np.ndarray, radius: float, frustum: dict
) -> tuple[Params, bool]:
    """24-knot ellipse sweep fitted to the carved SDF; returns the sweep and fold reliability."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares

    b = np.array([frustum["u"], frustum["v"], frustum["n"]], dtype=float)
    if np.linalg.det(b) < 0:
        b[1] *= -1
    origin = np.asarray(frustum["origin"], dtype=float)
    zc = (poses[:, :3, 3] / radius - origin) @ b[2]
    zlo, zhi = float(zc.min() - 1.75), float(zc.max() + 8.8)
    z = np.linspace(zlo, zhi, KNOTS)
    raydir = np.cos(PHI)[:, None] * b[0] + np.sin(PHI)[:, None] * b[1]
    rho = np.linspace(0, 6.5, 180)
    lower = [-4, -4, np.log(0.09), -0.8, -0.8]
    upper = [4, 4, np.log(6), 0.8, 0.8]
    prof = []
    for zi in z:
        ax = origin + zi * b[2]
        c = np.zeros(2)
        if ev(ax[None])[0] >= -0.04:
            # Station centre outside the lumen: start from the nearest clearly negative point.
            grid = np.linspace(-3.5, 3.5, 45)
            gx, gy = np.meshgrid(grid, grid)
            xy = np.c_[gx.ravel(), gy.ravel()]
            ok = ev(ax + xy @ b[:2]) < -0.08
            if ok.any():
                c = xy[np.where(ok)[0][np.argmin(np.linalg.norm(xy[ok], axis=1))]]
        sol = None
        for _ in range(2):
            center = ax + c @ b[:2]
            vals = ev((center + rho[None, :, None] * raydir[:, None]).reshape(-1, 3)).reshape(
                len(PHI), -1
            )
            outside = vals > 0
            ix = outside.argmax(1)
            ok = outside.any(1) & (ix > 0) & (vals[:, 0] < 0)
            if ok.sum() < 18:
                break
            jj = np.where(ok)[0]
            k = ix[ok]
            v0, v1 = vals[jj, k - 1], vals[jj, k]
            rad = rho[k - 1] - v0 * (rho[k] - rho[k - 1]) / np.maximum(v1 - v0, 1e-8)
            pts = c + rad[:, None] * np.c_[np.cos(PHI[ok]), np.sin(PHI[ok])]

            def residual(t, pts=pts, c=c):
                d = pts - t[:2]
                pred = ellipse_radius(np.r_[0, 0, t[2:]], np.arctan2(d[:, 1], d[:, 0]))
                return np.r_[np.linalg.norm(d, axis=1) - pred, 0.12 * (t[:2] - c), 0.035 * t[3:]]

            t0 = np.r_[c, np.log(np.median(rad)), 0.01, 0] if sol is None else sol
            t0 = np.clip(
                t0,
                [-3.999, -3.999, np.log(0.091), -0.799, -0.799],
                [3.999, 3.999, np.log(5.999), 0.799, 0.799],
            )
            sol = least_squares(
                residual, t0, bounds=(lower, upper), loss="soft_l1", f_scale=0.12, max_nfev=45
            ).x
            c = sol[:2]
        if sol is None:
            sol = np.array(prof[-1]) if prof else np.zeros(5)
        prof.append(sol)
    prof = np.asarray(prof)
    prof[:, :2] = gaussian_filter1d(prof[:, :2], 0.6, axis=0, mode="nearest")
    prof[:, 2:] = gaussian_filter1d(prof[:, 2:], 0.42, axis=0, mode="nearest")
    p = dict(
        R_pose=radius,
        origin_R=origin,
        basis=b,
        z_R=z,
        profiles=prof,
        fold=dict(
            spacing_R=1.0, depth_frac=0.0, width_R=0.07, phase=0.0, taenia_rad=0.0, span_rad=np.pi
        ),
    )
    # Refine each station against the signed distance at the model surface.
    for i, zi in enumerate(z):
        t0 = prof[i].copy()
        ax = origin + zi * b[2]

        def residual(t, ax=ax, t0=t0):
            x = ax + t[:2] @ b[:2] + ellipse_radius(t, PHI)[:, None] * raydir
            return np.r_[ev(x), 0.12 * (t - t0)]

        box = np.array([0.25, 0.25, 0.25, 0.2, 0.2])
        prof[i] = least_squares(
            residual, t0, bounds=(t0 - box, t0 + box), loss="soft_l1", f_scale=0.10, max_nfev=25
        ).x
    p["profiles"] = prof
    reliable, p["fold"] = _fit_folds(p, ev, zc, zlo, zhi, raydir)
    return p, reliable


def _fit_folds(p, ev, zc, zlo, zhi, raydir):
    """Haustral folds from radial deficits of the SDF along the fitted sweep."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    fine = np.linspace(max(zlo, zc.min() - 0.3), min(zhi, zc.max() + 4.5), 180)
    q = interp_profile(p, fine)
    pred = ellipse_radius(q[:, None, :], PHI[None])
    cent = centreline(p, fine)
    scales = np.linspace(0.45, 1.3, 36)
    x = (
        cent[:, None, None]
        + (pred[:, :, None] * scales[None, None])[:, :, :, None] * raydir[None, :, None]
    )
    vals = ev(x.reshape(-1, 3)).reshape(len(fine), len(PHI), len(scales))
    ix = (vals > 0).argmax(-1)
    good = (ix > 0) & (vals[:, :, 0] < 0)
    ix = np.maximum(ix, 1)
    low = np.take_along_axis(vals, (ix - 1)[..., None], 2)[..., 0]
    high = np.take_along_axis(vals, ix[..., None], 2)[..., 0]
    obs = scales[ix - 1] - low * (scales[ix] - scales[ix - 1]) / np.maximum(high - low, 1e-6)
    deficit = np.where(good, np.clip(1 - obs, 0, 0.6), 0)
    signal = np.percentile(deficit, 70, axis=1)
    sig = signal - gaussian_filter1d(signal, 9)
    peaks, _ = find_peaks(sig, prominence=0.025, distance=12)
    arc = arc_length(p, fine)
    reliable = len(peaks) >= 2 and np.mean(good) > 0.65
    theta = 0.0
    if len(peaks):
        row = deficit[peaks[np.argmax(sig[peaks])]]
        angles = np.linspace(-np.pi, np.pi, 60, endpoint=False)
        scores = []
        for a in angles:
            waves = []
            for sec in range(3):
                dph = np.angle(np.exp(1j * (PHI - a - np.pi / 3 - sec * 2 * np.pi / 3)))
                waves.append(np.cos(np.pi / 2 * np.clip(np.abs(dph) / (np.pi / 2), 0, 1)) ** 2)
            scores.append(max(float(row @ w) for w in waves))
        theta = float(angles[np.argmax(scores)])
    spacing = float(np.clip(np.median(np.diff(arc[peaks])), 0.4, 2.5)) if len(peaks) > 1 else 1.0
    depth = float(np.clip(np.median(signal[peaks]), 0.025, 0.40)) if len(peaks) else 0.06
    width = 0.07
    if len(peaks):
        widths = []
        for k in peaks:
            inds = np.where(sig[max(0, k - 8) : k + 9] > 0.5 * sig[k])[0]
            widths.append(len(inds) * np.median(np.diff(arc)) / 2.355)
        width = float(np.clip(np.median(widths), 0.035, 0.20))
    fold = dict(
        spacing_R=spacing,
        depth_frac=depth,
        width_R=width,
        phase=float((arc[peaks[0]] / spacing) % 1) if len(peaks) else 0.0,
        taenia_rad=theta,
        span_rad=float(np.pi),
    )
    return bool(reliable), fold


def finalize_profile(
    p: Params, reliable: bool, poses: np.ndarray, key: str, fold_prior: list[dict]
) -> Params:
    """Folds (fitted, or drawn from the training patterns) and the 0.18 R camera containment."""
    p = json_roundtrip(p)
    if not reliable:
        p["fold"] = draw_fold(key, fold_prior)
    p, _ = contain_cameras(p, poses, 0.18)
    return p


def json_roundtrip(p: Params) -> Params:
    """Deep copy with plain lists (values are unchanged: float64 round-trips exactly)."""
    import json

    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [conv(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    return json.loads(json.dumps(conv(p)))


def strict_refit(
    base: Params,
    ev: Callable[[np.ndarray], np.ndarray],
    frustum_slope: np.ndarray,
    poses: np.ndarray,
    cam: Calibration,
    field: np.ndarray,
) -> tuple[Params, tuple]:
    """Low-order sweep (linear log-radius, scaled axis offsets, constant anisotropy) refitted to the
    SDF; trials in order of SDF error, stopping at the first with one opening in every frame."""
    from scipy.optimize import least_squares

    radius = base["R_pose"]
    zz = np.array(base["z_R"])
    b = np.array(base["basis"])
    orig = np.array(base["profiles"])
    d = poses[:, :3, 3] / radius - np.array(base["origin_R"])
    zc = d @ b[2]
    xy = d @ b[:2].T
    z = np.linspace(zc.min() - 0.5, zc.max() + 3.5, 24)
    ph2 = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    dirs = np.cos(ph2)[:, None] * b[0] + np.sin(ph2)[:, None] * b[1]
    ec = 0.5 * np.log(frustum_slope[0] / frustum_slope[1])
    e0 = np.array([np.clip(ec, -0.6, 0.6), 0.0])
    phase = dict(base["fold"])["phase"]

    def build(a, slope, alpha, ecc):
        p = json_roundtrip(base)
        q = orig.copy()
        q[:, :2] *= alpha
        q[:, 2] = np.log(np.maximum(a + slope * (zz - zz[0]), 0.08))
        q[:, 3:] = ecc[None]
        p["profiles"] = q
        p["fold"]["phase"] = phase
        qq = interp_profile(p, zc)
        dd = xy - qq[:, :2]
        rho = np.linalg.norm(dd, axis=1)
        rad = ellipse_radius(qq, np.arctan2(dd[:, 1], dd[:, 0])) * (1 - p["fold"]["depth_frac"])
        q[:, 2] += np.log(max(1.0, float(np.max((rho + 0.18) / rad))))
        p["profiles"] = q
        return p

    def surface_points(p):
        rad = ellipse_radius(interp_profile(p, z)[:, None, :], ph2[None])
        return (centreline(p, z)[:, None] + rad[:, :, None] * dirs[None]).reshape(-1, 3)

    sol = least_squares(
        lambda t: ev(surface_points(build(t[0], t[1], 0, e0))),
        [0.7, 0.07],
        bounds=([0.12, 0.0], [2.8, 0.5]),
        loss="soft_l1",
        f_scale=0.15,
        max_nfev=35,
    )
    trials = []
    for alpha in (0.5, 0.0):
        for af, bf in [
            (1, 1),
            (1, 0.5),
            (1, 0),
            (0.8, 0),
            (1.2, 0),
            (1.5, 0),
            (1.1, 1.2),
            (0.8, 0.5),
        ]:
            for ecc in (e0, np.zeros(2)):
                p = build(sol.x[0] * af, sol.x[1] * bf, alpha, ecc)
                trials.append((float(np.median(np.abs(ev(surface_points(p))))), p))
    trials.sort(key=lambda x: x[0])
    best = None
    for error, p in trials:
        v, f, _, _ = closed_sweep(p)
        rank = (
            *opening_rank(opening_counts(probe_depth(v, f, poses, cam, field), field, radius)),
            error,
        )
        if best is None or rank < best[0]:
            best = (rank, p)
        if rank[:2] == (0, 0):
            break
    return best[1], best[0]


def seed_from_sdf(
    sdf: np.ndarray,
    lo: np.ndarray,
    step: float,
    frustum: dict,
    poses: np.ndarray,
    radius: float,
    cam: Calibration,
    field: np.ndarray,
    key: str,
    fold_prior: list[dict],
) -> tuple[Params, dict[str, Any]]:
    """SDF route: profile fit, folds, containment and the opening check / low-order refit.

    ``poses``: window camera-to-world (rebased, reconstruction timeline); ``key``: window name
    (seeds the fold draw).
    """
    ev = sdf_sampler(sdf, lo, step)
    p, reliable = fit_profile(ev, poses, radius, frustum)
    base = finalize_profile(p, reliable, poses, key, fold_prior)
    v, f, _, _ = closed_sweep(base)
    rank = opening_rank(opening_counts(probe_depth(v, f, poses, cam, field), field, radius))
    info: dict[str, Any] = dict(fold_fitted=reliable, base_rank=rank, refit=False)
    if rank == (0, 0):
        return json_roundtrip(base), info
    refit, refit_rank = strict_refit(base, ev, np.asarray(frustum["slope"]), poses, cam, field)
    v, f, _, _ = closed_sweep(refit)
    final_rank = opening_rank(opening_counts(probe_depth(v, f, poses, cam, field), field, radius))
    info.update(refit_rank=final_rank, refit_search=refit_rank)
    if final_rank == rank:
        # Equal opening ranks: the candidate closer to the carved surface is kept.
        from colonoworld.tube.seed_sdf import carve_mesh

        ref_v, ref_f = carve_mesh(sdf, lo, step, radius)
        errors = [mesh_to_carve_median(c, poses, ref_v, ref_f) for c in (base, refit)]
        info["mesh_to_carve"] = errors
        rank, final_rank = (*rank, errors[0]), (*final_rank, errors[1])
    if final_rank < rank:
        info["refit"] = True
        return json_roundtrip(refit), info
    return json_roundtrip(base), info


# ----------------------------------------------------------------------------- direct route
def lumen_masks(video: str, video_indices: np.ndarray, cam: Calibration, field: np.ndarray):
    """Dark-lumen masks (cue grid) of the given frames."""
    import cv2

    from colonoworld.tube.build_from_video import lumen_mask

    wanted = set(int(i) for i in video_indices)
    frames = {}
    import av

    with av.open(str(video)) as container:
        container.streams.video[0].thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                frames[i] = cv2.resize(
                    frame.to_ndarray(format="rgb24", threads=1),
                    (276, 208),
                    interpolation=cv2.INTER_AREA,
                )
            if i >= max(wanted):
                break
    fs = field[::4, ::4].astype(bool)
    return np.array([lumen_mask(frames[int(j)], cam, fs) for j in video_indices], np.uint8)


def fit_profile_direct(
    video: str,
    cam: Calibration,
    all_poses: np.ndarray,
    video_indices: np.ndarray,
    keyframes: dict[str, Any],
    field: np.ndarray,
) -> Params:
    """Ellipse profiles fitted directly to the accepted keyframe wall points (``R = 0.3``)."""
    import cv2
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares

    from colonoworld.tube.seed_depth import DEPTH_H, DEPTH_W, gray, pixel_rays, rect_maps

    vi = np.asarray(video_indices)
    pc = all_poses[vi].astype(float)
    inv0 = np.linalg.inv(pc[0])
    poses = inv0[None] @ pc
    idx, kp, depth, ok = keyframes["idx"], keyframes["P"], keyframes["depth"], keyframes["ok"]
    k_depth, support = keyframes["Kd"], keyframes["support"]
    js = np.flatnonzero(ok & (idx >= vi.min() - 6) & (idx <= vi.max() + 6))
    if not len(js):
        raise ValueError("no accepted keyframe depth in the window")
    keymask = lumen_masks(video, idx[js], cam, field)
    _, mx, my, valid0, _, _ = rect_maps(cam)
    import av

    wanted, keygray = set(map(int, idx[js])), {}
    with av.open(str(video)) as container:
        container.streams.video[0].thread_count = 1
        for j, frame in enumerate(container.decode(video=0)):
            if j in wanted:
                rgb = cv2.remap(
                    frame.to_ndarray(format="rgb24", threads=1),
                    mx,
                    my,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                rgb[~valid0] = 0
                keygray[j] = gray(rgb)
            if j >= max(wanted):
                break
    rx = cv2.resize((mx + 0.5) / 4 - 0.5, (DEPTH_W, DEPTH_H))
    ry = cv2.resize((my + 0.5) / 4 - 0.5, (DEPTH_W, DEPTH_H))
    rect_field = (
        cv2.resize(
            cv2.remap(field.astype(np.float32), mx, my, cv2.INTER_NEAREST),
            (DEPTH_W, DEPTH_H),
            interpolation=cv2.INTER_AREA,
        )
        == 1
    )
    rays = pixel_rays(k_depth)
    points = []
    for a, k in enumerate(js):
        dep = depth[k].astype(np.float32)
        lm = (
            cv2.remap(cv2.dilate(keymask[a], np.ones((9, 9), np.uint8)), rx, ry, cv2.INTER_NEAREST)
            > 0
        )
        spec = (
            cv2.dilate((keygray[int(idx[k])] >= 235).astype(np.uint8), np.ones((7, 7), np.uint8))
            > 0
        )
        good = support & rect_field & np.isfinite(dep) & (dep > 0.02) & (dep < 30) & ~lm & ~spec
        if good.sum() < 100:
            continue
        good &= dep < 2.5 * np.median(dep[good])
        good[1::2] = False
        good[:, 1::2] = False
        q = inv0 @ kp[k]
        points.append((rays[good] * dep[good, None]) @ q[:3, :3].T + q[:3, 3])
    if not points:
        raise ValueError("no wall observations in the window")
    r = DIRECT_RADIUS
    x = np.concatenate(points) / r
    campos = poses[:, :3, 3] / r
    forward = np.mean(poses[:, :3, 2], axis=0)
    forward /= np.linalg.norm(forward)
    u = np.cross([0, 1, 0], forward)
    if np.linalg.norm(u) < 0.1:
        u = np.cross([1, 0, 0], forward)
    u /= np.linalg.norm(u)
    v = np.cross(forward, u)
    b = np.stack([u, v, forward])
    zc = campos @ forward
    zz = np.linspace(zc.min() - 1.75, zc.max() + 8.8, KNOTS)
    xz = x @ forward
    xy = x @ b[:2].T
    profiles = []
    for z in zz:
        obs = xy[np.abs(xz - z) < 0.55]
        if len(obs) >= 40:

            def residual(t, obs=obs):
                d = obs - t[:2]
                pred = ellipse_radius(np.r_[0, 0, t[2:]], np.arctan2(d[:, 1], d[:, 0]))
                return np.r_[np.linalg.norm(d, axis=1) - pred, 0.3 * t[:2], 0.2 * t[3:]]

            rad = np.clip(np.median(np.linalg.norm(obs, axis=1)), 0.2, 5.5)
            sol = least_squares(
                residual,
                [0, 0, np.log(rad), 0, 0],
                bounds=([-4, -4, np.log(0.09), -0.8, -0.8], [4, 4, np.log(6), 0.8, 0.8]),
                loss="soft_l1",
                f_scale=0.12,
                max_nfev=60,
            ).x
        elif profiles:
            sol = profiles[-1].copy()
        else:
            sol = np.zeros(5)
        profiles.append(sol)
    q = gaussian_filter1d(np.array(profiles), 0.8, axis=0, mode="nearest")
    return dict(
        R_pose=r,
        origin_R=[0.0, 0.0, 0.0],
        basis=b.tolist(),
        z_R=zz.tolist(),
        profiles=q.tolist(),
        fold=dict(
            spacing_R=1.0,
            depth_frac=0.06,
            width_R=0.07,
            phase=0.0,
            taenia_rad=0.0,
            span_rad=float(np.pi),
        ),
    )


def _low_order_trials(
    p: Params, poses: np.ndarray, ev: Callable[[np.ndarray], np.ndarray] | None = None
) -> list[tuple[Params, dict]]:
    """Low-order sweeps (linear radius, scaled axis offsets) ranked by the median SDF magnitude at
    their surface (``ev`` given; the radius line is first fitted to the SDF) or else by the
    distance to the profile."""
    from scipy.optimize import least_squares

    zz = np.array(p["z_R"])
    orig = np.array(p["profiles"])
    ecc = np.median(orig[:, 3:], axis=0)
    b_ax = np.array(p["basis"])
    zc = (poses[:, :3, 3] / p["R_pose"] - np.array(p["origin_R"])) @ b_ax[2]
    zs = np.linspace(zc.min() - 0.5, zc.max() + 3.5, 24)
    ph = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    dirs = np.cos(ph)[:, None] * b_ax[0] + np.sin(ph)[:, None] * b_ax[1]

    def build(a, slope, alpha, ec):
        trial = copy.deepcopy(p)
        q = orig.copy()
        q[:, :2] *= alpha
        q[:, 2] = np.log(np.maximum(a + slope * (zz - zz[0]), 0.08))
        q[:, 3:] = ec
        trial["profiles"] = q
        return contain_cameras(trial, poses, 0.36)[0]

    def surface_points(trial):
        q = interp_profile(trial, zs)
        rad = ellipse_radius(q[:, None, :], ph[None])
        return (centreline(trial, zs)[:, None] + rad[:, :, None] * dirs[None]).reshape(-1, 3)

    radius = np.exp(orig[:, 2])
    a = float(np.clip(np.median(radius), 0.12, 2.8))
    b = float(np.clip((radius[-1] - radius[0]) / (zz[-1] - zz[0]), 0, 0.5))
    if ev is not None:
        a, b = least_squares(
            lambda t: ev(surface_points(build(t[0], t[1], 0, ecc))),
            [a, b],
            bounds=([0.12, 0], [2.8, 0.5]),
            loss="soft_l1",
            f_scale=0.15,
            max_nfev=35,
        ).x
    trials = []
    for alpha in (0.5, 0.0):
        for af, bf in [
            (1, 1),
            (1, 0.5),
            (1, 0),
            (0.8, 0),
            (1.2, 0),
            (1.5, 0),
            (1.1, 1.2),
            (0.8, 0.5),
        ]:
            for ec in (ecc, np.zeros(2)):
                trial = build(a * af, b * bf, alpha, ec)
                if ev is not None:
                    error = float(np.median(np.abs(ev(surface_points(trial)))))
                else:
                    error = float(np.mean(np.abs(np.array(trial["profiles"]) - orig)))
                trials.append((error, trial, dict(family="low_order", radius_factor=af)))
    return [(t, m) for _, t, m in sorted(trials, key=lambda x: x[0])]


def enlargement_search(
    p: Params,
    poses: np.ndarray,
    cam: Calibration,
    field: np.ndarray,
    masks: np.ndarray,
    ev: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[Params, dict[str, Any]]:
    """First candidate (radius x1 ... x2, then low-order sweeps) whose camera projections show one
    opening, at most 20 % of the dark lumen ``masks`` nearer than 3 R, no missing lumen depth or
    holes and no wall crossing, with every camera inside."""
    radius = p["R_pose"]
    fs = field[::8, ::8]
    lm = masks[:, ::2, ::2].astype(bool)
    trials = []
    for scale in RADIUS_STEPS:
        trial = copy.deepcopy(p)
        trial["profiles"] = np.asarray(p["profiles"]).copy()
        trial["profiles"][:, 2] += np.log(scale)
        trials.append((trial, dict(family="radius", radius_factor=scale)))

    def candidates():
        yield from trials
        yield from _low_order_trials(p, poses, ev)

    for trial, meta in candidates():
        v, f, _, _ = closed_sweep(trial)
        d = probe_depth(v, f, poses, cam, field)
        cs = opening_counts(d, field, radius)
        finite = np.isfinite(d)
        n_lumen = int(lm.sum())
        blocked = int((lm & finite & (d.astype(np.float32) < 3 * radius)).sum()) / max(n_lumen, 1)
        hole = float(np.max((~finite & fs[None]).sum((1, 2)) / fs.sum()))
        ok = (
            all(c["openings_8R"] == 1 for c in cs)
            and not any(c["deep"] > 1 for c in cs)
            and n_lumen > 0
            and blocked <= 0.20
            and int((lm & ~finite).sum()) == 0
            and hole <= 0.02
            and not any(c["wall_cross"] for c in cs)
            and inside_sweep(trial, poses[:, :3, 3]).all()
        )
        if ok:
            return json_roundtrip(trial), dict(meta, blocked_fraction=blocked)
    raise RuntimeError("no seed candidate passes the opening and blockage checks")


def seed_from_keyframes(
    video: str,
    cam: Calibration,
    all_poses: np.ndarray,
    video_indices: np.ndarray,
    keyframes: dict[str, Any],
    field: np.ndarray,
    key: str,
    fold_prior: list[dict],
) -> tuple[Params, dict[str, Any]]:
    """Direct route: keyframe-wall profile fit, containment, folds and the enlargement search."""
    vi = np.asarray(video_indices)
    pc = all_poses[vi].astype(float)
    poses = np.linalg.inv(pc[0])[None] @ pc
    p = fit_profile_direct(video, cam, all_poses, vi, keyframes, field)
    p, _ = contain_cameras(p, poses, 0.36)
    p["fold"] = draw_fold(key, fold_prior)
    p, _ = contain_cameras(p, poses, 0.36)
    return enlargement_search(p, poses, cam, field, lumen_masks(video, vi, cam, field))


def seed_from_sdf_search(
    sdf: np.ndarray,
    lo: np.ndarray,
    step: float,
    frustum: dict,
    poses: np.ndarray,
    radius: float,
    cam: Calibration,
    field: np.ndarray,
    masks: np.ndarray,
    key: str,
    fold_prior: list[dict],
) -> tuple[Params, dict[str, Any]]:
    """SDF route with the enlargement search: the profile fitted to the carved SDF, 0.36 R
    containment, folds (fitted or drawn), then :func:`enlargement_search` with SDF-ranked
    low-order sweeps. ``masks``: dark-lumen masks of the window frames (cue grid)."""
    ev = sdf_sampler(sdf, lo, step)
    p, reliable = fit_profile(ev, poses, radius, frustum)
    p, _ = contain_cameras(json_roundtrip(p), poses, 0.36)
    if not reliable:
        p["fold"] = draw_fold(key, fold_prior)
        p, _ = contain_cameras(p, poses, 0.36)
    return enlargement_search(p, poses, cam, field, masks, ev)


# ----------------------------------------------------------------------------- re-render
def seed_surface(p: Params, poses: np.ndarray, field: np.ndarray) -> dict[str, Any]:
    """Seed surface arrays (closed sweep in pose units) for the given poses."""
    v, f, chart, rings = closed_sweep(p)
    return dict(V=v, F=f, chart=chart, poses=poses, R=float(p["R_pose"]), field_mask=field)


def native_poses(all_poses: np.ndarray, first: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive decoded frames from ``first``; poses rebased to the first."""
    ids = np.arange(first, first + frames)
    return ids, np.linalg.inv(all_poses[ids[0]])[None] @ all_poses[ids]


def render_seed(
    p: Params, poses: np.ndarray, cam: Calibration, field: np.ndarray, texture: str
) -> tuple[np.ndarray, np.ndarray]:
    """Seed tube frames ``(T, 832, 1104, 3)`` uint8 and stride-8 depth ``(T, 104, 138)`` float16.

    Shading uses the sweep's reference radius as light radius; pixels outside ``field`` are black.
    """
    from colonoworld.tube.render import TubeScene, load_texture

    v, f, chart, _ = closed_sweep(p)
    tube = TubeModel(
        centerline=np.zeros((0, 3)),
        frames=np.zeros((0, 3, 3)),
        radii=np.zeros((0, 0)),
        angles=np.zeros(0),
        arclength=np.zeros(0),
        vertices=v,
        faces=f,
        chart=np.stack(chart),
    )
    scene = TubeScene(tube, cam, field, load_texture(texture), threads=1)
    frames, depth = [], []
    for pose in poses:
        img, dep = scene.frame(pose, float(p["R_pose"]))
        frames.append(img)
        depth.append(dep[::8, ::8].astype(np.float16))
    return np.asarray(frames), np.asarray(depth)


# ----------------------------------------------------------------------------- pipeline
def build_training_seed(
    video: str,
    cam: Calibration,
    camera_npz: str,
    start: int,
    frames: int,
    key: str,
    fold_prior: list[dict],
    texture: str,
    keyframes: dict[str, Any] | None = None,
    disparity_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    route: str = "sdf",
) -> dict[str, Any]:
    """Seed sweep of a training window from the raw clip and the camera/depth estimator outputs.

    Args:
        video: real clip.
        cam: KB4 calibration.
        camera_npz: estimator output with ``relative_c2w`` and ``timestamp_seconds`` per decoded
            frame and the first frame's depth (``frame0_depth``, ``frame0_depth_support``,
            ``_meta.frame0_median_depth``).
        start, frames: window start on the 30 Hz timeline and number of 24 fps window frames.
        key: window name (seeds the fold draw).
        fold_prior: training fold patterns (``fold_prior.json`` ``parameters``).
        texture: mucosa texture image.
        keyframes: precomputed :func:`colonoworld.tube.seed_depth.keyframe_depth` output, else it is
            computed with ``disparity_fn`` (see
            :func:`~colonoworld.tube.seed_depth.load_depth_anything`).
        route: ``"sdf"`` (reconstruction, carve, opening check and low-order refit),
            ``"sdf_search"`` (reconstruction and carve, then the enlargement search) or
            ``"direct"`` (profiles from keyframe walls, then the enlargement search).

    Returns:
        ``parameters`` (seed sweep), ``surface`` (``V``, ``F``, ``poses``, ``video_indices``,
        ``field_mask``, ``R``), native-rate seed ``frames`` and stride-8 ``depth``, and ``info``.
    """
    from colonoworld.tube import seed_depth as sd
    from colonoworld.tube import seed_radius, seed_sdf

    with np.load(camera_npz) as z:
        all_poses = z["relative_c2w"].astype(np.float64)
        vi = sd.window_indices(z["timestamp_seconds"], start, frames)
    if keyframes is None:
        keyframes = sd.keyframe_depth(video, cam, all_poses, disparity_fn)
    _, full = sd.decode_small(video, [vi[0]], keep_full=[vi[0]])
    field = sd.window_field(full[int(vi[0])], cam)
    info: dict[str, Any] = dict(route=route)
    if route in ("sdf", "sdf_search"):
        rr = seed_radius.reference_radius(
            full[int(vi[0])], all_poses[vi], cam, seed_radius.load_frame0_depth(camera_npz)
        )
        radius = float(rr["R"])
        c = sd.collect_evidence(video, cam, all_poses, vi, keyframes, radius, field)
        rec = seed_sdf.reconstruct_sdf(c)
        masks = seed_sdf.largest_dark_masks(video, vi, cam, field)
        frustum = seed_sdf.fit_lumen_frustum(masks, c["P"], radius, cam)
        carved, lo = seed_sdf.carve_lumen(
            rec["sdf"], rec["lo"], rec["step"], c["P"], radius, frustum
        )
        if route == "sdf":
            p, fit_info = seed_from_sdf(
                carved, lo, rec["step"], frustum, c["P"], radius, cam, field, key, fold_prior
            )
        else:
            masks = lumen_masks(video, vi, cam, field)
            p, fit_info = seed_from_sdf_search(
                carved, lo, rec["step"], frustum, c["P"], radius, cam, field, masks, key, fold_prior
            )
        info.update(radius=rr["info"], fit=fit_info)
    elif route == "direct":
        p, fit_info = seed_from_keyframes(
            video, cam, all_poses, vi, keyframes, field, key, fold_prior
        )
        info.update(fit=fit_info)
    else:
        raise ValueError(route)
    ids, poses = native_poses(all_poses, int(vi[0]), frames)
    images, depth = render_seed(p, poses, cam, field, texture)
    v, f, _, _ = closed_sweep(p)
    surface = dict(V=v, F=f, poses=poses, video_indices=ids, field_mask=field, R=float(p["R_pose"]))
    return dict(parameters=p, surface=surface, frames=images, depth=depth, info=info)
