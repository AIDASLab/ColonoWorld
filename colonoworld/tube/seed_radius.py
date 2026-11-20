# SPDX-License-Identifier: CC-BY-NC-4.0
"""Reference radius ``R`` of a training window (causal tube fit from inference-time inputs).

The fit uses only what exists for a window at inference time: its camera trajectory, its first
frame (lens field and dark lumen) and the frame-0 depth of the pose/depth estimator (radius).

1. **Base centreline** (:func:`trajectory_centreline`): principal curve of the time-smoothed camera
   positions (Hastie-Stuetzle iterations with local-linear regression), extended straight at both
   ends. A trajectory shorter than ``0.3`` pose units is *stationary*.
2. **Frame-0 lumen** (:func:`frame0_lumen`): darkest blob after radial brightness normalisation,
   giving a lumen ray and a confidence in ``[0, 1]``.
3. **Decision**:

   - no confident lumen (``conf < 0.3``): stationary windows use a straight axis through the
     frame-0 camera along its optical axis (``axis_f0``), otherwise the base centreline (``Ct``);
   - stationary: straight axis along the frame-0 lumen ray (``axis_f0lumen``);
   - ambiguous (``|optical axis . path tangent| < 0.3`` at frame 0, a sideways path): the same
     frame-0 lumen axis (``axis_f0lumen_amb``);
   - otherwise the base curve is corrected so that frame 0 sees its far end on the lumen ray
     (:func:`lumen_centreline`, ``Cl0_f0``).

4. **Vanishing-point gate** (lumen-confident windows): frame 0 of the folded tube is ray cast;
   when the farthest 1 % of the field lies more than ``0.05`` image widths from the detected lumen,
   the fit is replaced by the frame-0 lumen axis (``axis_f0lumen_gate``).

The radius is the median distance of the frame-0 depth points to the centreline, at least
``cam_dist_max / centre_frac`` (camera within the central part of the tube) and ``1.05``
``cam_dist_max``, at most ``3``. Lengths are in pose units (first-frame median depth of the clip).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from colonoworld.tube.build_from_video import CUE_SIZE, _gray, content_mask, specular_mask
from colonoworld.tube.camera import Calibration, pixel_grid_rays, rays_from_clip

CLIP_W, CLIP_H = 1104, 832
CUE_W, CUE_H = CUE_SIZE
DEPTH_W, DEPTH_H = 512, 384  # estimator depth grid (0.8 x the 640x480 rectified chart)
RECT_W, RECT_H = 640, 480
RADIUS_GRID = (138, 104)  # frame-0 depth samples used for the radius
GATE_STRIDE = 8  # gate depth grid stride (138x104)
LUMEN_CONF_MIN = 0.3
AMBIGUOUS = 0.3
VP_GATE = 0.05  # image widths
FOLD_CLEARANCE = 1.65  # fold radius >= 1.65 x the largest camera distance from the axis
R_CAP = 3.0
EXT = 60.0  # straight extension of every centreline (pose units)
FOLDS = dict(period=1.1, depth=0.28, width=0.07)  # haustral folds of the gate mesh (x R0)
N_PHI = 192
MESH_BEHIND, MESH_AHEAD = 10.0, 40.0  # gate mesh range around the cameras (x R0)


# ----------------------------------------------------------------------------- centreline object
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rmf(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimising frame of a polyline (double reflection): tangents, n1, n2 per segment."""
    a = _unit(np.diff(points, axis=0))
    n = np.zeros_like(a)
    ref = np.array([0.0, 1.0, 0.0]) if abs(a[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = ref - (ref @ a[0]) * a[0]
    n[0] = _unit(n0 - (n0 @ a[0]) * a[0])
    for j in range(len(a) - 1):
        v1 = points[j + 1] - points[j]
        c1 = v1 @ v1
        r_l = n[j] - (2 / c1) * (v1 @ n[j]) * v1
        t_l = a[j] - (2 / c1) * (v1 @ a[j]) * v1
        v2 = a[j + 1] - t_l
        c2 = v2 @ v2
        n[j + 1] = r_l - (2 / c2) * (v2 @ r_l) * v2 if c2 > 1e-18 else r_l
        n[j + 1] = _unit(n[j + 1] - (n[j + 1] @ a[j + 1]) * a[j + 1])
    return a, n, np.cross(a, n)


class Centreline:
    """Polyline centreline ``P (K+1, 3)`` with radius ``R``, segment frames and arc length ``S``."""

    def __init__(self, points: np.ndarray, R: float) -> None:
        self.P = np.asarray(points, float)
        self.R = float(R)
        self.a, self.n1, self.n2 = rmf(self.P)
        self.L = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.S = np.r_[0, np.cumsum(self.L)]
        self.core = self.P

    def closest(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance of points ``X (N, 3)`` to the polyline and arc length of the closest point."""
        X = np.asarray(X, float)
        if len(X) > 2048:
            parts = [self.closest(X[i : i + 2048]) for i in range(0, len(X), 2048)]
            return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        best = np.full(len(X), np.inf)
        s = np.zeros(len(X))
        for j0 in range(0, len(self.a), 256):
            A, a, L = self.P[:-1][j0 : j0 + 256], self.a[j0 : j0 + 256], self.L[j0 : j0 + 256]
            w = X[:, None, :] - A[None]
            u = np.clip((w * a[None]).sum(-1), 0, L[None])
            d = np.linalg.norm(w - u[..., None] * a[None], axis=-1)
            k = d.argmin(1)
            dk = d[np.arange(len(X)), k]
            upd = dk < best
            best[upd] = dk[upd]
            s[upd] = self.S[j0 + k[upd]] + u[np.arange(len(X)), k][upd]
        return best, s

    def min_curvature_radius(self, s_lo: float, s_hi: float) -> float:
        """Smallest radius of curvature of the segments with midpoints in ``[s_lo, s_hi]``."""
        mid = 0.5 * (self.S[:-1] + self.S[1:])
        idx = np.where((mid >= s_lo) & (mid <= s_hi))[0]
        if len(idx) < 2:
            return np.inf
        aa = self.a[idx]
        ang = np.arccos(np.clip((aa[:-1] * aa[1:]).sum(1), -1, 1))
        return float((self.L[idx[:-1]] / np.maximum(ang, 1e-9)).min())

    def kappa_R(self, ds: float) -> float:
        """``R / min radius of curvature`` over the fitted part (straight extensions excluded)."""
        rc = self.min_curvature_radius(self.S[0] + EXT - 0.5 * ds, self.S[-1] - EXT + 0.5 * ds)
        return self.R / rc if np.isfinite(rc) else 0.0


# ----------------------------------------------------------------------------- polyline helpers
def _smooth_positions(p: np.ndarray) -> np.ndarray:
    """Gaussian time smoothing (sigma 3 frames, edge padding) for 7 or more positions."""
    if len(p) < 7:
        return p
    kk = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    kk /= kk.sum()
    return np.stack(
        [np.convolve(np.pad(p[:, i], 9, mode="edge"), kk, mode="valid") for i in range(3)], 1
    )


def _arc_resample(pts: np.ndarray, ds: float) -> np.ndarray:
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    keep = np.r_[True, np.diff(sl) > 1e-9]
    pts, sl = pts[keep], sl[keep]
    n = max(int(np.ceil(sl[-1] / ds)), 1)
    sq = np.linspace(0, sl[-1], n + 1)
    return np.stack([np.interp(sq, sl, pts[:, k]) for k in range(3)], 1)


def _extend(pts: np.ndarray, a0: np.ndarray, a1: np.ndarray, ext: float, ds: float) -> np.ndarray:
    n = max(int(np.ceil(ext / ds)), 1)
    k = np.arange(1, n + 1)[:, None] * ds
    return np.vstack([(pts[0] - k * a0)[::-1], pts, pts[-1] + k * a1])


def _extend_robust(pts: np.ndarray, ext: float, ds: float, scale: float) -> np.ndarray:
    """Straight extension along the end chords over ``scale`` pose units (2 samples to half the
    curve)."""
    m = int(np.clip(round(scale / ds), 2, max(2, len(pts) // 2)))
    m = min(m, len(pts) - 1)
    return _extend(pts, _unit(pts[m] - pts[0]), _unit(pts[-1] - pts[-1 - m]), ext, ds)


# ----------------------------------------------------------------------------- frame-0 depth
def load_frame0_depth(npz_path: str) -> dict[str, Any]:
    """Frame-0 depth of the pose/depth estimator output (``frame0_depth``, its support, median)."""
    with np.load(npz_path, allow_pickle=False) as z:
        meta = json.loads(z["_meta"].item())
        return dict(
            depth=z["frame0_depth"].astype(np.float64),
            support=z["frame0_depth_support"].astype(bool),
            median_depth=float(meta["frame0_median_depth"]),
        )


def frame0_points(cam: Calibration, frame0_depth: dict[str, Any]) -> np.ndarray:
    """Camera-frame 3-D points of the frame-0 depth on the 138x104 clip grid (median depth = 1)."""
    w, h = RADIUS_GRID
    d, ok = pixel_grid_rays(cam, w, h)
    d, ok = d.reshape(-1, 3), ok.reshape(-1)
    cw, ch = cam["clip_size"]
    k = np.diag([RECT_W / cw, RECT_H / ch, 1.0]) @ np.array(cam["K_clip"])
    kd = np.diag([0.8, 0.8, 1.0]) @ k
    with np.errstate(divide="ignore", invalid="ignore"):
        p = d / d[:, 2:3]
    uv = np.nan_to_num(p @ kd.T, nan=-1, posinf=-1, neginf=-1)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = ok & (d[:, 2] > 1e-3) & (u >= 0) & (u < DEPTH_W) & (v >= 0) & (v < DEPTH_H)
    z = np.full(len(d), np.nan)
    z[ok] = frame0_depth["depth"][v[ok], u[ok]] / frame0_depth["median_depth"]
    support = frame0_depth["support"][np.clip(v, 0, DEPTH_H - 1), np.clip(u, 0, DEPTH_W - 1)]
    ok &= np.isfinite(z) & (z > 0) & support
    dist = np.where(ok, z / np.maximum(d[:, 2], 1e-6), np.nan)
    return (d * np.nan_to_num(dist)[:, None])[ok]


def estimate_radius(
    points: np.ndarray, cam: Calibration, frame0_depth: dict[str, Any], positions: np.ndarray
) -> tuple[float, float]:
    """``(R_depth, cam_dist_max)``: median distance of the frame-0 depth points (``0 < z < 3``) to
    the centreline clipped to ``[0.3, 3]`` (1 with too few points), at least the largest camera
    distance."""
    tmp = Centreline(points, 1.0)
    X = frame0_points(cam, frame0_depth)
    X = X[(X[:, 2] > 0) & (X[:, 2] < 3.0)]
    R = float(np.median(tmp.closest(X)[0])) if len(X) > 50 else 1.0
    dmax = float(tmp.closest(np.asarray(positions))[0].max())
    return max(float(np.clip(R, 0.3, 3.0)), dmax), dmax


def _contained_radius(R_depth: float, dmax: float, centre_frac: float) -> float:
    R = float(min(max(R_depth, dmax / centre_frac), R_CAP))
    return max(R, 1.05 * dmax)


# ----------------------------------------------------------------------------- centrelines
def _resample_curve(curve, u0: float, u1: float, ds: float) -> np.ndarray:
    pts = curve(np.linspace(u0, u1, 2000))
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    sq = np.linspace(0, sl[-1], max(int(np.ceil(sl[-1] / ds)), 1) + 1)
    return np.stack([np.interp(sq, sl, pts[:, k]) for k in range(3)], 1)


def initial_curve(
    positions: np.ndarray, rotations: np.ndarray, ds: float = 0.05, ridge: float = 1e-3
) -> tuple[np.ndarray, bool]:
    """Polynomial principal curve of the smoothed positions (PCA axis + ridge-regularised lateral
    polynomials of degree 1-3 by span), or a short optical-axis segment for a stationary path.
    Returns ``(points, stationary)``."""
    p = _smooth_positions(np.asarray(positions, float))
    mu = p.mean(0)
    zmean = _unit(np.asarray(rotations)[:, :3, 2].mean(0))
    X = p - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    a = Vt[0] * (1 if Vt[0] @ zmean >= 0 else -1)
    u = X @ a
    span = float(u.max() - u.min())
    if span < 0.3:
        L = max(span, ds * 4)
        pts = mu[None] + np.linspace(-L / 2, L / 2, max(int(L / ds), 4) + 1)[:, None] * zmean[None]
        return _extend(pts, _unit(pts[1] - pts[0]), _unit(pts[-1] - pts[-2]), 0.0, ds), True
    ref = [0.0, 1.0, 0.0] if abs(a[1]) < 0.9 else [1.0, 0.0, 0.0]
    b1 = _unit(np.cross(a, ref))
    b2 = np.cross(a, b1)
    deg = 1 if span < 2 else (2 if span < 6 else 3)
    half = max(span / 2, 1e-9)
    V = np.vander((u - u.mean()) / half, deg + 1)
    A = V.T @ V + ridge * len(u) * np.diag(np.r_[np.ones(deg), 0.0])
    coef = [np.linalg.solve(A, V.T @ (X @ b)) for b in (b1, b2)]

    def curve(uq: np.ndarray) -> np.ndarray:
        Vq = np.vander((uq - u.mean()) / half, deg + 1)
        return mu + uq[:, None] * a + (Vq @ coef[0])[:, None] * b1 + (Vq @ coef[1])[:, None] * b2

    pts = _resample_curve(curve, u.min(), u.max(), ds)
    return _extend(pts, _unit(pts[1] - pts[0]), _unit(pts[-1] - pts[-2]), 0.0, ds), False


def trajectory_centreline(
    positions: np.ndarray,
    rotations: np.ndarray,
    cam: Calibration,
    frame0_depth: dict[str, Any],
    h: float = 0.45,
    n_iter: int = 4,
    centre_frac: float = 0.6,
    ds: float = 0.05,
    kappa_max: float = 0.6,
) -> tuple[Centreline, dict[str, Any]]:
    """Principal curve of the camera trajectory (base centreline ``Ct``).

    Starting from :func:`initial_curve`, positions are projected onto the curve ``n_iter`` times and
    the curve is re-estimated by Gaussian local-linear regression of positions on arc length
    (bandwidth ``h``). While ``R / min radius of curvature > kappa_max`` the bandwidth grows by 1.4
    (at most 6 times).
    """
    f = _smooth_positions(np.asarray(positions, float))
    init, stationary = initial_curve(positions, rotations)
    stationary = stationary or len(init) < 4

    def core_fn(h: float) -> np.ndarray:
        if stationary:
            return init
        c = _arc_resample(init, ds)
        for _ in range(n_iter):
            _, lam = Centreline(c, 1.0).closest(f)
            lo, hi = lam.min(), lam.max()
            if hi - lo < 2 * ds:
                return c
            g = np.arange(lo, hi + 1e-9, ds)
            dl = lam[None] - g[:, None]
            w = np.exp(-0.5 * (dl / h) ** 2)
            S0, S1, S2 = w.sum(1), (w * dl).sum(1), (w * dl * dl).sum(1)
            T0, T1 = w @ f, (w * dl) @ f
            den = np.maximum(S0 * S2 - S1 * S1, 1e-12)
            c = _arc_resample((S2[:, None] * T0 - S1[:, None] * T1) / den[:, None], ds)
        return c

    for _ in range(7):
        core = core_fn(h)
        pts = _extend_robust(core, EXT, ds, max(h, 0.2))
        R_depth, dmax = estimate_radius(pts, cam, frame0_depth, positions)
        cl = Centreline(pts, _contained_radius(R_depth, dmax, centre_frac))
        if cl.kappa_R(ds) <= kappa_max or stationary:
            break
        h *= 1.4
    cl.core = core
    kind = "stationary_optical_axis" if stationary else "principal_curve"
    return cl, dict(kind=kind, R=cl.R, R_depth=R_depth, cam_dist_max=dmax)


def _base_core(base: Centreline, ds: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Base core resampled at ``ds`` and extended past the look-ahead distances ``(2R, 4R)``."""
    D = (2.0 * base.R, 4.0 * base.R)
    core0 = _arc_resample(_extend_robust(_arc_resample(base.core, ds), max(D) + ds, ds, 0.3), ds)
    return core0, D


def frame0_tangent(poses: np.ndarray, base: Centreline, ds: float = 0.1) -> np.ndarray:
    """Tangent of the base core at the projection of the (smoothed) frame-0 camera."""
    core0, _ = _base_core(base, ds)
    bcl = Centreline(core0, 1.0)
    _, lam = bcl.closest(_smooth_positions(np.asarray(poses, float)[:, :3, 3]))
    kt = np.clip(np.round(lam / ds).astype(int), 0, len(core0) - 1)
    return bcl.a[np.clip(kt, 0, len(core0) - 2)][0]


def lumen_centreline(
    poses: np.ndarray,
    lumen_ray: np.ndarray,
    conf: float,
    sign0: float,
    cam: Calibration,
    frame0_depth: dict[str, Any],
    base: Centreline,
    ds: float = 0.1,
    w_lum: float = 4.0,
    w_smooth: float = 0.02,
    centre_frac: float = 0.75,
    n_irls: int = 3,
    huber: float = 0.3,
    kappa_max: float = 0.6,
) -> Centreline:
    """Base core corrected so that the centreline points ``2R`` and ``4R`` ahead of the frame-0
    camera (direction ``sign0`` along the tangent) lie on its lumen ray.

    Linear least squares over the core points ``c_k`` (spacing ``ds``): stay near the smoothed
    camera path (weight 1), lumen-ray constraints ``[l]x (c_j - p_0)`` (weight ``w_lum * conf``,
    Huber IRLS) and bending energy ``w_smooth |c_{k+1} - 2 c_k + c_{k-1}|^2 / ds^3``. The bending
    weight grows x4 (up to 7 times) while ``R / min radius of curvature > kappa_max``.
    """
    P = np.asarray(poses, float)
    p, Rw = P[:, :3, 3], P[:, :3, :3]
    T = len(p)
    f = _smooth_positions(p)
    core0, D = _base_core(base, ds)
    K = len(core0)
    _, lam = Centreline(core0, 1.0).closest(f)
    kt = np.clip(np.round(lam / ds).astype(int), 0, K - 1)
    rays = np.full((T, 3), np.nan)
    rays[0] = lumen_ray
    lw = np.einsum("tij,tj->ti", Rw, np.nan_to_num(rays))
    l0 = lw[0] / np.linalg.norm(lw[0])
    lx = np.array([[0, -l0[2], l0[1]], [l0[2], 0, -l0[0]], [-l0[1], l0[0], 0]])
    pairs = [j for j in (int(np.round((lam[0] + sign0 * d) / ds)) for d in D) if 0 <= j < K]
    n_rows = T + len(pairs) + K - 2
    A = np.zeros((3 * n_rows, 3 * K))
    b = np.zeros(3 * n_rows)
    W0 = np.zeros(3 * n_rows)
    eye = np.eye(3)
    for t in range(T):
        A[3 * t : 3 * t + 3, 3 * kt[t] : 3 * kt[t] + 3] = eye
        b[3 * t : 3 * t + 3] = f[t]
        W0[3 * t : 3 * t + 3] = np.sqrt(1.0 / T)
    lum_idx = list(range(T, T + len(pairs)))
    for r, j in zip(lum_idx, pairs, strict=True):
        A[3 * r : 3 * r + 3, 3 * j : 3 * j + 3] = lx
        b[3 * r : 3 * r + 3] = lx @ p[0]
        W0[3 * r : 3 * r + 3] = np.sqrt(w_lum * conf / max(len(pairs), 1))
    for r, k in enumerate(range(1, K - 1), start=T + len(pairs)):
        A[3 * r : 3 * r + 3, 3 * (k + 1) : 3 * (k + 1) + 3] = eye
        A[3 * r : 3 * r + 3, 3 * k : 3 * k + 3] = -2 * eye
        A[3 * r : 3 * r + 3, 3 * (k - 1) : 3 * (k - 1) + 3] = eye
        W0[3 * r : 3 * r + 3] = np.sqrt(w_smooth / ds**3)
    n_smooth = 3 * (K - 2)
    gain = 1.0
    for _ in range(8):
        W = W0.copy()
        W[-n_smooth:] *= np.sqrt(gain)
        hub = np.ones(n_rows)
        for _ in range(n_irls):
            Wh = W * np.repeat(hub, 3)
            x = np.linalg.lstsq(A * Wh[:, None], b * Wh, rcond=None)[0]
            if not lum_idx:
                break
            res = np.linalg.norm((A @ x - b).reshape(-1, 3)[lum_idx], axis=1)
            hub[lum_idx] = np.where(res <= huber, 1.0, np.sqrt(huber / np.maximum(res, 1e-9)))
        core = _arc_resample(x.reshape(K, 3), 0.05)
        pts = _extend_robust(core, EXT, 0.05, max(max(D) / 2, 0.3))
        R_depth, dmax = estimate_radius(pts, cam, frame0_depth, p)
        cl = Centreline(pts, _contained_radius(R_depth, dmax, centre_frac))
        cl.core = core
        if cl.kappa_R(0.05) <= kappa_max:
            break
        gain *= 4
    cl.fit = dict(R=cl.R, R_depth=R_depth, cam_dist_max=dmax, smooth_gain=gain)
    return cl


def frame0_axis(
    poses: np.ndarray, direction: np.ndarray, cam: Calibration, frame0_depth: dict[str, Any]
) -> Centreline:
    """Straight axis through the frame-0 camera along a camera-frame ``direction``."""
    P = np.asarray(poses, float)
    dirw = P[0, :3, :3] @ direction
    o = P[0, :3, 3]
    pts = _extend_robust(
        o[None] + np.arange(-0.5, 1.5 + 1e-9, 0.05)[:, None] * dirw[None], EXT, 0.05, 0.3
    )
    R_depth, dmax = estimate_radius(pts, cam, frame0_depth, P[:, :3, 3])
    cl = Centreline(pts, _contained_radius(R_depth, dmax, 0.6))
    cl.fit = dict(R=cl.R, R_depth=R_depth, cam_dist_max=dmax)
    return cl


# ----------------------------------------------------------------------------- frame-0 lumen
def frame0_lumen(
    small: np.ndarray, cam: Calibration, field: np.ndarray, r_max_frac: float = 0.8, q: float = 2.0
) -> dict[str, Any] | None:
    """Dark-lumen blob of the cue-grid frame ``small`` (276x208 RGB).

    Grey image with specular highlights inpainted, Gaussian blur (sigma 9), divided by the median
    brightness per radius ring (12 rings); the darkest ``q`` % inside ``r_max_frac`` of the field
    radius, weighted by darkness, largest-weight connected blob. Returns its centroid ``xy_clip``
    (clip pixels) and ``conf = ramp(contrast, 0.15..0.5) * ramp(concentration, 0.3..0.8)``, or
    ``None`` without a dark blob.
    """
    import cv2

    g8 = np.clip(_gray(small), 0, 255).astype(np.uint8)
    hole = (specular_mask(small) | ~field).astype(np.uint8)
    g = cv2.inpaint(g8, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 9)
    K = np.asarray(cam["K_clip"], float)
    yy, xx = np.mgrid[:CUE_H, :CUE_W]
    cx, cy = (K[0, 2] + 0.5) / 4 - 0.5, (K[1, 2] + 0.5) / 4 - 0.5
    r = np.hypot(xx - cx, yy - cy)
    rmax = r[field].max()
    bins = np.minimum((r / rmax * 12).astype(int), 11)
    prof = np.array(
        [
            np.median(blur[field & (bins == k)]) if (field & (bins == k)).sum() > 20 else np.nan
            for k in range(12)
        ]
    )
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    norm = blur / np.maximum(prof[bins], 1.0)
    inner = field & (r < r_max_frac * rmax)
    thr = np.percentile(norm[inner], q)
    dark = (inner & (norm <= thr)).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None
    wts = np.maximum(thr - norm, 0) + 1e-3
    best = max(range(1, n), key=lambda i: wts[lab == i].sum())
    blob = lab == best
    w = wts * blob
    x, y = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
    contrast = float(1 - blur[blob].mean() / max(np.median(blur[inner]), 1e-6))
    concentration = float(w.sum() / (wts * dark).sum())
    conf = float(
        np.clip((contrast - 0.15) / 0.35, 0, 1) * np.clip((concentration - 0.3) / 0.5, 0, 1)
    )
    return dict(xy_clip=[float((x + 0.5) * 4 - 0.5), float((y + 0.5) * 4 - 0.5)], conf=conf)


# ----------------------------------------------------------------------------- vanishing-point gate
def radius_folds(R0: float):
    """Haustral-fold radius ``R(s, phi)``: crescent ring ridges between three taeniae."""
    period, depth, width = FOLDS["period"] * R0, FOLDS["depth"], FOLDS["width"]

    def f(s: np.ndarray, phi: np.ndarray) -> np.ndarray:
        x = (s % period) - period / 2
        ridge = np.exp(-0.5 * (x / (width * R0)) ** 2)
        crescent = 0.55 + 0.45 * np.cos(1.5 * phi) ** 2
        lobes = 1 - 0.06 * np.cos(3 * phi)
        return R0 * lobes * (1 - depth * ridge * crescent)

    return f


def fold_mesh(cl: Centreline, R0: float, s_lo: float, s_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Swept fold mesh (rings perpendicular to the centreline) between arc lengths s_lo..s_hi."""
    ds = min(0.012, 0.015 * R0)
    s = np.arange(max(s_lo, cl.S[0]), min(s_hi, cl.S[-1]) + 1e-9, ds)
    j = np.clip(np.searchsorted(cl.S, s, side="right") - 1, 0, len(cl.a) - 1)
    c = cl.P[j] + (s - cl.S[j])[:, None] * cl.a[j]
    n1, n2 = cl.n1[j], cl.n2[j]
    phi = np.linspace(-np.pi, np.pi, N_PHI + 1)
    Rr = radius_folds(R0)(s[:, None], phi[None])
    dirs = np.cos(phi)[None, :, None] * n1[:, None] + np.sin(phi)[None, :, None] * n2[:, None]
    V = c[:, None] + Rr[..., None] * dirs
    ns, npp = len(s), N_PHI + 1
    i = np.arange(ns - 1)[:, None] * npp + np.arange(N_PHI)[None]
    F = np.concatenate(
        [
            np.stack([i, i + npp, i + 1], -1).reshape(-1, 3),
            np.stack([i + 1, i + npp, i + npp + 1], -1).reshape(-1, 3),
        ]
    )
    return V.reshape(-1, 3).astype(np.float32), F.astype(np.int32)


def gate_depth(
    cl: Centreline, R0: float, poses: np.ndarray, cam: Calibration, field: np.ndarray
) -> np.ndarray:
    """Frame-0 ray distance of the fold tube (radius ``R0``) on the stride-8 clip grid (float16,
    NaN outside the field or without a hit). The mesh spans ``10 R0`` behind to ``40 R0`` ahead of
    the arc-length range of all window cameras."""
    import drjit as dr
    import mitsuba as mi

    P = np.asarray(poses, float)
    sc = cl.closest(P[:, :3, 3])[1]
    cdm = R0 / FOLD_CLEARANCE if R0 > cl.R else 0.0
    R0 = max(cl.R, FOLD_CLEARANCE * cdm)
    V, F = fold_mesh(cl, R0, float(sc.min()) - MESH_BEHIND * R0, float(sc.max()) + MESH_AHEAD * R0)
    mi.set_variant("llvm_ad_rgb")
    dr.set_thread_count(1)
    mesh = mi.Mesh("tube", len(V), len(F))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(np.ascontiguousarray(V).reshape(-1))
    params["faces"] = mi.UInt32(np.ascontiguousarray(F.astype(np.uint32)).reshape(-1))
    params.update()
    scene = mi.load_dict({"type": "scene", "tube": mesh})
    rays, ok = pixel_grid_rays(cam, CLIP_W, CLIP_H)
    rays, ok = rays.reshape(-1, 3).astype(np.float32), ok.reshape(-1)
    d = rays @ P[0, :3, :3].T.astype(np.float32)
    o = P[0, :3, 3].astype(np.float32)
    ray = mi.Ray3f(
        mi.Point3f(*[np.full(len(d), x, np.float32) for x in o]),
        mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]),
    )
    si = scene.ray_intersect_preliminary(ray, coherent=True)
    t = np.array(si.t, np.float64)
    pid = np.array(si.prim_index, np.int64)
    hit = np.isfinite(t) & (t < 1e30) & ok & (pid >= 0) & (pid < len(F))
    depth = np.where(hit, t, np.nan).astype(np.float32).reshape(CLIP_H, CLIP_W)
    depth[~field] = np.nan
    return depth[::GATE_STRIDE, ::GATE_STRIDE].astype(np.float16)


def vanishing_point(depth0: np.ndarray, field: np.ndarray, q: float = 1.0) -> tuple[float, float]:
    """Mean clip pixel of the farthest ``q`` % of the gate samples inside the field (eroded by two
    samples); misses count as infinitely far."""
    import cv2

    m = field[::GATE_STRIDE, ::GATE_STRIDE][: depth0.shape[0], : depth0.shape[1]]
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    z = np.where(np.isfinite(depth0), depth0.astype(np.float64), 1e30)[m]
    yy, xx = np.nonzero(m)
    n = min(max(int(np.ceil(q / 100 * len(z))), 3), len(z) - 1)
    far = np.argpartition(-z, n)[:n]
    return float(xx[far].mean() * GATE_STRIDE), float(yy[far].mean() * GATE_STRIDE)


def vp_gate(depth0: np.ndarray, field: np.ndarray, lumen_xy: list[float]) -> dict[str, Any]:
    """Distance (image widths) between the tube vanishing point and the detected lumen."""
    vx, vy = vanishing_point(depth0, field)
    err = float(np.hypot(vx - lumen_xy[0], vy - lumen_xy[1]) / CLIP_W)
    return dict(vp_xy=[vx, vy], real_xy=list(lumen_xy), err_w=err, ok=bool(err <= VP_GATE))


# ----------------------------------------------------------------------------- public API
def reference_radius(
    frame0: np.ndarray, poses: np.ndarray, cam: Calibration, frame0_depth: dict[str, Any]
) -> dict[str, Any]:
    """Reference radius of a window.

    Args:
        frame0: first frame of the window, full-resolution RGB ``(832, 1104, 3)`` uint8.
        poses: ``(T, 4, 4)`` camera-to-world poses of the window frames in the clip gauge
            (translation in units of the clip's first-frame median depth).
        cam: KB4 calibration dict.
        frame0_depth: :func:`load_frame0_depth` of the clip.

    Returns:
        ``dict(R, info, centreline, gate_depth)``: radius, decision record (``pick``, ``R``,
        ``folds_R0``, ``cam_dist_max``, ``lumen_conf``, ``lumen_xy``, ``ambiguous``, ``fallback``,
        ``vp_gate``, ``vp_gate_initial``, ``fit``), centreline points (clip gauge) and the final
        frame-0 gate depth (``None`` without a confident lumen).
    """
    import cv2

    P = np.asarray(poses, np.float64)
    pos, rot = P[:, :3, 3], P[:, :3, :3]
    small = cv2.resize(frame0, CUE_SIZE, interpolation=cv2.INTER_AREA)
    field_small = content_mask(frame0, cam)
    field = (
        cv2.resize(field_small.astype(np.uint8), (CLIP_W, CLIP_H), interpolation=cv2.INTER_NEAREST)
        > 0
    )
    base, base_info = trajectory_centreline(pos, rot, cam, frame0_depth)
    base.fit = base_info
    stationary = base_info["kind"] == "stationary_optical_axis"

    lumen = frame0_lumen(small, cam, field_small)
    ray, conf, xy = None, 0.0, None
    if lumen is not None:
        d, ok = rays_from_clip(np.array([lumen["xy_clip"]]), cam)
        if ok[0]:
            ray, conf, xy = d[0], lumen["conf"], list(lumen["xy_clip"])
    confident = conf >= LUMEN_CONF_MIN
    info: dict[str, Any] = dict(lumen_conf=float(conf), lumen_xy=xy, ambiguous=False)

    if not confident:  # trajectory only
        if stationary:
            cl, pick = frame0_axis(P, np.array([0.0, 0.0, 1.0]), cam, frame0_depth), "axis_f0"
        else:
            cl, pick = base, "Ct"
    else:
        ray_u = ray / np.linalg.norm(ray)
        ambiguous = False
        if not stationary:
            tangent = frame0_tangent(P, base)
            info["opt_dot_t"] = float(P[0, :3, 2] @ tangent)
            info["lum_dot_t"] = float((P[0, :3, :3] @ ray_u) @ tangent)
            ambiguous = abs(info["opt_dot_t"]) < AMBIGUOUS
        info["ambiguous"] = bool(ambiguous)
        if stationary or ambiguous:
            cl = frame0_axis(P, ray_u, cam, frame0_depth)
            pick = "axis_f0lumen_amb" if ambiguous else "axis_f0lumen"
        else:
            sign0 = float(np.sign(info["lum_dot_t"]) or 1.0)
            cl = lumen_centreline(P, ray, conf, sign0, cam, frame0_depth, base)
            pick = "Cl0_f0"

    cdm = float(cl.closest(pos)[0].max())
    R0 = max(cl.R, FOLD_CLEARANCE * cdm)
    depth0, gate, fallback = None, None, False
    if confident:
        depth0 = gate_depth(cl, R0, P, cam, field)
        gate = vp_gate(depth0, field, xy)
        info["vp_gate_initial"] = gate
        if not gate["ok"]:
            cl, pick, fallback = frame0_axis(P, ray_u, cam, frame0_depth), "axis_f0lumen_gate", True
            cdm = float(cl.closest(pos)[0].max())
            R0 = max(cl.R, FOLD_CLEARANCE * cdm)
            depth0 = gate_depth(cl, R0, P, cam, field)
            gate = vp_gate(depth0, field, xy)
    info.update(
        pick=pick,
        R=cl.R,
        folds_R0=R0,
        cam_dist_max=cdm,
        fallback=fallback,
        vp_gate=gate,
        fit={k: v for k, v in cl.fit.items() if isinstance(v, (int, float, str, bool))},
    )
    return dict(R=cl.R, info=info, centreline=cl.P, gate_depth=depth0)
