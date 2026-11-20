# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shape regularisation of a lumen-fit training tube.

A training window first gets a profile-sweep wall fitted to its own video (``params``, see
:mod:`colonoworld.tube.wall_fit`). This module turns that fit into the final training tube:

- **Cross-section clip (X).** Per-vertex radius deviation from the ring mean is clipped to
  ``+-cross_section_clip`` (mean-preserving per ring).
- **Short-stub extension (E).** Tubes shorter than ``min_length_diameters`` local diameters are
  extended at both ends by circular rings along the end tangent.
- **Plausible shape (D).** Ring radii are clipped to ``[radius_lo, radius_hi]`` times the median
  radius along the camera arc, the axial radius slope is capped (Lipschitz) and the centreline is
  locally smoothed wherever its bend radius is smaller than the tube radius.
- **Anatomy relief (A).** Three taeniae, staggered haustral pouches with semilunar folds, three
  rectal valves in the rectum and a three-lobed transverse section, faded toward the open ends and
  near inward features that come close to the camera.
- **Final clip (F).** The cross-section clip again on the final surface.

Candidates are tried in a fixed order and the first one that passes every check is kept:

====  ==========================================================================================
A     plausible shape, clip everywhere, extension
B     plausible shape, clip outside the near-camera region, extension
C     near-camera rings kept as fitted, plausible shape elsewhere, relief outside the region
D     fitted centreline and radius, relief and clip outside the near-camera region (no bend check)
T     the fitted surface unchanged (always accepted)
====  ==========================================================================================

Each tier is tried with relief scale 1 and 0 (0 only when the segment is unknown) and, where needed,
with mild inflation steps until the camera keeps ``0.3 R`` clearance. The checks render the
candidate (same renderer and exposure rule as every tube video) and require a single lumen, no
wall crossing, clearance, no back-face hits, a training-like brightness, a bend radius at least the
tube radius, and camera following (dense-flow agreement with the real video) no worse than the
fitted tube.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.camera import Calibration, rays_from_clip
from colonoworld.tube.model import TubeModel
from colonoworld.tube.wall_fit import (
    Params,
    arc_length,
    centreline,
    ellipse_radius,
    fold_factor,
    interp_profile,
    sweep_mesh,
    tube_from_surface,
)

DEFAULTS: dict[str, Any] = dict(
    cross_section_clip=0.15,
    min_length_diameters=3.3,
    radius_lo=0.62,
    radius_hi=1.24,
    radius_slope=0.25,
    transverse_lobe=0.12,
    inflation_cap=1.3,
    inflation_steps=[1.1, 1.2, 1.35],
    exposure_divisor=1.38,
    min_clearance=0.3,
    luma_bounds=[37.05697452995954, 135.44725543887705],
    motion_tolerance=0.015,
)
LUMA = np.array([0.299, 0.587, 0.114])
FLOW_SIZE = (160, 120)
FARNEBACK = (0.5, 3, 21, 3, 5, 1.2, 0)


# ----------------------------------------------------------------------------- small helpers
def _c1(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 1, np.cos(np.pi * np.clip(np.abs(x), 0, 1) / 2) ** 2, 0.0)


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * a))


def _smooth01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _arc(x: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]


def sample_times(n_frames: int, stride: int = 4) -> np.ndarray:
    """Frames used by the camera-following check: every ``stride``-th frame plus the last."""
    return np.unique(np.r_[np.arange(0, n_frames, stride), n_frames - 1]).astype(int)


def decode_small(path: str | Path, indices: np.ndarray, size: tuple[int, int] = FLOW_SIZE) -> list:
    """Decode the given frame indices of a video, resized (area) to ``size`` (W, H)."""
    import av
    import cv2

    wanted = [int(i) for i in indices]
    keys = set(wanted)
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        for t, frame in enumerate(container.decode(video=0)):
            if t in keys:
                im = frame.to_ndarray(format="rgb24", threads=1)
                found[t] = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            if t >= max(wanted):
                break
    if set(found) != keys:
        raise ValueError(f"frames missing in {path}")
    return [found[t] for t in wanted]


def _rotmin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from colonoworld.tube.synthetic_colon import minimal_rotation

    return minimal_rotation(a, b)


def _surface_distance(v: np.ndarray, f: np.ndarray, x: np.ndarray) -> np.ndarray:
    from colonoworld.tube.synthetic_colon import surface_distance

    return surface_distance(v, f, x)


# ----------------------------------------------------------------------------- fitted mesh
def fitted_mesh(
    p: Params, ds: float = 0.065, n_phi: int = 72
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Open side-wall mesh of a lumen-fit sweep (72 vertices per ring, 0.065 R axial step).

    Fits marked ``source_sidewall_candidate`` use the finer sweep of
    :func:`colonoworld.tube.wall_fit.sweep_mesh` instead.
    """
    if p.get("source_sidewall_candidate"):
        v, f, chart, _ = sweep_mesh(p)
        return v, f, chart
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
    faces = faces.reshape(-1, 3)
    edges = np.linalg.norm(
        np.roll(v.reshape(ns, n_phi, 3), -1, axis=1) - v.reshape(ns, n_phi, 3), axis=2
    )
    circ = edges.sum(1)
    sv = np.repeat(arc_length(p, zz), n_phi)
    tv = (np.c_[np.zeros(ns), np.cumsum(edges, axis=1)[:, :-1]] / circ[:, None]).ravel()
    fv = np.repeat(circ / 2, n_phi)
    return (
        (v * p["R_pose"]).astype(np.float32),
        faces.astype(np.int32),
        (sv.astype(np.float32), tv.astype(np.float32), fv.astype(np.float32)),
    )


def _grid_faces(ns: int, nph: int) -> np.ndarray:
    i = np.arange(ns - 1)[:, None]
    j = np.arange(nph)[None]
    a = i * nph + j
    b = i * nph + (j + 1) % nph
    c, d = a + nph, b + nph
    return (
        np.stack([np.stack([a, b, c], -1), np.stack([b, d, c], -1)], 2)
        .reshape(-1, 3)
        .astype(np.int64)
    )


def near_camera_weight(c: np.ndarray, near: np.ndarray, rc: float) -> np.ndarray:
    """1 within two local diameters of the camera arc, smoothstep fade to 0 over one more."""
    s0 = _arc(c)
    d_ = 2 * rc
    i0, i1 = int(near.min()), int(near.max())
    idx = np.arange(len(c))
    d = np.where(idx < i0, s0[i0] - s0, np.where(idx > i1, s0 - s0[i1], 0.0))
    x = np.clip(1 - (d - 2 * d_) / d_, 0, 1)
    return x * x * (3 - 2 * x)


def _clip_mean_zero(dev: np.ndarray, limit: float) -> np.ndarray:
    """Per-ring ``clip(dev - lam)`` with ``lam`` chosen (bisection) so the ring mean stays zero."""
    lo = dev.min(1) - limit
    hi = dev.max(1) + limit
    for _ in range(50):
        lam = 0.5 * (lo + hi)
        m = np.clip(dev - lam[:, None], -limit, limit).mean(1)
        lo = np.where(m > 0, lam, lo)
        hi = np.where(m > 0, hi, lam)
    return np.clip(dev - (0.5 * (lo + hi))[:, None], -limit, limit)


def _nearest_rings(c: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.array([np.argmin(np.linalg.norm(c - x, axis=1)) for x in poses[:, :3, 3]])


def _edit_rings(p0: Params, poses: np.ndarray, everywhere: bool, cfg: dict, state: dict) -> tuple:
    """Cross-section clip and short-stub extension of the fold-free fitted mesh."""
    v, f, chart = fitted_mesh(p0)
    cs, ct, cf = [np.asarray(x) for x in chart]
    nph = int((cs == cs[0]).sum())
    state["nph"] = nph
    ns = len(v) // nph
    if ns * nph != len(v) or not np.array_equal(np.asarray(f), _grid_faces(ns, nph)):
        raise ValueError("fitted mesh is not a ring grid")
    x = np.asarray(v, float).reshape(ns, nph, 3)
    c = x.mean(1)
    off = x - c[:, None]
    rad = np.linalg.norm(off, axis=2)
    rm = rad.mean(1)
    near = _nearest_rings(c, poses)
    rc = float(np.median(rm[near.min() : near.max() + 1]))
    w = near_camera_weight(c, near, rc)
    cw = np.ones(ns) if everywhere else 1 - w
    dev = rad / rm[:, None] - 1
    dc = _clip_mean_zero(dev, cfg["cross_section_clip"])
    rad2 = rm[:, None] * (1 + dev + cw[:, None] * (dc - dev))
    off = off * (rad2 / np.maximum(rad, 1e-12))[..., None]
    x = c[:, None] + off
    s = _arc(c)
    dd = 2 * rc
    add = [0, 0]
    if s[-1] < cfg["min_length_diameters"] * dd:
        need = (cfg["min_length_diameters"] * dd - s[-1]) / 2
        parts = []
        for end in (0, 1):
            idx = [0, 1, 2, 3] if end == 0 else [-1, -2, -3, -4]
            c0 = c[idx[0]]
            t = c[idx[0]] - c[idx[3]]
            t /= np.linalg.norm(t)
            step = float(np.linalg.norm(c[idx[0]] - c[idx[1]]))
            n = int(np.ceil(need / max(step, 1e-9)))
            add[end] = n
            ring = x[idx[0]] - c0
            ring = (
                ring
                / np.linalg.norm(ring, axis=1, keepdims=True)
                * np.linalg.norm(ring, axis=1).mean()
            )
            parts.append(np.stack([c0 + (k + 1) * step * t + ring for k in range(n)]))
        dcs0 = float(cs[nph] - cs[0])
        dcs1 = float(cs[-nph] - cs[-2 * nph])
        x = np.concatenate([parts[0][::-1], x, parts[1]])
        cs = np.r_[
            np.repeat(cs[0] - dcs0 * np.arange(add[0], 0, -1), nph),
            cs,
            np.repeat(cs[-1] + dcs1 * np.arange(1, add[1] + 1), nph),
        ]
        ct = np.r_[np.tile(ct[:nph], add[0]), ct, np.tile(ct[-nph:], add[1])]
        cf = np.r_[np.tile(cf[:nph], add[0]), cf, np.tile(cf[-nph:], add[1])]
    state["edits"] = dict(
        dev_abs_max_before=float(np.abs(dev).max()),
        dev_abs_max_after=float(np.abs(rad2 / rm[:, None] - 1).max()),
        length_over_D_before=float(s[-1] / dd),
        rings_added=add,
    )
    return (
        x.reshape(-1, 3).astype(np.asarray(v).dtype),
        _grid_faces(len(x), nph).astype(np.asarray(f).dtype),
        (cs.astype(np.float32), ct.astype(np.float32), cf.astype(np.float32)),
    )


# ----------------------------------------------------------------------------- shape (D)
def _lipschitz(s: np.ndarray, r: np.ndarray, k: float) -> np.ndarray:
    d = np.abs(s[:, None] - s[None, :])
    lo = (r[None, :] + k * d).min(1)
    hi = (r[None, :] - k * d).max(1)
    return 0.5 * (lo + hi)


def _tangent(c: np.ndarray, s: np.ndarray, sig: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    t = np.gradient(gaussian_filter1d(c, sig, axis=0, mode="nearest"), s, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _curvature(c: np.ndarray, s: np.ndarray, sig: float) -> np.ndarray:
    t = _tangent(c, s, sig)
    return np.linalg.norm(np.gradient(t, s, axis=0), axis=1)


def plausible_shape(c, s, rm, near, R, cfg):
    """Radius clip + slope cap + bend radius >= tube radius (up to 120 local smoothing passes)."""
    from scipy.ndimage import gaussian_filter1d

    rc_ = float(np.median(rm[near.min() : near.max() + 1]))
    rcl = np.clip(rm, cfg["radius_lo"] * rc_, cfg["radius_hi"] * rc_)
    ds = float(np.median(np.diff(s)))
    sig = max(1.0, 0.1 * 2 * rc_ / ds)
    cn = c.copy()
    t0 = _tangent(c, s, sig)
    passes = 0
    for passes in range(120):  # noqa: B007 (pass count is reported)
        s = _arc(cn)
        r = _lipschitz(s, rcl, cfg["radius_slope"])
        bad = _curvature(cn, s, sig) * r > 1.0
        if not bad.any():
            break
        w = gaussian_filter1d(bad.astype(float), max(1.0, 0.5 * rc_ / ds), mode="nearest")
        w = np.clip(w / max(w.max(), 1e-12), 0, 1)
        cs_ = gaussian_filter1d(cn, max(1.0, 0.5 * rc_ / ds), axis=0, mode="nearest")
        cn = cn + w[:, None] * (cs_ - cn)
    s = _arc(cn)
    r = _lipschitz(s, rcl, cfg["radius_slope"])
    kr = float((_curvature(cn, s, sig) * r).max())
    rot = _rotmin(t0, _tangent(cn, s, sig))
    info = dict(Rc=rc_, bend_passes=passes + 1, max_curv_x_r=kr)
    return cn, r, rot, s, info


def _near_camera_blend(c, s, rm, near, R, cfg, infl, keep_fitted):
    """Fitted rings near the camera blended into the plausible shape elsewhere."""
    cn, r, rot, s2, info = plausible_shape(c, s, rm, near, R, cfg)
    rm_o = rm / infl
    w = near_camera_weight(c, near, info["Rc"] / infl)
    if keep_fitted:
        cb, rb = c.copy(), rm.copy()
    else:
        cb = w[:, None] * c + (1 - w[:, None]) * cn
        rb = w * rm_o + (1 - w) * r
    s0 = _arc(c)
    ds = float(np.median(np.diff(s0)))
    sig = max(1.0, 0.1 * 2 * info["Rc"] / ds)
    sb = _arc(cb)
    kr = float((_curvature(cb, sb, sig) * rb).max())
    rot = _rotmin(_tangent(c, s0, sig), _tangent(cb, sb, sig))
    if keep_fitted:
        rot = np.broadcast_to(np.eye(3), (len(c), 3, 3)).copy()
    else:
        rot = rot.copy()
        rot[w >= 1 - 1e-9] = np.eye(3)
    return cb, rb, rot, sb, dict(info, max_curv_x_r=kr), w


# ----------------------------------------------------------------------------- relief (A)
def anatomy_relief(s, ph, rm, G, seg, rng, cam_s, cfg):
    """Taeniae, haustra, semilunar folds, rectal valves (fractions of the ring radius)."""
    from colonoworld.tube.colon_anatomy import ANATOMY

    tp = ANATOMY["taenia"]
    hp = ANATOMY["haustra"][seg]
    phase = rng.uniform(0, 2 * np.pi)
    inward = np.zeros_like(ph)
    outward = np.zeros_like(ph)
    for off in np.deg2rad(tp["offsets_deg"]):
        arc_cm = _wrap(ph - (phase + off)) * (rm / G)[:, None]
        inward -= tp["depth_fraction"][seg] * _c1(arc_cm / (tp["width_cm"] / 2))
    if hp["spacing_cm"]:
        cycle = s / G / hp["spacing_cm"] + rng.uniform()
        for q, off in enumerate(ANATOMY["haustra_sector_offsets_cycle"]):
            centre = np.deg2rad(ANATOMY["haustra_sector_centers_relative_deg"][q])
            awin = _c1(_wrap(ph - (phase + centre)) / (np.pi / 3))
            cyc = np.mod(cycle + off, 1.0)
            pouch = np.sin(np.pi * cyc) ** 2
            flong = _c1(np.minimum(cyc, 1 - cyc) / hp["fold_halfwidth_cycle"])
            outward += hp["haustral_bulge_fraction"] * pouch[:, None] * awin
            inward -= hp["semilunar_depth_fraction"] * flong[:, None] * awin
    if seg == "rectum":
        u = rng.uniform(-1.5, 1.5)
        for valve in ANATOMY["houston"]:
            sv = cam_s + (valve["position_cm"] - 9.0 + u) * G
            ang = phase + (np.pi / 2 if valve["world"][0] > 0 else -np.pi / 2)
            half = np.pi * valve["extent_fraction"]
            wa = _c1((s - sv) / (valve["axial_width_cm"] / 2 * G))
            wc = _c1(_wrap(ph - ang) / half)
            inward -= (valve["depth_cm"] / (rm / G))[:, None] * wa[:, None] * wc
    if seg == "transverse":
        outward = outward + cfg["transverse_lobe"] * 0.5 * (1 - np.cos(3 * (ph - phase)))
    return inward, outward


def build_surface(p, poses, seg, seed, scale, infl, tier, cfg):
    """Candidate surface of one tier (before the final cross-section clip)."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    state: dict[str, Any] = {}
    p0 = copy.deepcopy(p)
    p0["fold"]["depth_frac"] = 0.0
    v0, f, chart = _edit_rings(p0, poses, tier == "A", cfg, state)
    cs = np.asarray(chart[0])
    nph = int((cs == cs[0]).sum())
    ns = len(v0) // nph
    x = v0.reshape(ns, nph, 3)
    c = x.mean(1)
    off = x - c[:, None]
    rad = np.linalg.norm(off, axis=2)
    rm = rad.mean(1)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    ph = 2 * np.pi * np.asarray(chart[1])[: ns * nph].reshape(ns, nph)
    G = float(p["R_pose"]) / 2.5
    R = float(p["R_pose"])
    rng = np.random.default_rng(seed)
    near = _nearest_rings(c, poses)
    rm = rm * infl
    off = off * infl
    if tier in ("A", "B"):
        cn, r, rot, s, sinfo = plausible_shape(c, s, rm, near, R, cfg)
        w = np.zeros(len(c))
    else:
        cn, r, rot, s, sinfo, w = _near_camera_blend(c, s, rm, near, R, cfg, infl, tier == "D")
    a0, a1 = s[near].min(), s[near].max()
    off = np.einsum("nij,nkj->nki", rot, off) * (r / rm)[:, None, None]
    dirs = off / np.linalg.norm(off, axis=2, keepdims=True)
    dd = 2 * float(np.median(r[near.min() : near.max() + 1]))
    if seg:
        inward, outward = anatomy_relief(s, ph, r, G, seg, rng, 0.5 * (a0 + a1), cfg)
        k = (1 - w)[:, None]
        inward, outward = inward * k, outward * k
    else:
        inward, outward = np.zeros_like(ph), np.zeros_like(ph)
    l0 = max(min(dd, a0 - s[0]), 1e-9)
    l1 = max(min(dd, s[-1] - a1), 1e-9)
    endw = _smooth01((s - s[0]) / l0) * _smooth01((s[-1] - s) / l1)
    endw[(s >= a0) & (s <= a1)] = 1.0
    inward *= scale * endw[:, None]
    outward *= scale * endw[:, None]
    tree = cKDTree(poses[:, :3, 3])
    atten = np.ones_like(inward)

    def surf(a):
        fac = 1 + outward + inward * a
        if fac.min() <= 0.2:
            raise ValueError("relief folds the wall through itself")
        fac = fac / fac.mean(1, keepdims=True)
        xn = cn[:, None] + off * fac[..., None]
        r2 = np.linalg.norm(xn - cn[:, None], axis=2).mean(1)
        return cn[:, None] + (xn - cn[:, None]) * (r / r2)[:, None, None]

    for _ in range(8):  # fade inward features that come close to and face a camera
        xn = surf(atten)
        dv, ci = tree.query(xn.reshape(-1, 3))
        dv = dv.reshape(ns, nph)
        rel = xn - poses[ci.reshape(ns, nph), :3, 3]
        deriv = np.einsum("ijk,ijk->ij", rel, dirs) / np.maximum(dv, 1e-12)
        unsafe = (dv < 0.4 * R) & (deriv > 0.15) & (inward < 0) & (atten > 1e-3)
        if not unsafe.any():
            break
        red = gaussian_filter(unsafe.astype(float), (2.0, 1.0), mode=("nearest", "wrap"))
        red = np.clip(red / max(red.max(), 1e-12), 0, 1)
        atten *= 1 - red
    xn = surf(atten)
    vn = np.asarray(v0, float).copy()
    vn[: ns * nph] = xn.reshape(-1, 3)
    info = dict(sinfo, edits=state["edits"], scale=scale, inflation=float(infl))
    return vn, f, chart, info, state["nph"]


def final_clip(vn, poses, everywhere, nph, cfg):
    """Cross-section clip on the final surface."""
    vn = np.asarray(vn, float).copy()
    ns = len(vn) // nph
    x = vn[: ns * nph].reshape(ns, nph, 3)
    c = x.mean(1)
    off = x - c[:, None]
    rad = np.linalg.norm(off, axis=2)
    rm = rad.mean(1)
    near = _nearest_rings(c, poses)
    w = near_camera_weight(c, near, float(np.median(rm[near.min() : near.max() + 1])))
    cw = np.ones(ns) if everywhere else 1 - w
    dev = rad / rm[:, None] - 1
    dc = _clip_mean_zero(dev, cfg["cross_section_clip"])
    rad2 = rm[:, None] * (1 + dev + cw[:, None] * (dc - dev))
    vn[: ns * nph] = (c[:, None] + off * (rad2 / np.maximum(rad, 1e-12))[..., None]).reshape(-1, 3)
    return vn


# ----------------------------------------------------------------------------- checks
def lumen_counts(depth: np.ndarray, field: np.ndarray, R: float) -> list[dict[str, Any]]:
    """Per-frame lumen openings and wall-crossing flags from stride-8 depth."""
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
                openings=int((area > 0.01 * fs.sum()).sum()),
                wall_cross=bool(miss > 0.5 or near > 0.5),
            )
        )
    return out


def lumen_summary(counts: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        zero_lumen_frames=sum(c["openings"] == 0 for c in counts),
        multi_lumen_frames=sum(c["openings"] > 1 for c in counts),
        wall_cross_frames=sum(c["wall_cross"] for c in counts),
    )


def _flow_field_mask(real, cam):
    import cv2

    w, h = FLOW_SIZE
    yy, xx = np.mgrid[:h, :w]
    cw, ch = cam["clip_size"]
    uv = np.stack([(xx + 0.5) * cw / w - 0.5, (yy + 0.5) * ch / h - 0.5], -1)
    _, valid = rays_from_clip(uv.reshape(-1, 2), cam)
    mask = valid.reshape(h, w)
    mx = np.max([f.max(-1) for f in (real[0], real[len(real) // 2], real[-1])], 0)
    dark = (mx < 14).astype(np.uint8)
    _, lab = cv2.connectedComponents(dark, connectivity=4)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    border = np.isin(lab, list(edge)) & (dark > 0)
    border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return cv2.erode((mask & ~border).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _photometric_mask(rgb, field):
    import cv2

    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    spec = (hsv[..., 2] > 0.95) & (hsv[..., 1] < 0.15)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thr = np.percentile(gray[field], 2)
    bad = spec | (gray <= thr)
    return field & ~(cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-4 or np.std(b[ok]) < 1e-4:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _similarity(flow, mask, cam):
    """Least-squares translation, radial expansion and rotation of a flow field."""
    w, h = FLOW_SIZE
    yy, xx = np.mgrid[:h, :w]
    k = np.asarray(cam["K_clip"])
    cw, ch = cam["clip_size"]
    x = ((xx + 0.5) - (k[0, 2] + 0.5) * w / cw) / 80
    y = ((yy + 0.5) - (k[1, 2] + 0.5) * h / ch) / 80
    x, y = x[mask], y[mask]
    u, v = flow[mask].T
    one, zero = np.ones(len(x)), np.zeros(len(x))
    design = np.concatenate([np.stack([one, zero, x, -y], 1), np.stack([zero, one, y, x], 1)])
    return np.linalg.lstsq(design, np.r_[u, v], rcond=None)[0]


def flow_agreement(real: list, tube: list, cam: Calibration) -> dict[str, float]:
    """Dense-flow agreement between real and tube frames (160x120, Farneback).

    ``block_cos`` is the mean cosine of median flows over active 20x15 blocks; ``radial_r`` etc. are
    correlations of the per-pair translation/expansion/rotation fits.
    """
    import cv2

    w, h = FLOW_SIZE
    field = _flow_field_mask(real, cam)
    if field.mean() <= 0.2:
        raise ValueError("insufficient field")
    masks = [[_photometric_mask(f, field) for f in seq] for seq in (real, tube)]
    grays = [[cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq] for seq in (real, tube)]
    cosines, ratios, rcoef, tcoef, mags, coverage = [], [], [], [], [], []
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for t in range(len(real) - 1):
        flows = [cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, *FARNEBACK) for g in grays]
        mask = field.copy()
        for k, f in enumerate(flows):
            target = (
                cv2.remap(
                    masks[k][t + 1].astype(np.uint8),
                    xx + f[..., 0],
                    yy + f[..., 1],
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                > 0
            )
            mask &= masks[k][t] & target
        coverage.append(mask.sum() / field.sum())
        if mask.sum() <= 500:
            raise ValueError("insufficient flow support")
        mags.append([float(np.linalg.norm(f[mask], axis=1).mean()) for f in flows])
        rcoef.append(_similarity(flows[0], mask, cam))
        tcoef.append(_similarity(flows[1], mask, cam))
        for by in range(8):
            for bx in range(8):
                sl = np.s_[by * 15 : (by + 1) * 15, bx * 20 : (bx + 1) * 20]
                m = mask[sl]
                if m.sum() < 60:
                    continue
                a, b = [np.median(f[sl][m], axis=0) for f in flows]
                ma, mb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if max(ma, mb) >= 0.1:
                    c = float(np.dot(a, b) / (ma * mb)) if min(ma, mb) >= 0.1 else 0.0
                    cosines.append(float(np.clip(c, -1, 1)))
                if ma >= 0.1:
                    ratios.append(mb / ma)
    rc, tc, m = np.asarray(rcoef), np.asarray(tcoef), np.asarray(mags)
    out = dict(
        block_cos=float(np.mean(cosines)) if cosines else float("nan"),
        magnitude_ratio=float(np.median(ratios)) if ratios else float("nan"),
        valid_fraction_min=float(np.min(coverage)),
        real_p90_flow=float(np.percentile(m[:, 0], 90)),
        tube_p90_flow=float(np.percentile(m[:, 1], 90)),
        both_static_fraction=float(np.mean((m < 0.2).all(1))),
    )
    for j, name in enumerate(("horizontal", "vertical", "radial", "rotation")):
        out[name + "_r"] = _pearson(rc[:, j], tc[:, j])
    return out


def follows_camera(r: dict[str, float]) -> bool:
    """Balanced camera-following verdict (static pair, or agreeing motion)."""
    static = (
        r["both_static_fraction"] >= 0.8 and max(r["real_p90_flow"], r["tube_p90_flow"]) <= 0.35
    )
    moving = r["block_cos"] >= 0.35 and r["radial_r"] >= 0.40 and 0.4 <= r["magnitude_ratio"] <= 2.5
    return bool(r["valid_fraction_min"] >= 0.4 and (static or moving))


def _backface_fraction(scene, vertices, pose) -> float:
    mi = scene.mi
    d = scene.rays @ pose[:3, :3].T.astype(np.float32)
    o = pose[:3, 3].astype(np.float32)
    ray = mi.Ray3f(
        mi.Point3f(*[np.full(len(d), x, np.float32) for x in o]),
        mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]),
    )
    si = scene.scene.ray_intersect_preliminary(ray, coherent=True)
    t = np.array(si.t, np.float32)
    pid = np.array(si.prim_index, np.int64)
    hit = scene.ok & np.isfinite(t) & (pid >= 0) & (pid < len(scene.F)) & scene.field.ravel()
    tri = vertices[scene.F[pid[hit]]]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    sg = np.sign((fn * d[hit]).sum(1))
    return float((sg != np.sign(np.median(sg))).mean())


def check_candidate(vn, f, chart, surface, cam, real_small, times, base, zero0, info, texture, cfg):
    """Render a candidate (training renderer and exposure rule) and evaluate every check."""
    from colonoworld.tube.render import TubeScene, light_radius_for, load_texture, write_tube_video

    poses, field, R = surface["poses"], surface["field_mask"].astype(bool), float(surface["R"])
    clear = _surface_distance(vn, f, poses[:, :3, 3]) / R
    tube = tube_from_surface(vn, f, chart)
    scene = TubeScene(tube, cam, field, load_texture(str(texture)))
    light = light_radius_for(scene, poses, field) / cfg["exposure_divisor"]
    frames, depth, luma = [], [], []
    for pose in poses:
        img, dep = scene.frame(pose, light)
        depth.append(dep[::8, ::8].astype(np.float16))
        frames.append(img)
        luma.append(float(img[field].mean(0) @ LUMA))
    backface = max(
        _backface_fraction(scene, vn, poses[t]) for t in (0, len(poses) // 2, len(poses) - 1)
    )
    summary = lumen_summary(lumen_counts(np.asarray(depth), field, R))
    lo, hi = cfg["luma_bounds"]
    guards = dict(
        one_lumen=summary["multi_lumen_frames"] == 0 and summary["zero_lumen_frames"] <= zero0,
        zero_wall_cross=summary["wall_cross_frames"] == 0,
        clearance=bool(clear.min() >= cfg["min_clearance"]),
        backface=backface < 1e-3,
        luma=lo <= float(np.mean(luma)) <= hi,
        bend=bool(info.get("max_curv_x_r", 0.0) <= 1.0),
    )
    motion, motion_error = None, None
    if base is not None:
        with tempfile.TemporaryDirectory() as tmp:  # the check compares the encoded tube video
            path = Path(tmp) / "candidate.mp4"
            write_tube_video(path, np.asarray(frames))
            tube_small = decode_small(path, times)
        try:
            motion = flow_agreement(real_small, tube_small, cam)
            motion["follows"] = follows_camera(motion)
            guards["motion"] = bool(
                motion["block_cos"] >= base["block_cos"] - cfg["motion_tolerance"]
                and int(motion["follows"]) >= int(base["follows"])
            )
        except ValueError as e:
            motion_error = repr(e)
            guards["motion"] = False
    return dict(
        guards=guards,
        backface_frac=backface,
        luma=float(np.mean(luma)),
        clearance_min=float(clear.min()),
        summary=summary,
        light_radius=light,
        motion=motion,
        motion_error=motion_error,
        info=info,
    )


# ----------------------------------------------------------------------------- public API
def regularize_training_tube(
    params: Params,
    surface: dict[str, np.ndarray],
    original_depth: np.ndarray,
    real_small: list,
    original_tube_small: list,
    cam: Calibration,
    segment: str | None,
    window_id: str,
    texture: str | Path,
    cfg: dict[str, Any] | None = None,
) -> tuple[TubeModel, dict[str, Any]]:
    """Final training tube of one window.

    Args:
        params: lumen-fit sweep parameters of the window (a ``base_parameters`` key marks a closed
            cecal-pole fit, which is kept unchanged).
        surface: the fitted surface: ``V``, ``F``, ``poses``, ``R``, ``field_mask``, ``chart_s``,
            ``chart_t``, ``chart_f``.
        original_depth: stride-8 depth of the fitted tube's video (to count its zero-lumen frames).
        real_small / original_tube_small: real and fitted-tube frames at :func:`sample_times`,
            decoded with :func:`decode_small` (160x120).
        cam: fisheye calibration of the window.
        segment: anatomical segment of the window (``None`` = unknown: no relief).
        window_id: window identifier (seeds the relief phases).
        texture: mucosa texture path (renderer of the checks).
        cfg: ``training_shape`` section of ``configs/tube.yaml``.

    Returns:
        ``(tube, info)`` with ``info["tier"]``, ``info["scale"]`` and the per-candidate checks.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    poses, R = surface["poses"], float(surface["R"])
    field = surface["field_mask"].astype(bool)
    times = sample_times(len(poses))
    try:
        base = flow_agreement(real_small, original_tube_small, cam)
        base["follows"] = follows_camera(base)
    except ValueError:
        base = None
    zero0 = lumen_summary(lumen_counts(original_depth.astype(np.float32), field, R))[
        "zero_lumen_frames"
    ]
    seed = int(hashlib.sha256(window_id.encode()).hexdigest()[:8], 16)
    closed_pole = "base_parameters" in params
    dil = (
        float((params.get("static_dilation") or {}).get("factor", 1.0)) if not closed_pole else 1.0
    )
    scales = (1.0, 0.0) if segment else (0.0,)
    ladder = [] if (closed_pole or base is None) else [(t, s) for t in "ABCD" for s in scales]
    tries: list[dict[str, Any]] = []
    skip = False
    cap = cfg["inflation_cap"]
    for tier, scale in ladder + [("T", None)]:
        if tier == "T":
            vn, f = surface["V"], surface["F"]
            chart = (surface["chart_s"], surface["chart_t"], surface["chart_f"])
            info: dict[str, Any] = dict(tier="T")
        else:
            if skip:
                continue
            if tier == "D":
                steps = [1.0]
            else:
                steps = (
                    [1.0]
                    if dil <= cap
                    else sorted({min(dil, cap) / dil, *np.linspace(min(dil, cap) / dil, 1.0, 4)})
                )
                steps = steps + list(cfg["inflation_steps"])
            for infl in steps:
                vn, f, chart, info, nph = build_surface(
                    params, poses, segment, seed, scale, infl, tier, cfg
                )
                vn = final_clip(vn, poses, tier == "A", nph, cfg)
                clear = _surface_distance(vn, f, poses[:, :3, 3]) / R
                if clear.min() >= cfg["min_clearance"]:
                    break
            info = dict(info, tier=tier)
            if clear.min() < cfg["min_clearance"]:
                tries.append(
                    dict(tier=tier, scale=scale, fail="clearance", clear_min=float(clear.min()))
                )
                continue
        t = check_candidate(
            vn, f, chart, surface, cam, real_small, times, base, zero0, info, texture, cfg
        )
        t.update(tier=tier, scale=scale)
        tries.append(t)
        if t["motion_error"]:
            skip = True
        need = {k: v for k, v in t["guards"].items() if not (tier == "D" and k == "bend")}
        if tier == "T" or all(need.values()):
            tube = tube_from_surface(vn, f, chart)
            tube.meta.update(kind="training_tube", tier=tier, scale=scale)
            return tube, dict(tier=tier, scale=scale, tries=tries, poses=poses)
    raise RuntimeError("unreachable: the fitted surface is always accepted")
