# SPDX-License-Identifier: CC-BY-NC-4.0
"""Reference full colon ("donor template") for the synthetic colon shapes.

The synthetic shapes (:mod:`colonoworld.tube.colon_anatomy`) borrow three things from one
reference colon assembled from fitted training tubes: the fold relief texture of its wall, the
cecal-dwell rotations and the withdrawal step pattern of its camera path. The reference colon is
built here from

- a schematic anatomical course (control points per segment, scaled to 85 %),
- one fitted training tube per anatomical segment (chosen from a pool of placed window tubes by
  how close its calibre is to the reference gauge); its cross-sections and folds are laid along
  the course, and its own centreline wiggle (mean curvature removed) is added as a residual,
- a blind cecal cap (median of fitted cecal caps) with three taenia ridges, plus an appendiceal
  orifice dimple and ileocecal valve lips,
- a withdrawal camera: axial timing from a route table, small lateral/heading residuals replayed
  from smooth training windows, a base view looking slightly ahead, per-frame view controls
  checked by a depth probe (single visible lumen, no wall crossing) under the training rotation
  rate limits,
- a 189-frame cecal dwell that replays the rotations of a training cecum window.

All lengths are in pose units with ``reference_radius`` as the chain gauge (``R/2.5`` per cm).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.camera import rays_from_clip
from colonoworld.tube.wall_fit import ellipse_radius

SEGMENTS = (
    "rectum",
    "sigmoid",
    "descending",
    "splenic flexure",
    "transverse",
    "hepatic flexure",
    "ascending",
    "cecum",
)
NPHI = 96
CHUNK_STRIDE = 188


# ----------------------------------------------------------------------------- small helpers
def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0, 1)
    return u**3 * (10 - 15 * u + 6 * u * u)


def _interp(grid: np.ndarray, values: np.ndarray, s: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    flat = values.reshape(len(grid), -1)
    ans = np.stack([np.interp(s, grid, flat[:, k]) for k in range(flat.shape[1])], -1)
    return ans.reshape(np.asarray(s).shape + values.shape[1:])


def _bishop(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rotation-minimising frames (columns: normal, binormal, tangent)."""
    t = _unit(np.gradient(c, s, axis=0))
    n = np.empty_like(t)
    v = np.eye(3)[np.argmin(abs(t[0]))]
    n[0] = _unit(v - v @ t[0] * t[0])
    for i in range(1, len(t)):
        cross = np.cross(t[i - 1], t[i])
        d = np.clip(t[i - 1] @ t[i], -1, 1)
        n[i] = (
            n[i - 1]
            + np.cross(cross, n[i - 1])
            + np.cross(cross, np.cross(cross, n[i - 1])) / max(1 + d, 1e-9)
        )
        n[i] = _unit(n[i] - n[i] @ t[i] * t[i])
    return np.stack([n, np.cross(t, n), t], axis=-1)


def _frame_at(sgrid, c, b, s):
    bb = _interp(sgrid, b, s)
    t = _unit(bb[..., 2])
    right = _unit(bb[..., 0] - (bb[..., 0] * t).sum(-1, keepdims=True) * t)
    return _interp(sgrid, c, s), np.stack([right, np.cross(t, right), t], -1)


def _fold_factor_arc(fold: dict[str, float], s: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Haustral fold factor ``1 - fold`` on physical arc ``s`` (units of the window radius)."""
    h, phase = fold["spacing_R"], fold["phase"]
    out = np.zeros((len(s), len(ph)))
    first = int(np.floor(s.min() / h - phase)) + 0
    last = int(np.ceil(s.max() / h - phase)) + 3
    for j in range(first, last):
        sj = (j - 1 + phase) * h
        for sec in [j % 3] if j % 3 else [0, 1]:
            dp = np.angle(
                np.exp(1j * (ph - (fold["taenia_rad"] + np.pi / 3 + sec * 2 * np.pi / 3)))
            )
            x = np.clip(abs(dp) / (fold["span_rad"] / 2), 0, 1)
            a = np.cos(np.pi * x / 2) ** 2 * (x < 1)
            g = np.exp(-0.5 * ((s[:, None] - sj - 0.08 * np.cos(dp)[None]) / fold["width_R"]) ** 2)
            out = np.maximum(out, fold["depth_frac"] * g * a[None])
    return 1 - out


def opening_counts(depth: np.ndarray, field: np.ndarray, radius: float) -> list[dict[str, Any]]:
    """Per-frame lumen openings (deep regions) and wall-crossing flags of a stride-8 depth map."""
    import cv2

    fs = field[::8, ::8] if field.shape == (832, 1104) else field
    out = []
    medwin = float(np.nanmedian(np.where(fs, depth, np.nan)))
    for t, d in enumerate(depth.astype(np.float32)):
        valid = fs & np.isfinite(d)
        med = float(np.median(d[valid]))
        _, _, st, _ = cv2.connectedComponentsWithStats((valid & (d > 2 * med)).astype(np.uint8), 8)
        area = st[1:, cv2.CC_STAT_AREA]
        miss = float((fs & ~np.isfinite(d)).sum() / fs.sum())
        near = float((valid & (d < 0.1 * medwin)).sum() / fs.sum())
        far = cv2.connectedComponents((valid & (d >= 8 * radius)).astype(np.uint8), 8)[0] - 1
        out.append(
            dict(
                t=t,
                opening_count=int((area > 0.01 * fs.sum()).sum()),
                opening_8R_count=int(far),
                missing_fraction=miss,
                near_fraction=near,
                wall_cross=bool(miss > 0.5 or near > 0.5),
            )
        )
    return out


# ----------------------------------------------------------------------------- inputs
def schematic_course(ctrl: dict[str, list]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Natural cubic spline through the course control points (cm).

    Returns ``(arc, points, bounds)``: arc length of 40000 samples, the samples and the arc
    position of each segment's first control point plus the total length.
    """
    from scipy.interpolate import CubicSpline

    pts, starts = [], []
    for name in SEGMENTS:
        starts.append(len(pts))
        pts.extend(ctrl[name])
    pts = np.array(pts)
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    sp = CubicSpline(u, pts, bc_type="natural")
    uu = np.linspace(0, u[-1], 40000)
    x = sp(uu)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    b = np.r_[[np.interp(u[i], uu, arc) for i in starts], arc[-1]]
    return arc, x, b


def residual_blocks(
    windows: list[tuple[str, np.ndarray, float]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Small lateral/heading residuals of 189-frame training windows.

    Per window (``(id, poses, radius)``): positions in units of its radius relative to frame 0,
    high-passed by a sigma-4 Gaussian (axial component removed); headings relative to the
    sigma-4 smoothed orientation. Windows with small residuals (``selection`` ranges) are kept.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    out = []
    for wid, poses, radius in windows:
        pos = poses[:, :3, 3] / float(radius)
        pos = (pos - pos[0]) @ poses[0, :3, :3]
        wp = pos - gaussian_filter1d(pos, 4, axis=0, mode="reflect")
        wp -= wp[0]
        wp[:, 2] = 0
        quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
        for i in range(1, len(quat)):
            if np.dot(quat[i - 1], quat[i]) < 0:
                quat[i] *= -1
        smooth = gaussian_filter1d(quat, 4, axis=0, mode="reflect")
        smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
        rel = Rotation.from_quat(smooth).as_matrix().transpose(0, 2, 1) @ poses[:, :3, :3]
        wr = Rotation.from_matrix(rel).as_rotvec()
        wr -= wr[0]
        out.append(
            dict(
                id=wid,
                position_R_max=float(np.linalg.norm(wp, axis=1).max()),
                heading_deg_max=float(np.rad2deg(np.linalg.norm(wr, axis=1)).max()),
                heading_step_deg_max=float(
                    np.rad2deg(np.linalg.norm(np.diff(wr, axis=0), axis=1)).max()
                ),
                position=wp,
                heading=wr,
            )
        )
    pos_lo, pos_hi = selection["position_R_max"]
    head_lo, head_hi = selection["heading_deg_max"]
    step = selection["heading_step_deg_max"]
    return [
        r
        for r in out
        if pos_lo < r["position_R_max"] <= pos_hi
        and head_lo < r["heading_deg_max"] <= head_hi
        and r["heading_step_deg_max"] <= step
    ]


# ----------------------------------------------------------------------------- course + wall
def _donor_fields(rec, loc, ph):
    """Cross-section ellipses and folds of a donor window at signed arc ``loc`` from its centre."""
    from scipy.interpolate import PchipInterpolator

    p = rec["parameters"]
    z = np.asarray(p["z_R"])
    q = np.asarray(p["profiles"])
    zz = np.linspace(z[0], z[-1], 2400)
    ip = PchipInterpolator(z, q, axis=0)
    cc = np.c_[ip(zz)[:, :2], zz]
    ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))]
    sc = float(np.interp(rec["source_z_center"], zz, ss))
    source_s = sc + loc
    half = rec["source_span_R"] / 2
    support = np.interp([rec["source_z_center"] - half, rec["source_z_center"] + half], zz, ss)
    zs = np.interp(np.clip(source_s, *support), ss, zz)
    qq = ip(zs)
    radial = ellipse_radius(qq[:, None, :], ph[None])
    folds = _fold_factor_arc(p["fold"], source_s, ph)
    return radial, np.log(folds)


def _donor_residual(rec, loc, halfspan):
    """Donor centreline wiggle with its mean curvature removed, tapered to zero at the ends."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import PchipInterpolator

    p = rec["parameters"]
    zz = np.linspace(p["z_R"][0], p["z_R"][-1], 2400)
    q = PchipInterpolator(p["z_R"], p["profiles"], axis=0)(zz)
    c = np.c_[q[:, :2], zz]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    sc = float(np.interp(rec["source_z_center"], zz, s))
    half = rec["source_span_R"] / 2
    support = np.interp([rec["source_z_center"] - half, rec["source_z_center"] + half], zz, s)
    lo = max(support[0], sc - halfspan)
    hi = min(support[1], sc + halfspan)
    tt = np.linspace(lo, hi, 500)
    cc = _interp(s, c, tt)
    t = _unit(np.gradient(cc, tt, axis=0))
    k = np.gradient(t, tt, axis=0)
    mean = np.trapezoid(k, tt, axis=0) / (hi - lo)
    k -= mean
    v = cumulative_trapezoid(k, tt, axis=0, initial=0)
    d = cumulative_trapezoid(v, tt, axis=0, initial=0)
    d -= (tt - lo)[:, None] / (hi - lo) * (d[-1] - d[0]) + d[0]
    x = sc + loc
    u = np.clip((x - lo) / (hi - lo), 0, 1)
    weight = _quintic(u / 0.25) * _quintic((1 - u) / 0.25)
    return _interp(tt, d, x)[:, :2] * weight[:, None]


def _joins(s, b, radius):
    owner = np.clip(np.searchsorted(b[1:], s, side="right"), 0, 7)
    left, right, wt = owner.copy(), owner.copy(), np.zeros(len(s))
    for i, edge in enumerate(b[1:-1]):
        half = min(0.10 * radius, 0.18 * (edge - b[i]), 0.18 * (b[i + 2] - edge))
        hit = abs(s - edge) < half
        left[hit], right[hit] = i, i + 1
        wt[hit] = _quintic((s[hit] - edge + half) / (2 * half))
    return left, right, wt


def choose_donors(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per segment, the pool window whose calibre near its centre is closest to the gauge."""
    from scipy.interpolate import PchipInterpolator

    chosen = []
    for anatomy in SEGMENTS:
        ranked = []
        for k, r in enumerate(pool):
            if r["anatomy"] != anatomy:
                continue
            p = r["parameters"]
            qq = PchipInterpolator(p["z_R"], p["profiles"], axis=0)(
                r["source_z_center"] + np.linspace(-0.5, 0.5, 21)
            )
            ranked.append((float(abs(np.median(qq[:, 2])) + 0.1 * np.std(qq[:, 2])), k, r))
        if not ranked:
            raise ValueError("no donor window for " + anatomy)
        chosen.append(min(ranked, key=lambda a: a[:2])[2])
    return chosen


def build_reference_wall(
    course: tuple[np.ndarray, np.ndarray, np.ndarray],
    chosen: list[dict[str, Any]],
    cap_fits: list[dict[str, float]],
    radius: float,
    scale: float,
) -> dict[str, Any]:
    """Reference colon wall: course + donor sections/folds/residuals + blind cecal cap."""
    arc, x, bounds = course
    g = radius / 2.5
    length = float(arc[-1] * scale * g)
    s = np.unique(np.r_[np.arange(0, length, 0.04 * radius), bounds * scale * g, length])
    c_target = _interp(arc, x, s / (scale * g)) * scale * g
    c_target -= x[0] * scale * g
    b0 = _bishop(c_target, s)
    b = bounds * scale * g
    ph = np.linspace(-np.pi, np.pi, NPHI, endpoint=False)
    left, right, wt = _joins(s, b, radius)
    displacement = np.zeros((len(s), 2))
    for k, rec in enumerate(chosen):
        center = (b[k] + b[k + 1]) / 2
        halfspan = (b[k + 1] - b[k]) / (2 * radius)
        loc = (s - center) / radius
        d0 = _donor_residual(rec, loc, halfspan)
        w = np.where(left == k, 1 - wt, 0) + np.where(right == k, wt, 0)
        displacement += w[:, None] * d0
    c = c_target + radius * np.einsum("nij,nj->ni", b0[:, :, :2], displacement)
    target_s = s.copy()
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    b = np.interp(b, target_s, s)
    length = float(s[-1])
    frames = _bishop(c, s)
    # Sections and folds on the realised arc.
    left, right, wt = _joins(s, b, radius)
    lograd = np.zeros((len(s), NPHI))
    for k, rec in enumerate(chosen):
        center = (b[k] + b[k + 1]) / 2
        loc = (s - center) / radius
        rad0, lf = _donor_fields(rec, loc, ph)
        w = np.where(left == k, 1 - wt, 0) + np.where(right == k, wt, 0)
        lograd += w[:, None] * (np.log(rad0) + lf)
    rad = np.exp(lograd) * radius
    # Blind cecal cap: cos/sin dome with three taenia ridges converging on the pole.
    cap = dict(
        dome_length_R=float(np.median([f["dome_length_R"] for f in cap_fits])),
        fold_depth_R=float(np.median([f["fold_depth_R"] for f in cap_fits])),
        fold_width_R=float(np.median([f["fold_width_R"] for f in cap_fits])),
        taenia_rad=chosen[-1]["parameters"]["fold"]["taenia_rad"],
    )
    neck = length - cap["dome_length_R"] * radius
    body = s[s <= neck]
    neck = float(body[-1])
    cap_len = length - neck
    angle = np.linspace(0, np.pi / 2, 65, endpoint=False)[1:]
    ss = np.r_[body, neck + cap_len * np.sin(angle)]
    centers = _interp(s, c, ss)
    ring_frames = _frame_at(s, c, frames, ss)[1]
    radial = np.vstack(
        [_interp(s, rad, body), _interp(s, rad, neck)[None] * np.cos(angle)[:, None]]
    )
    directions = (
        np.cos(ph)[None, :, None] * ring_frames[:, None, :, 0]
        + np.sin(ph)[None, :, None] * ring_frames[:, None, :, 1]
    )
    rings = centers[:, None] + radial[:, :, None] * directions
    xx = radial[len(body) :] / radius * np.cos(ph)
    yy = radial[len(body) :] / radius * np.sin(ph)
    ridges = np.zeros_like(xx)
    for k in range(3):
        a = cap["taenia_rad"] + k * 2 * np.pi / 3
        along = xx * np.cos(a) + yy * np.sin(a)
        perp = -xx * np.sin(a) + yy * np.cos(a)
        ridges += (
            np.exp(-0.5 * (perp / cap["fold_width_R"]) ** 2)
            * 0.5
            * (1 + np.tanh(along / cap["fold_width_R"]))
        )
    u = angle / (np.pi / 2)
    fade = (1 - np.cos(angle) ** 2) ** 2 * (u * u * (3 - 2 * u))
    disp = -cap["fold_depth_R"] * ridges * fade[:, None] * radius
    rings[len(body) :] += disp[:, :, None] * ring_frames[len(body) :, None, :, 2]
    pole = c[-1] - 1.5 * cap["fold_depth_R"] * radius * frames[-1, :, 2]
    ns = len(ss)
    verts = np.vstack([rings.reshape(-1, 3), pole]).astype(np.float32)
    ii = np.arange((ns - 1) * NPHI).reshape(-1, NPHI)
    jj = np.roll(ii, -1, axis=1)
    faces = np.stack(
        [np.stack([ii, jj, ii + NPHI], -1), np.stack([jj, jj + NPHI, ii + NPHI], -1)], -2
    ).reshape(-1, 3)
    aa = (ns - 1) * NPHI + np.arange(NPHI)
    bb = (ns - 1) * NPHI + np.roll(np.arange(NPHI), -1)
    faces = np.vstack([faces, np.c_[aa, bb, np.full(NPHI, ns * NPHI)]]).astype(np.int32)
    face_cap = np.r_[np.repeat(ss[:-1] >= neck, 2 * NPHI), np.ones(NPHI, bool)]
    return dict(
        V=verts,
        F=faces,
        face_cap=face_cap,
        s_ring=ss,
        radial=radial,
        frames_ring=ring_frames,
        s=s,
        C=c,
        B=frames,
        anatomy_bounds=b,
        cap_neck=neck,
    )


# ----------------------------------------------------------------------------- camera
def withdrawal_camera(
    wall: dict[str, Any],
    blocks: list[dict[str, Any]],
    camera_s: np.ndarray,
    radius: float,
    lookahead: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Poses along ``camera_s`` with replayed training residual blocks (189 frames, stride 188).

    Returns ``(poses, residual headings)``; the base view looks ``lookahead`` radii ahead.
    """
    from scipy.spatial.transform import Rotation

    s, c, frames = wall["s"], wall["C"], wall["B"]
    n = len(camera_s)
    xy = np.zeros((n, 2))
    rot = np.zeros((n, 3))
    for k in range((n - 1) // CHUNK_STRIDE):
        member = blocks[k % len(blocks)]
        idx = slice(CHUNK_STRIDE * k, CHUNK_STRIDE * k + CHUNK_STRIDE + 1)
        wp = np.asarray(member["position"])[:, :2]
        wr = np.asarray(member["heading"])
        if k == 0:
            xy[idx] = wp
            rot[idx] = wr
        else:  # keep each block's increments; a 16-frame quintic connector removes the jump
            oldxy = xy[CHUNK_STRIDE * k].copy()
            oldrot = rot[CHUNK_STRIDE * k].copy()
            xy[idx] = wp
            rot[idx] = wr
            wt = 1 - _quintic(np.arange(16) / 15)
            xy[CHUNK_STRIDE * k : CHUNK_STRIDE * k + 16] += wt[:, None] * (oldxy - wp[0])
            rot[CHUNK_STRIDE * k : CHUNK_STRIDE * k + 16] += wt[:, None] * (oldrot - wr[0])
    cen, b = _frame_at(s, c, frames, camera_s)
    position_basis = b.copy()
    if lookahead > 0:
        view = _unit(_interp(s, c, np.minimum(camera_s + lookahead * radius, s[-1])) - cen)
        right = _unit(b[:, :, 0] - (b[:, :, 0] * view).sum(1, keepdims=True) * view)
        b = np.stack([right, np.cross(view, right), view], axis=-1)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, 3] = cen + radius * np.einsum("nij,nj->ni", position_basis[:, :, :2], xy)
    poses[:, :3, :3] = b @ Rotation.from_rotvec(rot).as_matrix()
    return poses, rot


class _DepthProbe:
    """Stride-8 ray-cast depth of the wall mesh (Mitsuba, CPU)."""

    def __init__(self, wall: dict[str, Any], field: np.ndarray, cam: dict, radius: float):
        import drjit as dr
        import mitsuba as mi

        mi.set_variant("llvm_ad_rgb")
        dr.set_thread_count(1)
        self.mi, self.wall, self.radius = mi, wall, radius
        mesh = mi.Mesh("probe", len(wall["V"]), len(wall["F"]))
        prm = mi.traverse(mesh)
        prm["vertex_positions"] = mi.Float(wall["V"].ravel())
        prm["faces"] = mi.UInt32(wall["F"].astype(np.uint32).ravel())
        prm.update()
        self.scene = mi.load_dict({"type": "scene", "wall": mesh})
        yy, xx = np.mgrid[:832:8, :1104:8]
        r, ok = rays_from_clip(np.c_[xx.ravel(), yy.ravel()], cam)
        self.rays = r.astype(np.float32)
        self.field = field[::8, ::8].astype(bool) & ok.reshape(104, 138)

    def run(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mi = self.mi
        depth, terminal = [], []
        for pose in poses:
            d = self.rays @ pose[:3, :3].T.astype(np.float32)
            o = pose[:3, 3].astype(np.float32)
            ray = mi.Ray3f(
                mi.Point3f(*[np.full(len(d), v, np.float32) for v in o]),
                mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]),
            )
            si = self.scene.ray_intersect_preliminary(ray, coherent=True)
            dep = np.asarray(si.t, np.float32).reshape(104, 138)
            pid = np.asarray(si.prim_index, np.int64).reshape(104, 138)
            valid = self.field & np.isfinite(dep) & (pid >= 0) & (pid < len(self.wall["F"]))
            pid = np.where(valid, pid, 0)
            far = valid & (dep >= np.quantile(dep[valid], 0.9))
            cap = valid & self.wall["face_cap"][pid]
            terminal.append(bool((cap & far).sum() / max(1, far.sum()) >= 0.9))
            dep[~valid] = np.nan
            depth.append(dep.astype(np.float16))
        return np.asarray(depth), np.asarray(terminal)

    def flags(self, depth: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        """Bad frames: wall crossing, several openings, or no opening unless the pole is seen."""
        cs = opening_counts(depth, self.field, self.radius)
        return np.array(
            [
                c["wall_cross"] or c["opening_count"] > 1 or (c["opening_count"] == 0 and not term)
                for c, term in zip(cs, terminal, strict=True)
            ]
        )


def refine_view(
    wall: dict[str, Any],
    field: np.ndarray,
    cam: dict,
    radius: float,
    poses: np.ndarray,
    heading: np.ndarray,
    controls: np.ndarray,
    rotation_stats: dict[str, float],
    angle_limit: float = 45.0,
    rate_limit: float = 8.0,
) -> np.ndarray:
    """Per-frame base-view controls, repaired where the depth probe flags a frame.

    ``controls`` (rotation vectors) tilt the base view; frames whose view is flagged (or whose
    rotation rate exceeds the limits) and a 40-frame collar are re-optimised by dynamic
    programming over a grid of view tilts (up to ``angle_limit`` degrees), with transitions
    limited to ``rate_limit`` degrees per frame and penalised above the training p99 rate.
    Camera positions are unchanged.
    """
    from scipy.spatial.transform import Rotation

    probe = _DepthProbe(wall, field, cam, radius)
    original = poses.copy()
    n = len(poses)
    cache: dict[int, np.ndarray] = {}
    r_train = Rotation.from_rotvec(heading).as_matrix()
    r_view = original[:, :3, :3] @ r_train.transpose(0, 2, 1)
    train_p99 = float(rotation_stats["p99"])
    if rate_limit > rotation_stats["max"]:
        raise ValueError("rate limit above the training maximum")
    vectors = [np.zeros(3)]
    diag = 0.70710678
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [diag, diag], [diag, -diag]]
    directions += [[-diag, diag], [-diag, -diag]]
    for angle in [2, 5, 8, 12, 16, 20, 25, 30, 40, 45]:
        for d in directions:
            vectors.append(np.array([d[0] * angle, d[1] * angle, 0.0]))
    base_deg = np.asarray(vectors)
    selected = np.asarray(controls, dtype=np.float64).copy()
    if np.max(np.linalg.norm(selected, axis=1)) > np.deg2rad(angle_limit) + 1e-6:
        raise ValueError("view controls exceed the angle limit")
    initial = original.copy()
    initial[:, :3, :3] = r_view @ Rotation.from_rotvec(selected).as_matrix() @ r_train
    frozen_controls = selected.copy()
    d0, cap0 = probe.run(initial)
    initial_flags = probe.flags(d0, cap0)
    initial_rates = np.degrees(
        Rotation.from_matrix(
            initial[:-1, :3, :3].transpose(0, 2, 1) @ initial[1:, :3, :3]
        ).magnitude()
    )
    repair = np.flatnonzero(initial_flags)
    high0 = np.flatnonzero(initial_rates > rate_limit + 1e-5)
    if np.quantile(initial_rates, 0.99) > train_p99 + 1e-5:
        high0 = np.flatnonzero(initial_rates > train_p99 + 1e-5)
    repair = np.unique(np.r_[repair, high0, np.minimum(n - 1, high0 + 1)])
    active = np.zeros(n, bool)
    for t in repair:
        active[max(0, t - 40) : min(n, t + 41)] = True
    for round_index in range(8):
        halfwidth = 4.0 / 2**round_index
        x, y = np.meshgrid(
            np.linspace(-halfwidth, halfwidth, 9), np.linspace(-halfwidth, halfwidth, 9)
        )
        offsets = np.c_[x.ravel(), y.ravel(), np.zeros(81)]
        grid, matrices, local = [], [], []
        for t in range(n):
            if not active[t]:
                ctrl = np.rad2deg(frozen_controls[t : t + 1])
                matrix = initial[t : t + 1, :3, :3]
                bad = np.array([initial_flags[t]])
            else:
                ctrl = np.vstack([base_deg, np.rad2deg(selected[t])[None] + offsets])
                cand = Rotation.from_rotvec(np.deg2rad(ctrl)).as_matrix()
                q = np.repeat(original[t : t + 1], len(ctrl), axis=0)
                q[:, :3, :3] = r_view[t][None] @ cand @ r_train[t][None]
                matrix = q[:, :3, :3]
                if t not in cache:
                    dg, cg = probe.run(q[:81])
                    cache[t] = probe.flags(dg, cg)
                dl, cl = probe.run(q[81:])
                bad = np.r_[cache[t], probe.flags(dl, cl)]
            norm = np.linalg.norm(ctrl, axis=1)
            valid = norm <= angle_limit + 1e-7
            grid.append(ctrl)
            matrices.append(matrix)
            local.append(np.where(valid, bad.astype(float) * 1e6 + 0.02 * norm**2, np.inf))
        cost = local[0].copy()
        back = []
        for t in range(1, n):
            delta = np.linalg.norm(grid[t - 1][:, None] - grid[t][None, :], axis=-1)
            cosine = (
                matrices[t - 1].reshape(len(grid[t - 1]), 9)
                @ matrices[t].reshape(len(grid[t]), 9).T
                - 1
            ) / 2
            ang = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            transition = 2 * delta**2 + 50 * np.maximum(0.0, ang - train_p99) ** 2
            m = cost[:, None] + np.where(ang <= rate_limit + 1e-6, transition, np.inf)
            prev = np.argmin(m, axis=0)
            cost = m[prev, np.arange(len(grid[t]))] + local[t]
            back.append(prev)
        choice = np.empty(n, int)
        choice[-1] = int(np.argmin(cost))
        if not np.isfinite(cost[choice[-1]]):
            raise ValueError("no feasible view path")
        for t in range(n - 2, -1, -1):
            choice[t] = back[t][choice[t + 1]]
        selected = np.deg2rad(np.asarray([grid[t][k] for t, k in enumerate(choice)]))
        out = initial.copy()
        out[active, :3, :3] = (
            r_view[active] @ Rotation.from_rotvec(selected[active]).as_matrix() @ r_train[active]
        )
        dd, cc = probe.run(out)
        failed = np.flatnonzero(probe.flags(dd, cc))
        realized = np.degrees(
            Rotation.from_matrix(out[:-1, :3, :3].transpose(0, 2, 1) @ out[1:, :3, :3]).magnitude()
        )
        fast = np.flatnonzero(realized > rate_limit + 1e-5)
        p99_ok = bool(np.quantile(realized, 0.99) <= train_p99 + 1e-5)
        high = np.flatnonzero(realized > train_p99 + 1e-5)
        if not len(failed) and not len(fast) and p99_ok:
            break
        rate_extra = np.unique(np.r_[fast, high if not p99_ok else np.array([], int)])
        extra = np.unique(np.r_[failed, rate_extra, np.minimum(n - 1, rate_extra + 1)])
        for t in extra:
            active[max(0, t - 40) : min(n, t + 41)] = True
    return out


# ----------------------------------------------------------------------------- cecum
def add_cecal_landmarks(
    wall: dict[str, Any], radius: float, orifice: dict[str, float], valve: dict[str, float]
) -> np.ndarray:
    """Ileocecal valve lips (two inward bumps) and appendiceal orifice dimple on the cap.

    Returns the new ring radii; ``wall['V']`` is updated in place.
    """
    length = wall["s"][-1]
    s_valve = length - 0.1 * radius
    ph = np.linspace(-np.pi, np.pi, NPHI, endpoint=False)
    ss = wall["s_ring"]
    rings = wall["V"][:-1].reshape(-1, NPHI, 3).astype(float).copy()
    radial = wall["radial"].copy()
    delta = np.angle(np.exp(1j * (ph - valve["phi_rad"])))
    for side in [-1, 1]:
        center = s_valve / radius + side * 0.28 * valve["width_R"]
        bump = (
            0.5
            * valve["height_R"]
            * np.exp(-0.5 * ((ss / radius - center) / (0.24 * valve["width_R"])) ** 2)[:, None]
            * np.exp(-0.5 * (delta / (valve["length_R"] / max(valve["wall_radius_R"], 0.1))) ** 2)[
                None
            ]
        )
        drad = -bump * radius
        radial += drad
        direction = (
            np.cos(ph)[None, :, None] * wall["frames_ring"][:, None, :, 0]
            + np.sin(ph)[None, :, None] * wall["frames_ring"][:, None, :, 1]
        )
        rings += drad[:, :, None] * direction
    if radial.min() <= 0:
        raise ValueError("valve relief closes the lumen")
    neck = wall["cap_neck"]
    cap = ss > neck
    rho = np.clip(radial / np.maximum(_interp(ss, wall["radial"], neck), 1e-9), 0, 1)
    angle = np.arcsin(np.clip((ss - neck) / (length - neck), 0, 1))
    fade = (
        (1 - rho * rho) ** 2
        * (angle / (np.pi / 2))[:, None] ** 2
        * (3 - 2 * angle / (np.pi / 2))[:, None]
    )
    x = radial / radius * np.cos(ph)
    y = radial / radius * np.sin(ph)
    dd = (x - orifice["x_R"]) ** 2 + (y - orifice["y_R"]) ** 2
    disp = orifice["depth_R"] * np.exp(-0.5 * dd / orifice["radius_R"] ** 2) * fade * radius
    disp[~cap] = 0
    rings += disp[:, :, None] * wall["frames_ring"][:, None, :, 2]
    wall["V"] = np.vstack([rings.reshape(-1, 3), wall["V"][-1]]).astype(np.float32)
    return radial


def cecal_dwell(
    wall: dict[str, Any],
    first_pose: np.ndarray,
    first_s: float,
    window_poses: np.ndarray,
    window_radius: float,
    radius: float,
    start_inset: float,
    frames: int = 189,
) -> tuple[np.ndarray, np.ndarray]:
    """Cecal inspection: the rotations of a training cecum window, trend removed, ending exactly
    at the first withdrawal pose; lateral sigma-4 high-pass residuals; slow axial pull-back."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    p = window_poses
    if len(p) != frames:
        raise ValueError("dwell window length")
    u = np.linspace(0, 1, frames)
    raw = Rotation.from_matrix(p[0, :3, :3].T @ p[:, :3, :3]).as_rotvec()
    rot = raw - u[:, None] * raw[-1]
    rot -= rot[0] + u[:, None] * (rot[-1] - rot[0])
    pos = (p[:, :3, 3] - p[0, :3, 3]) @ p[0, :3, :3] / window_radius
    xy = pos[:, :2] - gaussian_filter1d(pos[:, :2], 4, axis=0, mode="reflect")
    xy -= xy[0] + u[:, None] * (xy[-1] - xy[0])
    length = wall["s"][-1]
    cs = np.linspace(length - start_inset * radius, first_s, frames)
    if not ((cs >= wall["anatomy_bounds"][-2]).all() and (np.diff(cs) < 0).all()):
        raise ValueError("dwell leaves the cecum")
    cen, b = _frame_at(wall["s"], wall["C"], wall["B"], cs)
    q = np.tile(np.eye(4), (frames, 1, 1))
    q[:, :3, 3] = cen + radius * np.einsum("nij,nj->ni", b[:, :, :2], xy)
    q[:, :3, :3] = first_pose[:3, :3] @ Rotation.from_rotvec(rot).as_matrix()
    if not np.allclose(q[-1], first_pose, rtol=0, atol=1e-9):
        raise ValueError("dwell does not meet the withdrawal start")
    q[-1] = first_pose
    return q, cs


# ----------------------------------------------------------------------------- public API
def reference_frame_to_clip(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Reference frame on the 1104x832 clip canvas.

    A frame already on the canvas is returned unchanged; a native-resolution frame (e.g. the
    REAL-Colon frame named in ``cfg['reference_frame_source']``) is mapped with the reference
    camera's native-to-clip affine. Only its lens content mask is used.
    """
    import cv2

    width, height = cfg.get("reference_frame_clip_size", (1104, 832))
    if image.shape[:2] == (height, width):
        return image
    a = np.asarray(cfg["reference_frame_native_to_clip"], dtype=float)[:2]
    return cv2.warpAffine(image, a, (width, height), flags=cv2.INTER_AREA)


def load_donor_inputs(
    cfg: dict[str, Any], reference_frame: np.ndarray | None = None
) -> dict[str, Any]:
    """Load the per-window inputs listed in ``cfg`` (the ``donor_template`` config section).

    Files under ``cfg['inputs']``: ``window_fits/<window>.json`` (fitted profile-sweep
    parameters ``z_R``, ``profiles``, ``fold``), ``window_poses/<window>.npz`` (``poses``, ``R``)
    and ``view_controls.npy``. Window names use ``__`` for ``:``. ``reference_frame`` (RGB) is the
    user-provided reference frame (see :func:`reference_frame_to_clip`); without it
    ``reference_frame.png`` is read from ``cfg['inputs']``.
    """
    import cv2

    root = Path(cfg["inputs"])

    def key(wid: str) -> str:
        return wid.replace(":", "__")

    pool = []
    for rec in cfg["donor_pool"]:
        params = json.loads((root / "window_fits" / f"{key(rec['window'])}.json").read_text())
        pool.append(dict(rec, parameters=params))

    def poses(wid: str) -> tuple[np.ndarray, float]:
        with np.load(root / "window_poses" / f"{key(wid)}.npz") as z:
            return z["poses"], float(z["R"])

    if reference_frame is None:
        frame = cv2.cvtColor(cv2.imread(str(root / "reference_frame.png")), cv2.COLOR_BGR2RGB)
    else:
        frame = reference_frame_to_clip(reference_frame, cfg)
    return dict(
        pool=pool,
        blocks=[(w, *poses(w)) for w in cfg["residual_block_windows"]],
        dwell=poses(cfg["dwell_window"]),
        reference_frame=frame,
        view_controls=np.load(root / "view_controls.npy"),
    )


def build_donor_template(
    cfg: dict[str, Any], inputs: dict[str, Any], cam: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Build the donor template arrays (see :data:`colon_anatomy.DONOR_KEYS`).

    Args:
        cfg: the ``donor_template`` config section.
        inputs: :func:`load_donor_inputs` output.
        cam: fisheye calibration of the synthetic videos.
    """
    import cv2

    from colonoworld.tube.build_from_video import content_mask

    radius = float(cfg["chain_reference_radius"])
    course = schematic_course(cfg["course"])
    chosen = choose_donors(inputs["pool"])
    wall = build_reference_wall(course, chosen, cfg["cap_fits"], radius, float(cfg["scale"]))
    field = (
        cv2.resize(
            content_mask(inputs["reference_frame"], cam).astype(np.uint8),
            (1104, 832),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    # Withdrawal timing: route table (fraction of time -> fraction of length), inset at both ends.
    n = int(cfg["withdrawal_frames"])
    length = float(wall["s"][-1])
    start = length - float(cfg["start_inset_R"]) * radius
    end = float(cfg["end_inset_R"]) * radius
    route = cfg["route"]
    camera_s = _interp(
        np.asarray(route["fraction"]),
        np.asarray(route["s_fraction"]) * length,
        np.linspace(0, 1, n),
    )
    camera_s = end + (camera_s - camera_s[-1]) * (start - end) / (camera_s[0] - camera_s[-1])
    blocks = residual_blocks(inputs["blocks"], cfg["residual_block_selection"])
    poses, heading = withdrawal_camera(wall, blocks, camera_s, radius, float(cfg["lookahead_R"]))
    poses = refine_view(
        wall,
        field,
        cam,
        radius,
        poses,
        heading,
        inputs["view_controls"],
        cfg["training_rotation_deg_per_frame"],
    )
    radial = add_cecal_landmarks(wall, radius, cfg["appendiceal_orifice"], cfg["ileocecal_valve"])
    window_poses, window_radius = inputs["dwell"]
    dwell, dwell_s = cecal_dwell(
        wall,
        poses[0].copy(),
        camera_s[0],
        window_poses,
        window_radius,
        radius,
        float(cfg["dwell_start_inset_R"]),
    )
    return dict(
        R_pose=np.float64(radius),
        field_mask=field,
        radial=radial,
        s_ring=wall["s_ring"],
        anatomy_bounds=wall["anatomy_bounds"],
        anatomy=np.array(SEGMENTS),
        C=wall["C"],
        s=wall["s"],
        B=wall["B"],
        poses=np.concatenate([dwell[:-1], poses]),
        camera_s=np.r_[dwell_s[:-1], camera_s],
    )


def save_donor_template(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write the donor template npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
