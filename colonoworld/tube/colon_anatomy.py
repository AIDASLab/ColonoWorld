# SPDX-License-Identifier: CC-BY-NC-4.0
"""Synthetic whole-colon wall geometry for full-procedure generation (cecum to rectum withdrawal).

A colon shape is built in two stages from a per-shape configuration
(``configs/full_colon_shapes.yaml``) and a donor template (a fitted colonoscopy tube that
supplies the camera field mask, a reference fold texture and a reference withdrawal camera path):

1. **Course stage.** A centreline is spline-interpolated through anatomical control points
   (rectum, sigmoid, descending, splenic flexure, transverse, hepatic flexure, ascending, cecum),
   scaled to a target length and relieved at kinks. Ring radii follow literature diameters per
   segment and carry taeniae, haustra, semilunar folds, Houston valves, a cecal pouch with a dome,
   an ileocecal valve and an appendiceal orifice. Withdrawal deformations (dents, folds touching
   across the lumen, partial collapses, sigmoid pleats) modulate the rings. A camera path replays
   the donor withdrawal in the local ring frame with per-segment speeds matched to training videos.
2. **Wall stage.** Seeded random fields make every ring free-form: uneven fold depth, segmental
   and local distension, oval and triangular cross-sections, dents and bulges, optional contraction
   rings and flattened stretches. The wall is kept non-self-intersecting (centreline separation,
   contact dents, bend caps, a slope/curvature-limited minorant), the cecal dome and cecal features
   are rebuilt on the final wall, and the camera is moved laterally toward a smooth max-clearance
   path inside the lumen.

All lengths are in pose units; ``G`` pose units correspond to 1 cm.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTS = [
    "rectum",
    "sigmoid",
    "descending",
    "splenic flexure",
    "transverse",
    "hepatic flexure",
    "ascending",
    "cecum",
]
NPHI = 96  # samples per cross-section ring
CLIP_STRIDE = 188  # frames per generation chunk (chunks overlap by one frame)

_LEFT_COLON_TAENIAE = {
    "libera": [0.0, 0.0, 1.0],
    "mesocolica": [-1.0, 0.0, -1.0],
    "omentalis": [1.0, 0.0, -1.0],
}
_RIGHT_COLON_TAENIAE = {
    "libera": [0.0, 0.0, 1.0],
    "mesocolica": [1.0, 0.0, -1.0],
    "omentalis": [-1.0, 0.0, -1.0],
}

#: Anatomy parameters shared by every shape. Body frame: +x patient left, +y superior, +z anterior.
ANATOMY: dict[str, Any] = dict(
    radius_knots={
        "rectum": [(0, 0.48), (0.4, 0.64), (0.85, 0.50)],
        "sigmoid": [(0.5, 0.43)],
        "descending": [(0.5, 0.50)],
        "splenic flexure": [(0.5, 0.53)],
        "transverse": [(0.5, 0.57)],
        "hepatic flexure": [(0.5, 0.58)],
        "ascending": [(0.5, 0.62)],
        "cecum": [(0, 0.72), (0.5, 0.75), (1, 0.75)],
    },
    radius_lowpass_diameters=1.5,
    cecum=dict(
        length_cm=6.0,
        pouch_mean_units=0.745,
        pouch_ramp_radii=2.0,
        pouch_ramp_end_fraction=0.2,
        lobe_fraction=0.20,
        lobe_world=[0.7, -1.0, 0.0],
        pole_height_over_r=0.90,
        pole_drift_fraction=0.20,
        pole_drift_world=[1.0, -1.0, 0.0],
        dome_rings=24,
    ),
    taenia=dict(
        names=["libera", "mesocolica", "omentalis"],
        offsets_deg=[0.0, -120.0, 120.0],
        width_cm=0.80,
        rectal_spread_width_cm=2.40,
        rectal_fade_cm=1.0,
        angle_smooth_cm=2.0,
        depth_blend_cm=0.75,
        convergence_start_cm_above_pole=5.0,
        convergence_end_cm_above_pole=3.15,
        depth_fraction={
            "rectum": 0.0,
            "sigmoid": 0.038,
            "descending": 0.034,
            "splenic flexure": 0.036,
            "transverse": 0.045,
            "hepatic flexure": 0.040,
            "ascending": 0.042,
            "cecum": 0.040,
        },
        world={
            "rectum": _LEFT_COLON_TAENIAE,
            "sigmoid": _LEFT_COLON_TAENIAE,
            "descending": _LEFT_COLON_TAENIAE,
            "splenic flexure": _LEFT_COLON_TAENIAE,
            "transverse": {
                "libera": [0.0, -1.0, 0.0],
                "mesocolica": [0.0, 0.0, -1.0],
                "omentalis": [0.0, 1.0, 1.0],
            },
            "hepatic flexure": _RIGHT_COLON_TAENIAE,
            "ascending": _RIGHT_COLON_TAENIAE,
            "cecum": _RIGHT_COLON_TAENIAE,
        },
    ),
    # spacing_cm is fold-to-fold; semilunar_depth_fraction is inward,
    # haustral_bulge_fraction outward.
    haustra={
        "rectum": dict(
            spacing_cm=None,
            haustral_bulge_fraction=0.0,
            semilunar_depth_fraction=0.0,
            fold_halfwidth_cycle=0.18,
        ),
        "sigmoid": dict(
            spacing_cm=2.0,
            haustral_bulge_fraction=0.060,
            semilunar_depth_fraction=0.200,
            fold_halfwidth_cycle=0.22,
        ),
        "descending": dict(
            spacing_cm=3.0,
            haustral_bulge_fraction=0.040,
            semilunar_depth_fraction=0.065,
            fold_halfwidth_cycle=0.17,
        ),
        "splenic flexure": dict(
            spacing_cm=2.7,
            haustral_bulge_fraction=0.070,
            semilunar_depth_fraction=0.085,
            fold_halfwidth_cycle=0.18,
        ),
        "transverse": dict(
            spacing_cm=2.5,
            haustral_bulge_fraction=0.105,
            semilunar_depth_fraction=0.185,
            fold_halfwidth_cycle=0.19,
        ),
        "hepatic flexure": dict(
            spacing_cm=2.5,
            haustral_bulge_fraction=0.090,
            semilunar_depth_fraction=0.100,
            fold_halfwidth_cycle=0.19,
        ),
        "ascending": dict(
            spacing_cm=2.4,
            haustral_bulge_fraction=0.100,
            semilunar_depth_fraction=0.160,
            fold_halfwidth_cycle=0.19,
        ),
        "cecum": dict(
            spacing_cm=2.6,
            haustral_bulge_fraction=0.110,
            semilunar_depth_fraction=0.095,
            fold_halfwidth_cycle=0.20,
        ),
    },
    haustra_segment_blend_cm=0.45,
    haustra_rectosigmoid_rise_cm=0.90,
    haustra_sector_offsets_cycle=[0.0, 1 / 3, 2 / 3],
    haustra_sector_centers_relative_deg=[-60.0, 60.0, 180.0],
    houston=[
        dict(
            position_cm=6.0,
            world=[1.0, 0.0, 0.0],
            depth_cm=0.36,
            extent_fraction=0.58,
            axial_width_cm=1.15,
        ),
        dict(
            position_cm=9.0,
            world=[-1.0, 0.0, 0.0],
            depth_cm=0.52,
            extent_fraction=0.66,
            axial_width_cm=1.35,
        ),
        dict(
            position_cm=12.0,
            world=[1.0, 0.0, 0.0],
            depth_cm=0.42,
            extent_fraction=0.60,
            axial_width_cm=1.20,
        ),
    ],
    # Ileocecal valve: a labial slit on the posteromedial cecal wall.
    icv=dict(
        cm_above_pole=5.50,
        wall_world=[1.0, 0.0, -0.75],
        slit_length_cm=1.50,
        slit_gap_cm=0.16,
        lip_thickness_cm=0.25,
        mound_depth_cm=0.42,
        blind_pit_depth_cm=0.80,
        hole_half_rings=5,
        hole_half_angles=4,
        mound_rings=8,
        lip_rings=4,
        channel_rings=5,
        pit_rings=8,
        channel_exit_depth_cm=0.08,
        mound_overcurve_fraction=0.12,
        pit_rounding_start_fraction=0.55,
        pit_min_radius_fraction=0.05,
    ),
    # Appendiceal orifice: a crescent slit at the taenial convergence with crow's-foot folds.
    ao=dict(
        cm_above_pole=3.00,
        wall_world=[1.0, 0.0, -1.0],
        slit_length_cm=0.50,
        slit_gap_cm=0.08,
        lip_thickness_cm=0.07,
        mound_depth_cm=0.06,
        crescent_bow_cm=0.055,
        blind_pit_depth_cm=0.60,
        hole_half_rings=2,
        hole_half_angles=2,
        mound_rings=5,
        lip_rings=3,
        channel_rings=4,
        pit_rings=7,
        channel_exit_depth_cm=0.08,
        mound_overcurve_fraction=0.12,
        pit_rounding_start_fraction=0.55,
        pit_min_radius_fraction=0.05,
        crows_foot_depth_cm=0.13,
        crows_foot_width_cm=0.16,
        crows_foot_radius_cm=1.45,
        crows_foot_slopes=[-0.65, 0.0, 0.65],
        cecal_fold_to_icv_depth_cm=0.10,
        cecal_fold_width_cm=0.18,
    ),
    clearance=dict(
        vertex_guard_over_R=0.40,
        start_guard_over_local_r=0.84,
        attenuation_sigma_rings=2.0,
        attenuation_sigma_angles=1.0,
        max_iterations=8,
    ),
)

#: Wall-stage constants (minimum ring radius ratio, curvature and slope limits).
WALL_LIMITS = dict(
    minimum_radius_ratio=0.25,
    construction_headroom=0.020,
    self_contact_margin_R=0.02,
    radius_cap_rate_per_axial_R=1.5,
    bend_envelope_curvature_limit=0.75,
    final_radius_rate_limit_per_axial_R=2.9,
    final_radius_slope_change_limit_per_axial_R2=16.0,
)

DONOR_KEYS = (
    "R_pose",
    "field_mask",
    "radial",
    "s_ring",
    "anatomy_bounds",
    "anatomy",
    "C",
    "s",
    "B",
    "poses",
    "camera_s",
)


def load_donor(path: str | Path) -> dict[str, np.ndarray]:
    """Load the donor template (fitted reference tube and its withdrawal camera path)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in DONOR_KEYS}


# --------------------------------------------------------------------------- small helpers
def _interp_rows(q: np.ndarray, grid: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(q, grid, a[:, k]) for k in range(a.shape[1])], axis=1)


def _interp_frames(q: np.ndarray, grid: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Linearly blend rotation frames along ``grid`` and project back onto SO(3)."""
    ii = np.clip(np.searchsorted(grid, q), 1, len(grid) - 1)
    w = (q - grid[ii - 1]) / (grid[ii] - grid[ii - 1])
    out = []
    for j, x in zip(ii, w, strict=False):
        m = (1.0 - x) * frames[j - 1] + x * frames[j]
        u, _, vt = np.linalg.svd(m)
        out.append(u @ vt)
    return np.asarray(out)


def _angle_gaussian(angle: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    d = np.angle(np.exp(1j * (angle - centre)))
    return np.exp(-0.5 * (d / sigma) ** 2)


def _cos2_window(x: np.ndarray) -> np.ndarray:
    """Compact C1 window: cos^2(pi x / 2) for |x| < 1, else 0 (``x`` already non-negative)."""
    return np.where(x < 1, np.cos(np.pi * x / 2) ** 2, 0)


def _ring_directions(B: np.ndarray, cph: np.ndarray, sph: np.ndarray) -> np.ndarray:
    """Unit radial directions (rings, angles, 3) of every ring frame."""
    return cph[None, :, None] * B[:, None, :, 0] + sph[None, :, None] * B[:, None, :, 1]


def _segment_labels(bounds: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(np.searchsorted(bounds, s, side="right") - 1, n - 1)


def _world_angle(B: np.ndarray, v: Any) -> np.ndarray:
    """Ring angle of a body-frame direction projected onto every ring plane."""
    v = np.asarray(v, float)
    v /= np.linalg.norm(v)
    q = v[None] - (B[:, :, 2] @ v)[:, None] * B[:, :, 2]
    qn = np.linalg.norm(q, axis=1)
    a_ = np.arctan2(np.einsum("ij,ij->i", q, B[:, :, 1]), np.einsum("ij,ij->i", q, B[:, :, 0]))
    good = qn > 1e-4
    if not good.all():
        ig = np.flatnonzero(good)
        a_[~good] = np.interp(np.flatnonzero(~good), ig, np.unwrap(a_[good]))
    return a_


# --------------------------------------------------------------------------- course stage
def _shape_parameters(shape: dict[str, Any], G: float) -> dict[str, Any]:
    """Anatomy parameters with the shape's diameters and fold-depth adjustments applied."""
    params = copy.deepcopy(ANATOMY)
    for name, dcm in shape["diameter_cm"].items():
        if name == "cecum":
            params["radius_knots"][name] = [
                (f, dcm / 2 * G * r / 0.75) for f, r in params["radius_knots"][name]
            ]
            params["cecum"]["pouch_mean_units"] = dcm / 2 * G
        else:
            params["radius_knots"][name] = [(0.5, dcm / 2 * G)]
    if "splenic_semilunar_depth_fraction" in shape:
        params["haustra"]["splenic flexure"]["semilunar_depth_fraction"] = float(
            shape["splenic_semilunar_depth_fraction"]
        )
    for seg, gain in shape.get("haustra_depth_gain", {}).items():
        params["haustra"][seg]["semilunar_depth_fraction"] *= float(gain)
    return params


def _course_control_points(shape: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Control points (cm) from the anus to the cecum and the first index of each segment."""
    course = {n: np.asarray(shape["course"][n], float).copy() for n in SEGMENTS}
    if "rectum_bow_gain" in shape:
        # Deepen the sagittal (sacral) bow of the rectum about its anus-to-rectosigmoid chord.
        r_ = course["rectum"]
        a_, b_ = r_[0].copy(), r_[-1].copy()
        t_ = np.linspace(0.0, 1.0, len(r_))[:, None]
        dev = r_ - (a_ + t_ * (b_ - a_))
        dev[:, 0] = 0.0
        dev[:, 1] = 0.0
        course["rectum"] = r_ + (float(shape["rectum_bow_gain"]) - 1.0) * dev
    pts = []
    starts = []
    for n in SEGMENTS:
        starts.append(len(pts))
        pts.extend(course[n])
    return np.array(pts), starts


def _knot_positions(params: dict[str, Any], bounds: np.ndarray) -> list[float]:
    ks = []
    for i, n in enumerate(SEGMENTS):
        for f, _ in params["radius_knots"][n]:
            ks.append(bounds[i] + f * (bounds[i + 1] - bounds[i]))
    return ks


def _centreline(
    shape: dict[str, Any], params: dict[str, Any], R: float, G: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Arc-length sampled centreline ``(s, C, bounds, L, ds)`` at the target length."""
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d

    pts, starts = _course_control_points(shape)
    uu = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    sp = CubicSpline(uu, pts, bc_type="natural")
    fine = np.linspace(0, uu[-1], 60000)
    X = sp(fine)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    f_ = float(shape["target_length_cm"]) / arc[-1]
    X = X * f_
    arc = arc * f_
    uu = uu * f_
    fine = fine * f_
    bounds_cm = np.r_[[np.interp(uu[i], fine, arc) for i in starts], arc[-1]]
    L = arc[-1] * G
    bounds = bounds_cm * G
    ds = 0.04 * R
    s = np.r_[np.arange(0, L, ds), L]
    C = np.stack([np.interp(s / G, arc, X[:, k]) for k in range(3)], 1) * G
    C -= C[0]
    Cs = gaussian_filter1d(C, float(shape.get("smooth_units", 0.8)) / ds, axis=0, mode="nearest")
    a3 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))]
    bounds = np.interp(bounds, s, a3)
    CP = params["cecum"]
    ic = SEGMENTS.index("cecum")
    hd0 = CP["pole_height_over_r"] * CP["pouch_mean_units"]
    L = bounds[ic] + CP["length_cm"] * G - hd0
    bounds[-1] = L
    s = np.r_[np.arange(0, L, ds), L]
    C = np.stack([np.interp(s, a3, Cs[:, k]) for k in range(3)], 1)
    C -= C[0]
    s, C, bounds, L = _relieve_kinks(params, s, C, bounds, L, ds)
    s, C, bounds, L = _rescale_to_target_length(shape, s, C, bounds, L, ds, G)
    return s, C, bounds, L, ds


def _relieve_kinks(
    params: dict[str, Any], s: np.ndarray, C: np.ndarray, bounds: np.ndarray, L: float, ds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Where curvature x (1.3 x radius trend) exceeds 0.9, smooth the centreline locally and
    re-parameterise it by arc length (segment bounds follow)."""
    from scipy.ndimage import gaussian_filter1d

    ks_ = _knot_positions(params, bounds)
    kr_ = [r for n in SEGMENTS for _, r in params["radius_knots"][n]]
    for _ in range(300):
        rb = 1.3 * np.exp(np.interp(s, ks_, np.log(kr_)))
        Tk = np.gradient(
            gaussian_filter1d(C, 0.2 * np.median(rb) / ds, axis=0, mode="nearest"), s, axis=0
        )
        Tk /= np.linalg.norm(Tk, axis=1, keepdims=True)
        bad = np.linalg.norm(np.gradient(Tk, s, axis=0), axis=1) * rb > 0.9
        if not bad.any():
            break
        w = gaussian_filter1d(bad.astype(float), np.median(rb) / ds, mode="nearest")
        w = np.clip(w / w.max(), 0, 1)
        C = C + w[:, None] * (
            gaussian_filter1d(C, 0.5 * np.median(rb) / ds, axis=0, mode="nearest") - C
        )
        a_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        bounds = np.interp(bounds, s, a_)
        L = a_[-1]
        sn = np.r_[np.arange(0, L, ds), L]
        C = np.stack([np.interp(sn, a_, C[:, k]) for k in range(3)], 1)
        C -= C[0]
        s = sn
        ks_ = _knot_positions(params, bounds)
    return s, C, bounds, L


def _rescale_to_target_length(
    shape: dict[str, Any],
    s: np.ndarray,
    C: np.ndarray,
    bounds: np.ndarray,
    L: float,
    ds: float,
    G: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Scale the rectum..cecum-start part to the target length and translate the cecal tail."""
    ic = SEGMENTS.index("cecum")
    bi = bounds[ic]
    tail = L - bi
    f_ = (float(shape["target_length_cm"]) * G - tail) / bi
    c0 = np.array([np.interp(bi, s, C[:, k]) for k in range(3)])
    m_ = s <= bi
    C = np.where(m_[:, None], C * f_, C - c0 + c0 * f_)
    s = np.where(m_, s * f_, s + bounds[ic] * (f_ - 1))
    bi = bounds[ic]
    bounds = np.where(bounds <= bi, bounds * f_, bounds + bi * (f_ - 1))
    L = L + bi * (f_ - 1)
    sn = np.r_[np.arange(0, L, ds), L]
    C = np.stack([np.interp(sn, s, C[:, k]) for k in range(3)], 1)
    return sn, C, bounds, L


def _ring_frames(C: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rotation-minimising ring frames ``B[i] = [n1, n2, tangent]``."""
    T = np.gradient(C, s, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    n1 = np.cross(T[0], [0, 0, 1.0])
    n1 /= np.linalg.norm(n1)
    B = np.zeros((len(s), 3, 3))
    for i in range(len(s)):
        if i:
            n1 -= (n1 @ T[i]) * T[i]
            n1 /= np.linalg.norm(n1)
        B[i] = np.c_[n1, np.cross(T[i], n1), T[i]]
    return B


def _radius_trend(params: dict[str, Any], s: np.ndarray, bounds: np.ndarray, ds: float):
    """Log-space low-pass of the per-segment radius knots."""
    from scipy.ndimage import gaussian_filter1d

    ks = []
    kr = []
    for i, n in enumerate(SEGMENTS):
        for f, r in params["radius_knots"][n]:
            ks.append(bounds[i] + f * (bounds[i + 1] - bounds[i]))
            kr.append(r)
    rbase = np.exp(np.interp(s, ks, np.log(kr)))
    sig = params["radius_lowpass_diameters"] * 2 * np.median(rbase) / ds
    return np.exp(
        gaussian_filter1d(np.log(np.exp(np.interp(s, ks, np.log(kr)))), sig, mode="nearest")
    )


def _donor_fold_texture(
    donor: dict[str, np.ndarray],
    s: np.ndarray,
    bounds: np.ndarray,
    lab: np.ndarray,
    r0: np.ndarray,
    ds: float,
) -> np.ndarray:
    """High-pass fold relief of the donor tube, mirrored per segment onto the new rings."""
    from scipy.ndimage import gaussian_filter1d

    rad2 = donor["radial"]
    s2 = donor["s_ring"]
    b2 = donor["anatomy_bounds"]
    a2 = list(donor["anatomy"])
    ds2 = np.median(np.diff(s2))
    m2 = (
        rad2 / rad2.mean(1, keepdims=True)
        - 1
        + (
            rad2.mean(1, keepdims=True)
            / np.exp(
                gaussian_filter1d(np.log(rad2.mean(1)), 2 * np.median(rad2) / ds2, mode="nearest")
            )[:, None]
            - 1
        )
    )
    hp = m2 - gaussian_filter1d(m2, 2 * np.median(rad2) / ds2, axis=0, mode="nearest")
    hp = np.clip(hp, -0.08, 0.08)
    donor_fold = np.zeros((len(s), NPHI))
    for i, n in enumerate(SEGMENTS):
        j = a2.index("ascending" if n == "cecum" else n)
        m = lab == i
        src = (s2 >= b2[j]) & (s2 < b2[j + 1])
        S2 = s2[src] - b2[j]
        span = S2[-1]
        x = np.mod(s[m] - bounds[i], 2 * span)
        x = np.where(x > span, 2 * span - x, x)
        idx = np.clip(np.searchsorted(S2, x), 0, len(S2) - 1)
        donor_fold[m] = hp[src][idx]
    donor_fold = gaussian_filter1d(donor_fold, 1, axis=0)
    donor_fold -= gaussian_filter1d(donor_fold.mean(1), 4 * np.median(r0) / ds, mode="nearest")[
        :, None
    ]
    return donor_fold


def _cecal_pouch(
    CP: dict[str, Any],
    B: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    L: float,
    r0: np.ndarray,
    donor_fold: np.ndarray,
    ph: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular pouch lobe and the ring-mean radius target (widening into the cecum)."""
    ic = SEGMENTS.index("cecum")
    rr = np.interp(bounds[ic], s, r0)
    s0 = bounds[ic] - CP["pouch_ramp_radii"] * rr
    s1 = bounds[ic] + CP["pouch_ramp_end_fraction"] * (L - bounds[ic])
    w = np.clip((s - s0) / (s1 - s0), 0, 1)
    w = w * w * (3 - 2 * w)
    pouch_ang = _world_angle(B, CP["lobe_world"])
    pouch_shape = 1 + w[:, None] * CP["lobe_fraction"] * (
        np.maximum(0, np.cos(ph[None] - pouch_ang[:, None])) ** 2
    )
    ring_target = (r0 + w * (CP["pouch_mean_units"] - r0)) * np.where(
        w > 0, 1, (1 + donor_fold).mean(1)
    )
    ring_target = np.minimum(ring_target, CP["pouch_mean_units"])
    return pouch_shape, ring_target


def _replay_camera_path(
    shape: dict[str, Any],
    donor: dict[str, np.ndarray],
    s: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    bounds: np.ndarray,
    L: float,
    r0: np.ndarray,
    hd: float,
    R: float,
    G: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Donor withdrawal replayed in the local ring frame with per-segment matched speed.

    Returns ``(poses, camera_s, labels, local_radius_at_camera)``; the first 189 frames are a
    cecal dwell and the frame count is ``CLIP_STRIDE * k + 1``.
    """
    from scipy.ndimage import gaussian_filter1d

    rad2 = donor["radial"]
    s2 = donor["s_ring"]
    ds2 = np.median(np.diff(s2))
    C2 = donor["C"]
    sB2 = donor["s"]
    B2 = donor["B"]
    P2 = donor["poses"]
    cs2 = donor["camera_s"]
    r2 = np.exp(
        gaussian_filter1d(np.log(rad2.mean(1)), 1.5 * 2 * np.median(rad2) / ds2, mode="nearest")
    )
    Bc2 = _interp_frames(cs2, sB2, B2)
    Cc2 = _interp_rows(cs2, sB2, C2)
    rc2 = np.interp(cs2, s2, r2)
    Q2 = np.einsum("nji,njk->nik", Bc2, P2[:, :3, :3])
    o2 = np.einsum("nji,nj->ni", Bc2, P2[:, :3, 3] - Cc2) / rc2[:, None]
    adv2 = -np.diff(cs2) / rc2[1:]
    fl = (4.0 * G - hd) / (sB2[-1] - cs2[0])
    dw = np.array([L - (sB2[-1] - cs2[t]) * fl for t in range(189)])  # cecal dwell (189 frames)
    nw = len(cs2) - 189
    speed = shape["speed_match"]  # per segment: training median of local radius / step
    order = SEGMENTS

    def walk(fs: float) -> tuple[list[float], list[int]]:
        sc_ = list(dw)
        src_ = list(range(189))
        k_ = 0
        jp_ = 188
        am = float(np.median(adv2[189:]))
        while True:
            j_ = k_ % (2 * nw - 2)
            j_ = 189 + (j_ if j_ < nw else 2 * nw - 2 - j_)
            st_ = adv2[j_ - 1] if j_ > jp_ else adv2[j_]
            seg = order[
                min(int(np.searchsorted(bounds, sc_[-1], side="right")) - 1, len(order) - 1)
            ]
            st_ = st_ / am / float(speed[seg]) * fs
            nxt_ = sc_[-1] - st_ * np.interp(sc_[-1], s, r0)
            jp_ = j_
            if nxt_ <= 0.75 * R:
                break
            sc_.append(nxt_)
            src_.append(j_)
            k_ += 1
        return sc_, src_

    # Exactly CLIP_STRIDE * k + 1 frames via a global step factor (bisection).
    sc, src = walk(1.0)
    target = int(np.ceil((len(sc) - 1) / CLIP_STRIDE)) * CLIP_STRIDE + 1
    lo_, hi_ = 0.9, 1.0
    for _ in range(60):
        mid_ = 0.5 * (lo_ + hi_)
        n_ = len(walk(mid_)[0])
        if n_ == target:
            break
        lo_, hi_ = (mid_, hi_) if n_ > target else (lo_, mid_)
    sc, src = walk(mid_)
    if len(sc) != target:
        sc = sc[:target] if len(sc) > target else sc + [sc[-1]] * (target - len(sc))
        src = (src + [src[-1]] * target)[:target]
    sc = np.array(sc)
    src = np.array(src)
    clips = int(np.ceil((len(sc) - 1) / CLIP_STRIDE))
    N2 = clips * CLIP_STRIDE + 1
    d0 = 188
    withdraw = sc[d0:]
    tq = np.linspace(0, 1, N2 - d0)
    w2 = np.interp(tq, np.linspace(0, 1, len(withdraw)), withdraw)
    sc = np.r_[sc[:d0], w2]
    srcq = np.r_[
        src[:d0], src[d0:][np.round(np.linspace(0, len(withdraw) - 1, len(w2))).astype(int)]
    ]
    Bc = _interp_frames(sc, s, B)
    Cc = _interp_rows(sc, s, C)
    rc = np.interp(sc, s, r0)
    poses = np.tile(np.eye(4), (len(sc), 1, 1))
    poses[:, :3, :3] = np.einsum("nij,njk->nik", Bc, Q2[srcq])
    poses[:, :3, 3] = Cc + np.einsum("nij,nj->ni", Bc, o2[srcq]) * rc[:, None]
    poses[0, :3, :3] = poses[1, :3, :3]
    cam_lab = np.array(order)[_segment_labels(bounds, sc, len(order))]
    return poses, sc, cam_lab, rc


def _taenia_angles(
    params: dict[str, Any],
    B: np.ndarray,
    lab: np.ndarray,
    spole_s_cm: np.ndarray,
    ao_ang: np.ndarray,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous taenia angles from projected per-segment body directions, converging on the
    appendiceal orifice near the cecal pole. Returns ``(taenia_phase, taenia_angles)``."""
    from scipy.ndimage import gaussian_filter1d

    TP = params["taenia"]
    offsets = np.deg2rad(TP["offsets_deg"])
    raw_angles = np.zeros((len(B), 3))
    for i, n in enumerate(SEGMENTS):
        m = lab == i
        for q, bn in enumerate(TP["names"]):
            raw_angles[m, q] = _world_angle(B, TP["world"][n][bn])[m]
    phase_raw = np.angle(np.exp(1j * (raw_angles - offsets[None])).sum(1))
    taenia_phase = gaussian_filter1d(
        np.unwrap(phase_raw), TP["angle_smooth_cm"] * G / ds, mode="nearest"
    )
    conv = np.clip(
        (TP["convergence_start_cm_above_pole"] - spole_s_cm)
        / (TP["convergence_start_cm_above_pole"] - TP["convergence_end_cm_above_pole"]),
        0,
        1,
    )
    conv = conv * conv * (3 - 2 * conv)
    base_t = taenia_phase[:, None] + offsets[None]
    taenia_angles = base_t + conv[:, None] * np.angle(np.exp(1j * (ao_ang[:, None] - base_t)))
    return taenia_phase, taenia_angles


def _smooth_segment_values(
    params: dict[str, Any], lab: np.ndarray, key: str, G: float, ds: float, default: float = 0.0
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    x = np.array(
        [
            (
                params["haustra"][SEGMENTS[i]][key]
                if params["haustra"][SEGMENTS[i]][key] is not None
                else default
            )
            for i in lab
        ],
        float,
    )
    return gaussian_filter1d(x, params["haustra_segment_blend_cm"] * G / ds, mode="nearest")


def _taenia_relief(
    params: dict[str, Any],
    s: np.ndarray,
    lab: np.ndarray,
    rect_end: float,
    r0: np.ndarray,
    taenia_angles: np.ndarray,
    ph: np.ndarray,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inward taenia bands (spread and faded at the anal end). Returns ``(relief, width_cm)``."""
    from scipy.ndimage import gaussian_filter1d

    TP = params["taenia"]
    fade = np.clip((s - (rect_end - TP["rectal_fade_cm"] * G)) / (TP["rectal_fade_cm"] * G), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    fade[lab > SEGMENTS.index("rectum")] = 1
    width_cm = TP["width_cm"] + (1 - fade) * (TP["rectal_spread_width_cm"] - TP["width_cm"])
    tdepth = (
        gaussian_filter1d(
            np.array([TP["depth_fraction"][SEGMENTS[i]] for i in lab]),
            TP["depth_blend_cm"] * G / ds,
            mode="nearest",
        )
        * fade
    )
    relief = np.zeros((len(s), NPHI))
    for q in range(3):
        arc_cm = np.angle(np.exp(1j * (ph[None] - taenia_angles[:, q, None]))) * (r0 / G)[:, None]
        x = np.abs(arc_cm) / (width_cm[:, None] / 2)
        relief += 0.75 * tdepth[:, None] * _cos2_window(x)
    return relief, width_cm


def _haustra_relief(
    params: dict[str, Any],
    s: np.ndarray,
    lab: np.ndarray,
    rect_end: float,
    r0: np.ndarray,
    taenia_phase: np.ndarray,
    width_cm: np.ndarray,
    ph: np.ndarray,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Haustral bulges and semilunar folds in three sector windows between the taeniae (sector
    cycles offset by one third). Returns ``(outward bulge, inward fold relief)``."""
    spacing = _smooth_segment_values(params, lab, "spacing_cm", G, ds, 2.5)
    spacing = np.maximum(spacing, 0.25)
    cycle = np.r_[0, np.cumsum(np.diff(s) / G / ((spacing[:-1] + spacing[1:]) / 2))]
    bulge_amp = _smooth_segment_values(params, lab, "haustral_bulge_fraction", G, ds)
    fold_amp = _smooth_segment_values(params, lab, "semilunar_depth_fraction", G, ds)
    fold_hw = _smooth_segment_values(params, lab, "fold_halfwidth_cycle", G, ds, 0.18)
    colon_on = np.clip((s - rect_end) / (params["haustra_rectosigmoid_rise_cm"] * G), 0, 1)
    colon_on = colon_on * colon_on * (3 - 2 * colon_on)
    bulge_amp *= colon_on
    fold_amp *= colon_on
    bulge = np.zeros((len(s), NPHI))
    semilunar = np.zeros_like(bulge)
    for q, off in enumerate(params["haustra_sector_offsets_cycle"]):
        sector_center = taenia_phase + np.deg2rad(params["haustra_sector_centers_relative_deg"][q])
        da = np.angle(np.exp(1j * (ph[None] - sector_center[:, None])))
        sector_halfwidth = np.maximum(np.pi / 6, np.pi / 3 - 0.5 * width_cm / (r0 / G))
        awin = _cos2_window(np.abs(da) / sector_halfwidth[:, None])
        cyc = np.mod(cycle + off, 1.0)
        pouch = np.sin(np.pi * cyc) ** 2
        flong = _cos2_window(np.minimum(cyc, 1 - cyc) / fold_hw)
        bulge += bulge_amp[:, None] * pouch[:, None] * awin
        semilunar -= fold_amp[:, None] * flong[:, None] * awin
    return bulge, semilunar


def _houston_relief(
    params: dict[str, Any], B: np.ndarray, s: np.ndarray, r0: np.ndarray, ph: np.ndarray, G: float
) -> np.ndarray:
    """Rectal valves of Houston: compact axial and angular shelves."""
    relief = np.zeros((len(s), NPHI))
    for v in params["houston"]:
        av = _world_angle(B, v["world"])
        wa = _cos2_window(np.abs((s / G - v["position_cm"]) / (v["axial_width_cm"] / 2)))
        da = np.abs(np.angle(np.exp(1j * (ph[None] - av[:, None]))))
        half = np.pi * v["extent_fraction"]
        wc = np.where(da < half, np.cos(np.pi * da / (2 * half)) ** 2, 0)
        relief -= (v["depth_cm"] / (r0 / G))[:, None] * wa[:, None] * wc
    return relief


def _crow_field(
    params: dict[str, Any], ph: np.ndarray, qcm: np.ndarray, radius_cm: np.ndarray, aang: np.ndarray
) -> np.ndarray:
    """Crow's-foot folds around the appendiceal orifice plus a small fold toward the valve.

    ``qcm``: distance above the cecal pole (cm); ``radius_cm``: ring radius (cm); ``aang``: orifice
    ring angle.
    """
    AP = params["ao"]
    IP = params["icv"]
    u_ = np.asarray(qcm)[:, None] - AP["cm_above_pole"]
    v_ = (
        np.angle(np.exp(1j * (ph[None] - np.asarray(aang)[:, None])))
        * np.asarray(radius_cm)[:, None]
    )
    rho = np.hypot(u_, v_)
    support = np.where(
        rho < AP["crows_foot_radius_cm"],
        np.cos(np.pi * rho / (2 * AP["crows_foot_radius_cm"])) ** 2,
        0,
    )
    out = np.zeros_like(v_)
    for sl in AP["crows_foot_slopes"]:
        dl = np.abs(v_ - sl * u_) / np.sqrt(1 + sl * sl)
        out -= (
            (AP["crows_foot_depth_cm"] / np.asarray(radius_cm))[:, None]
            * _cos2_window(dl / AP["crows_foot_width_cm"])
            * support
        )
    along = (u_ >= 0) & (u_ <= IP["cm_above_pole"] - AP["cm_above_pole"])
    ends = np.sin(np.pi * np.clip(u_ / (IP["cm_above_pole"] - AP["cm_above_pole"]), 0, 1)) ** 2
    out -= (
        (AP["cecal_fold_to_icv_depth_cm"] / np.asarray(radius_cm))[:, None]
        * _cos2_window(np.abs(v_) / AP["cecal_fold_width_cm"])
        * along
        * ends
    )
    return out


def _attenuated_radial(
    params: dict[str, Any],
    C: np.ndarray,
    dirs: np.ndarray,
    r0: np.ndarray,
    pouch_shape: np.ndarray,
    ring_target: np.ndarray,
    inward: np.ndarray,
    outward: np.ndarray,
    poses: np.ndarray,
    rc0: float,
    R: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ring radii with anatomical relief; inward features that would protrude close to the camera
    path are faded locally. Returns ``(radial, fold)``."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    def radial_with_attenuation(a_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anatomical = outward + inward * a_
        fold_ = (1 + anatomical) / (1 + anatomical).mean(1, keepdims=True) - 1
        raw = r0[:, None] * (1 + fold_) * pouch_shape
        return raw * (ring_target / raw.mean(1))[:, None], fold_

    CL = params["clearance"]
    atten = np.ones_like(inward)
    camtree = cKDTree(poses[:, :3, 3])
    for _ in range(CL["max_iterations"]):
        radial, fold = radial_with_attenuation(atten)
        Xr = C[:, None, :] + radial[:, :, None] * dirs
        dv, ci = camtree.query(Xr.reshape(-1, 3))
        dv = dv.reshape(radial.shape)
        ci = ci.reshape(radial.shape)
        rel = Xr - poses[ci, :3, 3]
        deriv = np.einsum("ijk,ijk->ij", rel, dirs) / np.maximum(dv, 1e-12)
        unsafe = (
            (dv < CL["vertex_guard_over_R"] * R) & (deriv > 0.15) & (inward < 0) & (atten > 1e-3)
        )
        d0_ = np.linalg.norm(Xr - poses[0, :3, 3], axis=2)
        deriv0 = np.einsum("ijk,ijk->ij", Xr - poses[0, :3, 3], dirs) / np.maximum(d0_, 1e-12)
        unsafe |= (
            (d0_ < CL["start_guard_over_local_r"] * rc0)
            & (deriv0 > 0.15)
            & (inward < 0)
            & (atten > 1e-3)
        )
        if not unsafe.any():
            break
        red = gaussian_filter(
            unsafe.astype(float),
            (CL["attenuation_sigma_rings"], CL["attenuation_sigma_angles"]),
            mode=("nearest", "wrap"),
        )
        red = np.clip(red / max(red.max(), 1e-12), 0, 1)
        atten *= 1 - red
    return radial_with_attenuation(atten)


def _apply_deformations(
    shape: dict[str, Any],
    radial: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    r0: np.ndarray,
    ph: np.ndarray,
    G: float,
) -> list[dict[str, Any]]:
    """Withdrawal deformations, each multiplying the ring radii in place.

    pleats: sigmoid ring-mean modulation, wavelength growing toward the descending end;
    collapse: Gaussian along s (sigma_D local diameters), oval flattening along phi;
    kissing: two opposite folds nearly touching, narrow along s, |cos(phi - phi0)|^4;
    dent: smooth inward bulge from a neighbouring organ, Gaussian along s and angle.
    """
    deformations = []

    def spos(seg: str, f: float) -> float:
        i_ = SEGMENTS.index(seg)
        return bounds[i_] + f * (bounds[i_ + 1] - bounds[i_])

    for e in shape.get("deform", []):
        D = 2 * np.interp(spos(e.get("segment", "sigmoid"), e.get("at", 0.5)), s, r0)
        ph0 = np.deg2rad(e.get("phi_deg", 0.0))
        if e["kind"] == "pleats":
            i_ = SEGMENTS.index("sigmoid")
            m_ = (s >= bounds[i_]) & (s < bounds[i_ + 1])
            u_ = (s[m_] - bounds[i_]) / (bounds[i_ + 1] - bounds[i_])
            lam = (e["lambda_cm"][1] + (e["lambda_cm"][0] - e["lambda_cm"][1]) * u_) * G
            amp = e["depth"][1] + (e["depth"][0] - e["depth"][1]) * u_
            phase = 2 * np.pi * np.cumsum(np.r_[0, np.diff(s[m_]) / lam[1:]])
            mul = 1 - amp * 0.5 * (1 + np.cos(phase))
            radial[m_] *= mul[:, None]
        else:
            gs = np.exp(
                -0.5
                * ((s - spos(e["segment"], e.get("at", 0.5))) / (e.get("sigma_D", 0.5) * D)) ** 2
            )
            if e["kind"] == "collapse":
                ang = 1 - e.get("oval", 0.4) + e.get("oval", 0.4) * np.cos(ph - ph0) ** 2
            elif e["kind"] == "kissing":
                ang = np.abs(np.cos(ph - ph0)) ** 4
            elif e["kind"] == "dent":
                wd = np.deg2rad(e.get("width_deg", 70.0))
                ang = np.exp(-0.5 * (np.angle(np.exp(1j * (ph - ph0))) / (wd / 2)) ** 2)
            else:
                raise ValueError(e["kind"])
            radial *= 1 - e["depth"] * gs[:, None] * ang[None]
        deformations.append(dict(e, local_diameter_units=float(D)))
    return deformations


def _slit_rings(P: dict[str, Any], X0: np.ndarray, frame: tuple, G: float) -> list[np.ndarray]:
    """Loft rings of a slit feature attached to the wall rim ``X0``: a mound into the lumen, a lip,
    a channel back through the wall and a rounded blind pit."""
    Pc, no, ec, ea = frame
    uu_, vv_ = (X0 - Pc) @ ec, (X0 - Pc) @ ea
    psi = np.unwrap(np.arctan2(vv_ / max(np.ptp(vv_), 1e-9), uu_ / max(np.ptp(uu_), 1e-9)))
    A = P["slit_length_cm"] * G / 2
    gap = P["slit_gap_cm"] * G / 2
    lip = P["lip_thickness_cm"] * G
    bow = P.get("crescent_bow_cm", 0.0) * G

    def ell(c: np.ndarray, aa: float, bb: float, curve: float = 0.0) -> np.ndarray:
        co, si = np.cos(psi), np.sin(psi)
        return c + aa * co[:, None] * ec + (bb * si + curve * (1 - co * co))[:, None] * ea

    outerA, outerB = A + lip, gap + lip
    mound = P["mound_depth_cm"] * G
    rings = []
    outer = ell(Pc - no * mound, outerA, outerB, bow)
    for k in range(1, P["mound_rings"] + 1):
        t = k / P["mound_rings"]
        g = t * t * (3 - 2 * t)
        rings.append(
            (1 - g) * X0
            + g * outer
            - no * (mound * np.sin(np.pi * t)) * P["mound_overcurve_fraction"]
        )
    for k in range(1, P["lip_rings"] + 1):
        t = k / P["lip_rings"]
        g = t * t * (3 - 2 * t)
        rings.append(
            ell(
                Pc - no * (mound - lip * g),
                outerA + (A - outerA) * g,
                outerB + (gap - outerB) * g,
                bow,
            )
        )
    mouth_depth = mound - lip
    for k in range(1, P["channel_rings"] + 1):
        t = k / P["channel_rings"]
        dep = -mouth_depth + (mouth_depth + P["channel_exit_depth_cm"] * G) * t
        rings.append(ell(Pc + no * dep, A, gap, bow * (1 - t)))
    for k in range(1, P["pit_rings"] + 1):
        t = k / P["pit_rings"]
        d_ = P["channel_exit_depth_cm"] * G + P["blind_pit_depth_cm"] * G * t
        rs = P["pit_rounding_start_fraction"]
        f = np.sqrt(max(0.0, 1 - max(0.0, (t - rs) / (1 - rs)) ** 2))
        rings.append(
            ell(
                Pc + no * d_,
                A * max(f, P["pit_min_radius_fraction"]),
                gap * max(f, P["pit_min_radius_fraction"]),
                0.0,
            )
        )
    return rings


def _loft_faces(rim: np.ndarray, n_rings: int, first_new_vertex: int) -> np.ndarray:
    """Faces joining the wall rim to ``n_rings`` new rings and a closing tip vertex."""
    M = len(rim)
    ids = [rim]
    off = first_new_vertex
    for _ in range(n_rings):
        ids.append(off + np.arange(M))
        off += M
    tip = off
    Fl = []
    for a_, b_ in zip(ids[:-1], ids[1:], strict=False):
        for m in range(M):
            n = (m + 1) % M
            Fl += [[a_[m], a_[n], b_[m]], [a_[n], b_[n], b_[m]]]
    for m in range(M):
        Fl.append([ids[-1][m], ids[-1][(m + 1) % M], tip])
    return np.array(Fl)


def _flatten_near_camera(
    params: dict[str, Any],
    Vn: np.ndarray,
    Pc: np.ndarray,
    no: np.ndarray,
    n_body: int,
    rim_size: int,
    camtree: Any,
    poses: np.ndarray,
    rc0: float,
    R: float,
) -> np.ndarray:
    """Flatten the lumen-facing part of a feature where it would protrude close to the camera."""
    from scipy.ndimage import gaussian_filter

    CL = params["clearance"]
    requested = Vn.copy()
    inward_depth = np.maximum(0, -(Vn - Pc) @ no)
    pa = np.ones(len(Vn))
    body = n_body * rim_size
    for _ in range(CL["max_iterations"]):
        Vn = requested + inward_depth[:, None] * (1 - pa)[:, None] * no
        dv, ci = camtree.query(Vn)
        deriv = np.einsum("ij,j->i", Vn - poses[ci, :3, 3], no) / np.maximum(dv, 1e-12)
        d0_ = np.linalg.norm(Vn - poses[0, :3, 3], axis=1)
        deriv0 = (Vn - poses[0, :3, 3]) @ no / np.maximum(d0_, 1e-12)
        unsafe = (
            (
                ((dv < CL["vertex_guard_over_R"] * R) & (deriv > 0.15))
                | ((d0_ < CL["start_guard_over_local_r"] * rc0) & (deriv0 > 0.15))
            )
            & (inward_depth > 0)
            & (pa > 1e-3)
        )
        if not unsafe.any():
            break
        red = np.zeros_like(pa)
        rb = gaussian_filter(
            unsafe[:body].reshape(n_body, rim_size).astype(float),
            (1.0, CL["attenuation_sigma_angles"]),
            mode=("nearest", "wrap"),
        )
        if rb.max() > 0:
            red[:body] = np.clip(rb / rb.max(), 0, 1).ravel()
        pa *= 1 - red
    return requested + inward_depth[:, None] * (1 - pa)[:, None] * no


def _assemble_wall_mesh(
    params: dict[str, Any],
    s: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    radial: np.ndarray,
    L: float,
    hd: float,
    ao_ang: np.ndarray,
    ph: np.ndarray,
    poses: np.ndarray,
    rc0: float,
    R: float,
    G: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Tube rings, cecal dome, ileocecal valve and appendiceal orifice as one triangle mesh."""
    from scipy.spatial import cKDTree

    CP = params["cecum"]
    cph, sph = np.cos(ph), np.sin(ph)

    def ring(c: np.ndarray, Bm: np.ndarray, r: np.ndarray) -> np.ndarray:
        return c + r[:, None] * (cph[:, None] * Bm[:, 0] + sph[:, None] * Bm[:, 1])

    # Rings plus a cecal dome (its crow's-foot relief keeps the ring mean).
    Vs = [ring(C[i], B[i], radial[i]) for i in range(len(s))]
    chs = [np.full(NPHI, s[i] / R) for i in range(len(s))]
    re = radial[-1]
    nd = CP["dome_rings"]
    dim = np.asarray(CP["pole_drift_world"], float)
    dim -= (dim @ B[-1][:, 2]) * B[-1][:, 2]
    dim /= np.linalg.norm(dim)

    def dome_centre(th: float) -> np.ndarray:
        return (
            C[-1]
            + B[-1][:, 2] * hd * np.sin(th)
            + dim * CP["pole_drift_fraction"] * hd * (1 - np.cos(th))
        )

    q0 = hd / G
    crow0 = _crow_field(
        params, ph, np.array([q0]), np.array([re.mean() / G]), np.array([ao_ang[-1]])
    )[0]
    for k in range(1, nd):
        th = np.pi / 2 * k / nd
        ctr = dome_centre(th)
        qk = hd * (1 - np.sin(th)) / G
        ck = _crow_field(
            params,
            ph,
            np.array([qk]),
            np.array([max(re.mean() * np.cos(th) / G, 0.05)]),
            np.array([ao_ang[-1]]),
        )[0]
        rk = re * np.cos(th) * (1 + ck - crow0)
        rk *= (re.mean() * np.cos(th)) / rk.mean()
        Vs.append(ring(ctr, B[-1], rk))
        chs.append(np.full(NPHI, (L + hd * np.sin(th)) / R))
    apex = C[-1] + B[-1][:, 2] * hd + dim * CP["pole_drift_fraction"] * hd
    nr = len(Vs)
    Vw = np.vstack(Vs + [apex[None]])
    NA = len(Vw)
    chart_s = np.r_[np.concatenate(chs), (L + hd) / R]
    chart_t = np.r_[np.tile(ph / (2 * np.pi), nr), 0]

    def spole(i: int) -> float:
        """Distance above the cecal pole of ring ``i``."""
        if i < len(s):
            return (L - s[i]) + hd
        return hd * (1 - np.sin(np.pi / 2 * (i - len(s) + 1) / nd))

    def ring_centre(i: int) -> np.ndarray:
        if i < len(s):
            return C[i]
        return dome_centre(np.pi / 2 * (i - len(s) + 1) / nd)

    def angle_index(i: int, v: Any) -> int:
        Bm = B[min(i, len(s) - 1)]
        v = np.asarray(v, float)
        v -= (v @ Bm[:, 2]) * Bm[:, 2]
        return int(np.round(np.arctan2(v @ Bm[:, 1], v @ Bm[:, 0]) / (2 * np.pi) * NPHI)) % NPHI

    def hole(ic_: int, jc: int, ni: int, nj: int) -> tuple[set, np.ndarray]:
        ia, ib, ja, jb = ic_ - ni, ic_ + ni, jc - nj, jc + nj
        q = {(i, j % NPHI) for i in range(ia, ib) for j in range(ja, jb)}
        rim = (
            [(ia, j) for j in range(ja, jb)]
            + [(i, jb) for i in range(ia, ib)]
            + [(ib, j) for j in range(jb, ja, -1)]
            + [(i, ja) for i in range(ib, ia, -1)]
        )
        return q, np.array([i * NPHI + j % NPHI for i, j in rim])

    def frame_at_rim(X0: np.ndarray, i: int) -> tuple[np.ndarray, ...]:
        Pc = X0.mean(0)
        Bm = B[min(i, len(s) - 1)]
        no = Pc - ring_centre(i)
        no -= (no @ Bm[:, 2]) * Bm[:, 2]
        no /= np.linalg.norm(no)
        ea = Bm[:, 2] - (Bm[:, 2] @ no) * no
        ea /= np.linalg.norm(ea)
        ec = np.cross(ea, no)
        return Pc, no, ec, ea

    camtree = cKDTree(poses[:, :3, 3])
    extra = dict(V=[], F=[], cap=[], part=[], chart_s=[], chart_t=[])
    skip: set = set()

    def slit_part(i0: int, j0: int, P: dict, code: int, cap: bool) -> np.ndarray:
        """Cut a wall hole and loft a slit feature into it; returns the feature normal."""
        q, rim = hole(i0, j0, P["hole_half_rings"], P["hole_half_angles"])
        skip.update(q)
        X0 = Vw[rim]
        frame = frame_at_rim(X0, i0)
        rings = _slit_rings(P, X0, frame, G)
        Fl = _loft_faces(rim, len(rings), NA + sum(len(x) for x in extra["V"]))
        Vn = np.vstack(rings + [rings[-1].mean(0)[None]])
        Vn = _flatten_near_camera(
            params, Vn, frame[0], frame[1], len(rings), len(rim), camtree, poses, rc0, R
        )
        extra["V"].append(Vn)
        extra["F"].append(Fl)
        extra["cap"].append(np.full(len(Fl), cap))
        extra["part"].append(np.full(len(Vn), code))
        extra["chart_s"].append(np.tile(chart_s[rim], len(rings) + 1)[: len(Vn)])
        extra["chart_t"].append(np.tile(chart_t[rim], len(rings) + 1)[: len(Vn)])
        return frame[1]

    IP = params["icv"]
    AP = params["ao"]
    i_icv = int(np.argmin([abs(spole(i) / G - IP["cm_above_pole"]) for i in range(len(s))]))
    no_icv = slit_part(i_icv, angle_index(i_icv, IP["wall_world"]), IP, 1, False)
    i_ao = int(np.argmin([abs(spole(i) / G - AP["cm_above_pole"]) for i in range(nr - 1)]))
    no_ao = slit_part(i_ao, angle_index(i_ao, AP["wall_world"]), AP, 2, True)

    # Wall quads (minus feature holes), feature lofts oriented consistently, then the pole fan.
    F = []
    fc = []
    for i in range(nr - 1):
        for j in range(NPHI):
            if (i, j) in skip:
                continue
            a0 = i * NPHI + j
            a1 = i * NPHI + (j + 1) % NPHI
            b0 = a0 + NPHI
            b1 = a1 + NPHI
            F += [[a0, b0, a1], [a1, b0, b1]]
            fc += [i >= len(s) - 1] * 2
    de = {(int(a_), int(b_)) for f in F for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    for Fl, cap in zip(extra["F"], extra["cap"], strict=False):
        if (int(Fl[0][0]), int(Fl[0][1])) in de:
            Fl[:] = Fl[:, ::-1]
        F += Fl.tolist()
        fc += cap.tolist()
    for j in range(NPHI):
        F.append([(nr - 1) * NPHI + (j + 1) % NPHI, (nr - 1) * NPHI + j, NA - 1])
        fc.append(True)
    V = np.vstack([Vw] + extra["V"])
    mesh = dict(
        V=V.astype(np.float32),
        F=np.array(F, np.int32),
        chart_s=np.r_[chart_s, np.concatenate(extra["chart_s"])].astype(np.float32),
        chart_t=np.r_[chart_t, np.concatenate(extra["chart_t"])].astype(np.float32),
        chart_f=np.ones(len(V), np.float32),
        face_cap=np.array(fc),
        vertex_part=np.r_[np.zeros(NA, np.int8), np.concatenate(extra["part"])].astype(np.int8),
    )
    features = dict(
        ileocecal_valve=dict(ring=i_icv, normal=no_icv, params=IP),
        appendiceal_orifice=dict(ring=i_ao, normal=no_ao, params=AP),
    )
    return mesh, features


def _course_stage(shape: dict[str, Any], donor: dict[str, np.ndarray]) -> dict[str, Any]:
    """Centreline, ring radii with anatomical relief, cecal features and the camera path."""
    R = float(donor["R_pose"])
    G = R / 2.5  # pose units per cm
    params = _shape_parameters(shape, G)
    CP = params["cecum"]
    s, C, bounds, L, ds = _centreline(shape, params, R, G)
    B = _ring_frames(C, s)
    lab = _segment_labels(bounds, s, len(SEGMENTS))
    r0 = _radius_trend(params, s, bounds, ds)
    donor_fold = _donor_fold_texture(donor, s, bounds, lab, r0, ds)
    ph = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    pouch_shape, ring_target = _cecal_pouch(CP, B, s, bounds, L, r0, donor_fold, ph)
    hd = CP["pole_height_over_r"] * CP["pouch_mean_units"]
    poses, camera_s, labels, rc = _replay_camera_path(
        shape, donor, s, C, B, bounds, L, r0, hd, R, G
    )

    spole_s_cm = (L - s + hd) / G
    ao_ang = _world_angle(B, params["ao"]["wall_world"])
    taenia_phase, taenia_angles = _taenia_angles(params, B, lab, spole_s_cm, ao_ang, G, ds)
    rect_end = bounds[SEGMENTS.index("rectum") + 1]
    taenia, width_cm = _taenia_relief(params, s, lab, rect_end, r0, taenia_angles, ph, G, ds)
    bulge, semilunar = _haustra_relief(
        params, s, lab, rect_end, r0, taenia_phase, width_cm, ph, G, ds
    )
    houston = _houston_relief(params, B, s, r0, ph, G)
    crows = _crow_field(params, ph, spole_s_cm, r0 / G, ao_ang)
    inward = sum([taenia, semilunar, houston, crows])
    dirs = _ring_directions(B, np.cos(ph), np.sin(ph))
    radial, fold = _attenuated_radial(
        params, C, dirs, r0, pouch_shape, ring_target, inward, bulge, poses, rc[0], R
    )
    deformations = _apply_deformations(shape, radial, s, bounds, r0, ph, G)
    mesh, features = _assemble_wall_mesh(
        params, s, C, B, radial, L, hd, ao_ang, ph, poses, rc[0], R, G
    )
    arrays = dict(
        mesh,
        R=R,
        poses=poses,
        field_mask=donor["field_mask"],
        s_ring=s,
        centers=C,
        radial=radial,
        frames_ring=B,
        s=s,
        C=C,
        B=B,
        anatomy=np.array(SEGMENTS),
        anatomy_bounds=bounds,
        camera_s=camera_s,
        labels=labels,
        radius_trend=r0,
        fold=fold.astype(np.float32),
        taenia_angles=taenia_angles.astype(np.float32),
    )
    return dict(arrays=arrays, deformations=deformations, features=features, G=G, cecum=CP)


# --------------------------------------------------------------------------- wall stage
def _smooth_noise(
    rng: np.random.Generator, n: int, sigma_cm: float, G: float, ds: float, scale: float = 1.0
) -> np.ndarray:
    """Unit-variance Gaussian noise correlated over ``sigma_cm`` along the colon."""
    from scipy.ndimage import gaussian_filter1d

    x = gaussian_filter1d(rng.normal(size=n), max(1.0, sigma_cm * G / ds), mode="reflect")
    x -= x.mean()
    x /= max(float(x.std()), 1e-12)
    return scale * x


def _reshape_pleats_and_collapses(
    shape: dict[str, Any],
    radial: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    names: list[str],
    radius_trend: np.ndarray,
    ph: np.ndarray,
    G: float,
) -> np.ndarray:
    """Pleats fade continuously at both sigmoid junctions; partial collapses become one-sided."""
    wall = shape["wall"]
    for event in shape.get("deform", []):
        if event.get("kind") == "pleats":
            si = names.index("sigmoid")
            mask = (s >= bounds[si]) & (s < bounds[si + 1])
            u = (s[mask] - bounds[si]) / (bounds[si + 1] - bounds[si])
            wavelength = (
                float(event["lambda_cm"][1])
                + (float(event["lambda_cm"][0]) - float(event["lambda_cm"][1])) * u
            ) * G
            amplitude = (
                float(event["depth"][1]) + (float(event["depth"][0]) - float(event["depth"][1])) * u
            )
            phase = 2.0 * np.pi * np.cumsum(np.r_[0.0, np.diff(s[mask]) / wavelength[1:]])
            old_multiplier = 1.0 - amplitude * 0.5 * (1.0 + np.cos(phase))
            radial[mask] /= old_multiplier[:, None]
            edge_distance = np.minimum(s[mask] - bounds[si], bounds[si + 1] - s[mask]) / G
            taper = np.clip(edge_distance / 1.0, 0.0, 1.0)
            taper = taper * taper * (3.0 - 2.0 * taper)
            radial[mask] *= (1.0 - amplitude * taper * 0.5 * (1.0 + np.cos(phase)))[:, None]
            continue
        if event.get("kind") != "collapse":
            continue
        si = names.index(event["segment"])
        centre = bounds[si] + float(event.get("at", 0.5)) * (bounds[si + 1] - bounds[si])
        diameter = 2.0 * np.interp(centre, s, radius_trend)
        gs = np.exp(-0.5 * ((s - centre) / (float(event.get("sigma_D", 0.5)) * diameter)) ** 2)
        phi0 = np.deg2rad(float(event.get("phi_deg", 0.0)))
        oval = float(event.get("oval", 0.4))
        old_ang = 1.0 - oval + oval * np.cos(ph - phi0) ** 2
        old_mul = 1.0 - float(event["depth"]) * gs[:, None] * old_ang[None]
        radial /= np.maximum(old_mul, 1e-5)
        depth = float(wall.get("collapse_depth", event["depth"]))
        width = np.deg2rad(float(wall.get("collapse_width_deg", 85.0)))
        one_side = _angle_gaussian(ph, phi0, width / 2.355)
        radial *= 1.0 - depth * gs[:, None] * one_side[None]
    return radial


def _random_wall_fields(
    wall: dict[str, Any],
    radial: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    names: list[str],
    ph: np.ndarray,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded free-form wall fields: uneven fold depth, distension, oval/triangular modes, dents and
    bulges. Returns ``(radial, wall_relative_field, colon_weight)`` (weight 0 in the rectum)."""
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(int(wall["seed"]))
    corr = float(wall["correlation_cm"])

    def noise(sigma_cm: float, scale: float = 1.0) -> np.ndarray:
        return _smooth_noise(rng, len(s), sigma_cm, G, ds, scale)

    # Uneven fold relief, varying smoothly over a few cm.
    mean0 = radial.mean(axis=1)
    rel0 = radial / mean0[:, None] - 1.0
    fold_mod = np.clip(1.0 + noise(corr, float(wall["fold_depth_variation"])), 0.45, 1.75)
    radial = mean0[:, None] * (1.0 + rel0 * fold_mod[:, None])

    # Segment-to-segment distension plus a correlated within-segment field; the rectum stays smooth.
    seg_scale = rng.normal(0.0, float(wall["distension_segment_sigma"]), len(names))
    seg_field = np.zeros(len(s))
    for i in range(len(names)):
        seg_field[np.searchsorted(bounds, s, side="right") - 1 == i] = seg_scale[i]
    seg_field = gaussian_filter1d(seg_field, max(1.0, corr * G / ds), mode="nearest")
    distension = np.exp(seg_field + noise(corr, float(wall["distension_local_sigma"])))
    rect_end = bounds[names.index("rectum") + 1]
    colon_w = np.clip((s - rect_end) / (3.0 * G), 0.0, 1.0)
    colon_w = colon_w * colon_w * (3.0 - 2.0 * colon_w)
    distension = colon_w * distension + (1.0 - colon_w) * gaussian_filter1d(
        distension, 4.0 * G / ds, mode="nearest"
    )
    radial *= distension[:, None]

    # Non-circular modes (oval everywhere, triangular in the transverse colon).
    phase2 = np.cumsum(noise(corr, 0.012))
    phase3 = np.cumsum(noise(corr, 0.010))
    amp2 = np.clip(float(wall["oval_n2_amp"]) * (1.0 + noise(corr, 0.20)), 0.08, 0.30)
    ti = names.index("transverse")
    is_transverse = (s >= bounds[ti]) & (s < bounds[ti + 1])
    transverse_weight = gaussian_filter1d(
        is_transverse.astype(float), max(1.0, 1.5 * G / ds), mode="nearest"
    )
    amp3 = float(wall["transverse_n3_amp"]) * transverse_weight
    wall_rel = amp2[:, None] * np.cos(2.0 * (ph[None] - phase2[:, None]))
    wall_rel += amp3[:, None] * np.cos(3.0 * (ph[None] - phase3[:, None]))

    # Poisson-distributed dents, deep dents and bulges.
    for kind, sign in (("dent", -1.0), ("deep_dent", -1.0), ("bulge", 1.0)):
        count = int(rng.poisson(float(wall[f"{kind}_rate_per_m"]) * (s[-1] / G) / 100.0))
        for _ in range(count):
            margin = 0.08 if sign > 0 else 0.02
            centre = rng.uniform(s[0] + margin * (s[-1] - s[0]), s[-1] - margin * (s[-1] - s[0]))
            depth = rng.uniform(*wall[f"{kind}_depth_range"])
            sigma_s = rng.uniform(*wall[f"{kind}_width_cm_range"]) * G / 2.355
            centre_ph = rng.uniform(-np.pi, np.pi)
            sigma_ph = np.deg2rad(rng.uniform(*wall[f"{kind}_width_deg_range"])) / 2.355
            axial = np.exp(-0.5 * ((s - centre) / sigma_s) ** 2) * colon_w
            angular = _angle_gaussian(ph, centre_ph, sigma_ph)
            wall_rel[:] += sign * depth * axial[:, None] * angular[None]
    radial *= np.maximum(0.05, 1.0 + wall_rel)
    return radial, wall_rel, colon_w


def _calibre_lock(
    radial: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    names: list[str],
    radius_trend: np.ndarray,
    colon_w: np.ndarray,
    G: float,
    ds: float,
) -> np.ndarray:
    """Return each segment median ring-mean radius to the radius trend; smooth the rectum."""
    from scipy.ndimage import gaussian_filter1d

    rm = radial.mean(axis=1)
    lab = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, len(names) - 1)
    fac = np.ones(len(s))
    for i in range(len(names)):
        m = lab == i
        if m.any():
            fac[m] = np.median(radius_trend[m]) / np.median(rm[m])
    fac = gaussian_filter1d(fac, 1.5 * G / ds, mode="nearest")
    radial *= fac[:, None]
    rm2 = radial.mean(axis=1)
    sm = gaussian_filter1d(rm2, 4.0 * G / ds, mode="nearest")
    radial *= (1.0 + (1.0 - colon_w) * (sm / rm2 - 1.0))[:, None]
    rm3 = radial.mean(axis=1, keepdims=True)
    return rm3 + (radial - rm3) * (1.0 - 0.5 * (1.0 - colon_w))[:, None]


def _apply_wall_zones(
    zones: list[dict[str, Any]],
    radial: np.ndarray,
    s: np.ndarray,
    bounds: np.ndarray,
    names: list[str],
    ph: np.ndarray,
    G: float,
) -> np.ndarray:
    """Contraction rings and area-preserving flattened stretches with smoothstep onsets."""
    for z in zones:
        zi = names.index(z["segment"])
        zc = bounds[zi] + float(z["at"]) * (bounds[zi + 1] - bounds[zi])
        zL = 0.5 * float(z["length_cm"]) * G
        zo = float(z.get("onset_cm", 2.0)) * G
        zw = np.clip((zL + zo - np.abs(s - zc)) / zo, 0.0, 1.0)
        zw = zw * zw * (3.0 - 2.0 * zw)
        if z["kind"] == "contraction":
            zm = radial.mean(axis=1, keepdims=True)
            radial = (
                zm + (radial - zm) * (1.0 - float(z.get("relief_attenuation", 0.7)) * zw)[:, None]
            )
            radial *= (1.0 - float(z["radius_drop"]) * zw)[:, None]
        elif z["kind"] == "flatten":
            za = float(z["aspect"])
            zp = np.deg2rad(float(z["phi_deg"]))
            ze = 1.0 / np.sqrt(np.cos(ph - zp) ** 2 / za**2 + np.sin(ph - zp) ** 2) / np.sqrt(za)
            radial *= 1.0 + zw[:, None] * (ze[None] - 1.0)
        else:
            raise ValueError(z["kind"])
    return radial


def _radius_floor(radial: np.ndarray, minimum_ratio: float) -> None:
    """Pull every ring point to at least ``minimum_ratio`` x the ring mean (in place)."""
    for i in range(len(radial)):
        mu = float(radial[i].mean())
        lo = minimum_ratio * mu
        if radial[i].min() < lo:
            neg = radial[i] < mu
            factor = min(1.0, (mu - lo) / max(mu - radial[i].min(), 1e-12))
            radial[i, neg] = mu + factor * (radial[i, neg] - mu)


def _separate_branches(
    C: np.ndarray,
    B: np.ndarray,
    radial: np.ndarray,
    s: np.ndarray,
    contact_margin: float,
    ph: np.ndarray,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move remote centreline branches apart until both walls fit (arc timing unchanged)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    cph, sph = np.cos(ph), np.sin(ph)
    B_reference = B.copy()
    dirs = _ring_directions(B, cph, sph)
    for _ in range(100):
        rmax_now = radial.max(axis=1)
        tree_now = cKDTree(C)
        move = np.zeros_like(C)
        weight = np.zeros(len(C))
        violations = 0
        for i, j in tree_now.query_pairs(2.1 * float(rmax_now.max()) + contact_margin):
            if abs(s[i] - s[j]) <= 3.0 * (rmax_now[i] + rmax_now[j]):
                continue
            delta = C[j] - C[i]
            distance = float(np.linalg.norm(delta))
            direction = delta / max(distance, 1e-12)
            reach_i = float(np.max(radial[i] * np.einsum("ij,j->i", dirs[i], direction)))
            reach_j = float(np.max(radial[j] * np.einsum("ij,j->i", dirs[j], -direction)))
            excess = reach_i + reach_j + contact_margin - distance
            if excess <= 0.0:
                continue
            violations += 1
            move[i] -= 0.80 * excess * direction
            move[j] += 0.80 * excess * direction
            weight[i] += 1.0
            weight[j] += 1.0
        if violations == 0:
            break
        move /= np.maximum(weight, 1.0)[:, None]
        move = gaussian_filter1d(move, max(1.0, 0.4 * G / ds), axis=0, mode="nearest")
        C += move
        tangent_new = np.gradient(C, s, axis=0)
        tangent_new /= np.linalg.norm(tangent_new, axis=1, keepdims=True)
        normal_new = (
            B_reference[:, :, 0]
            - np.einsum("ij,ij->i", B_reference[:, :, 0], tangent_new)[:, None] * tangent_new
        )
        normal_new /= np.linalg.norm(normal_new, axis=1, keepdims=True)
        B = np.stack([normal_new, np.cross(tangent_new, normal_new), tangent_new], axis=2)
        dirs = _ring_directions(B, cph, sph)
    return C, B


def _rate_limited_minorant(raw_cap: np.ndarray, s: np.ndarray, rate: float) -> np.ndarray:
    """Greatest sampled minorant with |d cap / ds| <= rate."""
    cap = np.asarray(raw_cap, float).copy()
    for i in range(1, len(cap)):
        cap[i] = min(cap[i], cap[i - 1] + rate * (s[i] - s[i - 1]))
    for i in range(len(cap) - 2, -1, -1):
        cap[i] = min(cap[i], cap[i + 1] + rate * (s[i + 1] - s[i]))
    return cap


def _contact_dents(
    C: np.ndarray,
    dirs: np.ndarray,
    radial: np.ndarray,
    s: np.ndarray,
    contact_margin: float,
    floor_ratio: float,
    cap_rate: float,
    ds: float,
) -> None:
    """Dent the mutually facing walls of remote branches that still touch (in place)."""
    from scipy.spatial import cKDTree

    def accumulate(acc: np.ndarray, ring_index: int, angular_cap: np.ndarray) -> None:
        finite = np.isfinite(angular_cap)
        if not finite.any():
            return
        axial_cap = angular_cap[finite][None, :] + cap_rate * np.abs(s[:, None] - s[ring_index])
        acc[:, finite] = np.minimum(acc[:, finite], axial_cap)

    for _ in range(3):
        rmax_now = radial.max(axis=1)
        tree_now = cKDTree(C)
        acc = np.full_like(radial, np.inf)
        for i, j in tree_now.query_pairs(2.1 * float(rmax_now.max()) + contact_margin):
            if abs(s[i] - s[j]) <= 3.01 * ds:
                continue
            delta = C[j] - C[i]
            distance = float(np.linalg.norm(delta))
            direction = delta / max(distance, 1e-12)
            di = np.einsum("ij,j->i", dirs[i], direction)
            dj = np.einsum("ij,j->i", dirs[j], -direction)
            reach_i = float(np.max(radial[i] * di))
            reach_j = float(np.max(radial[j] * dj))
            excess = reach_i + reach_j + contact_margin - distance
            if excess <= 0.0:
                continue
            share_i = 0.5 * excess
            share_j = excess - share_i
            floor_i = floor_ratio * float(radial[i].mean())
            floor_j = floor_ratio * float(radial[j].mean())
            cap_i = np.where(di > 1e-5, np.maximum(floor_i, (reach_i - share_i) / di), np.inf)
            cap_j = np.where(dj > 1e-5, np.maximum(floor_j, (reach_j - share_j) / dj), np.inf)
            accumulate(acc, i, cap_i)
            accumulate(acc, j, cap_j)
        reduced = np.minimum(radial, acc)
        changed = int(np.any(reduced < radial - 1e-12, axis=1).sum())
        radial[:] = reduced
        if changed == 0:
            break


def _bend_cap(
    C: np.ndarray, s: np.ndarray, radial: np.ndarray, limit: float, cap_rate: float
) -> np.ndarray:
    """Keep outward relief on tight bends below the centreline bend radius."""
    tangent = np.gradient(C, s, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    curvature = np.linalg.norm(np.gradient(tangent, s, axis=0), axis=1)
    cap_raw = limit / np.maximum(curvature, 1e-12)
    return np.minimum(radial, _rate_limited_minorant(cap_raw, s, cap_rate)[:, None])


def _slope_curvature_minorant(
    raw_field: np.ndarray, s: np.ndarray, R: float, slope_limit: float, slope_change_limit: float
) -> np.ndarray:
    """Per angle, the largest field below ``raw_field`` with bounded slope and curvature."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    raw_u = np.asarray(raw_field, float) / R
    axial_u = s / R
    du = np.diff(axial_u)
    n = len(axial_u)
    rows, cols, values, upper = [], [], [], []
    constraint = 0
    for i, step in enumerate(du, 1):
        for sign in (1.0, -1.0):
            rows.extend((constraint, constraint))
            cols.extend((i, i - 1))
            values.extend((sign, -sign))
            upper.append(slope_limit * step)
            constraint += 1
    for i in range(1, n - 1):
        left, right = du[i - 1], du[i]
        half_span = 0.5 * (left + right)
        coefficients = {i - 1: 1.0 / left, i: -(1.0 / left + 1.0 / right), i + 1: 1.0 / right}
        for sign in (1.0, -1.0):
            for column, value in coefficients.items():
                rows.append(constraint)
                cols.append(column)
                values.append(sign * value)
            upper.append(slope_change_limit * half_span)
            constraint += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(constraint, n)).tocsr()
    upper = np.asarray(upper)
    result_field = np.empty_like(raw_u)
    objective = -np.ones(n)
    for angular_index in range(raw_u.shape[1]):
        bounds_lp = np.c_[np.zeros(n), raw_u[:, angular_index]]
        result = linprog(objective, A_ub=matrix, b_ub=upper, bounds=bounds_lp, method="highs")
        if not result.success:
            raise RuntimeError(f"radius minorant solve failed at angle {angular_index}")
        result_field[:, angular_index] = result.x
    return result_field * R


def _rebuild_cecal_dome(
    V: np.ndarray,
    chart_s_saved: np.ndarray,
    CP: dict[str, Any],
    C: np.ndarray,
    B: np.ndarray,
    radial: np.ndarray,
    s: np.ndarray,
    ph: np.ndarray,
    base_vertex_count: int,
    R: float,
) -> None:
    """Rebuild the dome rings and apex from the final end ring (in place); the endpoint wall field
    tapers to zero at the apex while the ring mean follows cos(theta)."""
    nring, nphi = radial.shape
    cph, sph = np.cos(ph), np.sin(ph)
    endpoint_mean = float(radial[-1].mean())
    endpoint_relative = radial[-1] / endpoint_mean - 1.0
    dome_height = float(CP["pole_height_over_r"]) * endpoint_mean
    dome_drift = np.asarray(CP["pole_drift_world"], float)
    dome_drift -= (dome_drift @ B[-1, :, 2]) * B[-1, :, 2]
    if np.linalg.norm(dome_drift) < 1e-12:
        dome_drift = B[-1, :, 0].copy()
    dome_drift /= np.linalg.norm(dome_drift)
    for k in range(1, int(CP["dome_rings"])):
        theta = 0.5 * np.pi * k / int(CP["dome_rings"])
        ct, st = np.cos(theta), np.sin(theta)
        centre = (
            C[-1]
            + B[-1, :, 2] * dome_height * st
            + dome_drift * float(CP["pole_drift_fraction"]) * dome_height * (1.0 - ct)
        )
        taper = ct * ct * (3.0 - 2.0 * ct)
        dome_radial = endpoint_mean * ct * (1.0 + endpoint_relative * taper)
        ring_index = nring + k - 1
        V[ring_index * nphi : (ring_index + 1) * nphi] = centre[None] + dome_radial[:, None] * (
            cph[:, None] * B[-1, :, 0] + sph[:, None] * B[-1, :, 1]
        )
        chart_s_saved[ring_index * nphi : (ring_index + 1) * nphi] = (s[-1] + dome_height * st) / R
    V[base_vertex_count - 1] = (
        C[-1]
        + B[-1, :, 2] * dome_height
        + dome_drift * float(CP["pole_drift_fraction"]) * dome_height
    )
    chart_s_saved[base_vertex_count - 1] = (s[-1] + dome_height) / R


def _transport_feature_vertices(
    V: np.ndarray,
    V_base: np.ndarray,
    feature_vertices: np.ndarray,
    chart_s: np.ndarray,
    chart_t: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    C_base: np.ndarray,
    B: np.ndarray,
    B_reference: np.ndarray,
    ratio: np.ndarray,
    base_ring_count: int,
) -> None:
    """Carry feature vertices with their anchor ring's radius change (in place)."""
    nphi = ratio.shape[1]
    tube_features = feature_vertices[chart_s[feature_vertices] <= s[-1]]
    if len(tube_features):
        ii = np.clip(np.searchsorted(s, chart_s[tube_features]), 1, len(s) - 1)
        choose_left = abs(chart_s[tube_features] - s[ii - 1]) < abs(chart_s[tube_features] - s[ii])
        ii[choose_left] -= 1
        jj = np.round(np.mod(chart_t[tube_features], 1.0) * nphi).astype(int) % nphi
        old_local = np.einsum("nji,nj->ni", B_reference[ii], V_base[tube_features] - C_base[ii])
        old_local[:, :2] *= ratio[ii, jj, None]
        V[tube_features] = C[ii] + np.einsum("nij,nj->ni", B[ii], old_local)
    dome_features = feature_vertices[chart_s[feature_vertices] > s[-1]]
    if len(dome_features):
        old_ring_chart = chart_s[np.arange(base_ring_count) * nphi]
        anchor = np.array([int(np.argmin(abs(old_ring_chart - chart_s[v]))) for v in dome_features])
        jj = np.round(np.mod(chart_t[dome_features], 1.0) * nphi).astype(int) % nphi
        old_centres = np.array([V_base[a * nphi : (a + 1) * nphi].mean(axis=0) for a in anchor])
        new_centres = np.array([V[a * nphi : (a + 1) * nphi].mean(axis=0) for a in anchor])
        old_local = np.einsum("ji,nj->ni", B_reference[-1], V_base[dome_features] - old_centres)
        old_radius = np.array(
            [
                np.linalg.norm((V_base[a * nphi + j] - oc) @ B_reference[-1, :, :2])
                for a, j, oc in zip(anchor, jj, old_centres, strict=False)
            ]
        )
        new_radius = np.array(
            [
                np.linalg.norm((V[a * nphi + j] - nc) @ B[-1, :, :2])
                for a, j, nc in zip(anchor, jj, new_centres, strict=False)
            ]
        )
        old_local[:, :2] *= (new_radius / np.maximum(old_radius, 1e-12))[:, None]
        V[dome_features] = new_centres + np.einsum("ij,nj->ni", B[-1], old_local)


def _rebuild_slit_features(
    V: np.ndarray,
    faces: np.ndarray,
    part: np.ndarray,
    features: dict[str, Any],
    B: np.ndarray,
    B_reference: np.ndarray,
    n_rings: int,
    G: float,
) -> None:
    """Rebuild the cecal slit lofts as nested rings around the transported wall rim (in place): the
    mound stays on the lumen side, the channel returns through the hole and the pit is capped."""
    vertex_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_faces.setdefault(int(vertex_index), []).append(face_index)
    for code, feature in (
        (1, features["ileocecal_valve"]),
        (2, features["appendiceal_orifice"]),
    ):
        P = feature["params"]
        ids = np.flatnonzero(part == code)
        rim_size = 4 * (int(P["hole_half_rings"]) + int(P["hole_half_angles"]))
        loft_rings = sum(
            int(P[k]) for k in ("mound_rings", "lip_rings", "channel_rings", "pit_rings")
        )
        anchor_ring = min(int(feature["ring"]), n_rings - 1)
        old_axis = np.asarray(feature["normal"], float)
        new_axis = B[anchor_ring] @ (B_reference[anchor_ring].T @ old_axis)
        new_axis /= np.linalg.norm(new_axis)
        first_ring = ids[:rim_size]
        rim = []
        for k, current in enumerate(first_ring):
            previous = first_ring[(k - 1) % rim_size]
            common = set(vertex_faces[int(current)]).intersection(vertex_faces[int(previous)])
            candidates = []
            for face_index in common:
                candidates.extend(int(v) for v in faces[face_index] if part[int(v)] == 0)
            rim.append(candidates[0])
        rim = np.asarray(rim, int)
        rim_points = V[rim]
        rim_centre = rim_points.mean(axis=0)
        planar = rim_points - rim_centre
        planar -= (planar @ new_axis)[:, None] * new_axis
        nm, nl = int(P["mound_rings"]), int(P["lip_rings"])
        nc, npit = int(P["channel_rings"]), int(P["pit_rings"])
        transported_rows = V[ids[:-1]].reshape(loft_rings, rim_size, 3)
        rim_spread = float(np.mean(np.linalg.norm(planar, axis=1)))
        channel_start = transported_rows[nm + nl]
        channel_planar = channel_start - channel_start.mean(axis=0)
        channel_planar -= (channel_planar @ new_axis)[:, None] * new_axis
        inner_scale = float(
            np.clip(
                np.mean(np.linalg.norm(channel_planar, axis=1)) / max(rim_spread, 1e-12),
                0.25,
                0.72,
            )
        )
        mound_depth = float(P["mound_depth_cm"]) * G
        lip = float(P["lip_thickness_cm"]) * G
        channel_exit = float(P["channel_exit_depth_cm"]) * G
        pit_depth = float(P["blind_pit_depth_cm"]) * G
        rebuilt = []
        for k in range(1, nm + 1):
            t = k / nm
            g = t * t * (3.0 - 2.0 * t)
            target = rim_centre + (1.0 - 0.02 * g) * planar - mound_depth * g * new_axis
            rebuilt.append((1.0 - g) * rim_points + g * target)
        for k in range(1, nl + 1):
            t = k / nl
            g = t * t * (3.0 - 2.0 * t)
            scale = 0.98 + (inner_scale - 0.98) * g
            depth = -mound_depth + lip * g
            rebuilt.append(rim_centre + scale * planar + depth * new_axis)
        mouth_depth = mound_depth - lip
        for k in range(1, nc + 1):
            t = k / nc
            depth = -mouth_depth + (mouth_depth + channel_exit) * t
            rebuilt.append(rim_centre + inner_scale * planar + depth * new_axis)
        for k in range(1, npit + 1):
            t = k / npit
            rounding_start = float(P["pit_rounding_start_fraction"])
            rounding = np.sqrt(
                max(0.0, 1.0 - max(0.0, (t - rounding_start) / (1.0 - rounding_start)) ** 2)
            )
            scale = inner_scale * max(rounding, float(P["pit_min_radius_fraction"]))
            rebuilt.append(rim_centre + scale * planar + (channel_exit + pit_depth * t) * new_axis)
        V[ids[:-1]] = np.asarray(rebuilt).reshape(-1, 3)
        V[ids[-1]] = np.asarray(rebuilt[-1]).mean(axis=0)


def _chord_radius_at_offsets(
    radial_rows: np.ndarray, offsets: np.ndarray, cph: np.ndarray, sph: np.ndarray
) -> np.ndarray:
    """Ray distance to the straight polygon chord bracketing each offset angle."""
    nphi = radial_rows.shape[1]
    angle = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi)
    u = angle * nphi / (2.0 * np.pi)
    j0 = np.floor(u).astype(int) % nphi
    j1 = (j0 + 1) % nphi
    row = np.arange(len(offsets))
    ray = np.c_[np.cos(angle), np.sin(angle)]
    p0 = radial_rows[row, j0, None] * np.c_[cph[j0], sph[j0]]
    p1 = radial_rows[row, j1, None] * np.c_[cph[j1], sph[j1]]
    edge = p1 - p0
    numerator = p0[:, 0] * edge[:, 1] - p0[:, 1] * edge[:, 0]
    denominator = ray[:, 0] * edge[:, 1] - ray[:, 1] * edge[:, 0]
    return numerator / denominator


def _point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distance from point ``p`` to each triangle ``tri[k]`` (k, 3, 3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = np.einsum("ij,ij->i", normal, normal)
    pa = p - a
    plane = abs(np.einsum("ij,ij->i", pa, normal)) / np.sqrt(np.maximum(n2, 1e-30))
    proj = p - (np.einsum("ij,ij->i", pa, normal) / np.maximum(n2, 1e-30))[:, None] * normal
    vp = proj - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", vp, ab)
    d21 = np.einsum("ij,ij->i", vp, ac)
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / np.maximum(den, 1e-30)
    v = (d00 * d21 - d01 * d20) / np.maximum(den, 1e-30)
    inside = (den > 1e-24) & (u >= 0) & (v >= 0) & (u + v <= 1)

    def segment_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        edge = y - x
        t = np.einsum("ij,ij->i", p - x, edge) / np.maximum(
            np.einsum("ij,ij->i", edge, edge), 1e-30
        )
        return np.linalg.norm(p - (x + np.clip(t, 0, 1)[:, None] * edge), axis=1)

    edges = np.minimum(
        segment_distance(a, b), np.minimum(segment_distance(b, c), segment_distance(c, a))
    )
    return np.where(inside, plane, edges)


class _TriangleClearance:
    """Exact point-to-mesh distance with a centroid k-d tree broad phase."""

    def __init__(self, V: np.ndarray, F: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        self.tri = V[F]
        centres = self.tri.mean(axis=1)
        self.broad = float(np.linalg.norm(self.tri - centres[:, None], axis=2).max(axis=1).max())
        self.tree = cKDTree(centres)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        _, near = self.tree.query(points, k=min(48, len(self.tri)))
        near = np.atleast_2d(near)
        out = np.empty(len(points))
        for qi, point in enumerate(points):
            upper = float(_point_triangle_distance(point, self.tri[np.atleast_1d(near[qi])]).min())
            ids = np.asarray(self.tree.query_ball_point(point, upper + self.broad + 1e-10), int)
            out[qi] = _point_triangle_distance(point, self.tri[ids]).min()
        return out


def _max_clearance_camera(
    wall: dict[str, Any],
    poses_base: np.ndarray,
    camera_s: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    radial: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    ph: np.ndarray,
    R: float,
    G: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the camera laterally toward a smooth, rate-limited max-clearance path inside the lumen
    (camera_s and timing unchanged); frames closer than 0.305 R to the wall or outside the lumen are
    rerouted. Returns ``(poses, camera_offset)``."""
    from scipy.ndimage import gaussian_filter1d

    cph, sph = np.cos(ph), np.sin(ph)
    base_centres = _interp_rows(camera_s, s, C)
    base_frames = _interp_frames(camera_s, s, B)
    base_local = np.einsum("nji,nj->ni", base_frames, poses_base[:, :3, 3] - base_centres)[:, :2]
    order = np.argsort(camera_s)
    sq, unique_idx = np.unique(camera_s[order], return_index=True)
    base_ring = _interp_rows(s, sq, base_local[order][unique_idx])

    max_offset = float(wall["camera_max_offset_R"])
    radii = np.linspace(0.0, max_offset, 5)
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    unit_candidates = np.vstack(
        ([0.0, 0.0], np.array([[r * np.cos(a), r * np.sin(a)] for r in radii[1:] for a in angles]))
    )

    def contained(points: np.ndarray, axial_s: np.ndarray) -> np.ndarray:
        local_xy = np.einsum(
            "nji,nj->ni", _interp_frames(axial_s, s, B), points - _interp_rows(axial_s, s, C)
        )[:, :2]
        wall_radius = _chord_radius_at_offsets(_interp_rows(axial_s, s, radial), local_xy, cph, sph)
        return wall_radius - np.linalg.norm(local_xy, axis=1) > 1e-8 * R

    target = np.zeros((len(s), 2))
    for i in range(len(s)):
        mu = radial[i].mean()
        cand = unit_candidates * mu
        wall_xy = radial[i, :, None] * np.c_[cph, sph]
        clearance = np.linalg.norm(cand[:, None, :] - wall_xy[None, :, :], axis=2).min(axis=1)
        penalty = float(wall["camera_replay_penalty"]) * np.linalg.norm(cand - base_ring[i], axis=1)
        wall_radius = _chord_radius_at_offsets(
            np.repeat(radial[i][None], len(cand), axis=0), cand, cph, sph
        )
        inside = np.linalg.norm(cand, axis=1) < wall_radius - 1e-8 * R
        score = clearance - penalty
        score[~inside] = -np.inf
        target[i] = cand[np.argmax(score)]
    target = gaussian_filter1d(
        target, max(1.0, float(wall["camera_offset_smooth_cm"]) * G / ds), axis=0, mode="nearest"
    )

    def limit_magnitude(target: np.ndarray) -> np.ndarray:
        limit = max_offset * radial.mean(axis=1)
        mag = np.linalg.norm(target, axis=1)
        return target * np.minimum(1.0, limit / np.maximum(mag, 1e-12))[:, None]

    rate = float(wall["camera_rate_limit_R_per_cm"])

    def rate_limit(target: np.ndarray) -> None:
        """Axial rate limit, applied forward then backward."""
        for indices, neighbour in ((range(1, len(s)), -1), (range(len(s) - 2, -1, -1), 1)):
            for i in indices:
                j = i + neighbour
                max_step = (
                    rate * (abs(s[i] - s[j]) / G) * 0.5 * (radial[i].mean() + radial[j].mean())
                )
                delta = target[i] - target[j]
                dn = np.linalg.norm(delta)
                if dn > max_step:
                    target[i] = target[j] + delta * (max_step / dn)

    target = limit_magnitude(target)
    rate_limit(target)
    camera_offset = _interp_rows(camera_s, s, target)
    poses = poses_base.copy()
    poses[:, :3, 3] = base_centres + np.einsum("nij,nj->ni", base_frames[:, :, :2], camera_offset)

    clearance_of = _TriangleClearance(V, F)
    tri_distance = clearance_of(poses[:, :3, 3])
    camera_inside = contained(poses[:, :3, 3], camera_s)
    for _ in range(5):
        bad_frames = np.flatnonzero((tri_distance < 0.305 * R) | ~camera_inside)
        if len(bad_frames) == 0:
            break
        bad_rings = np.unique(np.clip(np.searchsorted(s, camera_s[bad_frames]), 0, len(s) - 1))
        candidate_points, candidate_offsets, candidate_owner = [], [], []
        for ring_index in bad_rings:
            cand = unit_candidates * radial[ring_index].mean()
            candidate_points.append(
                C[ring_index] + np.einsum("ij,kj->ki", B[ring_index, :, :2], cand)
            )
            candidate_offsets.append(cand)
            candidate_owner.extend([ring_index] * len(cand))
        candidate_points = np.vstack(candidate_points)
        candidate_offsets = np.vstack(candidate_offsets)
        candidate_owner = np.asarray(candidate_owner)
        candidate_clearance = clearance_of(candidate_points)
        candidate_clearance[~contained(candidate_points, s[candidate_owner])] = -np.inf
        for ring_index in bad_rings:
            mask = candidate_owner == ring_index
            chosen = candidate_offsets[np.flatnonzero(mask)[np.argmax(candidate_clearance[mask])]]
            influence = np.exp(-0.5 * ((s - s[ring_index]) / (0.8 * G)) ** 2)[:, None]
            target = target * (1.0 - influence) + chosen[None] * influence
        target = gaussian_filter1d(target, max(1.0, 0.25 * G / ds), axis=0, mode="nearest")
        target = limit_magnitude(target)
        rate_limit(target)
        camera_offset = _interp_rows(camera_s, s, target)
        poses[:, :3, 3] = base_centres + np.einsum(
            "nij,nj->ni", base_frames[:, :, :2], camera_offset
        )
        tri_distance = clearance_of(poses[:, :3, 3])
        camera_inside = contained(poses[:, :3, 3], camera_s)
    return poses, camera_offset


def _wall_stage(shape: dict[str, Any], base: dict[str, Any]) -> dict[str, np.ndarray]:
    """Free-form wall fields, self-contact handling, cecal rebuild and max-clearance camera path."""
    wall = shape["wall"]
    LIM = WALL_LIMITS
    arrays = dict(base["arrays"])
    R = float(arrays["R"])
    G = float(base["G"])
    s = arrays["s_ring"].astype(float)
    C = arrays["C"].astype(float)
    B = arrays["B"].astype(float)
    radial_base = arrays["radial"].astype(float)
    radius_trend = arrays["radius_trend"].astype(float)
    bounds = arrays["anatomy_bounds"].astype(float)
    names = [str(x) for x in arrays["anatomy"].tolist()]
    ph = np.linspace(0.0, 2.0 * np.pi, radial_base.shape[1], endpoint=False)
    ds = float(np.median(np.diff(s)))

    radial = _reshape_pleats_and_collapses(
        shape, radial_base.copy(), s, bounds, names, radius_trend, ph, G
    )
    radial, wall_rel, colon_w = _random_wall_fields(wall, radial, s, bounds, names, ph, G, ds)
    radial = _calibre_lock(radial, s, bounds, names, radius_trend, colon_w, G, ds)
    radial = _apply_wall_zones(shape.get("wall_zones", []), radial, s, bounds, names, ph, G)

    # Non-self-intersecting wall: radius floor, branch separation, contact dents, bend cap and a
    # slope/curvature-limited minorant.
    min_ratio = LIM["minimum_radius_ratio"]
    _radius_floor(radial, min_ratio + LIM["construction_headroom"])
    C_base = C.copy()
    B_reference = B.copy()
    contact_margin = float(LIM["self_contact_margin_R"]) * R
    C, B = _separate_branches(C, B, radial, s, contact_margin, ph, G, ds)
    cap_rate = float(LIM["radius_cap_rate_per_axial_R"])
    dirs = _ring_directions(B, np.cos(ph), np.sin(ph))
    _contact_dents(C, dirs, radial, s, contact_margin, min_ratio + 0.005, cap_rate, ds)
    radial = _bend_cap(C, s, radial, float(LIM["bend_envelope_curvature_limit"]), cap_rate)
    radial = _slope_curvature_minorant(
        radial,
        s,
        R,
        float(LIM["final_radius_rate_limit_per_axial_R"]),
        float(LIM["final_radius_slope_change_limit_per_axial_R2"]),
    )

    # Rebuild tube rings, the cecal dome and the cecal features on the final wall.
    V_base = arrays["V"].astype(float)
    V = V_base.copy()
    nring, nphi = radial.shape
    V[: nring * nphi] = (C[:, None, :] + radial[:, :, None] * dirs).reshape(-1, 3)
    part = arrays["vertex_part"]
    chart_s = arrays["chart_s"].astype(float) * R
    chart_s_saved = arrays["chart_s"].astype(float).copy()
    feature_vertices = np.flatnonzero(part > 0)
    base_vertex_count = int(feature_vertices[0]) if len(feature_vertices) else len(V)
    _rebuild_cecal_dome(V, chart_s_saved, base["cecum"], C, B, radial, s, ph, base_vertex_count, R)
    _transport_feature_vertices(
        V,
        V_base,
        feature_vertices,
        chart_s,
        arrays["chart_t"].astype(float),
        s,
        C,
        C_base,
        B,
        B_reference,
        radial / np.maximum(radial_base, 1e-9),
        (base_vertex_count - 1) // nphi,
    )
    _rebuild_slit_features(V, arrays["F"], part, base["features"], B, B_reference, len(s), G)

    poses, camera_offset = _max_clearance_camera(
        wall,
        arrays["poses"].astype(float),
        arrays["camera_s"].astype(float),
        s,
        C,
        B,
        radial,
        V,
        arrays["F"],
        ph,
        R,
        G,
        ds,
    )
    arrays.update(
        V=V.astype(np.float32),
        radial=radial.astype(np.float64),
        poses=poses.astype(np.float64),
        C=C.astype(np.float64),
        centers=C.astype(np.float64),
        B=B.astype(np.float64),
        frames_ring=B.astype(np.float64),
        chart_s=chart_s_saved.astype(np.float32),
        wall_relative_field=wall_rel.astype(np.float32),
        camera_offset_wall=camera_offset.astype(np.float32),
    )
    return arrays


# --------------------------------------------------------------------------- public API
def build_colon_wall(
    shape: dict[str, Any], donor: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build one synthetic colon wall and its withdrawal camera path.

    Args:
        shape: one entry of ``shapes`` in ``configs/full_colon_shapes.yaml``.
        donor: arrays from :func:`load_donor`.

    Returns:
        ``(arrays, deformations)``. ``arrays`` holds the wall mesh ``V`` (float32, pose units),
        ``F``, the ring parametrisation (``s``/``s_ring`` arc length, ``C``/``centers`` centreline,
        ``B``/``frames_ring`` ring frames ``[n1, n2, tangent]``, ``radial`` (rings x 96) radii),
        texture chart (``chart_s``, ``chart_t``, ``chart_f``), ``poses`` (camera-to-world, one per
        frame, cecum to rectum), ``camera_s`` and ``labels`` (arc position and anatomical segment
        per frame), ``anatomy``/``anatomy_bounds``, ``radius_trend``, ``field_mask`` and auxiliary
        fields. ``deformations`` lists the configured withdrawal deformations with their
        ``local_diameter_units``; entries of kind ``collapse`` define where camera clearance is
        measured against the local ring radius.
    """
    base = _course_stage(shape, donor)
    return _wall_stage(shape, base), base["deformations"]
