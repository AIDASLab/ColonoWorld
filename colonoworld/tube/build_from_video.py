# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tube for a real video clip from its first frame and its camera path.

This builder reads nothing beyond frame 0 of the clip, so it can be used for held-out benchmark
clips without looking at the frames that will be compared against the generated video:

1. **Frame-0 cues.** The device content mask (rays with a lens ray that are not part of the black
   border) and the dark lumen region: grey image, specular highlights inpainted, Gaussian blur,
   normalised by the median radial brightness profile; the darkest 2 % of the inner field form the
   lumen candidates and the connected blob with the largest darkness weight is the lumen. Its mean
   viewing ray is the frame-0 lumen direction.
2. **Axis.** The principal direction of the camera path when the path is long enough (at least
   ``moving_R`` radii), oriented toward the lumen; otherwise the frame-0 lumen ray (or the
   camera's optical axis when no lumen is visible).
3. **Centreline.** The Gaussian-smoothed camera path, shifted toward the side on which the lumen is
   seen (by ``offset_gain * (r (1 - fold depth) - 0.36)``, scaled by the lumen's off-axis angle
   relative to ``offset_ref_deg``). Ahead of the front-most camera the centreline bends toward the
   frame-0 lumen ray (lateral slope at most ``max_slope``), so the tube can turn sharply.
4. **Radius.** The per-segment median wall radius of the training tubes, clamped to the segment's
   [p10, p90] range (segments with fewer than ``min_rows`` training tubes use the pooled prior).
5. **Folds.** One haustral fold pattern drawn deterministically (from ``key``) from the fold prior
   of the training tubes.
6. **Containment.** The sweep is inflated until a ``0.36 R`` ball around every camera fits inside.

The result is a :class:`~colonoworld.tube.wall_fit` profile sweep, i.e. the same representation and
renderer as the training tubes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.camera import Calibration, pixel_grid_rays, rays_from_clip
from colonoworld.tube.model import TubeModel
from colonoworld.tube.wall_fit import Params, axis_basis, contain_cameras, to_tube_model

R_POSE = 0.3  # reference radius in pose units (pose units = first-frame median scene depth)
CUE_SIZE = (276, 208)  # (W, H) of the cue grid, a quarter of the clip canvas

DEFAULT_RULE = dict(
    bend=1.0,
    offset_gain=0.5,
    offset_ref_deg=30.0,
    knots=24,
    behind=1.75,
    ahead=8.8,
    max_slope=0.75,
    smooth=0.6,
    kernel_R=0.4,
    ramp_R=1.5,
    moving_R=2.0,
    ball=0.36,
)


# ----------------------------------------------------------------------------- frame-0 cues
def _gray(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def specular_mask(small: np.ndarray) -> np.ndarray:
    """Bright, unsaturated pixels (specular highlights), dilated by 5x5."""
    import cv2

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    m = ((v > 225) & (s < 90)) | (v > 248)
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def content_mask(frame: np.ndarray, cam: Calibration) -> np.ndarray:
    """Cue-grid mask of pixels with a lens ray that are not the black device border."""
    import cv2

    w, h = CUE_SIZE
    _, ok = pixel_grid_rays(cam, w, h)
    mx = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA).max(-1)
    dark = (mx < 14).astype(np.uint8)
    _, lab = cv2.connectedComponents(dark, connectivity=4)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
    border = np.isin(lab, list(edge)) & (dark > 0)
    border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return ok & ~border


def lumen_mask(small: np.ndarray, cam: Calibration, mask: np.ndarray) -> np.ndarray:
    """Dark-lumen blob on the cue grid (empty when no dark component exists)."""
    import cv2

    g8 = np.clip(_gray(small), 0, 255).astype(np.uint8)
    g = cv2.inpaint(g8, (specular_mask(small) | ~mask).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    blur = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 9)
    w, h = CUE_SIZE
    yy, xx = np.mgrid[:h, :w]
    k = np.asarray(cam["K_clip"])
    cx, cy = (k[0, 2] + 0.5) / 4 - 0.5, (k[1, 2] + 0.5) / 4 - 0.5
    r = np.hypot(xx - cx, yy - cy)
    rmax = r[mask].max()
    bins = np.minimum((r / rmax * 12).astype(int), 11)
    prof = np.array(
        [
            np.median(blur[mask & (bins == b)]) if (mask & (bins == b)).sum() > 20 else np.nan
            for b in range(12)
        ]
    )
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    norm = blur / np.maximum(prof[bins], 1.0)
    inner = mask & (r < 0.8 * rmax)
    thr = np.percentile(norm[inner], 2.0)
    dark = (inner & (norm <= thr)).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    weight = np.maximum(thr - norm, 0) + 1e-3
    best = max(range(1, n), key=lambda i: weight[lab == i].sum())
    return lab == best


def lumen_direction(lumen: np.ndarray, cam: Calibration, min_pixels: int = 20) -> np.ndarray:
    """Mean unit viewing ray (camera frame) over lumen pixels; NaN when too few pixels."""
    ys, xs = np.nonzero(lumen)
    if len(xs) < min_pixels:
        return np.full(3, np.nan)
    uv = np.c_[(xs + 0.5) * 4 - 0.5, (ys + 0.5) * 4 - 0.5]
    rays, valid = rays_from_clip(uv, cam)
    rays = rays[valid]
    if not len(rays):
        return np.full(3, np.nan)
    d = rays.mean(0)
    return np.eye(3) @ (d / np.linalg.norm(d))


def frame0_lumen_direction(frame0: np.ndarray, cam: Calibration) -> np.ndarray:
    """Frame-0 lumen viewing direction (camera frame of frame 0)."""
    import cv2

    small = cv2.resize(frame0, CUE_SIZE, interpolation=cv2.INTER_AREA)
    clip_w, clip_h = cam.get("clip_size", (1104, 832))
    field = content_mask(frame0, cam).astype(np.uint8)
    field = cv2.resize(field, (clip_w, clip_h), interpolation=cv2.INTER_NEAREST) > 0
    return lumen_direction(lumen_mask(small, cam, field[::4, ::4].astype(bool)), cam)


# ----------------------------------------------------------------------------- priors
def segment_radius(segment: str, priors: dict[str, Any]) -> float:
    """Prior wall radius (units of R_POSE) for an anatomical segment."""
    seg = priors["segments"].get(segment) or priors["segments"]["all"]
    if seg["n"] < priors["min_rows"]:
        seg = priors["segments"]["all"]
    return float(np.clip(seg["radius_median"], seg["radius_p10"], seg["radius_p90"]))


def draw_fold(key: str, fold_prior: list[dict[str, float]]) -> dict[str, float]:
    """Deterministic draw of a fold pattern from the prior, seeded by ``key``."""
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    fold = copy.deepcopy(fold_prior[int(rng.integers(len(fold_prior)))])
    fold["phase"] = float(rng.uniform())
    return fold


# ----------------------------------------------------------------------------- geometry rule
def tube_parameters(
    poses: np.ndarray,
    lumen_dir: np.ndarray,
    radius_R: float,
    fold: dict[str, float],
    rule: dict[str, float] | None = None,
) -> Params:
    """Profile-sweep parameters from the camera path and the frame-0 lumen direction."""
    from scipy.ndimage import gaussian_filter1d

    rule = {**DEFAULT_RULE, **(rule or {})}
    poses = np.asarray(poses, float)
    c = poses[:, :3, 3] / R_POSE
    n = len(poses)
    d0 = np.asarray(lumen_dir, float)
    has_lumen = bool(np.isfinite(d0).all() and d0[2] > 0.2)
    f = d0 if has_lumen else poses[0, :3, 2].copy()
    pc = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]
    extent = float(np.ptp((c - c.mean(0)) @ pc))
    if extent >= rule["moving_R"]:
        f = pc * np.sign(pc @ f if abs(pc @ f) > 1e-9 else 1.0)
    basis = axis_basis(np.asarray(f, float))
    f = basis[2]
    zc = c @ f
    xy = c @ basis[:2].T
    sig = max(1.0, n / 40)
    xs = np.c_[
        gaussian_filter1d(xy[:, 0], sig, mode="nearest"),
        gaussian_filter1d(xy[:, 1], sig, mode="nearest"),
    ]
    # The camera sees the lumen off its optical axis -> the tube axis lies toward that side.
    off = np.zeros(2)
    if has_lumen:
        lat = d0 @ basis[:2].T
        nl = np.linalg.norm(lat)
        if nl > 1e-6:
            tan = nl / max(d0 @ f, 0.2)
            frac = float(np.clip(tan / np.tan(np.radians(rule["offset_ref_deg"])), 0, 1))
            gap = max(radius_R * (1 - fold["depth_frac"]) - rule["ball"], 0.0)
            off = lat / nl * (rule["offset_gain"] * gap * frac)
    zz = np.linspace(zc.min() - rule["behind"], zc.max() + rule["ahead"], int(rule["knots"]))
    front = zc.max()
    centres = []
    for z in zz:
        w = -0.5 * ((z - zc) / rule["kernel_R"]) ** 2
        w = np.exp(w - w.max())
        base = (w[:, None] * xs).sum(0) / w.sum() + off
        ramp = float(np.clip((z - front) / rule["ramp_R"], 0, 1))
        target = None
        s = z - zc[0]
        if has_lumen and s > 0 and d0 @ f >= 0.2:
            reach = c[0] + (s / (d0 @ f)) * d0
            o = reach @ basis[:2].T - xy[0]
            lim = rule["max_slope"] * s
            nrm = np.linalg.norm(o)
            if nrm > lim:
                o = o * lim / nrm
            target = xs[0] + o
        centres.append(base + rule["bend"] * ramp * (target - base) if target is not None else base)
    centres = gaussian_filter1d(np.array(centres), rule["smooth"], axis=0, mode="nearest")
    knots = int(rule["knots"])
    prof = np.c_[centres, np.full(knots, np.log(radius_R)), np.zeros((knots, 2))]
    p = dict(
        R_pose=float(R_POSE),
        origin_R=[0.0, 0.0, 0.0],
        basis=basis.tolist(),
        z_R=zz.tolist(),
        profiles=prof,
        fold=copy.deepcopy(fold),
    )
    p, _ = contain_cameras(p, poses, rule["ball"])
    p["profiles"] = np.asarray(p["profiles"]).tolist()
    return p


def build_tube_from_video(
    frame0: np.ndarray,
    cameras: np.ndarray,
    cam: Calibration,
    segment: str,
    key: str,
    cfg: dict[str, Any],
) -> tuple[TubeModel, Params]:
    """Build the tube of one clip.

    Args:
        frame0: ``(H, W, 3)`` uint8 RGB first frame of the clip.
        cameras: ``(T, 4, 4)`` camera-to-world poses relative to frame 0 (pose units).
        cam: fisheye calibration of the clip.
        segment: anatomical segment label of the clip (e.g. ``"transverse"``).
        key: clip identifier; seeds the fold draw.
        cfg: the ``video_tube`` section of ``configs/tube.yaml`` (rule, prior file paths).

    Returns:
        ``(tube, params)``: the meshed tube and its profile-sweep parameters.
    """
    priors = json.loads(Path(cfg["radius_prior"]).read_text())
    fold_prior = json.loads(Path(cfg["fold_prior"]).read_text())["parameters"]
    lumen_dir = frame0_lumen_direction(frame0, cam)
    radius = segment_radius(segment, priors)
    fold = draw_fold(key, fold_prior)
    params = tube_parameters(cameras, lumen_dir, radius, fold, cfg.get("rule"))
    tube = to_tube_model(params)
    tube.meta.update(segment=segment, key=key, lumen_dir=np.nan_to_num(lumen_dir).tolist())
    return tube, params
