# SPDX-License-Identifier: CC-BY-NC-4.0
"""Fused wall depth of a training window (first step of the training-tube seed).

Inputs are the real clip, its KB4 calibration and the per-frame camera poses of a monocular
pose/depth estimator (MegaSaM: ``relative_c2w`` per decoded frame, OpenCV axes, translation in
units of the first frame's median depth). Depth is measured on a rectified pinhole chart
(640x480 frames, 512x384 depth grid):

1. **Keyframe depth** (:func:`keyframe_depth`): every 6th frame of the clip, the relative
   disparity of a monocular depth network (Depth Anything ViT-L) is aligned to the poses with
   ``disparity = a D + b`` minimising the robust reprojection error against DIS optical flow to
   frames +-3 and +-6, plus a prior to the previous accepted keyframe.
2. **Wall evidence** (:func:`collect_evidence`): for every 12th frame around the window, a pose
   gate (epipolar agreement of the flow), two-view triangulation and the aligned keyframe depth
   are fused into wall points; dark-lumen masks give free-space rays.

All lengths of the returned evidence are in units of the window's reference radius ``R``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.build_from_video import lumen_mask
from colonoworld.tube.camera import Calibration, kb4_radius, rays_from_clip, theta_limit

DEPTH_H, DEPTH_W = 384, 512  # depth grid (0.8 x the 640x480 rectified chart)
RECT_W, RECT_H = 640, 480
SMALL = (276, 208)  # cue grid (clip / 4)
KEY_STEP = 6  # keyframe spacing (decoded frames)
FLOW_GAPS = (3, 6)
FB_MAX = 1.0  # forward-backward flow consistency (px)
SPECULAR = 235
MIN_PARALLAX_DEG = 1.0
Z_RANGE = (0.02, 30.0)
DA_SIZE = (504, 378)  # network input (w, h)
MAX_FLOW_PX = 3.0
MAX_FLOW_FRAC = 0.5
PRIOR_WEIGHT = 0.5
# Pose gate of a keyframe pair (j, j + 3).
GATE = dict(min_fb_ok=0.2, sampson_px=1.5, rot_unsup_deg=2.5, min_inlier=0.5, min_flow_px=1.0)


# ----------------------------------------------------------------------------- frame indices
def staged_indices(timestamps: np.ndarray) -> np.ndarray:
    """Nearest decoded frame for every step of a 30 Hz timeline starting at the first frame."""
    t = np.asarray(timestamps)
    q = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * 30 + 1e-8)) + 1) / 30
    right = np.searchsorted(t, q).clip(0, len(t) - 1)
    left = (right - 1).clip(0, len(t) - 1)
    return np.where(np.abs(t[left] - q) <= np.abs(t[right] - q), left, right)


def window_indices(timestamps: np.ndarray, start: int, frames: int) -> np.ndarray:
    """Decoded frame indices of a window: ``frames`` steps of 1/24 s from 30 Hz step ``start``."""
    ids = [start + (2 * i * 30 + 24 - 1) // (2 * 24) for i in range(frames)]
    return staged_indices(timestamps)[np.asarray(ids, np.int64)]


def window_field(frame0: np.ndarray, cam: Calibration) -> np.ndarray:
    """``(832, 1104)`` content mask of a window from its first full-resolution frame."""
    import cv2

    from colonoworld.tube.build_from_video import content_mask

    m = content_mask(frame0, cam).astype(np.uint8)
    return (
        cv2.resize(m, tuple(cam.get("clip_size", (1104, 832))), interpolation=cv2.INTER_NEAREST) > 0
    )


# ----------------------------------------------------------------------------- rectified chart
def project_to_clip(points: np.ndarray, cam: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points to clip pixels (KB4 forward model, then the native-to-clip affine)."""
    k = np.asarray(cam["K_native"], float)
    d = np.asarray(cam["D"], float)
    a = np.asarray(cam["native_to_clip"], float)
    xy = points[:, :2]
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(r, points[:, 2])
    ok = np.isfinite(theta) & (theta < theta_limit(tuple(d))) & (points[:, 2] > -1e9)
    rd = kb4_radius(theta, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(r[:, None] > 1e-12, xy / r[:, None], 0.0) * rd[:, None]
    native = n @ k[:2, :2].T + k[:2, 2]
    ok &= np.linalg.norm(native - k[:2, 2], axis=1) <= cam.get("field_radius_px", 1e9)
    clip = np.c_[native, np.ones(len(native))] @ a.T
    return clip[:, :2] / clip[:, 2:3], ok


def rectification(cam: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole ``K`` of the 640x480 rectified chart, clip remap coordinates and their validity."""
    kn = np.array(cam["K_native"])
    a = np.array(cam["native_to_clip"])
    d = np.array(cam["D"])
    cw, ch = cam["clip_size"]
    k = np.diag([RECT_W / cw, RECT_H / ch, 1.0]) @ np.array(cam["K_clip"])
    yy, xx = np.mgrid[:RECT_H, :RECT_W]
    xy = np.stack([(xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1]], -1)
    r = np.linalg.norm(xy, axis=-1)
    theta = np.arctan(r)
    rd = theta * (1 + d[0] * theta**2 + d[1] * theta**4 + d[2] * theta**6 + d[3] * theta**8)
    ratio = np.divide(rd, r, out=np.ones_like(r), where=r > 1e-12)
    native = xy * ratio[..., None] * [kn[0, 0], kn[1, 1]] + kn[:2, 2]
    native_h = np.concatenate([native, np.ones((RECT_H, RECT_W, 1))], axis=-1)
    clip = native_h @ a.T
    clip = clip[..., :2] / clip[..., 2, None]
    valid = (
        (theta < theta_limit(tuple(d)))
        & (clip[..., 0] >= 0)
        & (clip[..., 0] < cw - 1)
        & (clip[..., 1] >= 0)
        & (clip[..., 1] < ch - 1)
    )
    valid &= np.linalg.norm(native - kn[:2, 2], axis=-1) <= cam["field_radius_px"]
    return k, clip[..., 0].astype(np.float32), clip[..., 1].astype(np.float32), valid


@functools.lru_cache(maxsize=8)
def _rect_maps(cam_json: str) -> tuple:
    import cv2

    k, mx, my, valid = rectification(json.loads(cam_json))
    support = (
        cv2.resize(valid.astype(np.float32), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA) == 1
    )
    return k, mx, my, valid, support, np.diag([0.8, 0.8, 1.0]) @ k


def rect_maps(cam: Calibration) -> tuple:
    """``(K, map_x, map_y, valid, support, K_depth)`` of the rectified chart (cached)."""
    return _rect_maps(json.dumps(cam, sort_keys=True))


def decode_rectified(video: str | Path, wanted: Any, cam: Calibration) -> dict[int, np.ndarray]:
    """``{frame index: rectified 640x480 RGB}`` for the wanted decoded frames."""
    import av
    import cv2

    _, mx, my, valid, _, _ = rect_maps(cam)
    wanted = sorted(set(int(i) for i in wanted))
    keys, out = set(wanted), {}
    with av.open(str(video)) as container:
        for j, frame in enumerate(container.decode(video=0)):
            if j in keys:
                a = cv2.remap(
                    frame.to_ndarray(format="rgb24"),
                    mx,
                    my,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                a[~valid] = 0
                out[j] = a
            if j >= wanted[-1]:
                break
    return out


def decode_small(
    video: str | Path, wanted: Any, keep_full: Any = ()
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """``({index: 276x208 RGB (area)}, {index: full-resolution RGB})`` for the wanted frames."""
    import av
    import cv2

    wanted = set(int(i) for i in wanted)
    keep = set(int(i) for i in keep_full)
    small, full = {}, {}
    with av.open(str(video)) as container:
        for j, frame in enumerate(container.decode(video=0)):
            if j in wanted:
                a = frame.to_ndarray(format="rgb24")
                small[j] = cv2.resize(a, SMALL, interpolation=cv2.INTER_AREA)
                if j in keep:
                    full[j] = a
            if j >= max(wanted):
                break
    return small, full


def pixel_rays(k_depth: np.ndarray) -> np.ndarray:
    """``(384, 512, 3)`` pinhole rays with unit z of the depth grid."""
    yy, xx = np.mgrid[:DEPTH_H, :DEPTH_W].astype(np.float64)
    return np.stack(
        [
            (xx - k_depth[0, 2]) / k_depth[0, 0],
            (yy - k_depth[1, 2]) / k_depth[1, 1],
            np.ones_like(xx),
        ],
        -1,
    )


def gray(rgb: np.ndarray) -> np.ndarray:
    """Grey 512x384 image of a rectified frame (area resize)."""
    import cv2

    small = cv2.resize(rgb, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _flow_fb(gi: np.ndarray, gj: np.ndarray, dis: Any = None) -> tuple:
    """DIS flow i -> j, target coordinates and forward-backward error."""
    import cv2

    dis = dis or cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(gi, gj, None)
    b = dis.calc(gj, gi, None)
    yy, xx = np.mgrid[: gi.shape[0], : gi.shape[1]].astype(np.float32)
    xj, yj = xx + f[..., 0], yy + f[..., 1]
    bw = cv2.remap(b, xj, yj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)
    return f, xj, yj, np.hypot(f[..., 0] + bw[..., 0], f[..., 1] + bw[..., 1])


def _specular(g: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.dilate((g >= SPECULAR).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def triangulate(
    img_i: np.ndarray,
    img_j: np.ndarray,
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    k_depth: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Two-view depth of view ``i`` (NaN where unreliable) from forward-backward checked flow."""
    import cv2

    gi, gj = gray(img_i), gray(img_j)
    _, xj, yj, fb = _flow_fb(gi, gj)
    ok = support & (fb < FB_MAX) & ~_specular(gi)
    ok &= (xj >= 0) & (xj < DEPTH_W - 1) & (yj >= 0) & (yj < DEPTH_H - 1)
    ok &= cv2.remap(support.astype(np.uint8), xj, yj, cv2.INTER_NEAREST) > 0
    ri = pixel_rays(k_depth)
    rj = np.stack(
        [
            (xj - k_depth[0, 2]) / k_depth[0, 0],
            (yj - k_depth[1, 2]) / k_depth[1, 1],
            np.ones_like(xj),
        ],
        -1,
    ).astype(np.float64)
    t_ji = np.linalg.inv(pose_j) @ pose_i
    rot, t = t_ji[:3, :3], t_ji[:3, 3]
    rri = ri @ rot.T
    c = np.cross(rj, rri)
    d = np.cross(rj, t[None, None])
    z = -(c * d).sum(-1) / np.maximum((c * c).sum(-1), 1e-12)

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    parallax = np.degrees(np.arccos(np.clip((unit(rri) * unit(rj)).sum(-1), -1, 1)))
    ok &= (parallax > MIN_PARALLAX_DEG) & (z > Z_RANGE[0]) & (z < Z_RANGE[1])
    ok &= (rri * z[..., None] + t)[..., 2] > Z_RANGE[0]
    return np.where(ok, z, np.nan)


def fit_affine(disparity: np.ndarray, z_ref: np.ndarray) -> tuple[float, float, int, float] | None:
    """Robust ``a D + b`` disparity fit to a reference depth: ``(a, b, n, median |dz| / z)``."""
    from scipy.optimize import least_squares

    m = np.isfinite(z_ref) & np.isfinite(disparity) & (z_ref > 0)
    n = int(m.sum())
    if n < 20:
        return None
    k = np.random.default_rng(0).choice(n, min(n, 20000), replace=False)
    x, ly = disparity[m][k], np.log(z_ref[m][k])
    dmed = float(np.median(x))

    def fun(p):
        a = np.exp(p[0])
        return -np.log(np.maximum(a * x + p[1] * a * dmed, 1e-9)) - ly

    best = min(
        (
            [la, c]
            for la in np.log(np.geomspace(0.05, 20, 25) / dmed)
            for c in (-0.6, -0.3, 0.0, 0.3, 1.0)
        ),
        key=lambda p: np.sum(np.minimum(np.abs(fun(p)), 3)),
    )
    sol = least_squares(fun, best, loss="soft_l1", f_scale=0.1)
    a = float(np.exp(sol.x[0]))
    b = float(sol.x[1] * a * dmed)
    zf = 1 / np.maximum(a * disparity[m] + b, 1e-9)
    return a, b, n, float(np.median(np.abs(zf - z_ref[m]) / z_ref[m]))


# ----------------------------------------------------------------------------- keyframe depth
def load_depth_anything(code_dir: str | Path, weights: str | Path, threads: int = 6) -> Any:
    """Depth Anything (v1, ViT-L) as distributed with MegaSaM (``code_dir`` holds ``Depth-Anything``
    and ``torchhub``). Returns a callable ``rgb640 -> disparity (384, 512)``."""
    import os
    import sys

    import torch

    torch.set_num_threads(threads)
    code_dir = Path(code_dir)
    sys.path.insert(0, str(code_dir / "Depth-Anything"))
    cwd = os.getcwd()
    os.chdir(code_dir)  # the DINOv2 backbone is loaded from ./torchhub
    try:
        from depth_anything.dpt import DPT_DINOv2

        model = DPT_DINOv2(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], localhub=True
        )
    finally:
        os.chdir(cwd)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.eval()
    return functools.partial(depth_anything_disparity, model=model)


def depth_anything_disparity(rgb640: np.ndarray, model: Any) -> np.ndarray:
    """Relative disparity ``(384, 512)`` of a rectified 640x480 RGB frame."""
    import cv2
    import torch

    x = cv2.resize(rgb640, DA_SIZE, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    with torch.no_grad():
        d = model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()))[0].numpy()
    return cv2.resize(d, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_LINEAR).astype(np.float64)


def _flow_pairs(img_j, neighbours, pose_j, k_depth, support, seed, n_sub=6000):
    """Flow correspondences ``(pixels, targets, T_nj)`` from keyframe ``j`` to each neighbour."""
    import cv2

    gi = gray(img_j)
    rng = np.random.default_rng(seed)
    spec = _specular(gi)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = []
    for img_n, pose_n in neighbours:
        _, xj, yj, fb = _flow_fb(gi, gray(img_n), dis)
        ok = support & ~spec & (fb < FB_MAX)
        ok &= (xj >= 0) & (xj < DEPTH_W - 1) & (yj >= 0) & (yj < DEPTH_H - 1)
        iy, ix = np.nonzero(ok)
        if len(iy) < 200:
            continue
        k = rng.choice(len(iy), min(n_sub, len(iy)), replace=False)
        out.append(
            (
                np.c_[ix[k], iy[k]].astype(np.int64),
                np.c_[xj[iy[k], ix[k]], yj[iy[k], ix[k]]].astype(np.float64),
                np.linalg.inv(pose_n) @ pose_j,
            )
        )
    return out


def _fit_flow(disp, pairs, k_depth, init=None, z_prior=None):
    """``(a, b, n, median px error, rotation-only px error)`` of the flow-consistent alignment."""
    from scipy.optimize import least_squares

    rays, dv, tg, ts = [], [], [], []
    for pix, tgt, t in pairs:
        rays.append(
            np.c_[
                (pix[:, 0] - k_depth[0, 2]) / k_depth[0, 0],
                (pix[:, 1] - k_depth[1, 2]) / k_depth[1, 1],
                np.ones(len(pix)),
            ]
        )
        dv.append(disp[pix[:, 1], pix[:, 0]])
        tg.append(tgt)
        ts.append(t)
    zp = None
    if z_prior is not None:
        iy, ix = np.nonzero(np.isfinite(z_prior))
        if len(iy) > 200:
            k = np.random.default_rng(1).choice(len(iy), min(4000, len(iy)), replace=False)
            zp = (disp[iy[k], ix[k]], z_prior[iy[k], ix[k]])

    def proj(a, b):
        res = []
        for r, d, t, tr in zip(rays, dv, tg, ts, strict=True):
            z = 1 / np.maximum(a * d + b, 1e-6)
            xn = (r * z[:, None]) @ tr[:3, :3].T + tr[:3, 3]
            zz = np.maximum(xn[:, 2], 1e-4)
            res.append(
                np.c_[
                    xn[:, 0] / zz * k_depth[0, 0] + k_depth[0, 2] - t[:, 0],
                    xn[:, 1] / zz * k_depth[1, 1] + k_depth[1, 2] - t[:, 1],
                ].ravel()
            )
        return res

    dmed = float(np.median(np.concatenate(dv)))

    def fun(p):
        a = np.exp(p[0])
        b = p[1] * a * dmed
        res = proj(a, b)
        if zp is not None:
            z = 1 / np.maximum(a * zp[0] + b, 1e-6)
            res.append(PRIOR_WEIGHT * 20 * np.log(z / zp[1]))
        return np.concatenate(res)

    if init is None:
        best = None
        for la in np.log(np.geomspace(0.05, 20, 25) / dmed):
            for c in (-0.6, -0.3, 0.0, 0.3, 1.0):
                v = np.sum(np.minimum(np.abs(fun([la, c])), 30))
                if best is None or v < best[0]:
                    best = (v, [la, c])
        x0 = best[1]
    else:
        x0 = [np.log(max(init[0], 1e-9)), init[1] / (max(init[0], 1e-9) * dmed)]
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=1.5, max_nfev=200)
    a = float(np.exp(sol.x[0]))
    b = float(sol.x[1] * a * dmed)

    def median_px(a_, b_):
        return float(
            np.median(np.concatenate([np.hypot(*v.reshape(-1, 2).T) for v in proj(a_, b_)]))
        )

    n = int(sum(len(v) // 2 for v in proj(a, b)))
    return a, b, n, median_px(a, b), median_px(1e-12, 1e-9)


def _reproject_previous(z_prev, pose_prev, pose_k, k_depth):
    """Previous aligned keyframe depth re-projected into keyframe ``k`` (nearest, z-buffer)."""
    ok = np.isfinite(z_prev)
    x = pixel_rays(k_depth)[ok] * z_prev[ok][:, None]
    t = np.linalg.inv(pose_k) @ pose_prev
    xk = x @ t[:3, :3].T + t[:3, 3]
    xk = xk[xk[:, 2] > Z_RANGE[0]]
    u = np.round(xk[:, 0] / xk[:, 2] * k_depth[0, 0] + k_depth[0, 2]).astype(int)
    v = np.round(xk[:, 1] / xk[:, 2] * k_depth[1, 1] + k_depth[1, 2]).astype(int)
    inb = (u >= 0) & (u < DEPTH_W) & (v >= 0) & (v < DEPTH_H)
    out = np.full((DEPTH_H, DEPTH_W), np.inf)
    np.minimum.at(out, (v[inb], u[inb]), xk[inb, 2])
    return np.where(np.isfinite(out), out, np.nan)


def _to_depth(disp, a, b, support):
    d = a * disp + b
    z = np.where(support & (d > 1 / Z_RANGE[1]), 1 / np.maximum(d, 1e-9), np.nan)
    return np.where(z > Z_RANGE[0], z, np.nan)


def keyframe_depth(
    video: str | Path,
    cam: Calibration,
    poses: np.ndarray,
    disparity_fn: Callable[[np.ndarray], np.ndarray],
    step: int = KEY_STEP,
) -> dict[str, Any]:
    """Aligned depth of decoded frames ``0, step, 2 step, ...`` of a clip.

    ``poses``: ``(N, 4, 4)`` camera-to-world of every decoded frame. Returns ``idx``, ``P``,
    ``depth`` ``(n, 384, 512)`` float16 in pose units (NaN = unused), ``ok``, ``Kd``, ``support``
    and the per-keyframe alignment records.
    """
    n_vid = len(poses)
    keys = list(range(0, n_vid, step))
    need = set(keys)
    for j in keys:
        for g in FLOW_GAPS:
            need |= {jj for jj in (j + g, j - g) if 0 <= jj < n_vid}
    imgs = decode_rectified(video, need, cam)
    _, _, _, _, support, k_depth = rect_maps(cam)
    depth = np.full((len(keys), DEPTH_H, DEPTH_W), np.nan, np.float32)
    ok = np.zeros(len(keys), bool)
    records = []
    z_prev = pose_prev = ab_prev = None
    for i, j in enumerate(keys):
        disp = disparity_fn(imgs[j])
        rec: dict[str, Any] = dict(j=int(j))
        nbrs = [
            (imgs[jj], poses[jj])
            for g in FLOW_GAPS
            for jj in (j + g, j - g)
            if 0 <= jj < n_vid and jj in imgs
        ]
        pairs = _flow_pairs(imgs[j], nbrs, poses[j], k_depth, support, seed=j)
        zp = (
            _reproject_previous(z_prev, pose_prev, poses[j], k_depth)
            if z_prev is not None
            else None
        )
        fit = _fit_flow(disp, pairs, k_depth, init=ab_prev, z_prior=zp) if pairs else None
        if fit is not None:
            a, b, n, epx, erot = fit
            rec.update(a=a, b=b, n=n, flow_med_px=epx, rot_only_px=erot)
        if (
            fit is not None
            and a > 0
            and epx <= MAX_FLOW_PX
            and epx <= MAX_FLOW_FRAC * max(erot, 1e-6)
        ):
            z = _to_depth(disp, a, b, support)
            depth[i] = z
            ok[i] = True
            z_prev, pose_prev, ab_prev = z, poses[j], (a, b)
        records.append(rec)
    return dict(
        idx=np.array(keys),
        P=poses[keys],
        depth=depth.astype(np.float16),
        ok=ok,
        Kd=k_depth,
        support=support,
        align=records,
    )


# ----------------------------------------------------------------------------- pose gate
def _rot_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))


def pose_pair_accepted(
    gi: np.ndarray,
    gj: np.ndarray,
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    k_depth: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
) -> tuple[bool, dict[str, float]]:
    """Whether the images of a frame pair agree with the relative pose.

    Accepted when at least 20 % of the field has consistent flow, the median Sampson distance of
    the flow to the pose's epipolar geometry is at most 1.5 px and the rotation recovered from
    the images (essential matrix, or a homography when it explains the flow clearly better)
    differs from the pose rotation by at most 2.5 degrees (no verdict on near-static pairs).
    """
    import cv2

    f, xj, yj, fb = _flow_fb(gi, gj)
    h, w = gi.shape
    ok = support & ~_specular(gi) & (fb < 1.0) & (xj >= 0) & (xj < w - 1) & (yj >= 0)
    ok &= yj < h - 1
    t_ji = np.linalg.inv(pose_j) @ pose_i
    rot, t = t_ji[:3, :3], t_ji[:3, 3]
    flow_med = float(np.median(np.hypot(f[..., 0], f[..., 1])[support]))
    fb_ok = float(ok.sum() / max(support.sum(), 1))
    if ok.sum() < 200:
        return False, dict(fb_ok_frac=fb_ok, sampson_px=np.nan, rot_unsup_deg=np.nan)
    iy, ix = np.nonzero(ok)
    k = rng.choice(len(iy), min(6000, len(iy)), replace=False)
    p1 = np.c_[ix[k], iy[k]].astype(np.float64)
    p2 = np.c_[xj[iy[k], ix[k]], yj[iy[k], ix[k]]].astype(np.float64)
    ki = np.linalg.inv(k_depth)
    tn = t / max(np.linalg.norm(t), 1e-12)
    tx = np.array([[0, -tn[2], tn[1]], [tn[2], 0, -tn[0]], [-tn[1], tn[0], 0]])
    fm = ki.T @ (tx @ rot) @ ki
    h1, h2 = np.c_[p1, np.ones(len(p1))], np.c_[p2, np.ones(len(p2))]
    fx1, ftx2 = h1 @ fm.T, h2 @ fm
    num = (h2 * fx1).sum(1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    sampson = float(np.median(np.sqrt(num / np.maximum(den, 1e-12))))
    rot_unsup = np.nan
    inlier = np.nan
    e, inl = cv2.findEssentialMat(p1, p2, k_depth, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if e is not None and e.shape == (3, 3) and inl is not None:
        inlier = float(inl.mean())
        _, r_img, _, _ = cv2.recoverPose(e, p1, p2, k_depth, mask=inl.copy())
        if inlier >= GATE["min_inlier"] and flow_med >= GATE["min_flow_px"]:
            rot_unsup = _rot_deg(r_img @ rot.T)
    hm, hin = cv2.findHomography(p1, p2, cv2.RANSAC, 1.0)
    if hm is not None:
        rh = ki @ hm @ k_depth
        u, _, vt = np.linalg.svd(rh)
        rh = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        hf = float(hin.mean())
        if hf > (inlier if np.isfinite(inlier) else 0) + 0.1 and flow_med >= GATE["min_flow_px"]:
            rot_unsup = _rot_deg(rh @ rot.T)
    accepted = bool(
        fb_ok >= GATE["min_fb_ok"]
        and np.isfinite(sampson)
        and sampson <= GATE["sampson_px"]
        and (not np.isfinite(rot_unsup) or rot_unsup <= GATE["rot_unsup_deg"])
    )
    return accepted, dict(fb_ok_frac=fb_ok, sampson_px=sampson, rot_unsup_deg=rot_unsup)


# ----------------------------------------------------------------------------- wall evidence
def small_field(field: np.ndarray) -> np.ndarray:
    """Cue-grid (276x208) field mask, nearest."""
    import cv2

    return cv2.resize(field.astype(np.uint8), SMALL, interpolation=cv2.INTER_NEAREST) > 0


def rect_lumen(small: np.ndarray, cam: Calibration, field_small: np.ndarray) -> np.ndarray:
    """Dark-lumen mask of a cue-grid frame, dilated 9x9 and mapped to the rectified depth grid."""
    import cv2

    m = lumen_mask(small, cam, field_small)
    _, mx, my, _, _, _ = rect_maps(cam)
    x = cv2.resize((mx + 0.5) / 4 - 0.5, (DEPTH_W, DEPTH_H))
    y = cv2.resize((my + 0.5) / 4 - 0.5, (DEPTH_W, DEPTH_H))
    lm = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))
    return cv2.remap(lm, x, y, cv2.INTER_NEAREST) > 0


def _two_view_depth(imgs, j, neighbours, poses, k_depth, support):
    """Triangulated depth of keyframe ``j``: up to two neighbours, epipolar-checked, fused."""
    import cv2

    depths = []
    for jj in neighbours:
        if jj not in imgs:
            continue
        z = triangulate(imgs[j], imgs[jj], poses[j], poses[jj], k_depth, support)
        # Reprojection check of the depth against independently recomputed flow.
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f = dis.calc(gray(imgs[j]), gray(imgs[jj]), None)
        yy, xx = np.mgrid[:DEPTH_H, :DEPTH_W]
        t = np.linalg.inv(poses[jj]) @ poses[j]
        x = (pixel_rays(k_depth) * np.nan_to_num(z, nan=0)[..., None]) @ t[:3, :3].T + t[:3, 3]
        zz = np.maximum(x[..., 2], 1e-8)
        ex = x[..., 0] / zz * k_depth[0, 0] + k_depth[0, 2] - (xx + f[..., 0])
        ey = x[..., 1] / zz * k_depth[1, 1] + k_depth[1, 2] - (yy + f[..., 1])
        depths.append(np.where(np.hypot(ex, ey) < 1.5, z, np.nan))
    if not depths:
        return np.full((DEPTH_H, DEPTH_W), np.nan, np.float32)
    if len(depths) == 1:
        return depths[0].astype(np.float32)
    a, b = depths[:2]
    both = np.isfinite(a) & np.isfinite(b)
    agree = np.abs(a - b) / np.maximum(0.5 * (a + b), 1e-8) < 0.4
    z = np.where(both, np.where(agree, 0.5 * (a + b), np.nan), np.where(np.isfinite(a), a, b))
    return z.astype(np.float32)


def collect_evidence(
    video: str | Path,
    cam: Calibration,
    poses: np.ndarray,
    video_indices: np.ndarray,
    keyframes: dict[str, Any],
    radius: float,
    field: np.ndarray,
) -> dict[str, Any]:
    """Fused wall points and free-space lumen rays of a window.

    ``poses``: camera-to-world of every decoded frame; ``video_indices``: the window's decoded
    frames (the reconstruction timeline); ``keyframes``: output of :func:`keyframe_depth`;
    ``radius``: reference radius ``R`` (pose units); ``field``: ``(832, 1104)`` content mask.
    Returns the window poses ``P`` (rebased to the first window frame), wall observations ``ev``
    (one per accepted keyframe) and per-frame lumen constraints ``times``.
    """
    import cv2

    vi = np.asarray(video_indices)
    p_clip = poses[vi].astype(float)
    inv0 = np.linalg.inv(p_clip[0])
    p_win = inv0[None] @ p_clip
    idx = keyframes["idx"]
    mono = keyframes["depth"].astype(np.float32)
    mono_ok = keyframes["ok"]
    k_depth, support = keyframes["Kd"], keyframes["support"]
    n_vid = len(poses)
    small, _ = decode_small(video, set(vi.tolist()))
    keys = [
        int(j)
        for j in idx
        if vi.min() - 12 <= j <= vi.max() + 12 and (j // 6) % 2 == 0 and j + 3 < n_vid
    ]
    need = set(keys) | {jj for j in keys for jj in (j - 6, j + 3, j + 6) if 0 <= jj < n_vid}
    imgs = decode_rectified(video, need, cam)
    missing = set(keys) - set(small)
    if missing:
        small.update(decode_small(video, missing)[0])
    fs = small_field(field)
    rays = pixel_rays(k_depth)
    rng = np.random.default_rng(0)
    r = radius
    records, ev = [], []
    for j in keys:
        if j not in imgs or j + 3 not in imgs:
            continue
        accepted, gate = pose_pair_accepted(
            gray(imgs[j]), gray(imgs[j + 3]), poses[j], poses[j + 3], k_depth, support, rng
        )
        rec: dict[str, Any] = dict(j=j, pose_accepted=accepted, pose_pair=gate)
        records.append(rec)
        if not accepted:
            continue
        k = list(idx).index(j)
        lm = rect_lumen(small[j], cam, fs)
        tri = _two_view_depth(
            imgs, j, [jj for jj in (j - 6, j + 6) if jj in imgs], poses, k_depth, support
        )
        z = mono[k].copy() if mono_ok[k] else np.full_like(tri, np.nan)
        m = np.isfinite(tri) & np.isfinite(z) & ~lm & support
        alignment = None
        if m.sum() >= 300:
            fit = fit_affine(
                1 / np.maximum(np.nan_to_num(z, nan=1), 1e-8), np.where(m, tri, np.nan)
            )
            if fit is not None:
                a, b, _, err = fit
                new = 1 / np.maximum(a / np.maximum(z, 1e-8) + b, 1e-8)
                ratio = float(np.nanmedian(new[m] / z[m]))
                alignment = dict(scale_ratio=ratio, accepted=bool(err < 0.35 and 0.4 < ratio < 2.5))
                if alignment["accepted"]:
                    z = new.astype(np.float32)
        mt = np.isfinite(tri)
        agree = mt & np.isfinite(z) & (np.abs(tri - z) / np.maximum(tri, 1e-8) < 0.4)
        z = np.where(agree, 0.7 * tri + 0.3 * z, np.where(mt, tri, z))
        valid = np.isfinite(z) & support & ~lm & ~_specular(gray(imgs[j]))
        valid &= (z > 0.03 * r) & (z < 6 * r)
        if valid.any():
            valid &= z < 3 * np.median(z[valid])
        if valid.sum() < 200:
            rec.update(depth_accepted=False)
            continue
        cutoff = max(2 * np.percentile(z[valid], 70), 2 * r)
        zl = z[lm & support & np.isfinite(z) & (z < 6 * r)]
        if len(zl):
            cutoff = max(cutoff, float(np.percentile(zl, 80)))
        cutoff = min(cutoff, 6 * r)
        q = inv0 @ poses[j]
        sample = valid[::4, ::4]
        pts = (rays[::4, ::4][sample] * z[::4, ::4][sample, None]) @ q[:3, :3].T + q[:3, 3]
        rec.update(depth_accepted=True, alignment=alignment)
        ev.append(
            dict(
                j=j,
                Q=q,
                z=z / r,
                valid=valid,
                lumen=lm,
                cutoff=cutoff / r,
                points=pts / r,
                record=rec,
            )
        )
    if not ev:
        raise RuntimeError("no pose-gated depth observations in the window")
    # Lumen masks at every window frame; frames far from an accepted pose key carve nothing.
    times = []
    for t, j in enumerate(vi):
        nearest = min(records, key=lambda rr: abs(rr["j"] - j))
        accepted = nearest["pose_accepted"] and abs(nearest["j"] - j) <= 8
        cutoff = min(ev, key=lambda e: abs(e["j"] - j))["cutoff"]
        times.append(dict(t=t, j=int(j), Q=p_win[t], cutoff=cutoff, accepted=accepted))
    # Free-space lumen rays in the original fisheye camera (cue grid, dilated 9x9).
    domain = []
    for e in ev + times:
        m = lumen_mask(small[e["j"]], cam, fs)
        e["small_lumen"] = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if not e.get("accepted", True):
            continue
        y, x = np.nonzero(e["small_lumen"])
        uv = np.c_[(x + 0.5) * 4 - 0.5, (y + 0.5) * 4 - 0.5]
        rr, ok = rays_from_clip(uv, cam)
        ok &= rr[:, 2] > 0.05
        if ok.any():
            end = rr[ok] * (e["cutoff"] / rr[ok][:, 2])[:, None]
            domain.append(end @ e["Q"][:3, :3].T + e["Q"][:3, 3] / r)
    return dict(
        P=p_win,
        video_indices=vi,
        R=r,
        field=field,
        cam=cam,
        Kd=k_depth,
        ev=ev,
        times=times,
        domain_points=domain,
        records=records,
    )
