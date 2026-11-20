# SPDX-License-Identifier: CC-BY-NC-4.0
"""Training windows, quality gates and the joint-conditioning sampler.

Every training window is one row of a JSONL manifest (written by ``scripts/prepare_data.py``)::

    id, frames, video, tube_video, camera (npz: poses [T,4,4], field_mask, calibration json),
    conditions (npz with bit-packed condition masks, optional), strata (list), excluded, score,
    forward, segments (per-frame colon segment or null), polyp, tool, snare, water, nbi,
    light_labelled, light_unknown_frames, flags

Each draw of a training update is fully determined by ``(seed, update, draw)`` through SHA-256
based uniform numbers, so any rank can reproduce any draw without communication.

Sampling (one draw):

1. pick a stratum with weights clean .30 / polyp .25 / irrigation .15 / tool .15 / NBI .15
   (renormalised over non-empty pools);
2. pick a window in the stratum (clean windows are weighted x2 for the top-30% lateral/rotation
   tube motion score and x4 for forward-moving windows; NBI windows that change light between the
   first and last frame are weighted x8, other light-switch windows x2);
3. for the polyp, tool and irrigation strata an "entry" draw forces a playback direction in which
   the object is absent in frame 0 and appears later;
4. choose, per condition class, whether the mask stream carries the mask, an explicit empty mask,
   or nothing (condition dropout), and build the per-latent-frame global flags.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Window = dict[str, Any]

HEIGHT, WIDTH = 832, 1104
FPS = 24
STRATA = ("clean", "polyp", "irrigation", "tool", "nbi")
MASK_CLASSES = ("polyp", "tool", "water", "snare")
GLOBAL_CLASSES = ("polyp", "tool", "water", "blood", "snare")
FRAME_FLAGS = ("special_light", "dye", "tool", "irrigation")
GLOBAL_IN = 2 * len(FRAME_FLAGS) + len(GLOBAL_CLASSES)
SEGMENTS = ("cecum", "ascending", "transverse", "descending", "sigmoid", "rectum")
SEGMENT_NULL = len(SEGMENTS)
BASE_CAPTION = "Egocentric colonoscopy video showing the colon lumen and mucosal surface."
CLEAN_FLAGS = ("dye", "bubbles", "irrigation")


# --------------------------------------------------------------------------------------------
# Deterministic uniform numbers
# --------------------------------------------------------------------------------------------


def _sha(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def joint_u01(tag: str, *key: Any, namespace: str = "joint-v2") -> float:
    """Uniform [0, 1) from SHA-256 of ``namespace:tag:key...`` (53-bit mantissa)."""
    digest = _sha(namespace + ":" + tag + ":" + ":".join(map(str, key)))
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def joint_subseed(tag: str, *key: Any, namespace: str = "joint-v2") -> str:
    """Hex sub-seed of a draw, used to key the NBI and irrigation window pickers."""
    return _sha(namespace + ":" + tag + ":" + ":".join(map(str, key))).hex()


def nbi_u01(tag: str, *key: Any) -> float:
    digest = _sha("nbi-v1:" + tag + ":" + ":".join(map(str, key)))
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def irrigation_u01(tag: str, *key: Any) -> float:
    digest = _sha(json.dumps([tag, *key]))
    return int.from_bytes(digest[:8], "big") / 2**64


def dropout_kind(seed: int, update: int, micro: int = 0, sample: int = 0) -> str:
    """Tube/camera condition dropout of one microbatch: full .85, null_tube .10, null_both .05.

    ``null_tube`` removes the tube stream, ``null_both`` additionally zeroes the camera rays.
    The draw depends only on the update (and microbatch index), so all ranks agree.
    """
    digest = _sha(f"{seed}:{update}:{micro}:{sample}")
    u = int.from_bytes(digest[:8], "little") / 2**64
    return "full" if u < 0.85 else "null_tube" if u < 0.95 else "null_both"


def segment_null_draw(seed: int, update: int, draw: int, probability: float = 0.10) -> bool:
    """True when the colon-segment embedding of this draw is replaced by the null class."""
    digest = _sha(f"segment-embedding-null-v1:{seed}:{update}:{draw}")
    return int.from_bytes(digest[:8], "big") / 2**64 < probability


def caption_generic_draw(seed: int, update: int, draw: int) -> bool:
    """True (10%) when the caption omits the colon-segment clause."""
    digest = _sha(f"segment-caption-v1:{seed}:{update}:{draw}")
    return int.from_bytes(digest[:8], "big") < (2**64) // 10


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def read_manifest(path: str | Path) -> list[Window]:
    """Read a JSONL training manifest; relative paths resolve against the manifest folder."""
    path = Path(path)
    windows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("video", "tube_video", "camera", "conditions", "tube_latent"):
            if row.get(key) and not Path(row[key]).is_absolute():
                row[key] = str(path.parent / row[key])
        windows.append(row)
    ids = [w["id"] for w in windows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate window ids in manifest")
    return windows


def write_manifest(path: str | Path, windows: Iterable[Window]) -> None:
    Path(path).write_text("".join(json.dumps(w, sort_keys=True) + "\n" for w in windows))


# --------------------------------------------------------------------------------------------
# Window selection criteria (used by scripts/prepare_data.py)
# --------------------------------------------------------------------------------------------


@dataclass
class QualityGates:
    """Thresholds for admitting a window to training (defaults of the released model)."""

    max_blocked_fraction: float = 0.20
    min_valid_fraction: float = 0.4
    min_block_cos: float = 0.35
    min_radial_r: float = 0.4
    magnitude_ratio: tuple[float, float] = (0.4, 2.5)
    static_fraction: float = 0.8
    static_p90_flow: float = 0.35


def tube_fit_ok(fit: dict[str, Any], frames: int, gates: QualityGates) -> bool:
    """Tube wall-fit gate: lumen opening and camera inside the tube in every frame."""
    return (
        fit["blocked_fraction"] <= gates.max_blocked_fraction
        and fit["missing_lumen_depth_pixels"] == 0
        and fit["single_opening_frames"] == frames
        and fit["multi_lumen_frames"] == 0
        and fit["inside_frames"] == frames
        and fit["wall_cross_frames"] == 0
    )


def tube_agreement_ok(score: dict[str, float], gates: QualityGates) -> bool:
    """Real-video vs rendered-tube optical-flow agreement gate.

    Moving windows need direction agreement (block cosine), radial agreement and a matching
    flow magnitude; nearly static windows need both videos to be static.
    """
    keys = (
        "valid_fraction_min",
        "block_cos",
        "radial_r",
        "magnitude_ratio",
        "both_static_fraction",
        "real_p90_flow",
        "tube_p90_flow",
    )
    if score.get("error") or any(not math.isfinite(float(score.get(k, "nan"))) for k in keys):
        return False
    lo, hi = gates.magnitude_ratio
    moving = (
        score["block_cos"] >= gates.min_block_cos
        and score["radial_r"] >= gates.min_radial_r
        and lo <= score["magnitude_ratio"] <= hi
    )
    static = (
        score["both_static_fraction"] >= gates.static_fraction
        and score["real_p90_flow"] <= gates.static_p90_flow
        and score["tube_p90_flow"] <= gates.static_p90_flow
    )
    return score["valid_fraction_min"] >= gates.min_valid_fraction and (moving or static)


FLOW_SIZE = (160, 120)
FARNEBACK = (0.5, 3, 21, 3, 5, 1.2, 0)
FLOW_ACTIVE = 0.1  # px per 4 frames at 160x120: block counts as moving
FLOW_STATIC = 0.2  # px per 4 frames at 160x120: interval counts as static


def _flow_field(frame: np.ndarray, cam: dict[str, Any], frames: Sequence[np.ndarray]) -> np.ndarray:
    """Endoscope field at flow resolution minus the edge-connected black device border."""
    import cv2

    from colonoworld.models.camera import unproject_kb4

    w, h = FLOW_SIZE
    yy, xx = np.mgrid[:h, :w]
    cw, ch = cam["clip_size"]
    uv = np.stack([(xx + 0.5) * cw / w - 0.5, (yy + 0.5) * ch / h - 0.5], -1)
    _, valid = unproject_kb4(uv.reshape(-1, 2), cam)
    mask = valid.reshape(h, w)
    brightest = np.max([f.max(-1) for f in (frames[0], frames[len(frames) // 2], frames[-1])], 0)
    dark = (brightest < 14).astype(np.uint8)
    _, lab = cv2.connectedComponents(dark, connectivity=4)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    border = np.isin(lab, list(edge)) & (dark > 0)
    border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return cv2.erode((mask & ~border).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _photometric_mask(rgb: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Field minus specular highlights and the darkest 2% (flow is unreliable there)."""
    import cv2

    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    spec = (hsv[..., 2] > 0.95) & (hsv[..., 1] < 0.15)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bad = spec | (gray <= np.percentile(gray[field], 2))
    return field & ~(cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)


def _similarity(flow: np.ndarray, mask: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    """Least-squares (tx, ty, radial expansion, in-plane rotation) of a flow field."""
    w, h = FLOW_SIZE
    yy, xx = np.mgrid[:h, :w]
    k = np.asarray(cam["K_clip"])
    cw, ch = cam["clip_size"]
    x = (((xx + 0.5) - (k[0, 2] + 0.5) * w / cw) / 80)[mask]
    y = (((yy + 0.5) - (k[1, 2] + 0.5) * h / ch) / 80)[mask]
    u, v = flow[mask].T
    one, zero = np.ones(len(x)), np.zeros(len(x))
    design = np.concatenate([np.stack([one, zero, x, -y], 1), np.stack([zero, one, y, x], 1)])
    return np.linalg.lstsq(design, np.r_[u, v], rcond=None)[0]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-4 or np.std(b[ok]) < 1e-4:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def flow_agreement(
    real: Sequence[np.ndarray], tube: Sequence[np.ndarray], cam: dict[str, Any]
) -> dict[str, float]:
    """Optical-flow agreement of a real window and its rendered tube.

    ``real``/``tube`` are RGB frames 0, 4, 8, ... resized to 160x120 (``cv2.INTER_AREA``).
    Returns the block direction cosine, magnitude ratio, static fractions, per-motion-component
    correlations (``radial_r`` etc.) and the tube motion energies used for the motion score.
    """
    import cv2

    field = _flow_field(real[0], cam, real)
    if field.mean() <= 0.2:
        raise ValueError("insufficient endoscope field")
    masks = [[_photometric_mask(f, field) for f in seq] for seq in (real, tube)]
    grays = [[cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq] for seq in (real, tube)]
    cosines, ratios, real_coeff, tube_coeff, mags, coverage = [], [], [], [], [], []
    h, w = FLOW_SIZE[1], FLOW_SIZE[0]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for t in range(len(real) - 1):
        flows = [cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, *FARNEBACK) for g in grays]
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
                a, b = [np.median(f[sl][m], axis=0) for f in flows]
                ma, mb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if max(ma, mb) >= FLOW_ACTIVE:
                    c = float(np.dot(a, b) / (ma * mb)) if min(ma, mb) >= FLOW_ACTIVE else 0.0
                    cosines.append(float(np.clip(c, -1, 1)))
                if ma >= FLOW_ACTIVE:
                    ratios.append(mb / ma)
    rc, tc, m = np.asarray(real_coeff), np.asarray(tube_coeff), np.asarray(mags)
    out = dict(
        block_cos=float(np.mean(cosines)) if cosines else float("nan"),
        magnitude_ratio=float(np.median(ratios)) if ratios else float("nan"),
        valid_fraction_min=float(np.min(coverage)),
        real_p90_flow=float(np.percentile(m[:, 0], 90)),
        tube_p90_flow=float(np.percentile(m[:, 1], 90)),
        both_static_fraction=float(np.mean((m < FLOW_STATIC).all(1))),
    )
    for j, name in enumerate(("horizontal", "vertical", "radial", "rotation")):
        out[name + "_r"] = _pearson(rc[:, j], tc[:, j])
        out["tube_" + name + "_rms"] = float(np.sqrt(np.mean(tc[:, j] ** 2)))
    return out


def motion_energies(score: dict[str, float]) -> tuple[float, float, float]:
    """(lateral, rotation, radial) tube motion energies from :func:`flow_agreement`."""
    lateral = score["tube_horizontal_rms"] ** 2 + score["tube_vertical_rms"] ** 2
    return lateral, score["tube_rotation_rms"] ** 2, score["tube_radial_rms"] ** 2


MOTION_CLASSES = ("fwd", "bwd", "pan", "yaw", "pitch", "stationary")


def _motion_grid() -> tuple[np.ndarray, np.ndarray]:
    w, h = FLOW_SIZE
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    return yy, xx


def _project_kb4(x: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    """Project camera-frame points to the 160x120 motion grid (KB4 fisheye, clip intrinsics)."""
    w, h = FLOW_SIZE
    rho = np.linalg.norm(x[..., :2], axis=-1)
    theta = np.arctan2(rho, x[..., 2])
    t2 = theta * theta
    d = cam["D"]
    td = theta * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
    k = np.asarray(cam["K_clip"])
    fac = td / np.maximum(rho, 1e-12)
    return np.stack(
        [
            (x[..., 0] * fac * k[0, 0] + k[0, 2] + 0.5) * w / WIDTH - 0.5,
            (x[..., 1] * fac * k[1, 1] + k[1, 2] + 0.5) * h / HEIGHT - 0.5,
        ],
        -1,
    )


def _motion_rays(cam: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Unit rays and validity of the 160x120 motion grid."""
    w, h = FLOW_SIZE
    yy, xx = _motion_grid()
    k = np.asarray(cam["K_clip"])
    nx = ((xx + 0.5) * WIDTH / w - 0.5 - k[0, 2]) / k[0, 0]
    ny = ((yy + 0.5) * HEIGHT / h - 0.5 - k[1, 2]) / k[1, 1]
    rd = np.hypot(nx, ny)
    d = cam["D"]
    roots = np.polynomial.polynomial.polyroots(
        [1, 0, 3 * d[0], 0, 5 * d[1], 0, 7 * d[2], 0, 9 * d[3]]
    )
    limit = (
        min(
            [np.pi / 2 - 1e-7]
            + [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < np.pi / 2]
        )
        - 1e-9
    )

    def radius(th):
        return th * (1 + d[0] * th**2 + d[1] * th**4 + d[2] * th**6 + d[3] * th**8)

    valid = rd < radius(limit)
    lo, hi = np.zeros_like(rd), np.full_like(rd, limit)
    for _ in range(45):
        mid = (lo + hi) / 2
        less = radius(mid) < rd
        lo, hi = np.where(less, mid, lo), np.where(less, hi, mid)
    th = (lo + hi) / 2
    s = np.sin(th) / np.maximum(rd, 1e-12)
    return np.stack([nx * s, ny * s, np.cos(th)], -1), valid


def _geometric_flow(ray: np.ndarray, depth: np.ndarray, T: np.ndarray, cam: dict[str, Any]):
    yy, xx = _motion_grid()
    x = ray * depth[..., None]
    x = x @ T[:3, :3].T + T[:3, 3]
    f = _project_kb4(x, cam) - np.stack([xx, yy], -1)
    ok = np.isfinite(depth) & (depth > 0) & (x[..., 2] > 0) & np.isfinite(f).all(-1)
    return np.nan_to_num(f).astype(np.float32), ok


def _resize_depth(depth: np.ndarray) -> np.ndarray:
    """Stride-8 tube depth (104x138) -> 160x120 grid, ignoring invalid depth."""
    import cv2

    w, h = FLOW_SIZE
    yy, xx = _motion_grid()
    mx = (((xx + 0.5) * WIDTH / w - 0.5) / 8).astype(np.float32)
    my = (((yy + 0.5) * HEIGHT / h - 0.5) / 8).astype(np.float32)
    ok = np.isfinite(depth) & (depth > 0)
    num = cv2.remap(
        np.where(ok, depth, 0).astype(np.float32),
        mx,
        my,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    den = cv2.remap(
        ok.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return np.where(den > 0.99, num / np.maximum(den, 1e-9), np.nan)


def interval_motion(
    ray: np.ndarray,
    field: np.ndarray,
    depth: np.ndarray,
    poses: np.ndarray,
    t: int,
    j: int,
    cam: dict[str, Any],
) -> tuple[str, float]:
    """Motion class of the camera between frames t and j, and its mean geometric flow.

    The tube depth at frame t is moved by the relative pose: stationary if the median flow is
    below 0.15 px (160x120); rotation if the rotation-only flow dominates the translation part
    (yaw if |w_y| >= |w_x|, else pitch); otherwise pan for mostly lateral translation, else
    forward (fwd) or backward (bwd) along the optical axis.
    """
    import cv2

    D_ = _resize_depth(depth[t].astype(np.float32))
    T = np.linalg.inv(poses[j]) @ poses[t]
    fg, ok = _geometric_flow(ray, D_, T, cam)
    ok &= field
    gm = np.linalg.norm(fg, axis=-1)
    rot = T.copy()
    rot[:3, 3] = 0
    frot, _ = _geometric_flow(ray, np.ones(FLOW_SIZE[::-1]), rot, cam)
    ftrans = fg - frot
    dc = np.linalg.inv(poses[t]) @ poses[j]
    tr = dc[:3, 3]
    w = cv2.Rodrigues(np.ascontiguousarray(dc[:3, :3]))[0].ravel()
    if not ok.any() or np.median(gm[ok]) < 0.15:
        label = "stationary"
    elif np.mean(np.linalg.norm(frot[ok], axis=-1)) > np.mean(np.linalg.norm(ftrans[ok], axis=-1)):
        label = "yaw" if abs(w[1]) >= abs(w[0]) else "pitch"
    else:
        label = "pan" if np.linalg.norm(tr[:2]) > abs(tr[2]) else ("fwd" if tr[2] > 0 else "bwd")
    return label, float(gm[ok].mean()) if ok.any() else 0.0


def motion_class(
    poses: np.ndarray, depth: np.ndarray, field_mask: np.ndarray, cam: dict[str, Any]
) -> str:
    """Window motion class: majority of the 4-frame interval classes (ties: larger flow).

    ``poses`` [T,4,4] camera-to-world, ``depth`` [T,104,138] stride-8 tube depth (from
    ``render_tube_video(..., return_depth=True)``), ``field_mask`` [832,1104] endoscope field.
    Clean windows of class ``"fwd"`` get the forward sampling weight.
    """
    import cv2

    field = cv2.resize(field_mask.astype(np.uint8), FLOW_SIZE, interpolation=cv2.INTER_NEAREST) > 0
    ray, valid = _motion_rays(cam)
    field &= valid
    counts: dict[str, int] = {}
    mags: dict[str, float] = {}
    for t in range(0, len(poses) - 4, 4):
        label, mag = interval_motion(ray, field, depth, poses, t, t + 4, cam)
        counts[label] = counts.get(label, 0) + 1
        mags[label] = mags.get(label, 0.0) + mag
    return sorted(counts, key=lambda k: (-counts[k], -mags[k], MOTION_CLASSES.index(k)))[0]


def pose_ok(poses: np.ndarray, valid: np.ndarray | None = None) -> bool:
    """Camera-estimate gate: finite rigid poses for every frame."""
    poses = np.asarray(poses, dtype=np.float64)
    if not np.isfinite(poses).all() or (valid is not None and not np.asarray(valid).all()):
        return False
    rot = poses[:, :3, :3]
    return bool(
        np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-5)
        and np.allclose(np.linalg.det(rot), 1.0, atol=1e-5)
    )


def motion_score(lateral: float, rotation: float, radial: float) -> float:
    """Share of lateral and rotational tube motion energy; the top 30% are drawn twice."""
    return (lateral + rotation) / max(lateral + rotation + radial, 1e-12)


def mark_top_motion(windows: list[Window], fraction: float = 0.3) -> None:
    """Set ``top_motion`` on the ``ceil(fraction * N)`` highest-scoring windows (ties by id)."""
    ranked = sorted(windows, key=lambda w: (-w.get("score", 0.0), w["id"]))
    top = {w["id"] for w in ranked[: math.ceil(fraction * len(windows))]}
    for w in windows:
        w["top_motion"] = w["id"] in top


def majority_segment(values: Sequence[str | None]) -> str | None:
    """Strict (>50%) majority colon segment; any unknown frame or a tie gives None."""
    counts: dict[str | None, int] = {}
    for v in values:
        v = v if v in SEGMENTS else None
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    if not total or None in counts:
        return None
    label, count = max(counts.items(), key=lambda kv: kv[1])
    return label if 2 * count > total else None


# --------------------------------------------------------------------------------------------
# Window labels
# --------------------------------------------------------------------------------------------


def _state(window: Window, cls: str) -> str | None:
    info = window.get(cls)
    return info.get("state") if isinstance(info, dict) else None


def has_polyp(w: Window) -> bool:
    return _state(w, "polyp") == "present"


def has_tool(w: Window) -> bool:
    return _state(w, "tool") == "present"


def has_water(w: Window) -> bool:
    return _state(w, "water") == "present"


def has_snare(w: Window) -> bool:
    return bool(w.get("snare"))


def is_nbi(w: Window) -> bool:
    return w.get("nbi") is not None


def tool_trainable(w: Window) -> bool:
    return has_tool(w) and bool(w["tool"].get("trainable", True))


def water_trainable(w: Window) -> bool:
    return has_water(w) and bool(w["water"].get("trainable", True))


def is_clean(w: Window) -> bool:
    """Clean stratum: no polyp, tool, irrigation, special light, dye or bubbles."""
    if "clean" not in w.get("strata", ()) or w.get("excluded"):
        return False
    if _state(w, "polyp") != "absent" or _state(w, "tool") != "absent":
        return False
    if is_nbi(w) or has_water(w):
        return False
    flags = w.get("flags") or {}
    return not any(flags.get(k) for k in CLEAN_FLAGS)


def water_absent(w: Window) -> bool:
    return not has_water(w) and (_state(w, "water") == "absent" or is_clean(w))


def clean_weight(w: Window, top_weight: int = 2, forward_weight: int = 4) -> int:
    """Number of copies of a window in the draw list (only clean windows are repeated)."""
    if not is_clean(w):
        return 1
    return (top_weight if w.get("top_motion") else 1) * (forward_weight if w.get("forward") else 1)


def nbi_pick_weight(w: Window, directional: int = 8, switch: int = 2) -> int:
    frames = w["nbi"]["frames"]
    if bool(frames[0]) != bool(frames[-1]):
        return directional
    return switch if w["nbi"].get("switch") else 1


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """Joint-conditioning sampling recipe (defaults of the released model)."""

    seed: int = 20261027
    dropout_seed: int = 20261026031
    weights: dict[str, float] = field(
        default_factory=lambda: dict(clean=0.3, polyp=0.25, irrigation=0.15, tool=0.15, nbi=0.15)
    )
    top_weight: int = 2
    forward_weight: int = 4
    nbi_directional_weight: int = 8
    nbi_switch_weight: int = 2
    p_entry: dict[str, float] = field(
        default_factory=lambda: dict(polyp=0.5, tool=0.3, irrigation=0.5)
    )
    p_mask: dict[str, float] = field(default_factory=lambda: dict(polyp=0.8, tool=0.9, water=0.9))
    p_empty_negative: float = 0.3
    p_condition_unknown: float = 0.10
    p_flags_unknown: float = 0.10
    p_all_unknown: float = 0.05
    entry_retries: int = 8


@dataclass
class Draw:
    window: Window
    forced: str | None
    stratum: str


class JointSampler:
    """Stratified, hash-seeded window sampler over a manifest."""

    def __init__(self, windows: list[Window], cfg: SamplerConfig | None = None):
        self.cfg = cfg or SamplerConfig()
        c = self.cfg
        weighted = [
            w for w in windows for _ in range(clean_weight(w, c.top_weight, c.forward_weight))
        ]
        eligible = [w for w in weighted if not w.get("excluded")]

        def allowed(w: Window, name: str) -> bool:
            return name in w.get("strata", ())

        self.pools: dict[str, list[Window]] = dict(
            clean=[w for w in eligible if is_clean(w)],
            polyp=[w for w in eligible if allowed(w, "polyp") and has_polyp(w)],
            irrigation=[w for w in eligible if allowed(w, "irrigation") and water_trainable(w)],
            tool=[w for w in eligible if allowed(w, "tool") and tool_trainable(w)],
            nbi=[w for w in eligible if allowed(w, "nbi") and is_nbi(w)],
        )
        groups: dict[str, list[Window]] = {}
        for w in self.pools["irrigation"]:
            groups.setdefault(w["water"].get("event") or "window:" + w["id"], []).append(w)
        self.irrigation_events = [sorted(groups[k], key=lambda w: w["id"]) for k in sorted(groups)]

    # -- window picking -----------------------------------------------------------------------

    @staticmethod
    def _index(items: list, tag: str, seed: int, update: int, draw: int, attempt: int):
        if not items:
            raise ValueError(f"empty sampling pool: {tag}")
        return items[int(joint_u01(tag, seed, update, draw, attempt) * len(items))]

    @staticmethod
    def direction(w: Window, forced: str | None) -> str | None:
        """Irrigation windows are always played forward."""
        return "forward" if has_water(w) else forced

    def _positive(self, name: str, update: int, draw: int, attempt: int) -> Draw:
        seed, pool = self.cfg.seed, self.pools[name]
        options = []
        for w in pool:
            if w[name].get("entry_forward"):
                options.append((w, "forward"))
            if w[name].get("entry_reverse") and not has_water(w):
                options.append((w, "reverse"))
        if options and joint_u01(name + "-entry", seed, update, draw) < self.cfg.p_entry[name]:
            w, forced = self._index(options, name + "-entry-pick", seed, update, draw, attempt)
            return Draw(w, forced, name + ":entry")
        w = self._index(pool, name + "-positive", seed, update, draw, attempt)
        return Draw(w, self.direction(w, None), name + ":positive")

    def _nbi(self, update: int, draw: int, attempt: int) -> Draw:
        c = self.cfg
        sub = joint_subseed("nbi-pick", c.seed, update, draw, attempt)
        options = [
            w
            for w in self.pools["nbi"]
            for _ in range(nbi_pick_weight(w, c.nbi_directional_weight, c.nbi_switch_weight))
        ]
        w = options[int(nbi_u01("pick", sub, update, draw) * len(options))]
        return Draw(w, self.direction(w, None), "nbi")

    def _irrigation(self, update: int, draw: int, attempt: int) -> Draw:
        seed = self.cfg.seed
        event = self._index(self.irrigation_events, "irrigation-event", seed, update, draw, attempt)
        sub = joint_subseed("irrigation-pick", seed, update, draw, attempt)
        entry = [w for w in event if w["water"].get("entry_forward")]

        def weighted(options: list[Window], value: float) -> Window:
            weights = [float(w["water"].get("onset_weight", 1)) for w in options]
            threshold = value * sum(weights)
            for w, weight in zip(options, weights, strict=True):
                threshold -= weight
                if threshold < 0:
                    return w
            return options[-1]

        if entry and irrigation_u01("entry", sub, update, draw) < self.cfg.p_entry["irrigation"]:
            w = weighted(entry, irrigation_u01("entry-pick", sub, update, draw, 0))
            return Draw(w, "forward", "irrigation:entry")
        w = weighted(event, irrigation_u01("pos-pick", sub, update, draw, 0))
        return Draw(w, "forward", "irrigation:positive")

    def pick(self, update: int, draw: int, attempt: int = 0) -> Draw:
        """Pick stratum, window and forced playback direction of one draw."""
        c = self.cfg
        active = [(n, c.weights[n]) for n in STRATA if self.pools[n] and c.weights[n] > 0]
        if not active:
            raise ValueError("no trainable strata")
        position = joint_u01("stratum", c.seed, update, draw) * sum(wt for _, wt in active)
        name = active[-1][0]
        for candidate, weight in active:
            position -= weight
            if position < 0:
                name = candidate
                break
        if name == "clean":
            w = self._index(self.pools["clean"], "clean-pick", c.seed, update, draw, attempt)
            return Draw(w, None, "clean")
        if name in ("polyp", "tool"):
            return self._positive(name, update, draw, attempt)
        if name == "nbi":
            return self._nbi(update, draw, attempt)
        return self._irrigation(update, draw, attempt)

    # -- condition selection ------------------------------------------------------------------

    def condition_unknown(self, name: str, update: int, draw: int) -> bool:
        p = self.cfg.p_condition_unknown
        return joint_u01("condition-unknown-" + name, self.cfg.seed, update, draw) < p

    def all_unknown(self, update: int, draw: int) -> bool:
        return joint_u01("all-unknown", self.cfg.seed, update, draw) < self.cfg.p_all_unknown

    def choose_masks(self, w: Window, stratum: str, update: int, draw: int) -> dict[str, Any]:
        """Per class: ``"mask"`` (supply the mask), ``"empty"`` (explicit empty) or None."""
        seed = self.cfg.seed
        arm, _, local = stratum.partition(":")
        out: dict[str, Any] = dict.fromkeys(MASK_CLASSES)
        if self.all_unknown(update, draw):
            return out
        present = dict(polyp=has_polyp(w), tool=has_tool(w), water=has_water(w))
        absent = dict(
            polyp=_state(w, "polyp") == "absent",
            tool=_state(w, "tool") == "absent",
            water=water_absent(w),
        )
        owners = dict(polyp="polyp", tool="tool", water="irrigation")
        for cls in ("polyp", "tool", "water"):
            if self.condition_unknown(cls, update, draw):
                continue
            if present[cls]:
                entry = arm == owners[cls] and local in ("entry", "onset")
                if entry or joint_u01("mask-" + cls, seed, update, draw) < self.cfg.p_mask[cls]:
                    out[cls] = "mask"
            elif absent[cls]:
                if joint_u01("empty-" + cls, seed, update, draw) < self.cfg.p_empty_negative:
                    out[cls] = "empty"
        if not self.condition_unknown("snare", update, draw):
            if out["tool"] == "mask" and has_snare(w):
                out["snare"] = "mask"
            elif out["tool"] == "empty":
                out["snare"] = "empty"
        return out

    # -- global flags -------------------------------------------------------------------------

    def frame_flags(
        self,
        w: Window,
        choice: dict[str, Any],
        selected: np.ndarray,
        static: bool,
        presence: dict[str, np.ndarray],
        t: int,
        update: int,
        draw: int,
    ) -> np.ndarray:
        """Per-latent-frame global condition vector ``[t, 13]``.

        Columns: 0-3 values (special light, dye, tool, irrigation), 4-7 their known bits,
        8-12 class-known bits (polyp, tool, water, blood, snare). Unknown is value 0, known 0.
        ``presence`` holds the transformed per-output-frame occupancy of tool and water.
        """
        out = np.zeros((t, GLOBAL_IN), dtype=np.float32)
        light, light_known = light_flags(w, selected, static, t)
        out[:, 0], out[:, 4] = light, light_known
        unknown = {n: self.condition_unknown(n, update, draw) for n in (*MASK_CLASSES, "nbi")}
        if unknown["nbi"]:
            out[:, 0] = out[:, 4] = 0

        if choice["tool"] == "empty":
            out[:, 6] = 1
        elif has_tool(w):
            out[:, 2], out[:, 6] = _group_flag(presence["tool"], t)
        elif _state(w, "tool") == "absent":
            out[:, 6] = 1
        if unknown["tool"]:
            out[:, 2] = out[:, 6] = 0

        if choice["water"] == "empty" or water_absent(w):
            out[:, 7] = 1
        elif has_water(w):
            out[:, 3], out[:, 7] = _group_flag(presence["water"], t)
        if unknown["water"]:
            out[:, 3] = out[:, 7] = 0

        for cls in MASK_CLASSES:
            if choice[cls] in ("mask", "empty"):
                out[:, 2 * len(FRAME_FLAGS) + GLOBAL_CLASSES.index(cls)] = 1
        if joint_u01("flags-unknown", self.cfg.seed, update, draw) < self.cfg.p_flags_unknown:
            out[:, :8] = 0
        if self.all_unknown(update, draw):
            out[:] = 0
        out[:, :4] *= out[:, 4:8]
        return out


def latent_groups(values: np.ndarray, t: int) -> list[np.ndarray]:
    """Frame 0 alone, then groups of four frames (causal video VAE layout)."""
    return [values[:1]] + [values[1 + 4 * i : 5 + 4 * i] for i in range(t - 1)]


def _group_flag(values: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    groups = latent_groups(np.asarray(values, dtype=bool), t)
    return np.array([g.any() for g in groups], np.float32), np.ones(t, np.float32)


def select_frames(values: Sequence, selected: np.ndarray, static: bool) -> np.ndarray:
    """Map per-tube-frame values to output order (reversal, static repetition)."""
    out = np.asarray(values)[np.asarray(selected, dtype=int)]
    return np.repeat(out[:1], len(out), axis=0) if static else out


def light_flags(
    w: Window, selected: np.ndarray, static: bool, t: int
) -> tuple[np.ndarray, np.ndarray]:
    """Special-light value and known bit per latent frame.

    Labelled windows: a latent frame is known only when all its frames agree and none lies
    within the uncertainty band of a light switch. Clean and polyp windows are white light.
    """
    values, known = np.zeros(t, np.float32), np.zeros(t, np.float32)
    if not w.get("light_labelled", True):
        return values, known
    if is_nbi(w):
        frames = select_frames(np.asarray(w["nbi"]["frames"], bool), selected, static)
        uncertain = select_frames(np.asarray(w["nbi"]["uncertain"], bool), selected, static)
        groups, ugroups = latent_groups(frames, t), latent_groups(uncertain, t)
        sure = np.array([not g.any() for g in ugroups])
        on = np.array([g.all() for g in groups]) & sure
        off = np.array([(~g).all() for g in groups]) & sure
        return on.astype(np.float32), (on | off).astype(np.float32)
    if is_clean(w) or has_polyp(w):
        known[:] = 1
        if w.get("light_unknown_frames") is not None:
            unknown = select_frames(np.asarray(w["light_unknown_frames"], bool), selected, static)
            known = np.array([not g.any() for g in latent_groups(unknown, t)], np.float32)
    return values, known


def nbi_edge_uncertain(frames: Sequence[bool], edge: int = 15) -> np.ndarray:
    """Frames within ``edge`` frames of a light switch are uncertain."""
    f = np.asarray(frames, bool)
    u = np.zeros(len(f), bool)
    for i in np.flatnonzero(f[1:] != f[:-1]) + 1:
        u[max(0, i - edge) : min(len(f), i + edge)] = True
    return u


# --------------------------------------------------------------------------------------------
# Captions and segment labels
# --------------------------------------------------------------------------------------------


def base_caption(frames: int, fps: int = FPS, height: int = HEIGHT, width: int = WIDTH) -> str:
    return (
        f"{BASE_CAPTION} The video is {frames / fps:.1f} seconds long and is of {fps} FPS. "
        f"This video is of {height}x{width} resolution."
    )


def caption(frames: int, segment: str | None) -> str:
    text = base_caption(frames)
    if segment in SEGMENTS:
        text = text.rstrip() + " Colon segment: " + segment.replace("_", " ") + "."
    return text


def draw_caption(w: Window, seed: int, update: int, draw: int) -> str:
    segment = majority_segment(w.get("segments") or [None])
    return caption(int(w["frames"]), None if caption_generic_draw(seed, update, draw) else segment)


def segment_ids(
    w: Window, selected: np.ndarray, static: bool, seed: int, update: int, draw: int
) -> list[int]:
    """Colon-segment class per latent frame (output frames 0, 4, 8, ...), null 10% of draws."""
    labels = select_frames(
        np.asarray(
            [s if s in SEGMENTS else "" for s in (w.get("segments") or [None] * w["frames"])]
        ),
        selected,
        static,
    )
    ids = [SEGMENTS.index(v) if v in SEGMENTS else SEGMENT_NULL for v in labels[::4]]
    return [SEGMENT_NULL] * len(ids) if segment_null_draw(seed, update, draw) else ids


# --------------------------------------------------------------------------------------------
# Condition masks
# --------------------------------------------------------------------------------------------


class ConditionMasks:
    """Bit-packed per-window condition masks (``conditions`` npz of the manifest row).

    Arrays: ``<cls>`` uint8 [T,H,ceil(W/8)] bit-packed along width and ``<cls>_visible`` bool [T]
    for cls in polyp, tool, snare, water; ``water_loss_region`` (same packing) for irrigation.
    """

    def __init__(self, path: str | Path | None, frames: int):
        self.path, self.frames = path, frames
        self._cache: dict[str, np.ndarray] = {}

    def _load(self, key: str) -> np.ndarray | None:
        if key not in self._cache:
            if self.path is None:
                return None
            with np.load(self.path, allow_pickle=False) as z:
                if key not in z:
                    return None
                value = z[key]
            if value.dtype == np.uint8 and value.ndim == 3:
                value = np.unpackbits(value, axis=-1, count=WIDTH).astype(bool)
            self._cache[key] = value
        return self._cache[key]

    def folded(self, cls: str) -> np.ndarray:
        """Mask with invisible frames cleared, in tube-frame order [T,H,W] bool."""
        mask = self._load(cls)
        if mask is None:
            return np.zeros((self.frames, HEIGHT, WIDTH), bool)
        visible = self._load(cls + "_visible")
        return mask if visible is None else mask & np.asarray(visible, bool)[:, None, None]

    def presence(self, cls: str, with_snare: bool = False) -> np.ndarray:
        """Silhouette used for the tool/water global flags (tool includes the snare)."""
        mask = self.folded(cls)
        return mask | self.folded("snare") if with_snare else mask

    def loss_region(self) -> np.ndarray | None:
        return self._load("water_loss_region")


def mask_video(masks: ConditionMasks, choice: dict[str, Any]) -> np.ndarray | None:
    """Palette video of the selected condition masks in tube-frame order, or None."""
    if all(v is None for v in choice.values()):
        return None
    from colonoworld.conditions.palette import render_palette_video

    empty = np.zeros((masks.frames, HEIGHT, WIDTH), bool)
    selected = {
        cls: (empty if value == "empty" else masks.folded(cls))
        for cls, value in choice.items()
        if value is not None
    }
    return render_palette_video(selected)


def loss_weight_latent(
    pixel_weights: np.ndarray | None, latent_shape: Sequence[int], patch: int = 16
):
    """Max-pool pixel loss weights [T,H,W] onto the latent grid [t,h,w] (frame 0, then 4s)."""
    if pixel_weights is None:
        return None
    import torch
    import torch.nn.functional as F

    _, t, h, w = map(int, latent_shape)
    weights = torch.as_tensor(pixel_weights, dtype=torch.float32)
    pooled = F.max_pool2d(weights[:, None], kernel_size=patch, stride=patch)[:, 0]
    groups = [pooled[:1]] + [pooled[1 + 4 * i : 5 + 4 * i] for i in range(t - 1)]
    out = torch.stack([g.amax(dim=0) for g in groups], dim=0)
    assert tuple(out.shape) == (t, h, w)
    return out


# --------------------------------------------------------------------------------------------
# Condition rasterisers used when preparing training data
# --------------------------------------------------------------------------------------------


def rasterize_snare(
    frames: list[dict[str, Any]], width_px: int = 8, tip_radius: int = 8
) -> np.ndarray:
    """Snare wire polylines (normalised uv fragments + optional tip) -> bool [T,H,W]."""
    from PIL import Image, ImageDraw

    out = np.zeros((len(frames), HEIGHT, WIDTH), bool)
    for t, row in enumerate(frames):
        if not row.get("line"):
            continue
        im = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(im)
        for fragment in row["fragments_uv"]:
            pts = [(u * WIDTH, v * HEIGHT) for u, v in fragment]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=width_px, joint="curve")
        if row.get("tip_uv") is not None:
            x, y = row["tip_uv"][0] * WIDTH, row["tip_uv"][1] * HEIGHT
            draw.ellipse([x - tip_radius, y - tip_radius, x + tip_radius, y + tip_radius], fill=255)
        out[t] = np.asarray(im) > 0
    return out


def irrigation_ribbon(
    anchor: Sequence[float],
    target: np.ndarray,
    on: np.ndarray,
    width_anchor: float = 18.0,
    width_target: float = 8.0,
    end_radius: int = 12,
) -> np.ndarray:
    """Water-jet command ribbon from the nozzle anchor to the per-frame target -> bool [T,H,W]."""
    import cv2

    anchor = np.asarray(anchor, np.float64)
    target = np.asarray(target, np.float64)
    on = np.asarray(on, bool)
    out = np.zeros((len(on), HEIGHT, WIDTH), bool)
    for t in np.flatnonzero(on):
        point = target[t]
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        delta = point - anchor
        length = float(np.linalg.norm(delta))
        if length > 1e-9:
            normal = np.array([-delta[1], delta[0]]) / length
            polygon = np.array(
                [
                    anchor + normal * width_anchor / 2,
                    point + normal * width_target / 2,
                    point - normal * width_target / 2,
                    anchor - normal * width_anchor / 2,
                ]
            )
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1, lineType=cv2.LINE_8)
        cv2.circle(mask, tuple(np.rint(point).astype(int)), end_radius, 1, -1, lineType=cv2.LINE_8)
        out[t] = mask.astype(bool)
    return out


def irrigation_onset_weight(on: Sequence[bool]) -> int:
    """Windows whose jet starts between frames 6 and 48 (off at frame 0) are drawn 3x."""
    on = np.asarray(on, bool)
    onset = int(np.flatnonzero(on)[0]) if on.any() else -1
    return 3 if not on[0] and 6 <= onset <= 48 else 1


def entry_directions(occupied: Sequence[bool]) -> tuple[bool, bool]:
    """(forward, reverse) entry: object absent at the first played frame, present later."""
    occ = np.asarray(occupied, bool)
    forward = bool(not occ[0] and occ[1:].any())
    reverse = bool(not occ[-1] and occ[:-1].any())
    return forward, reverse


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, bool), axis=-1)


# --------------------------------------------------------------------------------------------
# Training draws
# --------------------------------------------------------------------------------------------


class EntryRejected(RuntimeError):
    """The augmented draw no longer shows the object entering the view."""


def decode_frames(path: str | Path, indices: Sequence[int] | None = None) -> np.ndarray:
    """Decode RGB frames (by ordinal) of a video with single-threaded PyAV -> [T,H,W,3] uint8."""
    import av

    frames: dict[int, np.ndarray] = {}
    wanted = None if indices is None else set(map(int, indices))
    last = None if indices is None else max(wanted)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        for j, frame in enumerate(container.decode(video=0)):
            if wanted is None or j in wanted:
                frames[j] = frame.reformat(format="rgb24", threads=1).to_ndarray()
            if last is not None and j >= last:
                break
    order = sorted(frames) if indices is None else list(map(int, indices))
    return np.stack([frames[j] for j in order])


def load_camera(path: str | Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Camera npz: ``poses`` [T,4,4] camera-to-world, ``calibration`` (json string, KB4),
    ``field_mask`` [h,w] endoscope field (resized to the training resolution)."""
    import cv2

    with np.load(path, allow_pickle=False) as z:
        poses = z["poses"].astype(np.float64)
        cam = json.loads(str(z["calibration"]))
        field = z["field_mask"].astype(np.uint8)
    field = cv2.resize(field, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    return poses, cam, field


@dataclass
class TrainingDraw:
    """CPU part of one training sample (latents are encoded on the GPU by the trainer)."""

    window_id: str
    stratum: str
    video: np.ndarray  # [T,H,W,3] uint8, augmented real frames
    tube: np.ndarray  # [T,H,W,3] uint8, augmented rendered tube
    mask_video: np.ndarray | None  # [T,H,W,3] uint8 palette video, or None (no mask stream)
    rays: np.ndarray  # [t,6,H/16,W/16] float32 Pluecker rays
    rays_valid: np.ndarray  # [H/16,W/16] bool
    latent_index: np.ndarray  # output frame of every latent frame
    caption: str
    segment_ids: list[int]
    global_flags: np.ndarray | None  # [t,13] float32, None when all zero
    loss_weights: np.ndarray | None  # [T,H,W] float32 pixel loss weights, or None
    handoff: dict[str, Any]
    plan: dict[str, Any]
    choice: dict[str, Any]
    tube_latent: str | None = None  # cached latent of the untransformed tube, when applicable


class JointDataset:
    """Produces the deterministic training draw ``(update, draw)``.

    ``draw`` is the global sample index inside an update: ``rank + micro * world_size``.
    """

    def __init__(
        self,
        windows: list[Window],
        sampler_cfg: SamplerConfig | None = None,
        augment: bool = True,
        static_schedule: str = "0.10",
        start_update: int = 0,
        handoff: bool = True,
        loss_weight: float = 3.0,
        loss_dilate_px: int = 25,
    ):
        self.sampler = JointSampler(windows, sampler_cfg)
        self.seed = self.sampler.cfg.seed
        self.augment, self.static_schedule = augment, static_schedule
        self.start_update, self.handoff = start_update, handoff
        self.loss_weight, self.loss_dilate_px = loss_weight, loss_dilate_px

    def plan(self, update: int, draw: int, attempt: int, forced: str | None) -> dict[str, Any]:
        from colonoworld.training import augmentation as A

        retries = self.sampler.cfg.entry_retries
        p = A.plan(
            self.seed,
            update,
            draw * (retries + 1) + attempt,
            self.augment,
            self.static_schedule,
            self.start_update,
        )
        return A.force_direction(p, forced)

    def __call__(self, update: int, draw: int) -> TrainingDraw:
        retries = self.sampler.cfg.entry_retries
        for attempt in range(retries):
            d = self.sampler.pick(update, draw, attempt)
            try:
                return self.build(
                    d.window, d.stratum, self.plan(update, draw, attempt, d.forced), update, draw
                )
            except EntryRejected:
                continue
        # Retry budget exhausted: keep the window and condition, drop the entry requirement.
        stratum = d.stratum.split(":")[0] + ":positive"
        p = self.plan(update, draw, retries, self.sampler.direction(d.window, None))
        return self.build(d.window, stratum, p, update, draw)

    def build(
        self, w: Window, stratum: str, p: dict[str, Any], update: int, draw: int
    ) -> TrainingDraw:
        from colonoworld.training import augmentation as A

        n = int(w["frames"])
        selected = A.selected_indices(n, p)
        static = p["static"]
        masks = ConditionMasks(w.get("conditions"), n)
        if stratum == "tool:entry":
            occ = select_frames(
                masks.presence("tool", has_snare(w)).reshape(n, -1).any(1), selected, static
            )
            if not (not occ[0] and occ[1:].any()):
                raise EntryRejected("tool entry lost by the temporal mapping")
        choice = self.sampler.choose_masks(w, stratum, update, draw)

        real = decode_frames(w["video"], w.get("frame_indices"))
        tube = decode_frames(w["tube_video"])
        poses, cam, field = load_camera(w["camera"])
        extra = mask_video(masks, choice)
        result = A.apply(real, tube, poses, cam, p, field, extra=extra)

        # Spatially transformed per-frame occupancy for the tool/water flags and the entry check.
        # It is evaluated in tube-frame order (static draws repeat frame 0), as in the released
        # model's training.
        tube_order = np.arange(n)
        presence: dict[str, np.ndarray] = {}
        tracks = dict(
            polyp=has_polyp(w) and stratum == "polyp:entry",
            tool=has_tool(w),
            water=has_water(w),
        )
        for cls, needed in tracks.items():
            if not needed:
                continue
            native = masks.presence(cls, cls == "tool" and has_snare(w))
            out = A.spatial_bool(select_frames(native, tube_order, static), result, p)
            occ = out.reshape(n, -1).any(axis=1)
            presence[cls] = occ
            if stratum == cls + ":entry" and not (not occ[0] and occ[1:].any()):
                raise EntryRejected(cls + " entry lost by the spatial transform")

        loss_weights = None
        arm, _, local = stratum.partition(":")
        if arm == "irrigation" and local != "negative" and choice["water"] == "mask":
            region = select_frames(masks.loss_region(), selected, static)
            region = A.spatial_bool(region, result, p)
            loss_weights = A.dilated_loss_weights(region, self.loss_weight, self.loss_dilate_px)

        t = 1 + (n - 1) // 4
        flags = self.sampler.frame_flags(w, choice, selected, static, presence, t, update, draw)
        handoff = A.handoff_params(self.seed, draw, update, enabled=self.handoff)
        return TrainingDraw(
            window_id=w["id"],
            stratum=stratum,
            video=result["real"],
            tube=result["tube"],
            mask_video=result["extra"],
            rays=result["rays"],
            rays_valid=result["valid"],
            latent_index=result["latent"],
            caption=draw_caption(w, self.seed, update, draw),
            segment_ids=segment_ids(w, selected, static, self.seed, update, draw),
            global_flags=flags if flags.any() else None,
            loss_weights=loss_weights,
            handoff=handoff,
            plan=p,
            choice=choice,
            tube_latent=w.get("tube_latent") if not (p["enabled"] or p["reverse"]) else None,
        )
