# SPDX-License-Identifier: CC-BY-NC-4.0
"""Lumen following: agreement of the generated dark lumen with the lumen of the conditioning tube.

Per frame, on a 160x120 grid:
    generated lumen  darkest connected component (8-connected, >= 15 px) of pixels darker than
                     0.45 x the median field luma inside the central ellipse; present when its area
                     is 0.5 %..60 % of the field.
    render lumen     largest connected component of tube-depth pixels deeper than 2 x the median
                     valid depth, covering > 1 % of the field (computed on the depth grid of the
                     tube render, then nearest-resized). In the default ``open_end_as_deep``
                     variant, field pixels without depth (looking out of the open far end of the
                     tube) are counted as deepest; ``frozen`` drops them.
Frame metrics (both lumens present): centroid distance / image diagonal, area ratio and its log,
IoU. Fractions: either / both / one side absent, dark (median luma < 20), washed out (median
luma > 200 or > 25 % of pixels above 245). Recall: share of render-present frames where the
generated lumen is present.
"""

from __future__ import annotations

import math

import numpy as np

W, H = 160, 120
_YY, _XX = np.mgrid[:H, :W]
INNER = np.hypot((_XX - 79.5) / 80, (_YY - 59.5) / 60) < 0.85
LUMA = np.array([0.299, 0.587, 0.114])
OPEN_END_DEPTH = np.float32(1e6)
VARIANTS = ("open_end_as_deep", "frozen")

COLUMNS = (
    "gen_present",
    "tube_present",
    "both",
    "centroid_dist_diag",
    "area_ratio",
    "log_area_ratio",
    "abs_log_area_ratio",
    "iou",
    "gen_area",
    "tube_area",
    "tube_missing_depth_fraction",
    "gen_absent",
    "tube_absent",
    "either_absent",
    "both_absent",
    "median_luma",
    "bright_pixel_fraction",
    "dark",
    "washed_out",
    "detector_exception",
    "open_end_view",
)
FRAME_METRICS = ("centroid_dist_diag", "area_ratio", "log_area_ratio", "abs_log_area_ratio", "iou")
FRACTIONS = (
    "gen_absent",
    "tube_absent",
    "either_absent",
    "both_absent",
    "dark",
    "washed_out",
    "detector_exception",
    "open_end_view",
)


# ----------------------------------------------------------------------------------- detectors
def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Bilinear resize of RGB frames ``(T, H, W, 3)`` to the 160x120 detector grid."""
    import cv2

    return np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames])


def generated_lumen(rgb: np.ndarray, field: np.ndarray):
    """``(mask, present, threshold, n_candidates)`` of the generated (dark) lumen."""
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
    present = bool(eligible) and 0.005 <= fraction <= 0.60
    return mask, present, threshold, len(eligible)


def render_lumen(depth: np.ndarray, field: np.ndarray):
    """``(mask, present, median_depth, n_candidates, missing_fraction)`` of the render lumen."""
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
    return mask, best is not None, median, len(ids), float(1 - valid.sum() / field.sum())


def centre(mask: np.ndarray) -> np.ndarray | None:
    """Mask centroid ``(x, y)`` or ``None`` for an empty mask."""
    yy, xx = np.nonzero(mask)
    return np.array([xx.mean(), yy.mean()]) if len(xx) else None


def _nearest(mask: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


# ------------------------------------------------------------------------------------ per frame
def measure_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    field_depth: np.ndarray,
    field: np.ndarray,
    variant: str = "open_end_as_deep",
) -> dict[str, float]:
    """Metrics of one frame.

    Args:
        rgb: generated frame on the 160x120 grid, uint8.
        depth: tube-render depth on its own grid.
        field_depth: boolean field mask on the depth grid.
        field: boolean field mask on the 160x120 grid.
        variant: ``open_end_as_deep`` or ``frozen`` (see module docstring).
    """
    rec = {name: float("nan") for name in COLUMNS}
    rec["detector_exception"] = 0.0
    rec["open_end_view"] = 0.0
    try:
        luma = rgb @ LUMA
        field_luma = luma[field]
        rec["median_luma"] = float(np.median(field_luma))
        rec["bright_pixel_fraction"] = float(np.mean(field_luma > 245))
        rec["dark"] = float(rec["median_luma"] < 20)
        rec["washed_out"] = float(rec["median_luma"] > 200 or rec["bright_pixel_fraction"] > 0.25)

        tube_depth = depth.astype(np.float32)
        valid = field_depth & np.isfinite(tube_depth) & (tube_depth > 0)
        missing = field_depth & ~valid
        original_missing = float(1 - valid.sum() / field_depth.sum())
        if variant == "open_end_as_deep":
            tube_depth[missing] = OPEN_END_DEPTH
        elif variant != "frozen":
            raise ValueError(f"unknown variant {variant}")
        tm_aux, tp, _, _, miss = render_lumen(tube_depth, field_depth)
        if variant == "open_end_as_deep":
            miss = original_missing
        rec["tube_present"], rec["tube_absent"] = float(tp), float(not tp)
        rec["tube_missing_depth_fraction"] = float(miss)
        rec["open_end_view"] = float(miss > 0.003)
        tm = _nearest(np.asarray(tm_aux, bool)) & field

        gm, gp, _, _ = generated_lumen(rgb, field)
        gm = np.asarray(gm, bool)
        rec["gen_present"], rec["gen_absent"] = float(gp), float(not gp)
        rec["either_absent"] = float(not gp or not tp)
        rec["both_absent"] = float(not gp and not tp)

        egm = gm if gp else np.zeros_like(gm)
        etm = tm if tp else np.zeros_like(tm)
        rec["gen_area"] = float(np.count_nonzero(egm))
        rec["tube_area"] = float(np.count_nonzero(etm))
        union = int(np.count_nonzero(egm | etm))
        if union:
            rec["iou"] = float(np.count_nonzero(egm & etm)) / union
        cg = centre(gm) if gp else None
        ct = centre(tm) if tp else None
        both = gp and tp and cg is not None and ct is not None
        rec["both"] = float(both)
        if both:
            rec["centroid_dist_diag"] = math.hypot(
                float(cg[0]) - float(ct[0]), float(cg[1]) - float(ct[1])
            ) / math.hypot(W, H)
            rec["area_ratio"] = rec["gen_area"] / rec["tube_area"]
            rec["log_area_ratio"] = math.log(rec["area_ratio"])
            rec["abs_log_area_ratio"] = abs(rec["log_area_ratio"])
    except Exception:
        rec["detector_exception"] = 1.0
    return rec


def measure_clip(
    frames: np.ndarray,
    tube_depth: np.ndarray,
    field_depth: np.ndarray,
    variant: str = "open_end_as_deep",
) -> np.ndarray:
    """Per-frame records ``(T, len(COLUMNS))``.

    Args:
        frames: generated frames ``(T, H, W, 3)`` uint8 at any size (resized bilinearly).
        tube_depth: tube-render depth ``(T, h, w)``.
        field_depth: boolean field mask ``(h, w)`` of the depth grid.
    """
    import cv2

    cv2.setNumThreads(1)
    small = frames if frames.shape[1:3] == (H, W) else resize_frames(frames)
    field_depth = np.asarray(field_depth, bool)
    field = _nearest(field_depth)
    rows = [
        measure_frame(small[t], tube_depth[t].astype(np.float32), field_depth, field, variant)
        for t in range(len(small))
    ]
    return np.array([[r[c] for c in COLUMNS] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------------- aggregation
def _finite(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _fmean(values) -> float:
    a = _finite(values)
    return float(a.mean()) if a.size else float("nan")


def _fmedian(values) -> float:
    a = _finite(values)
    return float(np.median(a)) if a.size else float("nan")


def clip_summary(data: np.ndarray) -> dict[str, float]:
    """Per-clip values: mean / median of frame metrics, fractions, lumen recall vs the render."""
    col = {name: i for i, name in enumerate(COLUMNS)}
    out: dict[str, float] = {}
    for m in FRAME_METRICS:
        out[f"{m}_mean"] = _fmean(data[:, col[m]])
        out[f"{m}_median"] = _fmedian(data[:, col[m]])
    for f in FRACTIONS:
        out[f] = _fmean(data[:, col[f]])
    out["n_both"] = int(np.sum(data[:, col["both"]] == 1))
    eligible = (data[:, col["tube_absent"]] == 0) & np.isfinite(data[:, col["gen_absent"]])
    recalled = eligible & (data[:, col["gen_absent"]] == 0)
    out["n_recall_eligible"] = int(eligible.sum())
    out["n_recalled"] = int(recalled.sum())
    out["lumen_recall_vs_render"] = (
        out["n_recalled"] / out["n_recall_eligible"] if out["n_recall_eligible"] else float("nan")
    )
    return out


def set_summary(clips: list[dict[str, float]]) -> dict[str, float]:
    """Set values: equal-weight mean over clips with a defined value, geometric-mean area ratio and
    pooled recall."""
    keys = [k for k in clips[0] if not k.startswith("n_")]
    out = {k: _fmean([c[k] for c in clips]) for k in keys}
    lm = out["log_area_ratio_mean"]
    out["area_ratio_geometric_mean"] = math.exp(lm) if math.isfinite(lm) else float("nan")
    den = sum(c["n_recall_eligible"] for c in clips)
    out["lumen_recall_vs_render_pooled"] = (
        sum(c["n_recalled"] for c in clips) / den if den else float("nan")
    )
    return out


def paired_difference(
    ours: dict[str, dict[str, float]], other: dict[str, dict[str, float]], metric: str
) -> dict[str, float]:
    """Mean within-clip difference ``ours - other`` of a clip metric over comparable clips."""
    diffs = [
        ours[c][metric] - other[c][metric]
        for c in sorted(ours)
        if c in other and math.isfinite(ours[c][metric]) and math.isfinite(other[c][metric])
    ]
    return {"value": _fmean(diffs), "n_clips": len(diffs)}
