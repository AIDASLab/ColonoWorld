# SPDX-License-Identifier: CC-BY-NC-4.0
"""Palette condition masks.

Spatial conditions (polyp, tool, irrigation water, blood, snare wire) are given to the model as one
RGB "palette" video: every class is painted in its own colour with a soft 3 px edge, in a fixed
draw order (later classes cover earlier ones). The palette video is VAE-encoded like the tube.
Frame-level conditions (special light, dye, tool, irrigation) are boolean flags per frame.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

PALETTE: dict[str, tuple[int, int, int]] = {
    "polyp": (255, 0, 0),
    "tool": (0, 255, 0),
    "water": (0, 0, 255),
    "blood": (255, 255, 0),
    "snare": (255, 0, 255),
}
MASK_CLASSES: tuple[str, ...] = ("polyp", "tool", "water", "blood", "snare")
DRAW_ORDER: tuple[str, ...] = ("polyp", "blood", "water", "tool", "snare")
FRAME_FLAGS: tuple[str, ...] = ("special_light", "dye", "tool", "irrigation")
EDGE_PX = 3.0


def soft_edge(mask: np.ndarray, edge_px: float = EDGE_PX) -> np.ndarray | None:
    """Alpha of a boolean mask: 1 inside, linear falloff over ``edge_px``; ``None`` if empty."""
    import cv2

    if not mask.any():
        return None
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - dist / edge_px, 0.0, 1.0)


def render_palette_frame(masks: Mapping[str, np.ndarray], edge_px: float = EDGE_PX) -> np.ndarray:
    """Paint one frame of boolean ``(H, W)`` class masks into a ``uint8`` RGB palette image."""
    unknown = set(masks) - set(MASK_CLASSES)
    if unknown:
        raise ValueError(f"unknown mask classes: {sorted(unknown)}")
    height, width = next(iter(masks.values())).shape[:2]
    out = np.zeros((height, width, 3), np.float32)
    for cls in DRAW_ORDER:
        if cls not in masks:
            continue
        alpha = soft_edge(np.asarray(masks[cls], bool), edge_px)
        if alpha is not None:
            colour = np.asarray(PALETTE[cls], np.float32)
            out = out * (1 - alpha[..., None]) + colour * alpha[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_palette_video(
    masks: Mapping[str, np.ndarray],
    visible: np.ndarray | None = None,
    field_mask: np.ndarray | None = None,
    edge_px: float = EDGE_PX,
) -> np.ndarray:
    """Render boolean ``(T, H, W)`` class masks into a ``uint8`` palette video ``(T, H, W, 3)``.

    Args:
        masks: class name -> boolean mask video; classes must be in ``MASK_CLASSES``.
        visible: optional ``(T,)`` boolean; frames that are not visible stay black.
        field_mask: optional ``(H, W)`` boolean endoscope field; pixels outside are set to black.
        edge_px: soft-edge width in pixels.
    """
    if not masks:
        raise ValueError("at least one mask class is required")
    first = next(iter(masks.values()))
    num_frames, height, width = first.shape
    out = np.zeros((num_frames, height, width, 3), np.uint8)
    for t in range(num_frames):
        if visible is not None and not visible[t]:
            continue
        frame = {cls: m[t] for cls, m in masks.items() if m[t].any()}
        if frame:
            out[t] = render_palette_frame(frame, edge_px)
            if field_mask is not None:
                out[t][~field_mask] = 0
    return out


def tool_flags(masks: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-frame ``tool`` flag: any tool or snare pixel in the frame."""
    num_frames = next(iter(masks.values())).shape[0]
    flag = np.zeros(num_frames, bool)
    for cls in ("tool", "snare"):
        if cls in masks:
            flag |= masks[cls].reshape(num_frames, -1).any(1)
    return flag


def update_condition(
    cond: dict[str, np.ndarray], edge_px: float = EDGE_PX
) -> dict[str, np.ndarray]:
    """Recompute the ``tool`` frame flag and the ``palette`` video of a condition dict in place.

    ``cond`` holds boolean ``(T, H, W)`` masks under ``MASK_CLASSES`` keys (missing classes are
    skipped), a ``(T, len(FRAME_FLAGS))`` boolean ``flags`` array and an ``(H, W)`` ``field_mask``.
    """
    masks = {cls: cond[cls] for cls in MASK_CLASSES if cls in cond}
    cond["flags"][:, FRAME_FLAGS.index("tool")] = tool_flags(masks)
    cond["palette"] = render_palette_video(masks, field_mask=cond["field_mask"], edge_px=edge_px)
    return cond


def empty_condition(num_frames: int, field_mask: np.ndarray) -> dict[str, np.ndarray]:
    """A condition dict with no masks and no flags set."""
    height, width = field_mask.shape
    cond = {cls: np.zeros((num_frames, height, width), bool) for cls in MASK_CLASSES}
    cond["flags"] = np.zeros((num_frames, len(FRAME_FLAGS)), bool)
    cond["palette"] = np.zeros((num_frames, height, width, 3), np.uint8)
    cond["field_mask"] = field_mask.astype(bool)
    cond["frame_flags"] = np.array(FRAME_FLAGS)
    return cond
