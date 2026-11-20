# SPDX-License-Identifier: CC-BY-NC-4.0
"""3D snare condition.

A polypectomy snare is modelled in the camera frame (x right, y down, z forward, millimetres):

* the sheath is a straight 2.5 mm cylinder that leaves the working channel at ``exit_mm`` (lower
  right of the scope tip) and extends by ``deployment * sheath_len_mm`` along a fixed direction;
  the direction is solved so that a sheath of ``direction_len_mm`` ends at an aim pixel;
* the wire is one continuous teardrop-shaped loop that opens forward from the sheath tip, tilted
  about its long axis, optionally turned (yaw) and partly closed over a frame range.

Both are projected with the Kannala-Brandt (KB4) fisheye model. The sheath is depth-buffered so
it hides the part of the wire behind it. The tool mask holds the sheath only; the wire is drawn
as an 8 px line in the snare mask.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SnareParams:
    """Snare geometry (millimetres, degrees) and deployment timing."""

    exit_mm: tuple[float, float] = (5.0, 5.0)
    sheath_len_mm: float = 11.0
    loop_len_mm: float = 20.0
    loop_width_mm: float = 14.0
    tilt_deg: float = 25.0
    yaw_deg: float = 0.0
    close: float = 0.0
    close_frames: tuple[int, int] = (120, 160)
    aim_fraction: float = 0.6
    aim_xy: tuple[float, float] | None = None
    direction_len_mm: float | None = None
    sheath_diam_mm: float = 2.5
    wire_px: int = 8
    sway_mm: tuple[float, float] = (0.25, 0.15)
    sway_rate: tuple[float, float] = (0.045, 0.037)
    ramp_frames: int = 15
    start_deployment: float = 0.3


def kb4_project(points: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Project camera-frame points ``(N, 3)`` to pixels ``(N, 2)`` with the KB4 fisheye model."""
    points = np.atleast_2d(points)
    r = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(r, points[:, 2])
    theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)
    scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 0)
    return np.stack(
        [K[0, 0] * points[:, 0] * scale + K[0, 2], K[1, 1] * points[:, 1] * scale + K[1, 2]], 1
    )


def kb4_unproject(pixel: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Unit camera ray of one pixel ``(x, y)`` (Newton inversion of the KB4 distortion)."""
    x, y = (pixel[0] - K[0, 2]) / K[0, 0], (pixel[1] - K[1, 2]) / K[1, 1]
    theta_d = math.hypot(x, y)
    theta = theta_d
    for _ in range(30):
        f = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)
        f -= theta_d
        df = 1 + 3 * D[0] * theta**2 + 5 * D[1] * theta**4 + 7 * D[2] * theta**6
        df += 9 * D[3] * theta**8
        theta -= f / df
    phi = math.atan2(y, x)
    return np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def deployment_schedule(
    num_frames: int, onset: int, offset: int, ramp_frames: int = 15, start: float = 0.3
) -> np.ndarray:
    """Per-frame sheath deployment in ``[0, 1]``; ``NaN`` where the snare is absent.

    The snare is present on frames ``onset <= t < offset``; deployment rises from ``start`` to 1
    with a smoothstep over ``ramp_frames`` frames (the tip is visible on the onset frame).
    """
    out = np.full(num_frames, np.nan)
    for t in range(max(onset, 0), min(offset, num_frames)):
        out[t] = start + (1 - start) * _smoothstep(min((t - onset + 1) / float(ramp_frames), 1.0))
    return out


def sheath_direction(
    params: SnareParams, K: np.ndarray, D: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed sheath direction and the aim pixel it was solved for.

    The aim pixel is ``aim_xy`` (image fraction) when given, otherwise ``aim_fraction`` of the way
    from the principal point to the lower-right corner. The direction is chosen so that a sheath of
    ``direction_len_mm`` (default ``sheath_len_mm``) starting at the channel exit ends on that ray.
    """
    height, width = size
    exit_point = np.array([params.exit_mm[0], params.exit_mm[1], 0.0])
    centre = [K[0, 2], K[1, 2]]
    if params.aim_xy is None:
        target = np.array(centre) + params.aim_fraction * (
            np.array([width - 1.0, height - 1.0]) - centre
        )
    else:
        target = np.array([params.aim_xy[0] * (width - 1), params.aim_xy[1] * (height - 1)])
    ray = kb4_unproject(target, K, D)
    length = params.sheath_len_mm if params.direction_len_mm is None else params.direction_len_mm
    b = ray @ exit_point
    lam = b + math.sqrt(b**2 - exit_point @ exit_point + length**2)
    return _unit(lam * ray - exit_point), target


def _sheath_depth(
    exit_point: np.ndarray,
    tip: np.ndarray,
    radius_mm: float,
    K: np.ndarray,
    D: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Range buffer of the sheath cylinder (``inf`` where empty), drawn far to near."""
    import cv2

    height, width = size
    centreline = exit_point + np.linspace(0, 1, 160)[:, None] * (tip - exit_point)
    zbuf = np.full((height, width), np.inf, np.float32)
    last = len(centreline) - 1
    for i in np.argsort(-np.linalg.norm(centreline, axis=1)):
        p = centreline[i]
        tangent = _unit(centreline[min(i + 1, last)] - centreline[max(i - 1, 0)])
        side = np.cross(tangent, _unit(p))
        side = _unit(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])
        if p[2] <= 0.5:
            continue
        u0, u1 = kb4_project(np.stack([p, p + side * radius_mm]), K, D)
        r_px = max(1, int(round(np.linalg.norm(u1 - u0))))
        disc = np.zeros((height, width), np.uint8)
        cv2.circle(disc, tuple(np.rint(u0).astype(int)), r_px, 1, -1)
        zbuf[disc.astype(bool)] = np.linalg.norm(p)
    return zbuf


def _wire_points(
    params: SnareParams, direction: np.ndarray, tip: np.ndarray, deployment: float, width: float
) -> np.ndarray:
    """3D teardrop loop ``(400, 3)`` that opens forward from the sheath tip."""
    tilt = math.radians(params.tilt_deg)
    a = direction
    b = _unit(np.cross(a, [0, 1.0, 0])) if abs(a[1]) < 0.99 else np.array([1.0, 0, 0])
    up = _unit(np.cross(b, a))
    long_axis = _unit(a * math.cos(tilt) - up * math.sin(math.radians(0.6 * params.tilt_deg)))
    if params.yaw_deg:
        yaw = math.radians(params.yaw_deg)
        long_axis = _unit(long_axis * math.cos(yaw) - b * math.sin(yaw))
    normal = _unit(np.cross(long_axis, b))
    short_axis = _unit(b * math.cos(tilt) + normal * math.sin(tilt))
    ang = np.linspace(0, 2 * math.pi - 0.13, 400)
    half = np.sin(ang / 2)
    lon = params.loop_len_mm * (1 - np.cos(ang)) / 2
    lat = width * 0.5 * params.loop_width_mm * np.sin(ang) * half / (4 / (3 * math.sqrt(3)))
    return tip[None] + deployment * (
        lon[:, None] * long_axis[None] + lat[:, None] * short_axis[None]
    )


def _wire_depth(
    points: np.ndarray, line_px: int, K: np.ndarray, D: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Range buffer of the wire polyline (``inf`` where empty), drawn far to near."""
    import cv2

    height, width = size
    uv = kb4_project(points, K, D)
    zbuf = np.full((height, width), np.inf, np.float32)
    rng = np.linalg.norm(points, axis=1)
    for i in np.argsort(-((rng[:-1] + rng[1:]) / 2)):
        if points[i, 2] <= 0.5 or points[i + 1, 2] <= 0.5:
            continue
        line = np.zeros((height, width), np.uint8)
        p0 = tuple(np.rint(uv[i]).astype(int))
        p1 = tuple(np.rint(uv[i + 1]).astype(int))
        cv2.line(line, p0, p1, 1, line_px, cv2.LINE_8)
        zbuf[line.astype(bool)] = (rng[i] + rng[i + 1]) / 2
    return zbuf


def render_snare(
    K: np.ndarray,
    D: np.ndarray,
    field_mask: np.ndarray,
    deployment: np.ndarray,
    params: SnareParams | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render the snare condition masks.

    Args:
        K: ``(3, 3)`` KB4 intrinsics in output pixels.
        D: ``(4,)`` KB4 distortion coefficients.
        field_mask: ``(H, W)`` boolean endoscope field.
        deployment: ``(T,)`` sheath deployment per frame, ``NaN`` where the snare is absent
            (see :func:`deployment_schedule`).
        params: snare geometry; defaults to :class:`SnareParams`.

    Returns:
        ``tool`` (sheath) and ``snare`` (wire) boolean masks ``(T, H, W)`` and a geometry record.
    """
    params = params or SnareParams()
    K, D = np.asarray(K, np.float64), np.asarray(D, np.float64)
    size = field_mask.shape
    num_frames = len(deployment)
    exit_point = np.array([params.exit_mm[0], params.exit_mm[1], 0.0])
    direction, aim = sheath_direction(params, K, D, size)
    tool = np.zeros((num_frames, *size), bool)
    snare = np.zeros((num_frames, *size), bool)
    frames = []
    c0, c1 = params.close_frames
    for t in range(num_frames):
        dep = deployment[t]
        if not np.isfinite(dep):
            continue
        sway = np.array(
            [
                params.sway_mm[0] * math.sin(t * params.sway_rate[0]),
                params.sway_mm[1] * math.sin(t * params.sway_rate[1]),
                0.0,
            ]
        )
        tip = exit_point + sway + direction * (params.sheath_len_mm * dep)
        z_sheath = _sheath_depth(exit_point, tip, params.sheath_diam_mm / 2, K, D, size)
        width = 1 - params.close * min(1, max(0, (t - c0) / max(1, c1 - c0)))
        wire = _wire_points(params, direction, tip, dep, width)
        z_wire = _wire_depth(wire, params.wire_px, K, D, size)
        tool[t] = np.isfinite(z_sheath) & field_mask
        snare[t] = np.isfinite(z_wire) & (z_wire < z_sheath) & field_mask
        frames.append(
            dict(frame=t, deployment=float(dep), loop_width_scale=width, sheath_tip_mm=tip.tolist())
        )
    record = dict(params=asdict(params), direction=direction.tolist(), aim_xy=aim.tolist())
    record["frames"] = frames
    return tool, snare, record
