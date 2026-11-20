# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tube video renderer.

A tube video is what the endoscope would see inside the tube wall when it follows the camera path:
each pixel's fisheye ray (see :mod:`colonoworld.tube.camera`) is intersected with the wall mesh
(Mitsuba 3, CPU/LLVM backend) and shaded with

- a mucosa texture attached to the wall chart (axial repeat ``2R``, an integer number of
  circumferential repeats around the section perimeter, trilinear mip filtering by footprint),
- a head-light term ``min((0.2 + 0.8 |n.v|) * 1.25 / max(d / R_light, 0.08)^2 + 0.012, 1)``
  with a far attenuation ``exp(-0.055 * max(d / R_light - 2, 0))`` and display gamma ``1/2.2``.

Exposure is set per clip: ``R_light`` = median over every 4th frame of the per-frame median ray
depth inside the device field, divided by ``exposure_divisor`` (1.38). The image is finally
multiplied by the endoscope's device-field alpha (an anti-aliased octagon intersected with the
calibrated lens disk), so tube videos share the black border of the real videos.

Training tubes, benchmark tubes and synthetic full-colon tubes are all rendered by this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from colonoworld.tube.camera import Calibration, rays_from_clip
from colonoworld.tube.model import TubeModel

LUMA = np.array([0.299, 0.587, 0.114])
EXPOSURE_DIVISOR = 1.38
CORNERS = ("tl", "tr", "bl", "br")


# ----------------------------------------------------------------------------- wall texture
@lru_cache(maxsize=4)
def load_texture(path: str) -> tuple[np.ndarray, ...]:
    """Load the square RGB texture as linear-light mip levels (finest first)."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    level = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255) ** 2.2
    levels = [level]
    while len(level) > 1:
        level = cv2.resize(level, (len(level) // 2, len(level) // 2), interpolation=cv2.INTER_AREA)
        levels.append(level)
    return tuple(levels)


def _footprint(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mip level from the screen-space derivatives of the texture coordinates."""
    uy, ux = np.gradient(u)
    vy, vx = np.gradient(v)
    rho = 1024 * np.maximum(np.hypot(ux, vx), np.hypot(uy, vy))
    return np.clip(np.log2(np.maximum(1.0, rho)), 0, 10).astype(np.float32)


def _lookup(
    levels: tuple[np.ndarray, ...], u: np.ndarray, v: np.ndarray, lod: np.ndarray
) -> np.ndarray:
    """Trilinear texture lookup with wrap-around addressing."""
    import cv2

    flat_u, flat_v = u.ravel(), v.ravel()
    lev = np.floor(lod.ravel()).astype(np.int32)
    mix = lod.ravel() - lev
    out = np.zeros((len(flat_u), 3), np.float32)
    for offset in (0, 1):
        ids = np.minimum(lev + offset, 10)
        weight = mix if offset else 1 - mix
        for k in np.unique(ids):
            m = np.where((ids == k) & (weight > 0))[0]
            if not len(m):
                continue
            n = len(levels[k])
            x = ((flat_u[m] % 1) * n).astype(np.float32).reshape(-1, 1)
            y = ((flat_v[m] % 1) * n).astype(np.float32).reshape(-1, 1)
            for start in range(0, len(m), 30000):  # cv2.remap needs maps below SHRT_MAX
                sl = slice(start, start + 30000)
                vals = cv2.remap(
                    levels[k], x[sl], y[sl], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
                ).reshape(-1, 3)
                out[m[sl]] += vals * weight[m[sl], None]
    return out.reshape(*u.shape, 3)


def sample_texture(
    levels: tuple[np.ndarray, ...], s: np.ndarray, t: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Linear albedo for chart coordinates ``(s, t, f)``.

    The circumferential repeat count is blended between ``floor(f)`` and ``floor(f) + 1`` so the
    texture closes seamlessly around sections of any perimeter.
    """
    f = np.maximum(f, 1)
    lo = np.floor(f)
    mix = (f - lo)[..., None]
    v = s / 2
    u0, u1 = t * lo, t * (lo + 1)
    a = _lookup(levels, u0, v, _footprint(u0, v))
    b = _lookup(levels, u1, v, _footprint(u1, v))
    return a + (b - a) * mix


# ----------------------------------------------------------------------------- device field
def _corner_distance(lines: dict[str, list[float]], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.min([lines[k][0] * x + lines[k][1] * y - lines[k][2] for k in CORNERS], 0)


@lru_cache(maxsize=8)
def _octagon_alpha_cached(lines_key: tuple, height: int, width: int, samples: int) -> np.ndarray:
    lines = {k: list(v) for k, v in lines_key}
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    d = _corner_distance(lines, xx, yy)
    alpha = (d > 0).astype(np.float32)
    ys, xs = np.nonzero(np.abs(d) < 1.0)
    o = (np.arange(samples) + 0.5) / samples - 0.5
    oy, ox = np.meshgrid(o, o, indexing="ij")
    for i in range(0, len(ys), 20000):
        sy = ys[i : i + 20000, None, None] + oy[None]
        sx = xs[i : i + 20000, None, None] + ox[None]
        alpha[ys[i : i + 20000], xs[i : i + 20000]] = (_corner_distance(lines, sx, sy) > 0).mean(
            (1, 2)
        )
    return alpha


def octagon_alpha(
    lines: dict[str, list[float]], size: tuple[int, int] = (832, 1104), samples: int = 16
) -> np.ndarray:
    """Pixel coverage of the device field, the canvas cut by four corner lines.

    Each line ``[nx, ny, c]`` keeps the side ``nx * x + ny * y > c`` (pixel centres at integer
    coordinates). Coverage near a line is estimated with ``samples x samples`` sub-pixel points.
    """
    key = tuple((k, tuple(float(x) for x in lines[k])) for k in CORNERS)
    return _octagon_alpha_cached(key, int(size[0]), int(size[1]), int(samples)).copy()


def lens_alpha(
    cam: Calibration, size: tuple[int, int] = (832, 1104), samples: int = 8
) -> np.ndarray:
    """Pixel coverage of the calibrated lens disk (pixels whose centre has no ray get 0)."""
    import cv2

    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    ok = np.asarray(rays_from_clip(np.c_[xx.ravel(), yy.ravel()].astype(float), cam)[1])
    ok = ok.reshape(height, width)
    alpha = ok.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    mixed = (
        (cv2.dilate(ok.astype(np.uint8), k) > 0)
        & (cv2.erode(ok.astype(np.uint8), k, borderType=cv2.BORDER_REPLICATE) == 0)
        & ok
    )
    ys, xs = np.nonzero(mixed)
    o = (np.arange(samples) + 0.5) / samples - 0.5
    oy, ox = np.meshgrid(o, o, indexing="ij")
    if len(ys):
        pts = np.stack(
            [(xs[:, None, None] + ox[None]).ravel(), (ys[:, None, None] + oy[None]).ravel()], 1
        )
        v = np.asarray(rays_from_clip(pts, cam)[1]).reshape(len(ys), samples * samples)
        alpha[ys, xs] = v.mean(1)
    return alpha


def pick_device_field(
    frame_rgb: np.ndarray, fields: dict[str, dict[str, list[float]]], min_margin: float = 0.11
) -> tuple[str, float, dict[str, float]]:
    """Choose the device-field variant that matches a real frame (e.g. the clip's first frame).

    A pixel is "bright" when its max RGB is at least 10. Each variant is scored by its agreement
    with the bright mask on the band where the variants disagree; the best variant must beat the
    second by ``min_margin``.

    Returns ``(variant, margin, scores)``.
    """
    names = sorted(fields)
    size = frame_rgb.shape[:2]
    inside = {k: octagon_alpha(fields[k], size) >= 0.5 for k in names}
    band = np.any([inside[k] for k in names], 0) & ~np.all([inside[k] for k in names], 0)
    bright = np.asarray(frame_rgb).max(-1) >= 10
    scores = {k: float((bright[band] == inside[k][band]).mean()) for k in names}
    best = max(names, key=lambda k: (round(scores[k], 4), inside[k].sum()))
    ranked = sorted(scores.values())
    margin = ranked[-1] - ranked[-2]
    if margin < min_margin:
        raise ValueError(f"ambiguous device field: margin={margin:.4f}, scores={scores}")
    return best, margin, scores


def pick_device_field_from_frames(
    frames: list[np.ndarray], fields: dict[str, dict[str, list[float]]], min_margin: float = 0.11
) -> tuple[str, float, dict[str, float]]:
    """Device-field variant from the most decisive of several real frames (largest margin)."""
    best = None
    for frame in frames:
        try:
            pick = pick_device_field(frame, fields, 0.0)
        except ValueError:
            continue
        if best is None or pick[1] > best[1]:
            best = pick
    if best is None or best[1] < min_margin:
        raise ValueError(f"ambiguous device field in every frame: {best}")
    return best


def device_field_alpha(
    cam: Calibration, lines: dict[str, list[float]], size: tuple[int, int] = (832, 1104)
) -> np.ndarray:
    """Device-field alpha: octagon coverage times calibrated lens-disk coverage."""
    return octagon_alpha(lines, size) * lens_alpha(cam, size)


def apply_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Multiply a uint8 image by the field alpha (rounded)."""
    return np.round(image.astype(np.float32) * alpha[..., None]).astype(np.uint8)


# ----------------------------------------------------------------------------- ray casting
class TubeScene:
    """Wall mesh loaded into Mitsuba with per-pixel fisheye rays."""

    def __init__(
        self,
        tube: TubeModel,
        cam: Calibration,
        field: np.ndarray,
        texture: tuple[np.ndarray, ...] | None,
        size: tuple[int, int] = (832, 1104),
        threads: int = 1,
    ) -> None:
        import drjit as dr
        import mitsuba as mi

        mi.set_variant("llvm_ad_rgb")
        dr.set_thread_count(threads)
        self.mi = mi
        self.size = size
        height, width = size
        V = np.asarray(tube.vertices)  # normals use the stored precision; Mitsuba takes float32
        F = np.asarray(tube.faces)
        self.F = F
        self.field = field.astype(bool)
        yy, xx = np.mgrid[:height, :width]
        rays, ok = rays_from_clip(np.c_[xx.ravel(), yy.ravel()], cam)
        self.rays = rays.astype(np.float32)
        self.ok = ok
        mesh = mi.Mesh("tube_wall", len(V), len(F))
        params = mi.traverse(mesh)
        params["vertex_positions"] = mi.Float(V.ravel())
        params["faces"] = mi.UInt32(F.astype(np.uint32).ravel())
        params.update()
        self.scene = mi.load_dict({"type": "scene", "wall": mesh})
        tri = V[F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        self.vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        s, t, f = (np.asarray(c, dtype=np.float32) for c in tube.chart_tuple)
        self.suv, self.tuv, self.fuv = s[F], t[F].copy(), f[F]
        seam = np.ptp(self.tuv, axis=1) > 0.5  # faces straddling t = 0/1
        self.tuv[seam] = np.where(self.tuv[seam] < 0.5, self.tuv[seam] + 1, self.tuv[seam])
        self.texture = texture

    def _intersect(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
        mi = self.mi
        d = self.rays @ pose[:3, :3].T.astype(np.float32)
        o = pose[:3, 3].astype(np.float32)
        ray = mi.Ray3f(
            mi.Point3f(*[np.full(len(d), x, np.float32) for x in o]),
            mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]),
        )
        si = self.scene.ray_intersect_preliminary(ray, coherent=True)
        distance = np.array(si.t, np.float32)
        pid = np.array(si.prim_index, np.int64)
        hit = self.ok & np.isfinite(distance) & (pid >= 0) & (pid < len(self.F))
        return d, distance, (si, pid, hit)

    def depth(self, pose: np.ndarray) -> np.ndarray:
        """Ray distance per pixel (NaN outside the field or without a hit)."""
        _, distance, (_, _, hit) = self._intersect(pose)
        dep = np.where(hit, distance, np.nan).reshape(self.size)
        dep[~self.field] = np.nan
        return dep

    def frame(self, pose: np.ndarray, light_radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Shaded uint8 RGB image and depth for one camera-to-world pose."""
        height, width = self.size
        d, distance, (si, pid, hit) = self._intersect(pose)
        pid = np.where(hit, pid, 0)
        b1 = np.where(hit, np.array(si.prim_uv.x), 0).astype(np.float32)
        b2 = np.where(hit, np.array(si.prim_uv.y), 0).astype(np.float32)
        b0 = 1 - b1 - b2
        tri = self.F[pid]
        normal = (
            b0[:, None] * self.vn[tri[:, 0]]
            + b1[:, None] * self.vn[tri[:, 1]]
            + b2[:, None] * self.vn[tri[:, 2]]
        )
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        cosv = np.abs((normal * d).sum(1))
        dist = np.where(hit, distance / light_radius, 1e6)
        shade = np.minimum((0.2 + 0.8 * cosv) * 1.25 / np.maximum(dist, 0.08) ** 2 + 0.012, 1)
        shade = shade * np.exp(-0.055 * np.maximum(dist - 2, 0))
        bc = np.stack([b0, b1, b2], axis=1)
        ss = (self.suv[pid] * bc).sum(1).reshape(height, width)
        tt = (self.tuv[pid] * bc).sum(1).reshape(height, width)
        ff = (self.fuv[pid] * bc).sum(1).reshape(height, width)
        albedo = sample_texture(self.texture, ss, tt, ff)
        img = (255 * np.clip(albedo * shade.reshape(height, width, 1), 0, 1) ** (1 / 2.2)).astype(
            np.uint8
        )
        img[~self.field | ~hit.reshape(height, width)] = 0
        dep = np.where(hit, distance, np.nan).reshape(height, width)
        dep[~self.field] = np.nan
        return img, dep


# ----------------------------------------------------------------------------- public API
def resolve_field_alpha(intrinsics: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    """Device-field alpha from ``intrinsics``.

    Uses ``intrinsics["field_alpha"]`` when given; otherwise the octagon of
    ``intrinsics["device_fields"][intrinsics["device_field"]]`` times the lens disk of
    ``intrinsics["calibration"]``.
    """
    if intrinsics.get("field_alpha") is not None:
        return np.asarray(intrinsics["field_alpha"], dtype=np.float32)
    lines = intrinsics["device_fields"][intrinsics["device_field"]]
    return device_field_alpha(intrinsics["calibration"], lines, size)


def light_radius_for(
    scene: TubeScene, cameras: np.ndarray, field: np.ndarray, stride: int = 4
) -> float:
    """Median over every ``stride``-th frame of the per-frame median ray depth inside the field."""
    meds = []
    for t in range(0, len(cameras), stride):
        d = scene.depth(cameras[t])
        meds.append(np.median(d[field & np.isfinite(d) & (d > 0)]))
    return float(np.median(meds))


def render_tube_video(
    tube: TubeModel,
    cameras: np.ndarray,
    intrinsics: dict[str, Any],
    size: tuple[int, int] = (832, 1104),
    return_depth: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Render the tube seen from a camera path.

    Args:
        tube: wall model.
        cameras: ``(T, 4, 4)`` camera-to-world poses in the tube's world frame.
        intrinsics: ``calibration`` (fisheye calibration dict), ``texture`` (path to the mucosa
            texture image), the device field (``field_alpha`` array, or ``device_fields`` +
            ``device_field`` variant name) and optionally ``exposure_divisor`` (default 1.38),
            ``exposure_stride`` (default 4) and ``threads`` (Mitsuba worker threads, default 1).
        size: output ``(H, W)``; must match the calibration's clip canvas.
        return_depth: also return the per-frame depth at stride 8 and render statistics.

    Returns:
        ``(T, H, W, 3)`` uint8 frames, or ``(frames, depth (T, H/8, W/8) float16, info)``.
    """
    cam = intrinsics["calibration"]
    clip_w, clip_h = cam.get("clip_size", (size[1], size[0]))
    if (clip_h, clip_w) != tuple(size):
        raise ValueError(f"size {size} does not match the calibration canvas {(clip_h, clip_w)}")
    cameras = np.asarray(cameras, dtype=np.float64)
    alpha = resolve_field_alpha(intrinsics, size)
    field = alpha > 0
    texture = load_texture(str(intrinsics["texture"]))
    scene = TubeScene(tube, cam, field, texture, size, threads=int(intrinsics.get("threads", 1)))
    median_depth = light_radius_for(
        scene, cameras, field, int(intrinsics.get("exposure_stride", 4))
    )
    light_radius = median_depth / float(intrinsics.get("exposure_divisor", EXPOSURE_DIVISOR))
    frames, depth = [], []
    for pose in cameras:
        img, dep = scene.frame(pose, light_radius)
        frames.append(apply_alpha(img, alpha))
        if return_depth:
            depth.append(dep[::8, ::8].astype(np.float16))
    frames = np.asarray(frames)
    if not return_depth:
        return frames
    info = dict(light_radius=light_radius, median_depth=median_depth)
    return frames, np.asarray(depth), info


def write_tube_video(path: str | Path, frames: np.ndarray, fps: int = 24, crf: int = 18) -> None:
    """Encode tube frames as H.264 (yuv420p, CRF 18, preset fast), the format used for training."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    container = av.open(str(path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.codec_context.thread_count = 1
    stream.options = {"crf": str(crf), "preset": "fast", "threads": "1"}
    for img in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for packet in stream.encode(frame.reformat(format="yuv420p", threads=1)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
