# SPDX-License-Identifier: CC-BY-NC-4.0
"""3D polyp condition.

A polyp is a small mesh placed on the fitted lumen wall: base centre ``c`` on the wall, wall normal
``n`` (pointing into the lumen), diameter ``D`` in tube pose units (``D = size_mm / 25 * R`` with
the tube radius ``R`` corresponding to a 25 mm lumen radius). Its silhouette is ray-cast on every
frame against the union of wall and polyp with the exact fisheye camera rays and per-frame poses,
so the wall occludes it correctly. The result is a boolean mask video for the ``polyp`` palette
class. Ray casting uses Mitsuba 3 (CPU ``llvm_ad_rgb`` variant), imported on first use.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MM_PER_R = 25.0
KINDS = ("sessile", "pedunculated", "flat")


# ----------------------------------------------------------------------------- camera rays


@functools.cache
def _theta_limit(coefficients: tuple[float, ...]) -> float:
    k = np.asarray(coefficients)
    roots = np.polynomial.polynomial.polyroots(
        [1, 0, 3 * k[0], 0, 5 * k[1], 0, 7 * k[2], 0, 9 * k[3]]
    )
    turning = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < np.pi / 2]
    return min([np.pi / 2 - 1e-7] + turning) - 1e-9


def _kb4_radius(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    return theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)


def pixel_rays(intrinsics: dict, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Unit camera rays of every pixel (row-major) and their validity.

    ``intrinsics`` holds ``K`` (3x3, output pixels) and ``D`` (4 KB4 coefficients). When the
    calibration was made at the native sensor resolution it may also hold ``K_native``,
    ``native_to_clip`` (3x3 affine with ``K = native_to_clip @ K_native``) and
    ``field_radius_px`` (native pixels); rays outside that radius are invalid.

    Returns:
        ``rays`` ``(H*W, 3)`` float64 (zero where invalid) and ``valid`` ``(H*W,)`` bool.
    """
    height, width = size
    yy, xx = np.mgrid[:height, :width]
    uv = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    d = np.asarray(intrinsics["D"], np.float64)
    if "K_native" in intrinsics:
        affine = np.asarray(intrinsics["native_to_clip"], np.float64)
        k = np.asarray(intrinsics["K_native"], np.float64)
    else:
        affine = np.eye(3)
        k = np.asarray(intrinsics["K"], np.float64)
    native = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(affine).T
    native = native[:, :2] / native[:, 2:3]
    xy = np.c_[native, np.ones(len(native))] @ np.linalg.inv(k).T
    xy = xy[:, :2] / xy[:, 2:3]
    rd = np.linalg.norm(xy, axis=1)
    hi = np.full(len(rd), _theta_limit(tuple(d)))
    lo = np.zeros(len(rd))
    valid = np.isfinite(rd) & (rd < _kb4_radius(hi, d))
    if "field_radius_px" in intrinsics:
        valid &= np.linalg.norm(native - k[:2, 2], axis=1) <= intrinsics["field_radius_px"]
    for _ in range(48):
        mid = (lo + hi) / 2
        less = _kb4_radius(mid, d) < rd
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    theta = (lo + hi) / 2
    scale = np.divide(np.sin(theta), rd, out=np.ones_like(rd), where=rd > 1e-12)
    rays = np.c_[xy * scale[:, None], np.cos(theta)]
    rays[~valid] = 0
    return rays, valid


# ----------------------------------------------------------------------------- polyp mesh


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _sphere(n_lat: int = 12, n_lon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    verts = np.array(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph]
    )
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            faces += [[a, a + n_lon, b], [b, a + n_lon, b + n_lon]]
    return verts, np.array(faces)


def polyp_mesh(
    kind: str, diameter: float, centre: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Polyp mesh in pose units on the wall at ``centre`` with outward ``normal``.

    ``sessile``: ellipsoid of height 0.6 D; ``flat``: height 0.15 D; ``pedunculated``: stalk plus
    spherical head. The base is sunk 0.15 D (flat: 0.05 D) into the wall.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    n = normal
    t1 = _unit(np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0]))
    t2 = np.cross(n, t1)
    sphere, sphere_faces = _sphere()
    parts = []

    def place(local: np.ndarray) -> None:
        world = centre + local[:, 0:1] * t1 + local[:, 1:2] * t2 + local[:, 2:3] * n
        parts.append((world, sphere_faces))

    a = diameter / 2
    if kind == "sessile":
        place(sphere * [a, a, 0.6 * diameter] - [0, 0, 0.15 * diameter])
    elif kind == "flat":
        place(sphere * [a, a, 0.15 * diameter] - [0, 0, 0.05 * diameter])
    else:
        stalk = 0.8 * diameter
        place(
            sphere * [0.15 * diameter, 0.15 * diameter, stalk / 2 + 0.15 * diameter]
            + [0, 0, stalk / 2 - 0.15 * diameter]
        )
        place(sphere * [a, a, a] + [0, 0, stalk + 0.4 * diameter])
    verts = np.vstack([p[0] for p in parts])
    offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    faces = np.vstack([p[1] + o for p, o in zip(parts, offsets, strict=True)])
    return verts, faces


# ----------------------------------------------------------------------------- ray casting


def _mitsuba():
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    return mi


def _scene(verts: np.ndarray, faces: np.ndarray):
    mi = _mitsuba()
    mesh = mi.Mesh("m", len(verts), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(verts.astype(np.float32).ravel())
    params["faces"] = mi.UInt32(faces.astype(np.uint32).ravel())
    params.update()
    return mi.load_dict({"type": "scene", "m": mesh})


def _cast(scene, rays: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First hit of camera rays from a camera-to-world pose: range, face id, world direction."""
    mi = _mitsuba()
    d = (rays @ pose[:3, :3].T).astype(np.float32)
    o = pose[:3, 3].astype(np.float32)
    origin = mi.Point3f(*[np.full(len(d), x, np.float32) for x in o])
    ray = mi.Ray3f(origin, mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]))
    hit = scene.ray_intersect_preliminary(ray, coherent=True)
    return np.array(hit.t, np.float32), np.array(hit.prim_index, np.int64), d


class PolypRenderer:
    """Ray-casts polyps on a fitted tube surface seen by a fisheye camera.

    Args:
        vertices: ``(V, 3)`` wall mesh vertices (pose units).
        faces: ``(F, 3)`` wall mesh triangles.
        poses: ``(T, 4, 4)`` camera-to-world poses (camera x right, y down, z forward).
        radius: tube radius ``R`` in pose units (25 mm).
        field_mask: ``(H, W)`` boolean endoscope field.
        intrinsics: camera intrinsics, see :func:`pixel_rays`.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        poses: np.ndarray,
        radius: float,
        field_mask: np.ndarray,
        intrinsics: dict,
    ) -> None:
        self.vertices, self.faces = vertices, faces
        self.poses, self.radius = poses, float(radius)
        self.field = field_mask.astype(bool)
        self.size = self.field.shape
        self.rays, self.valid = pixel_rays(intrinsics, self.size)

    @classmethod
    def from_files(cls, surface: str | Path, intrinsics: dict) -> PolypRenderer:
        """Build from a tube surface ``.npz`` (``V``, ``F``, ``poses``, ``R``, ``field_mask``)."""
        with np.load(surface, allow_pickle=False) as z:
            return cls(z["V"], z["F"], z["poses"], float(z["R"]), z["field_mask"], intrinsics)

    def diameter(self, size_mm: float) -> float:
        """Polyp diameter in pose units for a size in millimetres."""
        return size_mm / MM_PER_R * self.radius

    def wall_point(self, frame: int, pixel: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Wall point and lumen-facing normal hit by the ray of ``pixel`` (x, y) at ``frame``."""
        index = np.array([pixel[1] * self.size[1] + pixel[0]])
        t, prim, d = _cast(_scene(self.vertices, self.faces), self.rays[index], self.poses[frame])
        if not (self.valid[index[0]] and np.isfinite(t[0]) and 0 <= prim[0] < len(self.faces)):
            raise ValueError(f"pixel {tuple(pixel)} does not hit the wall at frame {frame}")
        tri = self.vertices[self.faces[prim]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        normal *= -np.sign((normal * d).sum(1, keepdims=True))
        centre = self.poses[frame][:3, 3] + d * t[:, None]
        return centre[0], normal[0]

    def render(
        self,
        kind: str,
        diameter: float,
        centre: np.ndarray,
        normal: np.ndarray,
        frames: Sequence[int] | None = None,
        pixels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Polyp silhouettes ``(len(frames), H*W)`` (or ``(len(frames), len(pixels))``)."""
        num_wall = len(self.faces)
        pverts, pfaces = polyp_mesh(
            kind, diameter, np.asarray(centre, float), np.asarray(normal, float)
        )
        scene = _scene(
            np.vstack([self.vertices, pverts]), np.vstack([self.faces, pfaces + len(self.vertices)])
        )
        frames = range(len(self.poses)) if frames is None else frames
        ok = self.valid & self.field.ravel()
        rays = self.rays
        if pixels is not None:
            ok, rays = ok[pixels], rays[pixels]
        out = []
        for t in frames:
            hit_t, prim, _ = _cast(scene, rays, self.poses[t])
            out.append(ok & np.isfinite(hit_t) & (prim >= num_wall))
        return np.stack(out)

    def choose_size(
        self,
        kind: str,
        centre: np.ndarray,
        normal: np.ndarray,
        sizes_mm: Sequence[float],
        target_area: float,
        grid_px: int = 4,
    ) -> tuple[float, list[tuple[float, float]]]:
        """Size whose median in-view area fraction is closest to ``target_area`` (coarse grid)."""
        height, width = self.size
        yy, xx = np.mgrid[:height:grid_px, :width:grid_px]
        grid = (yy * width + xx).ravel()
        scan = []
        for mm in sizes_mm:
            masks = self.render(kind, self.diameter(mm), centre, normal, pixels=grid)
            area = masks.sum(1) * grid_px**2 / (height * width)
            median = float(np.median(area[area > 0]))
            scan.append((abs(median - target_area), mm, median))
        return min(scan)[1], [(mm, median) for _, mm, median in scan]


def insert_polyp(
    renderer: PolypRenderer,
    centre: np.ndarray,
    normal: np.ndarray,
    size_mm: float,
    kind: str = "sessile",
    onset: int = 0,
    offset: int | None = None,
) -> np.ndarray:
    """Boolean polyp mask video ``(T, H, W)``; frames outside ``[onset, offset)`` are cleared."""
    masks = renderer.render(kind, renderer.diameter(size_mm), centre, normal)
    masks = masks.reshape(len(masks), *renderer.size)
    masks[:onset] = False
    if offset is not None:
        masks[offset:] = False
    return masks
