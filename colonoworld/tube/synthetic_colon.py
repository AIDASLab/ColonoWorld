# SPDX-License-Identifier: CC-BY-NC-4.0
"""Synthetic full colon and withdrawal camera path for full-procedure generation.

A full procedure is a cecum-to-rectum withdrawal through a whole synthetic colon:

1. :func:`colonoworld.tube.colon_anatomy.build_colon_wall` builds the anatomical wall of one shape
   (course of the eight segments, per-segment calibre, taeniae, haustra, semilunar folds, rectal
   valves, cecal pouch, wall deformations) and a base withdrawal path whose speed per segment
   matches the training videos, starting with a 189-frame dwell in the cecum.
2. :func:`replay_camera_path` makes the camera move like a real endoscope: per-frame viewing
   directions and lateral offsets (relative to the local wall) are replayed from training
   windows of the same segment (the *motion library*), cross-faded over 16 frames. Offsets are
   shrunk where the camera would come closer than the clearance target to the wall; where even
   the centreline is too close, or inside documented collapse zones, the camera moves toward the
   base path, which keeps maximal clearance.
3. The wall is rendered in 189-frame chunks (stride 188) with
   :func:`colonoworld.tube.render.render_tube_video`, exactly like the training tubes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from colonoworld.tube.model import TubeModel

SEGMENT_CLASSES = ("cecum", "ascending", "transverse", "descending", "sigmoid", "rectum")
NULL_SEGMENT = 6  # flexures and unknown labels
LIBRARY_ALIAS = {"hepatic flexure": "transverse", "splenic flexure": "descending"}
CROSSFADE = 16
DWELL_FRAMES = 189
CHUNK, STRIDE = 189, 188


# ----------------------------------------------------------------------------- geometry helpers
def _closest_triangle_distance(tris: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance from point ``p`` to each triangle (Ericson's closest-point test)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    v = vb / np.maximum(den, 1e-25)
    w = vc / np.maximum(den, 1e-25)
    out = a + v[:, None] * ab + w[:, None] * ac
    m = (d1 <= 0) & (d2 <= 0)
    out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    out[m] = c[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, 1e-25)
    out[m] = (a + t[:, None] * ab)[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, 1e-25)
    out[m] = (a + t[:, None] * ac)[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-25)
    out[m] = (b + t[:, None] * (c - b))[m]
    return np.linalg.norm(out - p, axis=1)


def surface_distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact distance from each point to the triangle mesh."""
    from scipy.spatial import cKDTree

    tri = vertices[faces].astype(float)
    cent = tri.mean(1)
    rad = np.linalg.norm(tri - cent[:, None], axis=2).max(1)
    rmax = rad.max()
    tree = cKDTree(cent)
    upper = cKDTree(vertices).query(points, workers=1)[0]
    dist = []
    for p, u in zip(points, upper, strict=True):
        idx = np.asarray(tree.query_ball_point(p, float(u + rmax + 1e-7)), int)
        idx = idx[np.linalg.norm(cent[idx] - p, axis=1) - rad[idx] <= u + 1e-7]
        dist.append(float(_closest_triangle_distance(tri[idx], p).min()))
    return np.array(dist)


def minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotations taking unit rows ``a`` to unit rows ``b`` by the smallest angle."""
    v = np.cross(a, b)
    c = np.einsum("ij,ij->i", a, b)
    k = np.zeros((len(a), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    out = (
        np.eye(3)[None]
        + k
        + np.einsum("nij,njk->nik", k, k) / (1 + np.maximum(c, -1 + 1e-6))[:, None, None]
    )
    for i in np.where(c < -1 + 1e-6)[0]:  # antiparallel: half turn about an orthogonal axis
        e = np.eye(3)[np.argmin(np.abs(a[i]))]
        u = np.cross(a[i], e)
        u /= np.linalg.norm(u)
        out[i] = 2 * np.outer(u, u) - np.eye(3)
    return out


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return minimal_rotation(a[None] / np.linalg.norm(a), b[None] / np.linalg.norm(b))[0]


def ring_tangents(centres: np.ndarray, s: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Unit tangents of a centreline smoothed over 0.2 local radii."""
    from scipy.ndimage import gaussian_filter1d

    ds = float(np.median(np.diff(s)))
    sigma = max(1.0, 0.2 * float(np.median(radius)) / ds)
    t = np.gradient(gaussian_filter1d(centres, sigma, axis=0, mode="nearest"), s, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _interp_rows(q: np.ndarray, grid: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(q, grid, a[:, k]) for k in range(a.shape[1])], 1)


def _interp_frames(q: np.ndarray, grid: np.ndarray, frames: np.ndarray) -> np.ndarray:
    ii = np.clip(np.searchsorted(grid, q), 1, len(grid) - 1)
    w = (q - grid[ii - 1]) / (grid[ii] - grid[ii - 1])
    out = []
    for j, x in zip(ii, w, strict=True):
        u, _, vt = np.linalg.svd((1 - x) * frames[j - 1] + x * frames[j])
        out.append(u @ vt)
    return np.asarray(out)


def _chord_radius(radial_rows: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Wall distance of the polygonal section along the direction of each ``xy``."""
    nphi = radial_rows.shape[1]
    ph = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    u = ph * nphi / (2 * np.pi)
    j0 = np.floor(u).astype(int) % nphi
    j1 = (j0 + 1) % nphi
    row = np.arange(len(xy))
    ray = np.c_[np.cos(ph), np.sin(ph)]
    a = 2 * np.pi * j0 / nphi
    b = 2 * np.pi * j1 / nphi
    p0 = radial_rows[row, j0, None] * np.c_[np.cos(a), np.sin(a)]
    p1 = radial_rows[row, j1, None] * np.c_[np.cos(b), np.sin(b)]
    e = p1 - p0
    return (p0[:, 0] * e[:, 1] - p0[:, 1] * e[:, 0]) / (ray[:, 0] * e[:, 1] - ray[:, 1] * e[:, 0])


def _inside_sections(points, centres, frames, radii, ref_radius):
    xy = np.einsum("nji,nj->ni", frames, points - centres)[:, :2]
    margin = _chord_radius(radii, xy) - np.linalg.norm(xy, axis=1)
    return margin > 1e-8 * ref_radius, margin


def _crossfade_weights(seed: np.ndarray) -> np.ndarray:
    """Linear ramps of ``CROSSFADE`` frames on each side of every seeded frame."""
    w = seed.astype(float).copy()
    for i in np.flatnonzero(seed):
        lo, hi = max(0, i - CROSSFADE), min(len(w), i + CROSSFADE + 1)
        j = np.arange(lo, hi)
        w[lo:hi] = np.maximum(w[lo:hi], np.maximum(0.0, 1.0 - np.abs(j - i) / (CROSSFADE + 1)))
    return w


# ----------------------------------------------------------------------------- motion library
def window_motion(
    vertices: np.ndarray, chart_s: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame wall-relative motion of one training tube.

    Returns ``(look, offset)``: the local tube tangent expressed in camera coordinates, and the
    camera's lateral offset from the nearest section centre in camera coordinates divided by the
    section's mean radius.
    """
    v = vertices.astype(float)
    nphi = int((chart_s == chart_s[0]).sum())
    ns = len(v) // nphi
    rings = v[: ns * nphi].reshape(ns, nphi, 3)
    c = rings.mean(1)
    rm = np.linalg.norm(rings - c[:, None], axis=2).mean(1)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    t = ring_tangents(c, s, rm)
    near = np.array([np.argmin(np.linalg.norm(c - x, axis=1)) for x in poses[:, :3, 3]])
    rot = poses[:, :3, :3]
    look = np.einsum("nji,nj->ni", rot, t[near])
    off = poses[:, :3, 3] - c[near]
    off -= np.einsum("ni,ni->n", off, t[near])[:, None] * t[near]
    offset = np.einsum("nji,nj->ni", rot, off) / rm[near][:, None]
    return look, offset


def build_motion_library(windows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Motion library from training tubes.

    Args:
        windows: dicts with ``segment`` (anatomical segment name), ``vertices`` and ``chart_s`` of
            the fitted training tube mesh (rings of equal ``chart_s``), and its ``poses``.

    Returns:
        Arrays for :func:`save_motion_library` (windows keep their order).
    """
    segs, lengths, looks, offsets = [], [], [], []
    for w in windows:
        look, offset = window_motion(w["vertices"], w["chart_s"], w["poses"])
        segs.append(w.get("segment") or "transverse")
        lengths.append(len(look))
        looks.append(look)
        offsets.append(offset)
    return dict(
        segment=np.array(segs),
        length=np.array(lengths, np.int64),
        look=np.concatenate(looks),
        offset=np.concatenate(offsets),
    )


def save_motion_library(path: str, library: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **library)


def load_motion_library(path: str) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load a motion library as ``{segment: [(look (T, 3), offset (T, 3)), ...]}``."""
    with np.load(path, allow_pickle=False) as z:
        ends = np.cumsum(z["length"])
        out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for seg, end, n in zip(z["segment"], ends, z["length"], strict=True):
            out.setdefault(str(seg), []).append(
                (z["look"][end - n : end], z["offset"][end - n : end])
            )
    return out


# ----------------------------------------------------------------------------- camera replay
def wall_chart_f(wall: dict[str, np.ndarray]) -> np.ndarray:
    """Per-vertex half section perimeter in units of the reference radius (texture repeats)."""
    from scipy.spatial import cKDTree

    v = wall["V"].astype(float)
    c = wall["C"].astype(float)
    nph = wall["radial"].shape[1]
    nr = len(wall["s_ring"])
    radius = float(wall["R"])
    ring = v[: nr * nph].reshape(nr, nph, 3)
    circ = np.linalg.norm(np.roll(ring, -1, axis=1) - ring, axis=2).sum(1)
    cf_ring = circ / (2 * radius)
    _, ri = cKDTree(c).query(v)
    cf = cf_ring[ri].astype(np.float32)
    cf[: nr * nph] = np.repeat(cf_ring, nph)
    return cf


def collapse_zones(
    wall: dict[str, np.ndarray], deformations: list[dict], camera_s: np.ndarray
) -> np.ndarray:
    """Frames within two sigma of a documented wall collapse."""
    names = [str(x) for x in wall["anatomy"]]
    bounds = wall["anatomy_bounds"].astype(float)
    trend = wall["radius_trend"].astype(float)
    s = wall["s_ring"].astype(float)
    mask = np.zeros(len(camera_s), bool)
    for event in deformations:
        if event.get("kind") != "collapse":
            continue
        si = names.index(event["segment"])
        centre = float(bounds[si] + float(event.get("at", 0.5)) * (bounds[si + 1] - bounds[si]))
        diameter = float(event.get("local_diameter_units", 2 * np.interp(centre, s, trend)))
        sigma = float(event.get("sigma_D", 0.5)) * diameter
        mask |= np.abs(camera_s - centre) <= 2 * sigma
    return mask


def replay_camera_path(
    wall: dict[str, np.ndarray],
    library: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    deformations: list[dict] | None = None,
    camera_replay_seed: int = 20261111,
    view_gain: float | None = None,
    clearance_target: float = 0.35,
    shrink: float = 0.7,
    passes: int = 8,
    min_clearance: float = 0.30,
) -> np.ndarray:
    """Withdrawal camera poses with endoscope-like view directions and wall offsets.

    Args:
        wall: arrays from :func:`~colonoworld.tube.colon_anatomy.build_colon_wall`.
        library: motion library (:func:`load_motion_library`).
        deformations: wall deformation events (collapse zones keep the base path).
        camera_replay_seed: RNG seed for the choice of replayed training windows.
        view_gain: when set, the off-axis components of the replayed look direction are scaled by
            this gain (the camera looks further toward the wall).
        clearance_target: offsets are shrunk (factor ``shrink``, up to ``passes`` times) until the
            wall distance is at least this many reference radii.
        min_clearance: final minimum wall distance (reference radii; local mean section radius
            inside collapse zones).

    Returns:
        ``(T, 4, 4)`` camera-to-world poses.
    """
    v = wall["V"].astype(float)
    f = wall["F"]
    radius = float(wall["R"])
    c = wall["C"].astype(float)
    s = wall["s_ring"].astype(float)
    rad = wall["radial"].astype(float)
    rm = rad.mean(1)
    frames_ring = wall["B"].astype(float)
    p0 = wall["poses"].copy()
    p0[0, :3, :3] = p0[1, :3, :3]
    cs = wall["camera_s"].astype(float)
    labels = [str(x) for x in wall["labels"]]
    tan = ring_tangents(c, s, rm)
    cc = np.stack([np.interp(cs, s, c[:, k]) for k in range(3)], 1)
    tc = np.stack([np.interp(cs, s, tan[:, k]) for k in range(3)], 1)
    tc /= np.linalg.norm(tc, axis=1, keepdims=True)
    rc = np.interp(cs, s, rm)
    bc = _interp_frames(cs, s, frames_ring)
    rr = _interp_rows(cs, s, rad)
    rm_frame = rr.mean(1)
    exempt = collapse_zones(wall, deformations or [], cs)

    def clearance(points):
        distance = surface_distance(v, f, points)
        return distance / np.where(exempt, rm_frame, radius)

    # Replayed per-frame look directions (camera coords) and offsets, cross-faded between windows.
    rng = np.random.default_rng(camera_replay_seed)
    n = len(p0)
    look = np.zeros((n, 3))
    offset = np.zeros((n, 3))
    t, prev = 0, None
    while t < n:
        group = LIBRARY_ALIAS.get(labels[t], labels[t])
        pool = library.get(group) or library["transverse"]
        a_w, d_w = pool[rng.integers(len(pool))]
        m = len(a_w)
        for j in range(m):
            if t + j >= n:
                break
            a, d = a_w[j], d_w[j]
            if prev is not None and j < len(prev[0]):
                x = (j + 1) / (len(prev[0]) + 1)
                a = (1 - x) * prev[0][j] + x * a
                d = (1 - x) * prev[1][j] + x * d
            if view_gain is None:
                look[t + j] = a / np.linalg.norm(a)
            else:
                a = np.asarray(a, float) / np.linalg.norm(a)
                a = np.array([view_gain * a[0], view_gain * a[1], a[2]])
                look[t + j] = a / np.linalg.norm(a)
            offset[t + j] = d
        overlap = min(CROSSFADE, m - 1)
        prev = (a_w[m - overlap :], d_w[m - overlap :])
        t += max(m - overlap, 1)

    poses = p0.copy()
    scale = np.ones(n)

    def place():
        for i in range(n):
            rs = p0[i, :3, :3]
            b = rs.T @ tc[i]
            q = np.eye(3) if i < DWELL_FRAMES else _rotation_between(look[i], b)
            rn = rs @ q
            off = rn @ (offset[i] * scale[i] * rc[i])
            off -= (off @ tc[i]) * tc[i]
            poses[i, :3, :3] = rn
            poses[i, :3, 3] = cc[i] + off

    for _ in range(passes):
        place()
        bad = surface_distance(v, f, poses[:, :3, 3]) / radius < clearance_target
        if not bad.any():
            break
        scale[bad] *= shrink
    place()
    inside, _ = _inside_sections(poses[:, :3, 3], cc, bc, rr, radius)
    current_bad = (surface_distance(v, f, poses[:, :3, 3]) / radius < clearance_target) | ~inside
    zero_inside, _ = _inside_sections(cc, cc, bc, rr, radius)
    zero_bad = (surface_distance(v, f, cc) / radius < clearance_target) | ~zero_inside
    scale[current_bad & ~zero_bad] = 0.0  # the centreline suffices: drop the offset entirely
    place()
    # Collapse zones, and frames where even the centreline is too close, move toward the base path.
    move_seed = exempt | (current_bad & zero_bad)
    if move_seed.any():
        target = p0[:, :3, 3].copy()
        scale[move_seed] = 0.0
        place()
        replay = poses[:, :3, 3].copy()
        seeds = move_seed.copy()
        weight = np.zeros(n)
        for _ in range(8):
            weight = np.maximum(weight, _crossfade_weights(seeds))
            poses[:, :3, 3] = replay + weight[:, None] * (target - replay)
            inside, _ = _inside_sections(poses[:, :3, 3], cc, bc, rr, radius)
            seeds = (clearance(poses[:, :3, 3]) < min_clearance) | ~inside
            if not seeds.any():
                break
    poses[0, :3, :3] = poses[1, :3, :3]
    inside, _ = _inside_sections(poses[:, :3, 3], cc, bc, rr, radius)
    if not inside.all() or clearance(poses[:, :3, 3]).min() < min_clearance * 0.999:
        raise ValueError("replayed camera path does not keep the required wall clearance")
    return poses


# ----------------------------------------------------------------------------- public API
def segment_runs(labels: np.ndarray) -> list[dict[str, Any]]:
    """Consecutive runs of per-frame segment labels.

    Each run is ``{"start", "end" (inclusive), "name", "segment_id"}``; ``segment_id`` indexes
    ``SEGMENT_CLASSES`` (cecum 0 ... rectum 5) and is ``NULL_SEGMENT`` (6) for flexures.
    """
    runs = []
    labels = [str(x) for x in labels]
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            name = labels[start]
            sid = SEGMENT_CLASSES.index(name) if name in SEGMENT_CLASSES else NULL_SEGMENT
            runs.append(dict(start=start, end=i - 1, name=name, segment_id=sid))
            start = i
    return runs


def chunk_ranges(n_frames: int, chunk: int = CHUNK, stride: int = STRIDE) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges of the generation chunks (one shared frame between chunks)."""
    return [(k * stride, k * stride + chunk) for k in range((n_frames - 1) // stride)]


def wall_to_tube(wall: dict[str, np.ndarray]) -> TubeModel:
    """Wrap the anatomical wall as a :class:`TubeModel` (vertices kept at double precision)."""
    s = wall["s_ring"].astype(float)
    names = [str(x) for x in wall["anatomy"]]
    bounds = wall["anatomy_bounds"].astype(float)
    ring_label = np.array(names)[
        np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, len(names) - 1)
    ]
    nphi = wall["radial"].shape[1]
    return TubeModel(
        centerline=wall["C"].astype(float),
        frames=wall["B"].astype(float).transpose(0, 2, 1),
        radii=wall["radial"].astype(float),
        angles=np.linspace(0, 2 * np.pi, nphi, endpoint=False),
        arclength=s,
        vertices=wall["V"].astype(float),
        faces=wall["F"],
        chart=np.stack([wall["chart_s"], wall["chart_t"], wall_chart_f(wall)]).astype(np.float32),
        labels=ring_label,
        meta=dict(kind="synthetic_colon", reference_radius=float(wall["R"])),
    )


def build_full_colon(
    shape_name: str, cfg: dict[str, Any]
) -> tuple[TubeModel, np.ndarray, list[dict[str, Any]]]:
    """Synthetic colon of one shape with its withdrawal camera path.

    Args:
        shape_name: one of ``cfg["shapes"]`` (standard, u_shaped, v_shaped, short_compact,
            twisted, long_sigmoid, textbook).
        cfg: ``configs/full_colon_shapes.yaml`` loaded with
            :func:`colonoworld.utils.config.load_config`.

    Returns:
        ``(tube, cameras (T, 4, 4), segments)`` where ``segments`` lists the anatomical segment of
        each run of frames (see :func:`segment_runs`).
    """
    from colonoworld.tube.colon_anatomy import build_colon_wall, load_donor

    shape = cfg["shapes"][shape_name]
    full = cfg["full_colon"]
    wall, deformations = build_colon_wall(shape, load_donor(cfg["donor_template"]))
    library = load_motion_library(full["motion_library"])
    replay = {**full["camera_replay"], **(shape.get("camera_replay") or {})}
    cameras = replay_camera_path(wall, library, deformations, **replay)
    tube = wall_to_tube(wall)
    tube.meta.update(shape=shape_name)
    return tube, cameras, segment_runs(wall["labels"])


def full_colon_intrinsics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Render settings (``intrinsics`` of :func:`render_tube_video`) for synthetic videos."""
    render = cfg["render"]
    return dict(
        calibration=cfg["full_colon"]["calibration"],
        texture=render["texture"],
        device_fields=render["device_fields"],
        device_field=cfg["full_colon"]["device_field"],
        exposure_divisor=render.get("exposure_divisor", 1.38),
        exposure_stride=render.get("exposure_stride", 4),
        threads=render.get("threads", 1),
    )
