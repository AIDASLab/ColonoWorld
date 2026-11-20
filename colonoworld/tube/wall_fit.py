# SPDX-License-Identifier: CC-BY-NC-4.0
"""Profile-sweep wall model fitted to a single video window.

A video tube is described by a small parameter dict (JSON serialisable)::

    R_pose    reference radius in pose units (pose units = first-frame median scene depth)
    origin_R  (3,) origin of the sweep, in units of R_pose
    basis     (3, 3) rows (u, v, axis): the sweep advances along ``axis``
    z_R       (n,) knot stations along the axis, in units of R_pose
    profiles  (n, 5) per knot: centre offset x, y (in the u/v plane), log radius, and the
              anisotropy vector (cos, sin) * log(aspect) / 2 of an elliptic cross-section
    fold      haustral fold field: spacing_R, depth_frac, width_R, phase, taenia_rad, span_rad

Between knots the profile is interpolated with a shape-preserving PCHIP spline, so the
centreline can bend sharply and the section can narrow, widen or flatten freely between stations.
Haustral folds are crescents between three taeniae that indent the section by up to
``depth_frac``. The mesh is the open side wall (no end caps) so the lumen stays visible.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from colonoworld.tube.model import TubeModel

Params = dict[str, Any]


def interp_profile(p: Params, z: np.ndarray) -> np.ndarray:
    """Profile rows ``(..., 5)`` at axial stations ``z`` (PCHIP between knots)."""
    from scipy.interpolate import PchipInterpolator

    q = np.asarray(p["profiles"])
    zz = np.asarray(p["z_R"])
    return PchipInterpolator(zz, q, axis=0)(z)


def ellipse_radius(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Radius of the elliptic section described by profile rows ``q`` at polar angle ``phi``."""
    q = np.asarray(q)
    e = np.hypot(q[..., 3], q[..., 4])
    a = 0.5 * np.arctan2(q[..., 4], q[..., 3])
    r = np.exp(q[..., 2])
    d = phi - a
    return r / np.sqrt(np.exp(-2 * e) * np.cos(d) ** 2 + np.exp(2 * e) * np.sin(d) ** 2)


def centreline(p: Params, z: np.ndarray) -> np.ndarray:
    """Section centres (units of R_pose) at axial stations ``z``."""
    q = interp_profile(p, z)
    b = np.asarray(p["basis"])
    return (
        np.asarray(p["origin_R"])
        + z[..., None] * b[2]
        + q[..., 0, None] * b[0]
        + q[..., 1, None] * b[1]
    )


def arc_length(p: Params, z: np.ndarray) -> np.ndarray:
    """Arc length of the centreline from the first knot to station ``z`` (units of R_pose)."""
    zz = np.linspace(p["z_R"][0], p["z_R"][-1], 1600)
    cc = centreline(p, zz)
    ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))]
    return np.interp(z, zz, ss)


def fold_factor(p: Params, z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Radial scale ``1 - fold`` on the grid ``z x phi``.

    Folds repeat every ``spacing_R`` of arc length; consecutive folds alternate between one and two
    crescents, each centred between two of the three taeniae.
    """
    f = p["fold"]
    ss = arc_length(p, z)
    space, depth, width = f["spacing_R"], f["depth_frac"], f["width_R"]
    out = np.zeros((len(z), len(phi)))
    for j, sj in enumerate(np.arange(-space + f["phase"] * space, ss.max() + space, space)):
        sectors = [j % 3] if j % 3 else [j % 3, (j + 1) % 3]
        for sec in sectors:
            dph = np.angle(np.exp(1j * (phi - (f["taenia_rad"] + np.pi / 3 + sec * 2 * np.pi / 3))))
            x = np.clip(np.abs(dph) / (f["span_rad"] / 2), 0, 1)
            amp = np.cos(np.pi * x / 2) ** 2 * (x < 1)
            crest = sj + 0.08 * np.cos(dph)
            g = np.exp(-0.5 * ((ss[:, None] - crest[None]) / width) ** 2)
            out = np.maximum(out, depth * g * amp[None])
    return 1 - out


def sweep_mesh(
    p: Params, ds: float = 0.045, n_phi: int = 96
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], dict[str, np.ndarray]]:
    """Open side-wall mesh of the sweep.

    Returns:
        ``(vertices (V, 3) float32 in pose units, faces (F, 3) int32, chart (s, t, f), rings)``
        where ``rings`` holds the stations ``z``, centres ``C``, radii ``radius`` (units of R_pose)
        and angles ``phi``.
    """
    zk = np.asarray(p["z_R"])
    z = np.linspace(zk[0], zk[-1], max(100, int((zk[-1] - zk[0]) / ds) + 1))
    ph = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    q = interp_profile(p, z)
    rr = ellipse_radius(q[:, None, :], ph[None]) * fold_factor(p, z, ph)
    b = np.asarray(p["basis"])
    c = centreline(p, z)
    v = c[:, None] + rr[:, :, None] * (
        np.cos(ph)[None, :, None] * b[0] + np.sin(ph)[None, :, None] * b[1]
    )
    v = v.reshape(-1, 3)
    ns = len(z)
    ii = np.arange((ns - 1) * n_phi).reshape(-1, n_phi)
    jj = np.roll(ii, -1, axis=1)
    faces = np.stack(
        [np.stack([ii, jj, ii + n_phi], -1), np.stack([jj, jj + n_phi, ii + n_phi], -1)], -2
    )
    vertices = (v * p["R_pose"]).astype(np.float32)
    faces = faces.reshape(-1, 3).astype(np.int32)
    # Wall chart: arc length, normalised perimeter position and half perimeter (units of R_pose).
    rings = vertices.reshape(ns, n_phi, 3) / p["R_pose"]
    edges = np.linalg.norm(np.roll(rings, -1, axis=1) - rings, axis=2)
    circ = edges.sum(1)
    sv = np.repeat(arc_length(p, z), n_phi).astype(np.float32)
    tv = (
        (np.c_[np.zeros(ns), np.cumsum(edges, axis=1)[:, :-1]] / circ[:, None])
        .ravel()
        .astype(np.float32)
    )
    fv = np.repeat(circ / 2, n_phi).astype(np.float32)
    return vertices, faces, (sv, tv, fv), dict(z=z, C=c, radius=rr, phi=ph)


def contain_cameras(p: Params, poses: np.ndarray, ball: float = 0.36) -> tuple[Params, float]:
    """Inflate the sweep until a ball of ``ball * R_pose`` around every camera fits inside.

    The test uses the fold-free ellipse shrunk by the maximal fold depth. Returns the new params and
    the applied radius factor (>= 1).
    """
    p = copy.deepcopy(p)
    b = np.array(p["basis"])
    cam = poses[:, :3, 3] / p["R_pose"]
    d = cam - np.array(p["origin_R"])
    z = d @ b[2]
    xy = d @ b[:2].T
    q = interp_profile(p, z)
    dd = xy - q[:, :2]
    rho = np.linalg.norm(dd, axis=1)
    phi = np.arctan2(dd[:, 1], dd[:, 0])
    limit = ellipse_radius(q, phi) * (1 - p["fold"]["depth_frac"])
    factor = max(1.0, float(np.max((rho + ball) / limit)))
    p["profiles"] = np.asarray(p["profiles"])
    p["profiles"][:, 2] += np.log(factor)
    return p, factor


def axis_basis(f: np.ndarray) -> np.ndarray:
    """Right-handed basis ``(u, v, f)`` with ``f`` normalised and ``u`` horizontal when possible."""
    f = f / np.linalg.norm(f)
    u = np.cross([0, 1, 0], f)
    if np.linalg.norm(u) < 0.1:
        u = np.cross([1, 0, 0], f)
    u /= np.linalg.norm(u)
    v = np.cross(f, u)
    return np.stack([u, v, f])


def to_tube_model(p: Params, ds: float = 0.045, n_phi: int = 96) -> TubeModel:
    """Mesh the sweep and wrap it as a :class:`TubeModel` (world units = pose units)."""
    vertices, faces, chart, rings = sweep_mesh(p, ds, n_phi)
    scale = float(p["R_pose"])
    basis = np.asarray(p["basis"], dtype=np.float64)
    return TubeModel(
        centerline=rings["C"] * scale,
        frames=np.repeat(basis[None], len(rings["z"]), axis=0),
        radii=rings["radius"] * scale,
        angles=rings["phi"],
        arclength=arc_length(p, rings["z"]) * scale,
        vertices=vertices,
        faces=faces,
        chart=np.stack(chart),
        meta=dict(kind="profile_sweep", parameters=_jsonable(p)),
    )


def tube_from_surface(
    vertices: np.ndarray, faces: np.ndarray, chart: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> TubeModel:
    """Wrap a fitted ring mesh (leading rings of equal chart ``s``) as a :class:`TubeModel`."""
    chart_s = np.asarray(chart[0])
    n_phi = int((chart_s == chart_s[0]).sum())
    ns = len(vertices) // n_phi
    rings = np.asarray(vertices[: ns * n_phi], dtype=np.float64).reshape(ns, n_phi, 3)
    centres = rings.mean(1)
    radial = rings - centres[:, None]
    tangent = np.gradient(centres, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    u = radial[:, 0] - (radial[:, 0] * tangent).sum(1, keepdims=True) * tangent
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    frames = np.stack([u, np.cross(tangent, u), tangent], 1)
    arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))]
    return TubeModel(
        centerline=centres,
        frames=frames,
        radii=np.linalg.norm(radial, axis=2),
        angles=np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
        arclength=arclength,
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        chart=np.stack([np.asarray(c, dtype=np.float32) for c in chart]),
        meta=dict(kind="fitted_surface"),
    )


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x
