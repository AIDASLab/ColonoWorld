# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tube model: a free-form 3D lumen wall around a centreline.

The wall is a sweep of cross-sections along a centreline. Each cross-section ``i`` sits at
``centerline[i]`` in the plane spanned by the first two axes of ``frames[i]`` and is described by
its wall distance ``radii[i, k]`` at the polar angles ``angles[k]``. Nothing constrains the
sweep to a pipe: the centreline may bend sharply, and radii may vary freely around and along the
lumen (narrowing, flattening, folds). The triangle mesh (``vertices``/``faces``) and its wall
texture chart are kept with the model so that every consumer renders exactly the same surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TubeModel:
    """Free-form lumen wall.

    Attributes:
        centerline: ``(N, 3)`` cross-section centres (world units of the camera path).
        frames: ``(N, 3, 3)`` per-section orthonormal frames; rows are (u, v, tangent).
        radii: ``(N, K)`` wall distance from the centre at each angle in ``angles``.
        angles: ``(K,)`` polar angles of the radii, measured from ``u`` toward ``v``.
        arclength: ``(N,)`` arc length of the cross-sections along the centreline.
        vertices: ``(V, 3)`` wall mesh vertices (float32 or float64; normals are computed at this
            precision).
        faces: ``(F, 3)`` int32 triangle indices (the mesh is an open or capped side wall).
        chart: ``(3, V)`` float32 wall chart per vertex: arc length ``s`` (in units of the
            texture scale), circumferential coordinate ``t`` in [0, 1), and half circumference
            ``f`` (sets the number of circumferential texture repeats).
        labels: optional ``(N,)`` anatomical segment name per cross-section.
        meta: free-form JSON-serialisable metadata (e.g. construction parameters).
    """

    centerline: np.ndarray
    frames: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    arclength: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    chart: np.ndarray
    labels: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chart_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The chart as the ``(s, t, f)`` triple consumed by the renderer."""
        return self.chart[0], self.chart[1], self.chart[2]

    def save(self, path: str | Path) -> None:
        """Write the model to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            centerline=self.centerline,
            frames=self.frames,
            radii=self.radii,
            angles=self.angles,
            arclength=self.arclength,
            vertices=self.vertices,
            faces=self.faces,
            chart=self.chart,
            meta=np.array(json.dumps(self.meta)),
        )
        if self.labels is not None:
            arrays["labels"] = np.asarray(self.labels)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> TubeModel:
        """Read a model written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            return cls(
                centerline=z["centerline"],
                frames=z["frames"],
                radii=z["radii"],
                angles=z["angles"],
                arclength=z["arclength"],
                vertices=z["vertices"],
                faces=z["faces"],
                chart=z["chart"],
                labels=z["labels"] if "labels" in z.files else None,
                meta=json.loads(str(z["meta"])),
            )
