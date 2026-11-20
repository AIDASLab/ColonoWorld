# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tube construction and rendering.

- :mod:`~colonoworld.tube.camera`: fisheye calibration and pose helpers.
- :mod:`~colonoworld.tube.wall_fit`: profile-sweep wall of a video window.
- :mod:`~colonoworld.tube.build_from_video`: tube of a real clip from frame 0 and its camera path.
- :mod:`~colonoworld.tube.colon_anatomy`: anatomical whole-colon wall.
- :mod:`~colonoworld.tube.synthetic_colon`: whole colon + withdrawal camera path.
- :mod:`~colonoworld.tube.render`: tube video renderer shared by all of the above.
"""

from colonoworld.tube.model import TubeModel

__all__ = ["TubeModel"]
