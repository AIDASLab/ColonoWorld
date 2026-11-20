# SPDX-License-Identifier: CC-BY-NC-4.0
"""Evaluation: camera control (read-back), FVD / FID / IS, lumen following.

Submodules are imported on use so that heavy optional dependencies (torch, PyAV, OpenCV, SciPy)
are loaded only by the metric that needs them.
"""

__all__ = ["camera_metrics", "fvd_fid_is", "lumen_following"]
