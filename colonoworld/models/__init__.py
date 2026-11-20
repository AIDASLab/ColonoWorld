# SPDX-License-Identifier: CC-BY-NC-4.0
"""ColonoWorld model: Cosmos video backbone with camera, tube, mask and global conditioning.

Public API::

    from colonoworld.models import load_model
    model = load_model(cfg)            # cfg = load_config("configs/model.yaml")
    video = model.generate(first_frame, cameras, tube_video, cond_mask=None, intrinsics=calib)
"""

from colonoworld.models.checkpoint import canonical_name, export_checkpoint, read_state
from colonoworld.models.conditions import (
    GLOBAL_IN,
    SEGMENT_CLASSES,
    SEGMENT_NULL,
    global_flags,
    segment_ids_from_labels,
)
from colonoworld.models.loader import ColonoWorldModel, build_prompts, load_model

__all__ = [
    "GLOBAL_IN",
    "SEGMENT_CLASSES",
    "SEGMENT_NULL",
    "ColonoWorldModel",
    "build_prompts",
    "canonical_name",
    "export_checkpoint",
    "global_flags",
    "load_model",
    "read_state",
    "segment_ids_from_labels",
]
