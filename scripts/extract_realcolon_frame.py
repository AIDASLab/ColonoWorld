# SPDX-License-Identifier: CC-BY-NC-4.0
"""Map a REAL-Colon frame onto the 1104x832 clip canvas used by ColonoWorld.

Start frames and reference frames are not distributed with the code. Take them from your own
REAL-Colon download (https://figshare.com/articles/media/REAL-colon_dataset/22202866):

    # cecum start frame of the full-procedure examples: video 003-003, frame 25114
    python scripts/extract_realcolon_frame.py --config configs/full_procedure.yaml \\
        --frame /path/to/REAL-Colon/<frame 25114 of video 003-003> \\
        --out ${COLONOWORLD_DATA}/full_procedure/cecum_frame0.png

The frame is warped with the native-to-clip affine of the recording device
(``frame_sources.<name>.native_to_clip`` in the config); a frame that is already 1104x832 is
copied unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from colonoworld.utils.config import load_config

CANVAS = (1104, 832)


def to_canvas(image: np.ndarray, native_to_clip: list[list[float]]) -> np.ndarray:
    """Warp a native-resolution RGB frame onto the clip canvas."""
    import cv2

    if image.shape[1::-1] == CANVAS:
        return image
    affine = np.asarray(native_to_clip, dtype=float)[:2]
    return cv2.warpAffine(image, affine, CANVAS, flags=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/full_procedure.yaml")
    parser.add_argument("--source", default="cecum_start", help="entry of frame_sources")
    parser.add_argument("--frame", required=True, help="REAL-Colon frame image")
    parser.add_argument("--out", required=True, help="output image (.png) or array (.npy)")
    args = parser.parse_args()

    import cv2

    cfg = load_config(args.config)
    source = cfg["frame_sources"][args.source]
    image = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.frame)
    rgb = to_canvas(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), source["native_to_clip"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".npy":
        np.save(out, rgb)
    else:
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"REAL-Colon video {source['video']} frame {source['frame']} -> {out}")


if __name__ == "__main__":
    main()
