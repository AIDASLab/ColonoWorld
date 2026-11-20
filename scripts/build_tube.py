# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build and render the tube of a real clip from its first frame and camera path.

Only frame 0 of the clip is read. Outputs ``tube.npz`` (TubeModel), ``tube_params.json``,
``cameras.npy`` and ``tube.mp4`` (the conditioning video, same frame count as the camera path).

Example:
    python scripts/build_tube.py --config configs/tube.yaml --video clip.mp4 \
        --camera-path poses.npz --calibration camera.json --segment transverse --out outputs/tube
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.tube.build_from_video import build_tube_from_video
from colonoworld.tube.camera import load_calibration, load_camera_path
from colonoworld.tube.render import pick_device_field, render_tube_video, write_tube_video
from colonoworld.utils.config import load_config


def read_first_frame(path: str) -> np.ndarray:
    """First frame of a video as uint8 RGB."""
    import av

    with av.open(path) as container:
        container.streams.video[0].thread_count = 1
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")
    raise ValueError(f"no frames in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/tube.yaml")
    parser.add_argument("--video", required=True, help="clip video (only frame 0 is read)")
    parser.add_argument("--camera-path", required=True, help="npz with relative_c2w (T, 4, 4)")
    parser.add_argument("--calibration", required=True, help="fisheye calibration JSON")
    parser.add_argument("--segment", required=True, help="anatomical segment of the clip")
    parser.add_argument("--key", default=None, help="clip identifier (seeds the fold draw)")
    parser.add_argument(
        "--device-field", default=None, help="device-field variant (default: from frame 0)"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame0 = read_first_frame(args.video)
    cam = load_calibration(args.calibration)
    cameras = load_camera_path(args.camera_path)
    key = args.key or Path(args.video).stem
    tube, params = build_tube_from_video(frame0, cameras, cam, args.segment, key, cfg["video_tube"])
    render = cfg["render"]
    variant = args.device_field
    if variant is None:
        variant, _, _ = pick_device_field(
            frame0, render["device_fields"], render["min_field_margin_eval"]
        )
    intrinsics = dict(
        calibration=cam,
        texture=render["texture"],
        device_fields=render["device_fields"],
        device_field=variant,
        exposure_divisor=render["exposure_divisor"],
        exposure_stride=render["exposure_stride"],
        threads=render["threads"],
    )
    frames = render_tube_video(tube, cameras, intrinsics)
    tube.meta.update(device_field=variant)
    tube.save(out / "tube.npz")
    (out / "tube_params.json").write_text(json.dumps(tube.meta["parameters"], indent=1))
    np.save(out / "cameras.npy", cameras)
    write_tube_video(out / "tube.mp4", frames)
    print(f"wrote {out / 'tube.mp4'} ({len(frames)} frames, device field {variant})")


if __name__ == "__main__":
    main()
