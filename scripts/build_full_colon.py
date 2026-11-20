# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build a synthetic whole colon with its cecum-to-rectum withdrawal and render the tube chunks.

Outputs ``tube.npz`` (TubeModel), ``cameras.npy`` (T, 4, 4 camera-to-world), ``segments.json``
(per-run anatomical segments) and, with ``--render``, ``chunk_XX.mp4`` tube videos of 189 frames
(stride 188, one shared frame between chunks) with ``chunk_XX_cameras.npy`` (poses relative to
the chunk's first frame), ready for the full-procedure chain.

Example:
    python scripts/build_full_colon.py --config configs/full_colon_shapes.yaml \
        --shape standard --out outputs/full_colon/standard --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.tube.camera import relative_poses
from colonoworld.tube.render import render_tube_video, write_tube_video
from colonoworld.tube.synthetic_colon import build_full_colon, chunk_ranges, full_colon_intrinsics
from colonoworld.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/full_colon_shapes.yaml")
    parser.add_argument("--shape", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render", action="store_true", help="render the tube chunks")
    parser.add_argument(
        "--chunks", default=None, help="comma-separated chunk indices (default all)"
    )
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tube, cameras, segments = build_full_colon(args.shape, cfg)
    tube.save(out / "tube.npz")
    np.save(out / "cameras.npy", cameras)
    ranges = chunk_ranges(len(cameras))
    (out / "segments.json").write_text(
        json.dumps(
            dict(shape=args.shape, frames=len(cameras), chunks=ranges, segments=segments), indent=1
        )
    )
    print(f"{args.shape}: {len(cameras)} frames, {len(ranges)} chunks")
    if not args.render:
        return
    wanted = range(len(ranges)) if args.chunks is None else [int(c) for c in args.chunks.split(",")]
    intrinsics = full_colon_intrinsics(cfg)
    for k in wanted:
        start, end = ranges[k]
        frames = render_tube_video(tube, cameras[start:end], intrinsics)
        write_tube_video(out / f"chunk_{k:02d}.mp4", frames)
        np.save(out / f"chunk_{k:02d}_cameras.npy", relative_poses(cameras[start:end]))
        print(f"chunk {k:02d}: frames {start}-{end - 1}")


if __name__ == "__main__":
    main()
