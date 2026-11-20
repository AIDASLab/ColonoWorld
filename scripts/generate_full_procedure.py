# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate a full colonoscopy procedure (cecum -> rectum withdrawal) as a chain of chunks.

Examples:
    # whole procedure through the standard colon
    python scripts/generate_full_procedure.py --config configs/full_procedure.yaml \
        --shape standard --frame0 cecum_frame0.png

    # regenerate chunk 8 with conditions, starting from the saved last frame of chunk 7
    python scripts/generate_full_procedure.py --config configs/full_procedure.yaml \
        --shape textbook --start-segment 8 --segments 1 \
        --frame0 outputs/full_procedure/chunks/chunk_07_last.npy --conditions conditions.json
"""

from __future__ import annotations

import argparse

from colonoworld.inference.chain import run_full_procedure
from colonoworld.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/full_procedure.yaml")
    parser.add_argument("--shape", help="synthetic colon shape")
    parser.add_argument("--frame0", help="start frame (image or uint8 .npy)")
    parser.add_argument("--segments", type=int, help="number of chunks to generate")
    parser.add_argument("--start-segment", type=int, help="first chunk to generate")
    parser.add_argument("--conditions", help="per-chunk conditions manifest (JSON)")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()
    overrides = list(args.overrides)
    for key, value in (
        ("chain.shape", args.shape),
        ("chain.frame0", args.frame0),
        ("chain.segments", args.segments),
        ("chain.start_segment", args.start_segment),
        ("chain.conditions", args.conditions),
        ("output.dir", args.out),
    ):
        if value is not None:
            overrides.append(f"{key}={value}")
    out = run_full_procedure(load_config(args.config, overrides))
    print(f"wrote {out / 'full_procedure.mp4'}")


if __name__ == "__main__":
    main()
