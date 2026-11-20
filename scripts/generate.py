# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate one video chunk from a first frame, camera poses and a tube.

Example:
    python scripts/generate.py --config configs/inference.yaml \
        inputs.first_frame=frame0.png inputs.cameras=cameras.npy inputs.intrinsics=calib.json \
        inputs.tube=tube.mp4 output.dir=outputs/chunk
"""

from __future__ import annotations

import argparse

from colonoworld.inference import run
from colonoworld.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()
    out = run(load_config(args.config, args.overrides))
    print(f"wrote {out / 'generated.mp4'}")


if __name__ == "__main__":
    main()
