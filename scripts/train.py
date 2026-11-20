# SPDX-License-Identifier: CC-BY-NC-4.0
"""Joint-conditioning training.

Example (8 GPUs, effective batch 8)::

    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_joint.yaml
"""

from __future__ import annotations

import argparse

from colonoworld.training.train import train
from colonoworld.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="ColonoWorld joint-conditioning training")
    p.add_argument("--config", default="configs/train_joint.yaml")
    p.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    a = p.parse_args()
    train(load_config(a.config, a.override))


if __name__ == "__main__":
    main()
