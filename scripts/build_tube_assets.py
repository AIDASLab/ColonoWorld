# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build the tube assets that are not distributed with the release.

Subcommands:
    texture          procedural mucosa texture (writes ``render.texture`` of configs/tube.yaml)
    donor-template   reference full colon used by the synthetic colon shapes
                     (writes the file named by ``donor_template`` in configs/full_colon_shapes.yaml)

Example:
    export COLONOWORLD_DATA=/path/to/data
    python scripts/build_tube_assets.py texture --config configs/tube.yaml
    python scripts/build_tube_assets.py donor-template --config configs/full_colon_shapes.yaml
"""

from __future__ import annotations

import argparse

from colonoworld.utils.config import load_config


def donor_template(args: argparse.Namespace) -> None:
    import cv2

    from colonoworld.tube.donor_template import (
        build_donor_template,
        load_donor_inputs,
        save_donor_template,
    )

    cfg = load_config(args.config, args.overrides)
    section = cfg["donor_template_build"]
    frame = None
    if args.reference_frame:
        frame = cv2.cvtColor(cv2.imread(args.reference_frame), cv2.COLOR_BGR2RGB)
    arrays = build_donor_template(
        section, load_donor_inputs(section, frame), cfg["full_colon"]["calibration"]
    )
    out = args.out or cfg["donor_template"]
    save_donor_template(out, arrays)
    print(f"wrote {out} ({len(arrays['poses'])} camera frames)")


def texture(args: argparse.Namespace) -> None:
    from colonoworld.tube.texture import save_mucosa_texture

    cfg = load_config(args.config, args.overrides)
    out = args.out or cfg["render"]["texture"]
    save_mucosa_texture(out)
    print(f"wrote {out}")


COMMANDS = {
    "texture": (texture, "configs/tube.yaml"),
    "donor-template": (donor_template, "configs/full_colon_shapes.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, default_config) in COMMANDS.items():
        p = sub.add_parser(name)
        p.add_argument("--config", default=default_config)
        p.add_argument("--out", default=None, help="output path (default: from the config)")
        if name == "donor-template":
            p.add_argument(
                "--reference-frame",
                default=None,
                help="REAL-Colon reference frame (see reference_frame_source in the config)",
            )
        p.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()
    COMMANDS[args.command][0](args)


if __name__ == "__main__":
    main()
