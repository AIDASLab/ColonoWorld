# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build a 3D polyp condition lifted onto the fitted tube wall (mask npz + palette video).

The polyp is placed either at an explicit wall point (``--centre`` and ``--normal``, pose units)
or where the ray of ``--pixel`` hits the wall at ``--place-frame``. Requires Mitsuba 3.

Example:
    python scripts/build_polyp_insert.py --config configs/conditions.yaml \
        --surface surface.npz --intrinsics camera.json --pixel 480 529 --place-frame 94 \
        --size-mm 16 --onset 13 --out out/polyp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.conditions import (
    PolypRenderer,
    empty_condition,
    insert_polyp,
    render_palette_video,
    update_condition,
)
from colonoworld.utils.config import load_config
from colonoworld.utils.io import save_npz, write_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/conditions.yaml")
    p.add_argument(
        "--surface", required=True, help="tube surface .npz (V, F, poses, R, field_mask)"
    )
    p.add_argument("--intrinsics", required=True, help="camera JSON (K, D[, K_native, ...])")
    p.add_argument("--centre", nargs=3, type=float, help="polyp base centre (pose units)")
    p.add_argument("--normal", nargs=3, type=float, help="wall normal into the lumen")
    p.add_argument("--pixel", nargs=2, type=int, metavar=("X", "Y"), help="placement pixel")
    p.add_argument("--place-frame", type=int)
    p.add_argument("--kind", choices=("sessile", "pedunculated", "flat"))
    p.add_argument("--size-mm", help='diameter in mm, or "auto"')
    p.add_argument("--onset", type=int, help="first frame with the polyp")
    p.add_argument("--offset", type=int, help="first frame without the polyp")
    p.add_argument(
        "--base-cond", help="condition .npz to add the polyp to (default: an empty condition)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    pcfg = cfg["polyp"]
    kind = args.kind or pcfg["kind"]
    size = args.size_mm if args.size_mm is not None else pcfg["size_mm"]
    onset = args.onset if args.onset is not None else pcfg["onset"]
    offset = args.offset if args.offset is not None else pcfg["offset"]

    cam = json.loads(Path(args.intrinsics).read_text())
    renderer = PolypRenderer.from_files(args.surface, cam)
    if args.centre is not None:
        centre, normal = np.array(args.centre), np.array(args.normal)
    else:
        frame = args.place_frame if args.place_frame is not None else pcfg["place_frame"]
        centre, normal = renderer.wall_point(frame, args.pixel)
    if str(size) == "auto":
        size, scan = renderer.choose_size(
            kind, centre, normal, pcfg["sizes_mm"], pcfg["target_area"]
        )
        print("size scan (mm, median area):", scan)
    size = float(size)

    masks = insert_polyp(renderer, centre, normal, size, kind, onset, offset)
    out = Path(args.out)
    save_npz(
        out / "polyp.npz",
        mask=masks,
        visible=masks.reshape(len(masks), -1).any(1),
        centre=centre,
        normal=normal,
        kind=kind,
        size_mm=size,
        diameter_over_R=size / 25.0,
        onset=onset,
    )
    edge = cfg["palette"]["edge_px"]
    if args.base_cond:
        with np.load(args.base_cond) as z:
            cond = {k: z[k] for k in z.files}
        cond["polyp"] = masks
        update_condition(cond, edge)
        save_npz(out / "cond.npz", **cond)
        palette = cond["palette"]
    else:
        palette = render_palette_video({"polyp": masks}, field_mask=renderer.field, edge_px=edge)
        cond = empty_condition(len(masks), renderer.field)
        cond["polyp"] = masks
        update_condition(cond, edge)
        save_npz(out / "cond.npz", **cond)
    write_video(out / "palette.mp4", palette, fps=float(cfg["fps"]))
    print(f"polyp {kind} {size} mm, visible frames {int(masks.any((1, 2)).sum())}, wrote {out}")


if __name__ == "__main__":
    main()
