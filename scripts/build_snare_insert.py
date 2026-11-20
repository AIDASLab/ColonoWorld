# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build a 3D snare condition (sheath tool mask + wire snare mask + palette video).

Example:
    python scripts/build_snare_insert.py --config configs/conditions.yaml \
        --intrinsics camera.json --base-cond cond.npz --out out/snare \
        --onset 30 --offset 170 --sheath-len-mm 16 --loop-len-mm 28 --loop-width-mm 22 \
        --tilt-deg -30 --yaw-deg 25 --aim-xy 0.62 0.80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.conditions import (
    SnareParams,
    deployment_schedule,
    empty_condition,
    render_snare,
    update_condition,
)
from colonoworld.utils.config import load_config
from colonoworld.utils.io import save_npz, write_video

PARAM_FLAGS = {
    "exit_mm": dict(nargs=2, type=float, metavar=("X", "Y"), help="channel exit (mm)"),
    "sheath_len_mm": dict(type=float, help="deployed sheath length"),
    "loop_len_mm": dict(type=float, help="wire loop length"),
    "loop_width_mm": dict(type=float, help="wire loop width"),
    "tilt_deg": dict(type=float, help="loop tilt about its long axis"),
    "yaw_deg": dict(type=float, help="loop turn toward the image left"),
    "close": dict(type=float, help="final loop-width reduction 0..1"),
    "close_frames": dict(nargs=2, type=int, metavar=("START", "END"), help="closing ramp"),
    "aim_fraction": dict(type=float, help="aim point fraction from centre to lower-right"),
    "aim_xy": dict(nargs=2, type=float, metavar=("FX", "FY"), help="aim point (image fraction)"),
    "direction_len_mm": dict(type=float, help="sheath length used to solve the direction"),
    "sheath_diam_mm": dict(type=float),
    "wire_px": dict(type=int),
    "ramp_frames": dict(type=int),
    "start_deployment": dict(type=float),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/conditions.yaml")
    p.add_argument("--intrinsics", required=True, help="JSON with K (3x3) and D (4 KB4 coefs)")
    p.add_argument("--base-cond", help="existing condition .npz to add the snare to")
    p.add_argument("--field-mask", help="(H, W) boolean .npy, used without --base-cond")
    p.add_argument("--num-frames", type=int, help="frame count, used without --base-cond")
    p.add_argument("--clear", nargs="*", default=[], help="mask classes to clear in the base")
    p.add_argument("--onset", type=int)
    p.add_argument("--offset", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    for name, kw in PARAM_FLAGS.items():
        p.add_argument("--" + name.replace("_", "-"), dest=name, **kw)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    scfg = dict(cfg["snare"])
    for name in PARAM_FLAGS:
        if getattr(args, name) is not None:
            scfg[name] = getattr(args, name)
    onset = args.onset if args.onset is not None else scfg.pop("onset")
    offset = args.offset if args.offset is not None else scfg.pop("offset")
    scfg.pop("onset", None)
    scfg.pop("offset", None)
    for key in ("exit_mm", "close_frames", "aim_xy"):
        if scfg.get(key) is not None:
            scfg[key] = tuple(scfg[key])
    params = SnareParams(**scfg)

    cam = json.loads(Path(args.intrinsics).read_text())
    if args.base_cond:
        with np.load(args.base_cond) as z:
            cond = {k: z[k] for k in z.files}
    else:
        cond = empty_condition(args.num_frames, np.load(args.field_mask).astype(bool))
    for cls in args.clear:
        cond[cls] = np.zeros_like(cond[cls])

    deployment = deployment_schedule(
        len(cond["flags"]), onset, offset, params.ramp_frames, params.start_deployment
    )
    tool, snare, record = render_snare(cam["K"], cam["D"], cond["field_mask"], deployment, params)
    cond["tool"], cond["snare"] = tool, snare
    update_condition(cond, cfg["palette"]["edge_px"])

    out = Path(args.out)
    save_npz(out / "cond.npz", **cond)
    write_video(out / "palette.mp4", cond["palette"], fps=float(cond.get("fps", cfg["fps"])))
    record.update(onset=onset, offset=offset)
    (out / "snare.json").write_text(json.dumps(record, indent=1))
    print(f"snare frames {len(record['frames'])}, wrote {out}")


if __name__ == "__main__":
    main()
