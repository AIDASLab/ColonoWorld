# SPDX-License-Identifier: CC-BY-NC-4.0
"""Fit, regularise and render the tubes of training windows.

Each line of the input JSONL describes one window:

    {"id": ..., "index": int, "segment": "transverse" | null,
     "video": real clip, "video_indices": [frame indices of the window],
     "calibration": calibration JSON, "seed_parameters": seed sweep parameters JSON,
     "seed_surface": npz with poses (T, 4, 4), R, field_mask (832, 1104),
     "seed_video": rendered seed tube video, "seed_depth": npz with depth (T, 104, 138)}

Instead of the seed fields, a window can give the raw inputs of the seed stage
(``colonoworld.tube.seed_fit.build_training_seed``):

    {"id": ..., "index": int, "segment": ..., "video": real clip, "calibration": ...,
     "camera_npz": camera/depth estimator output, "start": int, "frames": int,
     "route": "sdf" | "direct" (default "sdf"), "keyframes": optional keyframe-depth npz}

The seed (``<out>/<id>/seed/{parameters.json, surface.npz, seed.mp4, seed_depth.npz}``) is then
built first; keyframe depth is computed with the depth network of ``training_seed`` when no
``keyframes`` file is given.

Per window the script runs the lumen fit (``fit_training_tube``), the shape regularisation
(``regularize_training_tube``) and renders the final tube video with the device field picked from
the window's real frames. Outputs ``<out>/<id>/{tube.npz, parameters.json, tube.mp4, info.json}``.

Example:
    python scripts/fit_training_tubes.py --config configs/tube.yaml --windows windows.jsonl \
        --out outputs/training_tubes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.tube.camera import load_calibration
from colonoworld.tube.fit_training import fit_training_tube, load_window_frames
from colonoworld.tube.render import (
    pick_device_field_from_frames,
    render_tube_video,
    write_tube_video,
)
from colonoworld.tube.training_shape import decode_small, regularize_training_tube, sample_times
from colonoworld.utils.config import load_config


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return None if x.size > 64 else x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def read_frames(path: str, indices: list[int]) -> list[np.ndarray]:
    """Full-resolution RGB frames at the given indices."""
    import av

    wanted, out = set(indices), {}
    with av.open(path) as container:
        container.streams.video[0].thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max(wanted):
                break
    return [out[i] for i in indices]


def build_seed(w: dict, cfg: dict, dst: Path) -> dict:
    """Build the seed of a raw-input window; returns the window with the seed fields set."""
    from colonoworld.tube.seed_depth import load_depth_anything
    from colonoworld.tube.seed_fit import build_training_seed

    seed_cfg = cfg["training_seed"]
    keyframes, disparity_fn = None, None
    if w.get("keyframes"):
        with np.load(w["keyframes"]) as z:
            keyframes = {k: z[k] for k in ("idx", "P", "depth", "ok", "Kd", "support")}
    else:
        disparity_fn = load_depth_anything(
            seed_cfg["depth_code"], seed_cfg["depth_weights"], int(seed_cfg.get("depth_threads", 6))
        )
    prior = json.loads(Path(seed_cfg["fold_prior"]).read_text())["parameters"]
    key = str(w["id"]).replace(":", "__")
    seed = build_training_seed(
        w["video"],
        load_calibration(w["calibration"]),
        w["camera_npz"],
        int(w["start"]),
        int(w["frames"]),
        key,
        prior,
        seed_cfg["texture"],
        keyframes=keyframes,
        disparity_fn=disparity_fn,
        route=w.get("route", "sdf"),
    )
    sd = dst / "seed"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "parameters.json").write_text(json.dumps(_jsonable(seed["parameters"]), indent=1))
    np.savez_compressed(sd / "surface.npz", **seed["surface"])
    np.savez_compressed(sd / "seed_depth.npz", depth=seed["depth"])
    write_tube_video(sd / "seed.mp4", seed["frames"])
    return dict(
        w,
        video_indices=seed["surface"]["video_indices"].tolist(),
        seed_parameters=str(sd / "parameters.json"),
        seed_surface=str(sd / "surface.npz"),
        seed_video=str(sd / "seed.mp4"),
        seed_depth=str(sd / "seed_depth.npz"),
    )


def process(w: dict, cfg: dict, out: Path) -> dict:
    dst = out / str(w["id"]).replace(":", "__")
    dst.mkdir(parents=True, exist_ok=True)
    if "seed_parameters" not in w:
        w = build_seed(w, cfg, dst)
    cam = load_calibration(w["calibration"])
    seed_params = json.loads(Path(w["seed_parameters"]).read_text())
    with np.load(w["seed_surface"]) as z:
        poses, radius, field = z["poses"], float(z["R"]), z["field_mask"].astype(bool)
    with np.load(w["seed_depth"]) as z:
        seed_depth = z["depth"]
    fit_cfg = cfg["training_fit"]
    frames = load_window_frames(w["video"], w["video_indices"], w["seed_video"])
    seed = int(fit_cfg["search_seed_base"]) + int(w["index"])
    params, fitted, fit_info = fit_training_tube(
        seed_params, poses, radius, field, seed_depth, frames, cam, seed, fit_cfg, dst / "fit"
    )
    chosen = next(a for a in fit_info["attempts"] if a["index"] == fit_info["selected"])["record"]
    times = sample_times(len(poses))
    ids = np.asarray(w["video_indices"])
    chart = fitted.chart_tuple
    surface = dict(
        V=fitted.vertices,
        F=fitted.faces,
        poses=poses,
        R=radius,
        field_mask=field,
        chart_s=chart[0],
        chart_t=chart[1],
        chart_f=chart[2],
    )
    tube, shape_info = regularize_training_tube(
        params,
        surface,
        chosen["depth"],
        decode_small(w["video"], ids[times]),
        decode_small(chosen["video"], times),
        cam,
        w.get("segment"),
        str(w["id"]),
        fit_cfg["texture"],
        cfg["training_shape"],
    )
    render = cfg["render"]
    variant, _, _ = pick_device_field_from_frames(
        read_frames(w["video"], [int(i) for i in ids[::8]]),
        render["device_fields"],
        render["min_field_margin_training"],
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
    write_tube_video(dst / "tube.mp4", render_tube_video(tube, poses, intrinsics))
    tube.meta.update(device_field=variant, tier=shape_info["tier"])
    tube.save(dst / "tube.npz")
    # Wall surface with the window's camera path (input of scripts/build_polyp_insert.py).
    np.savez_compressed(
        dst / "surface.npz",
        V=tube.vertices,
        F=tube.faces,
        poses=poses,
        video_indices=ids,
        field_mask=field,
        R=np.float64(radius),
        chart_s=tube.chart[0],
        chart_t=tube.chart[1],
        chart_f=tube.chart[2],
    )
    (dst / "parameters.json").write_text(json.dumps(_jsonable(params), indent=1))
    info = dict(
        fit_selected=fit_info["selected"],
        tier=shape_info["tier"],
        scale=shape_info["scale"],
        device_field=variant,
    )
    (dst / "info.json").write_text(json.dumps(_jsonable(info), indent=1))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/tube.yaml")
    parser.add_argument("--windows", required=True, help="JSONL, one window per line")
    parser.add_argument("--out", required=True)
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(args.out)
    for line in Path(args.windows).read_text().splitlines():
        if line.strip():
            w = json.loads(line)
            print(w["id"], process(w, cfg, out), flush=True)


if __name__ == "__main__":
    main()
