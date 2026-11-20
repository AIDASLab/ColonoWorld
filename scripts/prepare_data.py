# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build the training manifest: videos + camera estimates -> tube videos -> condition masks.

Input: a JSONL index, one source window per line::

    {"id": ..., "video": clip.mp4, "frame_indices": [...] (optional), "camera": poses.npz
     (poses [T,4,4] camera-to-world, valid [T]), "calibration": calib.json (KB4),
     "field_mask": field.png (optional), "segments": [per-frame colon segment or null],
     "strata": ["clean", "polyp", "tool", "nbi"] (strata the window may be drawn in),
     "polyp": {"masks": npz (mask [T,H,W] bool, visible [T])} | {"absent": true},
     "tool": {"masks": npz, "heldout": false} | {"absent": true},
     "snare": snare_lines.json (per-frame wire polylines, optional),
     "water": {"track": npz (target_xy [T,2], visible [T]), "anchor_xy": [x, y],
               "jet_region": npz (jet [T,H,W] bool, reliable [T])} | {"absent": true},
     "nbi": [per-frame bool] (optional), "light_labelled": true,
     "flags": {"dye": false, "bubbles": false, "irrigation": false},
     "tube_fit": {...} (optional wall-fit QC)}

Output (``--out``): per window ``<id>/tube.mp4``, ``camera.npz``, ``conditions.npz`` (and
``tube_latent.pt`` with ``--cache-latents``), plus ``manifest.jsonl`` of the admitted windows
and ``rejected.json`` with the gate that rejected each other window.

Gates (see docs/training.md): finite rigid camera estimates; tube wall-fit QC (when provided);
real-vs-tube optical-flow agreement. Each window also gets its camera motion class from the
camera estimate and the rendered tube depth; forward-moving windows get the forward weight.
Windows with tool, irrigation, special light, dye or bubbles stay in the manifest but never
enter the clean stratum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colonoworld.training import data as D
from colonoworld.utils.config import load_config


def _load_mask(path: str, frames: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        mask = z["mask"]
        visible = z["visible"].astype(bool) if "visible" in z else np.ones(frames, bool)
    if mask.dtype == np.uint8 and mask.shape[-1] != D.WIDTH:
        mask = np.unpackbits(mask, axis=-1, count=D.WIDTH)
    return mask.astype(bool), visible


def _state(spec: dict | None, present: bool) -> str:
    if present:
        return "present"
    return "absent" if spec and spec.get("absent") else "unlabelled"


def build_conditions(row: dict, frames: int, out: Path) -> dict:
    """Write ``conditions.npz`` and return the label fields of the manifest row."""
    arrays: dict[str, np.ndarray] = {}
    labels: dict = {}
    occupancy: dict[str, np.ndarray] = {}
    for cls in ("polyp", "tool"):
        spec = row.get(cls) or {}
        if spec.get("masks"):
            mask, visible = _load_mask(spec["masks"], frames)
            arrays[cls], arrays[cls + "_visible"] = D.pack_mask(mask), visible
            occupancy[cls] = mask.reshape(frames, -1).any(1) & visible
        labels[cls] = dict(state=_state(spec, cls in arrays))
    if row.get("snare"):
        snare = D.rasterize_snare(json.loads(Path(row["snare"]).read_text()))
        arrays["snare"] = D.pack_mask(snare)
        if "tool" in occupancy:
            occupancy["tool"] |= snare.reshape(frames, -1).any(1)
    water = row.get("water") or {}
    if water.get("track"):
        with np.load(water["track"], allow_pickle=False) as z:
            target = z["target_xy"].astype(np.float64)
            on = z["visible"].astype(bool) & np.isfinite(target).all(1)
        ribbon = D.irrigation_ribbon(water["anchor_xy"], target, on)
        arrays["water"], arrays["water_visible"] = D.pack_mask(ribbon), on
        if water.get("jet_region"):
            with np.load(water["jet_region"], allow_pickle=False) as z:
                region = z["jet"].astype(bool) & z["reliable"].astype(bool)[:, None, None]
            arrays["water_loss_region"] = D.pack_mask(region)
        occupancy["water"] = (ribbon.reshape(frames, -1).any(1)) & on
        labels["water"] = dict(
            state="present",
            trainable=bool(water.get("jet_region")) and not water.get("heldout"),
            entry_forward=bool(not on[0] and on.any()),
            onset_weight=D.irrigation_onset_weight(on),
            event=water.get("event") or "window:" + row["id"],
        )
    else:
        labels["water"] = dict(state=_state(water, False))
    for cls in ("polyp", "tool"):
        if cls in occupancy:
            fwd, rev = D.entry_directions(occupancy[cls])
            labels[cls].update(entry_forward=fwd, entry_reverse=rev)
    if "tool" in arrays:
        labels["tool"]["trainable"] = not (row.get("tool") or {}).get("heldout", False)
    if arrays:
        np.savez_compressed(out / "conditions.npz", **arrays)
        labels["conditions"] = str(out / "conditions.npz")
    labels["snare"] = "snare" in arrays
    if row.get("nbi") is not None:
        frames_nbi = [bool(v) for v in row["nbi"]]
        labels["nbi"] = dict(
            frames=frames_nbi,
            uncertain=D.nbi_edge_uncertain(frames_nbi).tolist(),
            switch=len(set(frames_nbi)) > 1,
        )
    return labels


def prepare_window(row: dict, cfg: dict, out_root: Path) -> tuple[dict | None, str | None]:
    import cv2

    from colonoworld.tube.build_from_video import build_tube_from_video
    from colonoworld.tube.render import pick_device_field, render_tube_video, write_tube_video

    gates = D.QualityGates(**cfg.get("gates", {}))
    with np.load(row["camera"], allow_pickle=False) as z:
        poses = z["poses"].astype(np.float64)
        valid = z["valid"] if "valid" in z else None
    if not D.pose_ok(poses, valid):
        return None, "camera"
    poses = np.linalg.inv(poses[0])[None] @ poses
    frames = len(poses)
    if frames not in cfg.get("window_frames", (65, 189)):
        return None, "length"
    if row.get("tube_fit") is not None and not D.tube_fit_ok(row["tube_fit"], frames, gates):
        return None, "tube_fit"
    cam = json.loads(Path(row["calibration"]).read_text())
    real = D.decode_frames(row["video"], row.get("frame_indices"))
    segment = row.get("segment") or D.majority_segment(row.get("segments") or [None])
    if segment is None:
        return None, "segment"
    tube_model, _ = build_tube_from_video(real[0], poses, cam, segment, row["id"], cfg["tube"])
    render = cfg["render"]
    variant, _, _ = pick_device_field(
        real[0], render["device_fields"], render.get("min_field_margin", 0.11)
    )
    intrinsics = dict(render, calibration=cam, device_field=variant)
    tube, depth, _ = render_tube_video(
        tube_model, poses, intrinsics, (D.HEIGHT, D.WIDTH), return_depth=True
    )

    small = [cv2.resize(f, D.FLOW_SIZE, interpolation=cv2.INTER_AREA) for f in real[::4]]
    small_tube = [cv2.resize(f, D.FLOW_SIZE, interpolation=cv2.INTER_AREA) for f in tube[::4]]
    try:
        score = D.flow_agreement(small, small_tube, cam)
    except ValueError:
        return None, "flow_support"
    if cfg.get("agreement_gate", True) and not D.tube_agreement_ok(score, gates):
        return None, "agreement"

    out = out_root / row["id"]
    out.mkdir(parents=True, exist_ok=True)
    write_tube_video(out / "tube.mp4", tube)
    field = row.get("field_mask")
    if field:
        field = cv2.imread(field, cv2.IMREAD_GRAYSCALE)
        field = cv2.resize(field, (D.WIDTH, D.HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    else:
        field = tube.max(axis=(0, 3)) > 0
    np.savez(out / "camera.npz", poses=poses, calibration=json.dumps(cam), field_mask=field)
    window = dict(
        id=row["id"],
        frames=frames,
        video=str(Path(row["video"]).resolve()),
        frame_indices=row.get("frame_indices"),
        tube_video=str(out / "tube.mp4"),
        camera=str(out / "camera.npz"),
        strata=row.get("strata", ["clean"]),
        excluded=bool(row.get("excluded", False)),
        score=D.motion_score(*D.motion_energies(score)),
        motion_class=D.motion_class(poses, depth.astype(np.float32), field, cam),
        segments=row.get("segments") or [None] * frames,
        light_labelled=row.get("light_labelled", True),
        light_unknown_frames=row.get("light_unknown_frames"),
        flags={k: bool((row.get("flags") or {}).get(k)) for k in D.CLEAN_FLAGS},
        agreement=score,
    )
    window["forward"] = window["motion_class"] == "fwd"
    window.update(build_conditions(row, frames, out))
    return window, None


def cache_latents(windows: list[dict], model_cfg: dict) -> None:
    """VAE latents of the untransformed tubes (used by draws without spatial augmentation)."""
    import torch

    from colonoworld.models import load_model
    from colonoworld.training.train import encode_video

    backbone = load_model(model_cfg, device="cuda").backbone
    for w in windows:
        path = Path(w["tube_video"]).with_name("tube_latent.pt")
        torch.save(encode_video(backbone, D.decode_frames(w["tube_video"])).cpu(), path)
        w["tube_latent"] = str(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/train_joint.yaml")
    p.add_argument("--index", required=True, help="JSONL index of source windows")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--cache-latents", action="store_true")
    p.add_argument("--override", nargs="*", default=[])
    a = p.parse_args()
    cfg = load_config(a.config, a.override)
    prep = dict(cfg.get("prepare", {}))
    tube_cfg = load_config(prep.get("tube_config", "configs/tube.yaml"))
    prep.update(tube=tube_cfg["video_tube"], render=tube_cfg["render"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    windows, rejected = [], {}
    for line in Path(a.index).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        window, reason = prepare_window(row, prep, out)
        if window is None:
            rejected[row["id"]] = reason
        else:
            windows.append(window)
    if a.cache_latents:
        model_cfg = cfg["model"]
        if isinstance(model_cfg, str):
            model_cfg = load_config(model_cfg)
        cache_latents(windows, model_cfg)
    D.write_manifest(out / "manifest.jsonl", windows)
    (out / "rejected.json").write_text(json.dumps(rejected, indent=1))
    clean = sum(D.is_clean(w) for w in windows)
    print(f"{len(windows)} windows admitted ({clean} clean), {len(rejected)} rejected")


if __name__ == "__main__":
    main()
