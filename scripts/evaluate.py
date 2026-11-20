# SPDX-License-Identifier: CC-BY-NC-4.0
"""Evaluate generated videos.

Commands (each reads a CSV manifest, one row per clip, and writes ``rows.csv`` + ``summary.json``):

  camera     columns clip, reference_pose, readback
             reference_pose / readback: .npz with ``relative_c2w`` (readback empty = untracked)
  quality    columns clip, generated_video, reference_video, field_mask
  lumen      columns clip, generated_video, tube_aux
             tube_aux: .npz with ``depth`` (T,h,w) and ``field_mask`` (h,w) of the tube render
  procedure  --commands commands.npy (K,189,4,4) --readbacks <dir with chunk_000.npz, ...>

  python scripts/evaluate.py camera --config configs/eval.yaml --manifest clips.csv --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colonoworld.utils.config import load_config  # noqa: E402


def read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_outputs(out: Path, rows: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def cmd_camera(cfg: dict, args: argparse.Namespace) -> None:
    from colonoworld.eval import camera_metrics as cm

    c = cfg["camera"]
    rows = []
    for r in read_manifest(args.manifest):
        P = cm.load_readback(r["reference_pose"], c["readback_key"])
        Q = cm.load_readback(r["readback"], c["readback_key"]) if r.get("readback") else None
        rec = {"clip": r["clip"], **cm.score_clip_windowed(P, Q, c["windowed"])}
        s = cm.score_clip_summed(P, Q, c["summed"]["scale"], c["summed"]["cap"])
        rec.update({f"summed_{k}": v for k, v in s.items() if k != "untracked"})
        rows.append(rec)
    summary = {"windowed": cm.aggregate(rows)}
    summary["summed"] = {
        k: float(np.mean([r[f"summed_{k}_capped"] for r in rows]))
        for k in cm.SUMMED_KEYS
        if all(r[f"summed_{k}_capped"] is not None for r in rows)
    }
    write_outputs(Path(args.out), rows, summary)


def cmd_quality(cfg: dict, args: argparse.Namespace) -> None:
    from colonoworld.eval import fvd_fid_is as q

    c = cfg["quality"]
    ex = q.FeatureExtractor(
        c["i3d_weights"],
        c["inception_weights"],
        device=args.device,
        clip_len=c["clip_len"],
        window_stride=c["window_stride"],
        frame_every=c["frame_every"],
        resize=tuple(c["resize"]),
    )
    cache = Path(args.out) / "features"
    cache.mkdir(parents=True, exist_ok=True)

    def feats(tag: str, video: str, mask: str) -> dict[str, np.ndarray]:
        f = cache / f"{tag}.npz"
        if f.exists():
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        r = ex.video(video, mask, c["n_frames"])
        np.savez(f, **r)
        return r

    gen, ref, rows = {}, {}, []
    for r in read_manifest(args.manifest):
        gen[r["clip"]] = feats(f"gen_{r['clip']}", r["generated_video"], r["field_mask"])
        ref[r["clip"]] = feats(f"ref_{r['clip']}", r["reference_video"], r["field_mask"])
        rows.append({"clip": r["clip"], "n_i3d_windows": len(gen[r["clip"]]["i3d"])})
    summary = q.set_metrics(gen, ref, c["is_splits"])
    if len(ref) >= 4:
        summary["matched"] = q.matched_metrics(gen, ref, **c["matched"])
    write_outputs(Path(args.out), rows, summary)


def cmd_lumen(cfg: dict, args: argparse.Namespace) -> None:
    from colonoworld.eval import fvd_fid_is as q
    from colonoworld.eval import lumen_following as lf

    rows = []
    for r in read_manifest(args.manifest):
        with np.load(r["tube_aux"], allow_pickle=False) as z:
            depth, field = z["depth"], z["field_mask"].astype(bool)
        frames = q.decode_video(r["generated_video"], len(depth))
        data = lf.measure_clip(frames, depth, field, cfg["lumen"]["variant"])
        rows.append({"clip": r["clip"], **lf.clip_summary(data)})
    write_outputs(Path(args.out), rows, lf.set_summary(rows))


def cmd_procedure(cfg: dict, args: argparse.Namespace) -> None:
    from colonoworld.eval import camera_metrics as cm

    c = cfg["camera"]
    commands = np.load(args.commands)
    readbacks = []
    for k in range(len(commands)):
        f = Path(args.readbacks) / f"chunk_{k:03d}.npz"
        readbacks.append(
            cm.load_readback(f, c["readback_key"], orthonormalize=True) if f.exists() else None
        )
    res = cm.score_procedure(commands, readbacks, c["windowed"])
    write_outputs(Path(args.out), res["chunks"], {k: v for k, v in res.items() if k != "chunks"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["camera", "quality", "lumen", "procedure"])
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--manifest")
    ap.add_argument("--commands")
    ap.add_argument("--readbacks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    {"camera": cmd_camera, "quality": cmd_quality, "lumen": cmd_lumen, "procedure": cmd_procedure}[
        args.command
    ](cfg, args)


if __name__ == "__main__":
    main()
