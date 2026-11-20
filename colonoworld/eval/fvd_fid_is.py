# SPDX-License-Identifier: CC-BY-NC-4.0
"""FVD (I3D), FID and IS (Inception) of generated videos against real reference videos.

Feature extractors are the TorchScript networks used by StyleGAN-V (I3D, 400-d features) and
StyleGAN2-ADA (Inception ``inception-2015-12-05``: 2048-d pool features and 1008-way softmax).
They are downloaded by the user (see ``docs/evaluation.md``) and loaded with ``torch.jit.load``.

Per video (first ``n_frames`` frames in stream order, field mask multiplied into every frame):
    I3D:       16-frame windows with stride 1, frames bilinearly resized to 224x224, scaled to
               [-1, 1], ``return_features=True``.
    Inception: every frame (``frame_every=1``) as uint8 at native size (the network resizes to 299);
               pool features for FID and ``no_output_bias`` probabilities for IS.
Per set: FVD / FID = Frechet distance between Gaussians of the pooled generated and real features;
IS = exp(mean KL) over 10 consecutive splits of the generated probabilities (clips sorted by name).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------------- decoding
def decode_video(path: str | Path, n_frames: int = 189) -> np.ndarray:
    """First ``n_frames`` RGB frames ``(T, H, W, 3)`` uint8 in stream order (PyAV)."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    return np.stack(frames)


def load_field_mask(path: str | Path, hw: tuple[int, int] | None = None) -> np.ndarray:
    """Binary field-of-view mask ``(H, W)`` uint8, nearest-resized to ``hw`` when given."""
    from PIL import Image

    im = Image.open(path)
    if hw is not None and (im.size[1], im.size[0]) != tuple(hw):
        im = im.resize((hw[1], hw[0]), Image.NEAREST)
    return (np.asarray(im) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------------- extraction
class FeatureExtractor:
    """I3D window features and Inception frame features / probabilities of one video."""

    def __init__(
        self,
        i3d_weights: str | Path,
        inception_weights: str | Path,
        device: str | None = None,
        clip_len: int = 16,
        window_stride: int = 1,
        frame_every: int = 1,
        resize: tuple[int, int] = (224, 224),
        i3d_batch: int = 8,
    ) -> None:
        import torch

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inception = torch.jit.load(str(inception_weights), map_location=self.device).eval()
        self.i3d = torch.jit.load(str(i3d_weights), map_location=self.device).eval()
        self.clip_len, self.stride = clip_len, window_stride
        self.every, self.size, self.batch = frame_every, tuple(resize), i3d_batch

    def __call__(self, frames: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Features of ``frames`` ``(T, H, W, 3)`` uint8 with field ``mask`` ``(H, W)``."""
        torch, F = self.torch, self.torch.nn.functional
        mk = torch.from_numpy(mask.astype(np.float32)).to(self.device)[None, None]
        starts = list(range(0, len(frames) - self.clip_len + 1, self.stride))
        inc, probs, i3d = [], [], []
        with torch.no_grad():
            sub = frames[:: self.every]
            for s in range(0, len(sub), 16):
                x = torch.from_numpy(sub[s : s + 16]).to(self.device).permute(0, 3, 1, 2).float()
                xu = (x * mk).round().clamp(0, 255).to(torch.uint8)
                probs.append(self.inception(xu, no_output_bias=True).cpu().numpy())
                inc.append(self.inception(xu, return_features=True).cpu().numpy())
            resized = []
            for s in range(0, len(frames), 32):
                x = torch.from_numpy(frames[s : s + 32]).to(self.device).permute(0, 3, 1, 2)
                x = x.float() * mk
                x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
                resized.append(x / 127.5 - 1)
            allf = torch.cat(resized)
            for b in range(0, len(starts), self.batch):
                xb = torch.stack(
                    [
                        allf[s : s + self.clip_len].permute(1, 0, 2, 3)
                        for s in starts[b : b + self.batch]
                    ]
                )
                feats = self.i3d(xb, rescale=False, resize=False, return_features=True)
                i3d.append(feats.cpu().numpy())
        return {
            "inception_probs": np.concatenate(probs),
            "inception": np.concatenate(inc),
            "i3d": np.concatenate(i3d).reshape(len(starts), -1),
        }

    def video(self, path: str | Path, mask_path: str | Path, n_frames: int = 189):
        """Decode ``path``, apply the field mask at ``mask_path`` and extract features."""
        frames = decode_video(path, n_frames)
        if len(frames) != n_frames:
            raise ValueError(f"{path}: {len(frames)} frames, need {n_frames}")
        return self(frames, load_field_mask(mask_path, frames.shape[1:3]))


# --------------------------------------------------------------------------------- statistics
def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to feature rows ``a`` and ``b``."""
    from scipy import linalg

    m = a.mean(0) - b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = linalg.sqrtm(ca @ cb)
    s = s.real if np.iscomplexobj(s) else s
    return float(m @ m + np.trace(ca + cb - 2 * s))


def inception_score(probs: np.ndarray, splits: int = 10) -> dict[str, Any] | None:
    """Inception Score over consecutive splits of class probabilities."""
    n = len(probs)
    scores = []
    for i in range(splits):
        part = probs[i * n // splits : (i + 1) * n // splits]
        if len(part) == 0:
            return None
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n_frames": n}


def set_metrics(
    gen: dict[str, dict[str, np.ndarray]],
    ref: dict[str, dict[str, np.ndarray]],
    is_splits: int = 10,
) -> dict[str, Any]:
    """FVD, FID and IS of a generated set against a reference set ({clip name: features})."""
    a = np.concatenate([g["i3d"] for g in gen.values()])
    b = np.concatenate([r["i3d"] for r in ref.values()])
    ai = np.concatenate([g["inception"] for g in gen.values()])
    bi = np.concatenate([r["inception"] for r in ref.values()])
    gp = np.concatenate([g["inception_probs"] for _, g in sorted(gen.items())])
    return {
        "FVD": frechet_distance(a, b) if min(len(a), len(b)) > 2 else None,
        "FID": frechet_distance(ai, bi) if min(len(ai), len(bi)) > 2 else None,
        "IS": inception_score(gp, is_splits),
        "n_fvd_windows": [len(a), len(b)],
        "n_fid_frames": [len(ai), len(bi)],
    }


def half_splits(names: list[str], repeats: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """Fixed random half-splits ``(A, B)`` of the reference clip names."""
    names = sorted(names)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        perm = rng.permutation(len(names))
        h = len(names) // 2
        out.append(([names[i] for i in perm[:h]], [names[i] for i in perm[h:]]))
    return out


def matched_metrics(
    gen: dict[str, dict[str, np.ndarray]] | None,
    ref: dict[str, dict[str, np.ndarray]],
    repeats: int = 10,
    seed: int = 20261102,
) -> dict[str, Any]:
    """Same-size protocol: FD(gen_A, real_B) on fixed half-splits.

    With ``gen=None`` the values are FD(real_A, real_B).
    """
    fvd, fid = [], []
    src = ref if gen is None else gen
    for A, B in half_splits(list(ref), repeats, seed):
        m = set_metrics({a: src[a] for a in A}, {b: ref[b] for b in B})
        fvd.append(m["FVD"])
        fid.append(m["FID"])
    return {
        "FVD_mean": float(np.mean(fvd)),
        "FVD_std": float(np.std(fvd, ddof=1)),
        "FID_mean": float(np.mean(fid)),
        "FID_std": float(np.std(fid, ddof=1)),
    }
