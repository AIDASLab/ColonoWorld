# SPDX-License-Identifier: CC-BY-NC-4.0
"""Condition streams: tube tokens, mask tokens, segment embedding and the global flag adapter.

Tube and mask videos are VAE-encoded to the same latent grid as the generated video and
appended to the sequence as extra token rows. Each appended row copies the full mRoPE position of
the target token at the same latent time and spatial cell, so condition and target tokens are
aligned by position, and attention between all rows is dense and two-way. Condition rows carry
no noise and no loss; only target rows are decoded.

Global flags are a per-latent-frame vector of length :data:`GLOBAL_IN` mapped by a bias-free,
zero-initialised linear layer and added to every target token of that latent frame:

``[0:4]``  values: special light (NBI), dye, tool, irrigation
``[4:8]``  the corresponding "known" bits
``[8:13]`` class known: polyp, tool, water, blood, snare (the mask channel was supplied)

Unknown is value 0 and known 0; an all-zero vector is the same as no global condition.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

SEGMENT_CLASSES: tuple[str, ...] = (
    "cecum",
    "ascending",
    "transverse",
    "descending",
    "sigmoid",
    "rectum",
)
SEGMENT_NULL = len(SEGMENT_CLASSES)
FRAME_FLAGS: tuple[str, ...] = ("special_light", "dye", "tool", "irrigation")
GLOBAL_CLASSES: tuple[str, ...] = ("polyp", "tool", "water", "blood", "snare")
GLOBAL_IN = 2 * len(FRAME_FLAGS) + len(GLOBAL_CLASSES)
MASK_CHANNELS: tuple[str, ...] = ("polyp", "tool", "water", "snare")


class SegmentEmbedding(nn.Embedding):
    """Seven learned rows (six colon segments and a learned null row), zero initialised."""

    def __init__(self, width: int = 4096):
        super().__init__(SEGMENT_NULL + 1, width, _weight=torch.zeros(SEGMENT_NULL + 1, width))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class ConditionTokens(nn.Module):
    """Embeds patchified condition latents as appended token rows.

    ``y = W x + E_t[t] + E_h[h] + E_w[w] + e_type (+ segment[t])``, with a zero-initialised input
    projection so an untrained stream adds nothing.
    """

    def __init__(
        self,
        width: int = 4096,
        patch_dim: int = 192,
        max_time: int = 64,
        max_height: int = 64,
        max_width: int = 64,
        segment_embedding: bool = False,
    ):
        super().__init__()
        self.input = nn.Linear(patch_dim, width, bias=False)
        self.time = nn.Embedding(max_time, width)
        self.height = nn.Embedding(max_height, width)
        self.width = nn.Embedding(max_width, width)
        self.type_embedding = nn.Parameter(torch.zeros(width))
        self.segment_embedding = SegmentEmbedding(width) if segment_embedding else None
        if not self.input.weight.is_meta:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, seed: int = 0) -> None:
        """Zero input projection, N(0, 0.01^2) embeddings from a private generator."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.zeros_(self.input.weight)
        for param in (self.time.weight, self.height.weight, self.width.weight, self.type_embedding):
            param.copy_(torch.randn(param.shape, generator=generator) * 0.01)
        if self.segment_embedding is not None:
            self.segment_embedding.reset_parameters()

    def forward(
        self,
        patches: torch.Tensor,
        shape: tuple[int, int, int],
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed ``patches`` (t * hp * wp, patch_dim) of a latent with grid (t, h, w) ``shape``."""
        t, h, w = map(int, shape)
        hp, wp = (h + 1) // 2, (w + 1) // 2
        dev = patches.device
        tt, yy, xx = torch.meshgrid(
            torch.arange(t, device=dev),
            torch.arange(hp, device=dev),
            torch.arange(wp, device=dev),
            indexing="ij",
        )
        if len(patches) != tt.numel():
            raise ValueError("condition patch count does not match the latent grid")
        y = (
            self.input(patches.detach().to(self.input.weight.dtype))
            + self.time(tt.flatten())
            + self.height(yy.flatten())
            + self.width(xx.flatten())
            + self.type_embedding
        )
        if self.segment_embedding is not None:
            if segment_ids is None:
                segment_ids = torch.full((t,), SEGMENT_NULL, dtype=torch.long)
            ids = torch.as_tensor(segment_ids, device=dev, dtype=torch.long)
            if ids.shape != (t,) or bool(((ids < 0) | (ids > SEGMENT_NULL)).any()):
                raise ValueError(f"segment ids must be {t} values in 0..{SEGMENT_NULL}")
            y = y + self.segment_embedding(ids).repeat_interleave(hp * wp, dim=0).to(y.dtype)
        return y


class GlobalAdapter(nn.Linear):
    """Bias-free, zero-initialised ``Linear(GLOBAL_IN, width)`` for per-latent-frame flags."""

    def __init__(self, width: int = 4096):
        super().__init__(GLOBAL_IN, width, bias=False)
        nn.init.zeros_(self.weight)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def add_to(
        self,
        sequence: torch.Tensor,
        indexes: torch.Tensor,
        flags: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> None:
        """Add ``self(flags)`` in place to target rows ``indexes`` of a (t, hp, wp) token grid."""
        t, hp, wp = map(int, grid)
        flags = torch.as_tensor(flags, device=sequence.device)
        if flags.shape != (t, GLOBAL_IN) or indexes.numel() != t * hp * wp:
            raise ValueError(f"global flags must be ({t}, {GLOBAL_IN})")
        rows = self(flags.to(self.weight.dtype)).repeat_interleave(hp * wp, dim=0)
        sequence[indexes] = sequence[indexes] + rows.to(sequence.dtype)


# ----------------------------------------------------------------------------- inputs
def latent_groups(values: np.ndarray, num_latent: int) -> list[np.ndarray]:
    """Split a per-frame array into latent groups: frame 0 alone, then groups of four."""
    groups = [values[:1]] + [values[1 + 4 * i : 5 + 4 * i] for i in range(num_latent - 1)]
    if not all(len(g) for g in groups) or sum(map(len, groups)) != len(values):
        raise ValueError(f"{len(values)} frames do not form {num_latent} latent frames")
    return groups


def segment_ids_from_labels(labels: list[str | None] | np.ndarray) -> np.ndarray:
    """Per-frame segment names (T frames, 4n+1) -> per-latent-frame ids (frames 0, 4, 8, ...).

    Unknown or missing names map to the null id.
    """
    ids = []
    for value in list(labels)[::4]:
        name = value.strip().lower().replace(" ", "_") if isinstance(value, str) else None
        if name and name.endswith("_colon"):
            name = name[: -len("_colon")]
        ids.append(SEGMENT_CLASSES.index(name) if name in SEGMENT_CLASSES else SEGMENT_NULL)
    return np.asarray(ids, np.int64)


def palette_class_masks(palette: np.ndarray) -> dict[str, np.ndarray]:
    """Recover boolean class masks from a palette video by exact colour match (mask interiors)."""
    from colonoworld.conditions.palette import PALETTE

    out = {}
    for cls, colour in PALETTE.items():
        hit = (palette == np.asarray(colour, np.uint8)).all(-1)
        if hit.any():
            out[cls] = hit
    return out


def global_flags(
    num_frames: int,
    nbi: bool | np.ndarray | None = None,
    nbi_known: np.ndarray | None = None,
    choice: Mapping[str, str | None] | None = None,
    masks: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Per-latent-frame global flags, or ``None`` when every entry is zero.

    Args:
        num_frames: output frames (4n+1).
        nbi: per-frame special-light value, a single bool for all frames, or ``None`` (unknown).
        nbi_known: per-frame known bits for ``nbi`` (default: all known).
        choice: per mask channel (polyp, tool, water, snare): ``"mask"`` (supplied), ``"empty"``
            (supplied and known absent) or ``None`` (not supplied).
        masks: boolean (T, H, W) masks for channels whose choice is ``"mask"``.

    Light values are known for a latent frame only when all its frames are known and agree. Tool
    presence is any tool (or snare) pixel within the latent group; irrigation uses the water mask.
    """
    num_latent = 1 + (num_frames - 1) // 4
    out = np.zeros((num_latent, GLOBAL_IN), np.float32)
    choice = dict.fromkeys(MASK_CHANNELS) | dict(choice or {})
    masks = dict(masks or {})
    if nbi is not None:
        values = np.broadcast_to(np.asarray(nbi, bool), (num_frames,))
        known = np.ones(num_frames, bool) if nbi_known is None else np.asarray(nbi_known, bool)
        sure = np.asarray([g.all() for g in latent_groups(known, num_latent)], bool)
        groups = latent_groups(values, num_latent)
        on = np.asarray([g.all() for g in groups], bool) & sure
        off = np.asarray([(~g).all() for g in groups], bool) & sure
        out[:, 0] = on
        out[:, 4] = on | off

    def occupancy(cls: str) -> np.ndarray:
        frames = masks[cls].reshape(num_frames, -1).any(1)
        return np.asarray([g.any() for g in latent_groups(frames, num_latent)], bool)

    if choice["tool"] == "empty":
        out[:, 6] = 1
    elif choice["tool"] == "mask":
        occ = occupancy("tool")
        if choice["snare"] == "mask":
            occ = occ | occupancy("snare")
        out[:, 2] = occ
        out[:, 6] = 1
    if choice["water"] is not None:
        out[:, 3] = occupancy("water") if choice["water"] == "mask" else 0
        out[:, 7] = 1
    for cls in MASK_CHANNELS:
        if choice[cls] in ("mask", "empty"):
            out[:, 8 + GLOBAL_CLASSES.index(cls)] = 1
    out[:, :4] *= out[:, 4:8]
    return out if out.any() else None
