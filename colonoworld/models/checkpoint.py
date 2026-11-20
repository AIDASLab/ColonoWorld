# SPDX-License-Identifier: CC-BY-NC-4.0
"""ColonoWorld checkpoints.

A ColonoWorld checkpoint is a ``torch.save`` dict with the trained tensors only (the Cosmos base
weights are loaded separately)::

    {"ema": {name: tensor}, "raw": {name: tensor}, ...metadata}

Names are model parameter names (``net.<module>...``). The trained tensors are the generation
LoRA (rank 16 on the q/k/v/o projections of the generation expert), the camera input adapter,
the 36 per-block camera encoders, the tube and mask condition streams with the segment
embedding, and the global flag adapter. ``ema`` holds the exponential moving average
(decay 0.99) of ``raw`` and is the default for inference.

:func:`canonical_name` also accepts the training-time names of these tensors, so checkpoints
written by the training code load directly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\._checkpoint_wrapped_module\."), "."),
    (re.compile(r"^net\.tok_condition\.mask_stream\."), "net.mask_stream."),
    (re.compile(r"^net\.tok_condition\.global_adapter\."), "net.global_adapter."),
    (re.compile(r"^net\.tok_condition\."), "net.tube_stream."),
    (re.compile(r"^net\.plucker_adapter\."), "net.camera_adapter."),
    (re.compile(r"^net\.perblock_encoders\."), "net.camera_block_encoders."),
)
TRAINED_PREFIXES = (
    "net.camera_adapter.",
    "net.camera_block_encoders.",
    "net.tube_stream.",
    "net.mask_stream.",
    "net.global_adapter.",
)


def canonical_name(name: str) -> str:
    """Map a stored tensor name to the ColonoWorld parameter name."""
    for pattern, replacement in _RENAMES:
        name = pattern.sub(replacement, name)
    return name


def is_trained(name: str) -> bool:
    """Whether a parameter is trained by ColonoWorld (and therefore stored in its checkpoints)."""
    return name.startswith(TRAINED_PREFIXES) or ".lora_" in name


def read_state(path: str | Path, weights: str = "ema") -> dict[str, torch.Tensor]:
    """Read one weight set (``ema`` or ``raw``) of a checkpoint, with canonical names.

    Files holding a flat ``trainables`` dict (camera-stage checkpoints) are read as is.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if weights in data:
        state = data[weights]
    elif "trainables" in data:
        state = data["trainables"]
    else:
        raise KeyError(f"{path} has no '{weights}' weights")
    return {canonical_name(k): v for k, v in state.items()}


def merge_states(states: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Merge weight sets; later sets override earlier ones (e.g. camera stage, then final)."""
    out: dict[str, torch.Tensor] = {}
    for state in states:
        out.update(state)
    return out


@torch.no_grad()
def load_trained(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> list[str]:
    """Copy ``state`` into the trained parameters of ``model``.

    Every trained parameter of the model must be present; returns the loaded names.
    """
    params = {
        canonical_name(n): p for n, p in model.named_parameters() if is_trained(canonical_name(n))
    }
    missing = sorted(set(params) - set(state))
    unexpected = sorted(set(state) - set(params))
    if missing or unexpected:
        raise KeyError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    for name, param in params.items():
        value = state[name]
        local = param.to_local() if hasattr(param, "to_local") else param
        if local.shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f"bad tensor for {name}")
        local.copy_(value.to(local))
    return sorted(params)


def export_checkpoint(
    out: str | Path, sources: Iterable[str | Path], metadata: Mapping | None = None
) -> None:
    """Write a single ColonoWorld checkpoint from training outputs.

    Args:
        out: destination ``.pt`` file.
        sources: checkpoint files in load order; a later file overrides tensors of an earlier
            one. Files without separate raw/EMA sets contribute the same tensors to both.
        metadata: extra entries stored next to the weights.
    """
    sets = {}
    for weights in ("ema", "raw"):
        sets[weights] = merge_states(read_state(p, weights) for p in sources)
    torch.save(dict(sets, format="colonoworld-v1", **dict(metadata or {})), out)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge training checkpoints into one file.")
    parser.add_argument("--out", required=True)
    parser.add_argument("sources", nargs="+", help="checkpoint files in load order")
    args = parser.parse_args(argv)
    export_checkpoint(args.out, args.sources)


if __name__ == "__main__":
    main()
