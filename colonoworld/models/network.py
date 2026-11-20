# SPDX-License-Identifier: CC-BY-NC-4.0
"""ColonoWorld extensions of the Cosmos video network.

:func:`install` adds the conditioning modules to a freshly built (meta-device) Cosmos network and
switches its class to :class:`ColonoWorldNetwork`, which

1. appends tube and mask condition rows to the packed sequence before the forward pass,
2. after the stock vision projection, adds the input camera term to the target tokens, prepares
   the per-block camera payload, embeds the condition rows and adds the global flags,
3. decodes target rows only.

:func:`attach_block_hooks` adds the per-block camera term in front of every self-attention.
The current conditions live in ``net.colonoworld_state`` (:class:`ConditionState`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from colonoworld.models.camera import (
    PerBlockEncoders,
    PluckerEmbedder,
    block_payload,
    bounded_delta,
    pad_rays,
)
from colonoworld.models.conditions import ConditionTokens, GlobalAdapter

_PAYLOAD_KEY = "_colonoworld_camera_addition"


@dataclass
class ConditionState:
    """Conditions of the current forward pass.

    Attributes:
        rays: one ray tensor (T_latent, 6, H, W) per sample; required for every forward.
        tube: tube latent (C, T, H, W) or ``None`` (tube rows absent).
        mask: mask latent (C, T, H, W) or ``None`` (mask rows absent).
        flags: global flags (T_latent, GLOBAL_IN) or ``None``.
        segment_ids: per-latent-frame segment ids for the tube rows, or ``None`` (null id).
    """

    rays: list[torch.Tensor] | None = None
    tube: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    flags: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None
    block_payload: tuple[torch.Tensor, torch.Tensor] | None = field(default=None, repr=False)

    def replace(self, **changes) -> ConditionState:
        out = copy.copy(self)
        for key, value in changes.items():
            setattr(out, key, value)
        out.block_payload = None
        return out


def append_condition_rows(packed, latents: list[torch.Tensor], names: tuple[str, ...]):
    """Return a copy of a single-sample packed sequence with one block of condition rows per latent.

    Every appended row copies the full mRoPE position (latent time, spatial cell and frame-rate
    modulation) of the target row it mirrors. Condition rows are marked clean and carry no loss.
    """
    if len(packed.vision.tokens) != 1 or len(packed.sample_lens) != 1:
        raise ValueError("condition rows need a single unpadded video sample")
    if packed.attn_modes[-1] != "full" or packed.control_weights is not None:
        raise ValueError("condition rows need dense full attention")
    out = copy.copy(packed)
    vision = copy.copy(packed.vision)
    for latent in latents:
        if tuple(latent.shape[-4:]) != tuple(vision.tokens[0].shape[-4:]):
            raise ValueError("condition latent and video latent shapes differ")
    k = len(latents)
    n = len(vision.sequence_indexes)
    start = packed.sequence_length
    target = vision.sequence_indexes
    blocks = [
        torch.arange(start + i * n, start + (i + 1) * n, device=target.device, dtype=target.dtype)
        for i in range(k)
    ]
    out.sequence_length = start + k * n
    out.sample_lens = [packed.sample_lens[0] + k * n]
    out.split_lens = list(packed.split_lens)
    out.split_lens[-1] += k * n
    out.position_ids = torch.cat(
        [packed.position_ids] + [packed.position_ids[:, target]] * k, dim=1
    )
    vision.sequence_indexes = torch.cat([target] + blocks)
    vision.tokens = list(vision.tokens) + [x.detach().reshape_as(vision.tokens[0]) for x in latents]
    vision.token_shapes = list(vision.token_shapes) * (1 + k)
    vision.seconds_per_frame = list(vision.seconds_per_frame) * (1 + k)
    out.num_vision_items_per_sample = [1 + k]
    out.num_views_per_vision_item = [1] * (1 + k)
    vision.condition_mask = list(vision.condition_mask) + [
        torch.ones_like(vision.condition_mask[0]) for _ in range(k)
    ]
    vision.noisy_frame_indexes = list(vision.noisy_frame_indexes) + [
        vision.noisy_frame_indexes[0][:0] for _ in range(k)
    ]
    out.vision = vision
    out.vision_item_split_lens = []
    for key in ("_sequence_pack_metadata", "sequence_pack_metadata"):
        if hasattr(out, key):
            setattr(out, key, None)
    out._colonoworld_target = packed
    out._colonoworld_rows = list(zip(names, latents, blocks, strict=True))
    return out


class ColonoWorldNetwork(nn.Module):
    """Mixin placed in front of the Cosmos network class by :func:`install`."""

    colonoworld_state: ConditionState

    def forward(self, packed_seq, *args, **kwargs):  # noqa: D102
        state = self.colonoworld_state
        latents, names = [], []
        if state.tube is not None:
            latents.append(state.tube)
            names.append("tube")
        if state.mask is not None:
            latents.append(state.mask)
            names.append("mask")
        try:
            if latents:
                if kwargs.get("memory") is not None:
                    raise ValueError("condition rows do not support a text KV cache")
                packed_seq = append_condition_rows(packed_seq, latents, tuple(names))
            return super().forward(packed_seq, *args, **kwargs)
        finally:
            state.block_payload = None

    def _encode_vision(self, packed_seq, packed_sequence, target_dtype):
        state = self.colonoworld_state
        target = getattr(packed_seq, "_colonoworld_target", packed_seq)
        shapes = super()._encode_vision(target, packed_sequence, target_dtype)
        if state.rays is None or shapes is None or len(state.rays) != len(shapes):
            raise ValueError("every forward needs one camera ray tensor per video sample")
        indexes = target.vision.sequence_indexes
        adapter = self.camera_adapter
        embeds = torch.cat(
            [
                adapter(pad_rays(r, s, adapter.patch).to(dtype=adapter.proj.weight.dtype))
                for r, s in zip(state.rays, shapes, strict=True)
            ]
        ).to(dtype=target_dtype)
        if len(embeds) != len(indexes):
            raise ValueError("camera and video token counts differ")
        packed_sequence[indexes] = packed_sequence[indexes] + bounded_delta(
            embeds, packed_sequence[indexes]
        )
        if len(shapes) != 1:
            raise ValueError("ColonoWorld conditioning supports one video sample per forward")
        state.block_payload = block_payload(adapter, state.rays[0], shapes[0], indexes)
        for name, latent, rows in getattr(packed_seq, "_colonoworld_rows", ()):
            patches, _ = self.patchify_and_pack_latents(
                [latent], target.vision.token_shapes, latent.shape[-4]
            )
            if name == "tube":
                y = self.tube_stream(patches, shapes[0], segment_ids=state.segment_ids)
            else:
                y = self.mask_stream(patches, shapes[0])
            packed_sequence[rows] = y.to(packed_sequence.dtype)
        if state.flags is not None:
            t, h, w = map(int, shapes[0])
            self.global_adapter.add_to(
                packed_sequence, indexes, state.flags, (t, (h + 1) // 2, (w + 1) // 2)
            )
        return shapes

    def _decode_vision(self, packed_seq, *args, **kwargs):
        return super()._decode_vision(
            getattr(packed_seq, "_colonoworld_target", packed_seq), *args, **kwargs
        )


def install(net: nn.Module, width: int = 4096, patch: int = 2, num_blocks: int = 36) -> nn.Module:
    """Add the ColonoWorld modules to a meta-device Cosmos network and switch its class."""
    with torch.device("meta"):
        net.camera_adapter = PluckerEmbedder(width, patch)
        net.camera_block_encoders = PerBlockEncoders(num_blocks, width, patch)
        net.tube_stream = ConditionTokens(width, patch * patch * 48, segment_embedding=True)
        net.mask_stream = ConditionTokens(width, patch * patch * 48)
        net.global_adapter = GlobalAdapter(width)
    net.colonoworld_state = ConditionState()
    base = type(net)
    net.__class__ = type("ColonoWorld" + base.__name__, (ColonoWorldNetwork, base), {})
    return net


def _inject(pack: dict, indexes: torch.Tensor, addition: torch.Tensor) -> dict:
    """Add the bounded camera term to the generation-stream rows of ``indexes``."""
    if pack["is_sharded"]:
        raise ValueError("per-block camera conditioning needs context parallel size 1")
    full = pack["_full_indices"].long()
    gen = pack["full_only_seq"]
    indexes = indexes.to(device=full.device, dtype=torch.long)
    local = torch.searchsorted(full, indexes)
    if not torch.equal(full[local], indexes):
        raise ValueError("camera index outside the generation stream")
    out = dict(pack)
    out.pop(_PAYLOAD_KEY, None)
    out["full_only_seq"] = gen.index_copy(
        0, local, gen[local] + bounded_delta(addition.to(gen.dtype), gen[local])
    )
    return out


def attach_block_hooks(net: nn.Module) -> list:
    """Register the per-block camera hooks on every decoder layer (call after parallelisation)."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    layers = net.language_model.model.layers
    if len(layers) != len(net.camera_block_encoders):
        raise ValueError("one camera encoder per decoder block is required")
    handles = []
    for index, outer in enumerate(layers):
        inner = outer._checkpoint_wrapped_module if isinstance(outer, CheckpointWrapper) else outer

        def before_block(module, args, index=index):
            indexes, rays = net.colonoworld_state.block_payload
            encoder = net.camera_block_encoders[index]
            pack = dict(args[0])
            pack[_PAYLOAD_KEY] = (indexes, encoder(rays.to(dtype=encoder.weight.dtype)))
            return (pack, *args[1:])

        def before_attention(module, args):
            indexes, addition = args[0][_PAYLOAD_KEY]
            return (_inject(args[0], indexes, addition), *args[1:])

        def after_block(module, args, output):
            pack = dict(output[0])
            pack.pop(_PAYLOAD_KEY, None)
            return (pack, *output[1:])

        handles += [
            outer.register_forward_pre_hook(before_block),
            inner.self_attn.register_forward_pre_hook(before_attention),
            outer.register_forward_hook(after_block),
        ]
    return handles
