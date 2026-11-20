# SPDX-License-Identifier: CC-BY-NC-4.0
"""Cosmos model subclass with ColonoWorld conditioning and composable guidance.

The Cosmos framework (``cosmos_framework``) is imported lazily, so this module can be imported
without it. :func:`build_backbone` is the instantiation target used by :mod:`.loader`.

Guidance (applied inside every denoiser call, independently of text guidance):

* tube only: ``v = v_0 + s_t (v_T - v_0)`` with ``v_0`` the forward without tube rows,
  segment ids, mask rows and global flags (camera rays and text are kept). ``s_t = 1`` runs a
  single forward.
* tube and mask: ``v = v_0 + s_t (v_T - v_0) + s_m (v_TM - v_T)``, where ``v_TM`` is the full
  condition, ``v_T`` keeps the tube rows and segment ids but drops the mask rows and global flags,
  and ``v_0`` keeps only rays and text.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any

import torch

from colonoworld.models.network import ConditionState, attach_block_hooks, install


def compose_tube(full: torch.Tensor, null: torch.Tensor, scale: float) -> torch.Tensor:
    """Tube guidance; ``scale == 1`` returns ``full`` unchanged."""
    if scale == 1:
        return full
    return null + scale * (full - null)


def compose_tube_mask(
    v0: torch.Tensor, vt: torch.Tensor, vtm: torch.Tensor, s_t: float, s_m: float
) -> torch.Tensor:
    """Composable tube and mask guidance ``v0 + s_t (vt - v0) + s_m (vtm - vt)``."""
    return v0 + s_t * (vt - v0) + s_m * (vtm - vt)


@functools.cache
def model_class() -> type:
    """The ColonoWorld subclass of the Cosmos ``OmniMoTModel``."""
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    class ColonoWorldMoTModel(OmniMoTModel):
        """Cosmos MoT model with camera, tube, mask, segment and global conditioning."""

        tube_cfg: float = 1.0
        mask_cfg: float = 3.0

        def install_attention_dispatch(self, net: torch.nn.Module) -> None:
            super().install_attention_dispatch(net)
            install(
                net,
                width=net.vae2llm.out_features,
                patch=int(self.config.diffusion_expert_config.patch_spatial),
                num_blocks=len(net.language_model.model.layers),
            )
            initialize = net.init_weights

            def init_weights(*args, **kwargs):
                result = initialize(*args, **kwargs)
                for name in ("camera_adapter", "camera_block_encoders", "global_adapter"):
                    getattr(net, name).reset_parameters()
                net.tube_stream.reset_parameters()
                net.mask_stream.reset_parameters()
                return result

            net.init_weights = init_weights

        def build_net(self, *args, **kwargs) -> torch.nn.Module:
            if self.config.parallelism.context_parallel_shard_degree != 1:
                raise ValueError("ColonoWorld conditioning needs context parallel size 1")
            net = super().build_net(*args, **kwargs)
            net._colonoworld_hooks = attach_block_hooks(net)
            return net

        def _can_reuse_inference_text_kv(self, *args, **kwargs) -> bool:
            # Condition rows change with every guidance branch; the text KV cache is not used.
            return False

        @property
        def state(self) -> ConditionState:
            return self.net.colonoworld_state

        @contextmanager
        def condition_scope(self, state: ConditionState):
            """Set the conditions of every forward inside the scope."""
            previous = self.net.colonoworld_state
            self.net.colonoworld_state = state
            try:
                yield
            finally:
                self.net.colonoworld_state = previous

        def denoise(
            self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None
        ):
            base = functools.partial(
                super().denoise, net=net, video_temporal_causal=video_temporal_causal
            )
            state = self.state
            if state.mask is not None:
                full = base(data_batch_packed=data_batch_packed, memory=memory)
                with self.condition_scope(state.replace(mask=None, flags=None)):
                    vt = base(data_batch_packed=data_batch_packed, memory=None)
                with self.condition_scope(
                    state.replace(tube=None, mask=None, flags=None, segment_ids=None)
                ):
                    v0 = base(data_batch_packed=data_batch_packed, memory=None)
                result = dict(full)
                result["preds_vision"] = [
                    compose_tube_mask(a, b, c, self.tube_cfg, self.mask_cfg)
                    for c, b, a in zip(
                        full["preds_vision"], vt["preds_vision"], v0["preds_vision"], strict=True
                    )
                ]
                return result
            full = base(data_batch_packed=data_batch_packed, memory=memory)
            if self.tube_cfg == 1 or state.tube is None:
                return full
            with self.condition_scope(
                state.replace(tube=None, mask=None, flags=None, segment_ids=None)
            ):
                null = base(data_batch_packed=data_batch_packed, memory=None)
            result = dict(full)
            result["preds_vision"] = [
                compose_tube(p, b, self.tube_cfg)
                for p, b in zip(full["preds_vision"], null["preds_vision"], strict=True)
            ]
            return result

    return ColonoWorldMoTModel


def build_backbone(config: Any) -> Any:
    """Instantiation target: build the ColonoWorld Cosmos model from a framework model config."""
    return model_class()(config)
