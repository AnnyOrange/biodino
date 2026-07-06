# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Conservative residual multi-channel patch stem.

This stem keeps the official RGB PatchEmbed as the base path and only lets
channels beyond RGB contribute through a small residual branch:

    out = RGBPatchEmbed(ch0, ch1, ch2) + alpha * ExtraBranch(ch3...)

The output shape is the same as standard PatchEmbed, ``(B, H', W', D)``, so the
Transformer, RoPE, masking, and dense evaluators stay unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .patch_embed import PatchEmbed, make_2tuple


class ResidualMultiChannelStem(nn.Module):
    """RGB base stem plus a tiny residual branch for channels 4+.

    Args:
        extra_scale_init: Initial scalar multiplier for the extra-channel
            residual. A small non-zero value gives the extra branch gradients
            immediately while keeping the model very close to the RGB baseline.

    Channel semantics:
      * With ``channel_ids``: physical ids 0/1/2 are placed into RGB slots
        R/G/B, and ids >= 3 are pooled by the extra branch.
      * Without ``channel_ids``: positions 0/1/2 are RGB slots, positions >= 3
        are extra channels.

    Missing RGB slots are zero-filled, matching the conservative RGB packing
    path: 1ch -> [x,0,0], 2ch -> [x1,x2,0].
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: int = 768,
        extra_scale_init: float = 1e-3,
        **ignored,
    ):
        super().__init__()
        self.patch_size = make_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.extra_scale_init = float(extra_scale_init)
        self.rgb = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.extra = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        # FSDP fully_shard does not support scalar (0-D) parameters, so keep the
        # residual gate as a length-1 vector. It broadcasts exactly like a scalar.
        self.extra_scale = nn.Parameter(torch.full((1,), self.extra_scale_init, dtype=torch.float32))

    def _canonical_channel_ids(
        self,
        channel_ids: Optional[Tensor],
        batch_size: int,
        n_channels: int,
        device: torch.device,
    ) -> Tensor:
        if channel_ids is None:
            return torch.arange(n_channels, dtype=torch.long, device=device).expand(batch_size, -1)
        channel_ids = channel_ids.to(device=device, dtype=torch.long)
        if channel_ids.ndim == 1:
            if channel_ids.shape[0] != n_channels:
                raise ValueError(
                    f"channel_ids length ({channel_ids.shape[0]}) must match input channels ({n_channels})"
                )
            return channel_ids.expand(batch_size, -1)
        if channel_ids.ndim == 2:
            if channel_ids.shape != (batch_size, n_channels):
                raise ValueError(
                    "batched channel_ids must have shape "
                    f"({batch_size}, {n_channels}), got {tuple(channel_ids.shape)}"
                )
            return channel_ids
        raise ValueError(f"channel_ids must be 1D or 2D, got shape={tuple(channel_ids.shape)}")

    def _build_rgb_input(self, x: Tensor, valid_mask: Tensor, channel_ids: Tensor) -> Tensor:
        B, _, H, W = x.shape
        rgb_in = x.new_zeros(B, 3, H, W)
        for slot in range(3):
            match = (channel_ids == slot) & valid_mask
            if not match.any():
                continue
            # At most one physical channel should map to each RGB slot. If a
            # malformed sample duplicates ids, the first match is used.
            src_idx = match.float().argmax(dim=1)
            has_slot = match.any(dim=1)
            if has_slot.any():
                rgb_in[has_slot, slot] = x[has_slot, src_idx[has_slot]]
        return rgb_in

    def _extra_tokens(self, x: Tensor, extra_mask: Tensor) -> Tensor:
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size[0], W // self.patch_size[1]
        feat = self.extra(x.reshape(B * C, 1, H, W))
        feat = feat.reshape(B, C, self.embed_dim, Hp, Wp).permute(0, 1, 3, 4, 2)
        mask = extra_mask[:, :, None, None, None].to(dtype=feat.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (feat * mask).sum(dim=1) / denom

    def forward(
        self,
        x: Tensor,
        channel_ids: Optional[Tensor] = None,
        channel_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, C, _, _ = x.shape
        if channel_valid_mask is None:
            valid_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        else:
            valid_mask = channel_valid_mask.to(device=x.device, dtype=torch.bool)
            if valid_mask.shape != (B, C):
                raise ValueError(f"channel_valid_mask must have shape {(B, C)}, got {tuple(valid_mask.shape)}")

        channel_ids = self._canonical_channel_ids(channel_ids, B, C, x.device)
        rgb_tokens = self.rgb(self._build_rgb_input(x, valid_mask, channel_ids))
        extra_tokens = self._extra_tokens(x, valid_mask & (channel_ids >= 3))
        scale = self.extra_scale.to(device=x.device, dtype=rgb_tokens.dtype)
        return rgb_tokens + scale * extra_tokens

    def reset_parameters(self):
        self.rgb.reset_parameters()
        k = 1.0 / (self.patch_size[0] * self.patch_size[1])
        nn.init.uniform_(self.extra.weight, -math.sqrt(k), math.sqrt(k))
        with torch.no_grad():
            self.extra_scale.fill_(self.extra_scale_init)
