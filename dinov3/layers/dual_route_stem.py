# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Dual-route patch-embedding stem for the general microscopy DINOv3 base.

Each sample is routed to one of two stems, and BOTH stems return RGB-shaped
patch tokens ``(N, H', W', embed_dim)`` so the rest of the ViT (RoPE / masking /
Gram / dense eval) needs no change at all:

* **RGB route** — ``<=3`` highly-correlated channels (grayscale / joint colour,
  the ~94% majority). Uses a standard :class:`PatchEmbed` (Conv2d, 3 in-chans)
  that is initialised *exactly* from the pretrained DINOv3 stem, so the 1.7B
  prior and stem-level chromatic mixing are preserved.
* **Pool route** — independent multi-channel data (fluorescence / Cell Painting
  / multiplex IF). A *shared* per-channel Conv2d projection + a **content-derived
  channel-identity** embedding (#3) + masked attention pooling over the channel
  axis collapses ``C`` at the stem into a fixed ``H'×W'`` token grid (no
  ``C×H'×W'`` token explosion; scales to 40-channel IMC).

Routing is decided **per sample** from cross-channel Pearson correlation, which
is invariant to the per-channel affine normalisation applied upstream, so no
data-pipeline flag and no marker/modality label is required.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .patch_embed import PatchEmbed, make_2tuple


def _per_channel_standardize(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Standardize every ``(N, C)`` map over its spatial dims (mean 0, std 1).

    Makes the content statistics robust to the upstream per-channel
    ``(x - mean) / std`` normalisation: only the *shape* of each channel's
    distribution / its spatial structure is kept, not its absolute scale.
    """
    mu = x.mean(dim=(-2, -1), keepdim=True)
    var = x.var(dim=(-2, -1), unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


class ContentChannelDescriptor(nn.Module):
    """#3 — permutation-invariant, label-free channel identity.

    Derives a small set of **scale-invariant** statistics from each channel's
    own content and projects them to ``embed_dim``. Because the descriptor comes
    from content (not a filename index or a marker label), the same physical
    structure (e.g. a punctate nuclear stain) maps to the same identity
    regardless of which channel slot it occupies or which dataset it came from.
    """

    N_STATS = 6

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.N_STATS, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    @staticmethod
    def _stats(x: Tensor) -> Tensor:
        """``(N, C, H, W) -> (N, C, N_STATS)`` (detached; constants wrt input)."""
        z = _per_channel_standardize(x)  # mean 0, std 1 per (N, C)
        m3 = (z ** 3).mean(dim=(-2, -1))
        m4 = (z ** 4).mean(dim=(-2, -1))
        # signed-log / log1p tame the heavy tails of punctate channels.
        skew = torch.sign(m3) * torch.log1p(m3.abs())
        kurt = torch.log1p(m4.clamp_min(0.0))
        # texture / edge energy (mean abs gradient of the standardized map).
        gx = (z[..., 1:, :] - z[..., :-1, :]).abs().mean(dim=(-2, -1))
        gy = (z[..., :, 1:] - z[..., :, :-1]).abs().mean(dim=(-2, -1))
        grad = gx + gy
        # lag-1 spatial autocorrelation (z standardized => E[z_t * z_{t+1}]).
        ac_h = (z[..., 1:, :] * z[..., :-1, :]).mean(dim=(-2, -1))
        ac_w = (z[..., :, 1:] * z[..., :, :-1]).mean(dim=(-2, -1))
        ac = 0.5 * (ac_h + ac_w)
        # heavy-tail fraction + bounded peak (punctate vs diffuse).
        tail = (z.abs() > 2.0).float().mean(dim=(-2, -1))
        peak = z.amax(dim=(-2, -1)).clamp(0.0, 20.0) / 20.0
        feats = torch.stack([skew, kurt, grad, ac, tail, peak], dim=-1)
        return feats.detach()

    def forward(self, x: Tensor) -> Tensor:
        feats = self._stats(x.float())  # (N, C, N_STATS)
        return self.mlp(feats.to(self.mlp[0].weight.dtype)).to(x.dtype)  # (N, C, D)

    def reset_parameters(self):
        for mod in self.mlp:
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)


class ChannelPoolStem(nn.Module):
    """Shared per-channel Conv2d projection + content identity + masked pooling.

    ``(N, C, H, W)`` + ``valid_mask (N, C)`` -> ``(N, H', W', embed_dim)``.
    The same 1-input-channel Conv2d is applied to every channel, then the
    content identity (#3) is added per channel, then the channel axis is pooled
    away (masked attention by default, masked mean optional).
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: int = 768,
        pool_type: str = "attn",
    ):
        super().__init__()
        patch_HW = make_2tuple(patch_size)
        self.patch_size = patch_HW
        self.embed_dim = embed_dim
        assert pool_type in ("attn", "mean"), f"unknown pool_type {pool_type}"
        self.pool_type = pool_type
        # Shared across channels: a single 1-in-channel patch projection.
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.descriptor = ContentChannelDescriptor(embed_dim)
        if pool_type == "attn":
            self.query = nn.Parameter(torch.empty(1, 1, 1, 1, embed_dim))
            self.scale = embed_dim ** -0.5

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        N, C, H, W = x.shape
        Hp, Wp = H // self.patch_size[0], W // self.patch_size[1]
        # Per-channel content identity (#3): (N, C, D)
        ident = self.descriptor(x)
        # Shared per-channel projection: fold channels into the batch dim.
        feat = self.proj(x.reshape(N * C, 1, H, W))  # (N*C, D, Hp, Wp)
        feat = feat.reshape(N, C, self.embed_dim, Hp, Wp).permute(0, 1, 3, 4, 2)  # (N,C,Hp,Wp,D)
        feat = feat + ident[:, :, None, None, :]
        # Masked pool over the channel axis -> (N, Hp, Wp, D)
        if self.pool_type == "attn":
            logits = (feat * self.query).sum(dim=-1) * self.scale  # (N, C, Hp, Wp)
            invalid = ~valid_mask[:, :, None, None].expand_as(logits)
            logits = logits.masked_fill(invalid, float("-inf"))
            attn = torch.softmax(logits, dim=1).to(feat.dtype)  # (N, C, Hp, Wp)
            out = (attn[..., None] * feat).sum(dim=1)
        else:  # masked mean
            mf = valid_mask[:, :, None, None, None].to(feat.dtype)  # (N,C,1,1,1)
            out = (feat * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)
        return out  # (N, Hp, Wp, D)

    def reset_parameters(self):
        k = 1.0 / (self.patch_size[0] * self.patch_size[1])
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
        self.descriptor.reset_parameters()
        if self.pool_type == "attn":
            nn.init.trunc_normal_(self.query, std=0.02)


class DualRouteStem(nn.Module):
    """Route each sample to the RGB stem or the channel-pool stem (see module
    docstring). Output is RGB-shaped ``(B, H', W', embed_dim)``.

    Args:
        corr_threshold: mean abs cross-channel Pearson correlation above which a
            ``<=max_rgb_channels`` sample is treated as joint colour (RGB route).
        max_rgb_channels: samples with more valid channels always take the pool
            route (the RGB Conv2d only accepts 3 channels).
        pool_type: ``"attn"`` (default) or ``"mean"``.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: int = 768,
        corr_threshold: float = 0.5,
        max_rgb_channels: int = 3,
        pool_type: str = "attn",
        **ignored,
    ):
        super().__init__()
        self.patch_size = make_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.corr_threshold = corr_threshold
        self.max_rgb_channels = max_rgb_channels
        self.rgb = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.pool = ChannelPoolStem(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            pool_type=pool_type,
        )

    @staticmethod
    def _mean_abs_corr(vx: Tensor) -> float:
        """Mean abs off-diagonal Pearson correlation of channels ``vx (n,H,W)``."""
        n = vx.shape[0]
        if n <= 1:
            return 1.0
        f = vx.reshape(n, -1).float()
        f = f - f.mean(dim=1, keepdim=True)
        f = f / f.std(dim=1, keepdim=True).clamp_min(1e-6)
        corr = (f @ f.t()) / f.shape[1]  # (n, n)
        off = corr - torch.diag(torch.diagonal(corr))
        return float(off.abs().sum() / (n * (n - 1)))

    @torch.no_grad()
    def _route_and_rgb_input(self, x: Tensor, valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Per-sample route decision + a valid 3-channel RGB input for EVERY
        sample. Returns ``(is_rgb (N,) bool, rgb_in (N, 3, H, W))``.

        A 3-channel input is built for pool-route samples too (it is masked out
        in forward). This keeps the RGB stem's execution data-INDEPENDENT, which
        is required so all FSDP ranks issue identical collectives (see forward).
        """
        N, _, H, W = x.shape
        is_rgb = torch.zeros(N, dtype=torch.bool, device=x.device)
        rgb_in = x.new_zeros(N, 3, H, W)
        for i in range(N):
            valid = valid_mask[i]
            n = int(valid.sum())
            vx = x[i, :1] if n == 0 else x[i][valid]  # (max(n,1), H, W)
            if n <= self.max_rgb_channels and self._mean_abs_corr(vx) >= self.corr_threshold:
                is_rgb[i] = True
            v3 = vx[:3]
            if v3.shape[0] < 3:  # 1ch/2ch -> replicate last channel up to 3
                v3 = torch.cat([v3, v3[-1:].expand(3 - v3.shape[0], H, W)], dim=0)
            rgb_in[i] = v3
        return is_rgb, rgb_in

    def forward(
        self,
        x: Tensor,
        channel_ids: Optional[Tensor] = None,  # accepted for interface compat (#2 hook), unused in v1
        channel_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        N, C, H, W = x.shape
        if channel_valid_mask is None:
            valid_mask = torch.ones(N, C, dtype=torch.bool, device=x.device)
        else:
            valid_mask = channel_valid_mask.to(device=x.device, dtype=torch.bool)

        is_rgb, rgb_in = self._route_and_rgb_input(x, valid_mask)

        # BOTH stems ALWAYS run on the full batch on EVERY rank, then we select
        # per sample with a 0/1 convex combination. This is an FSDP-safety
        # requirement: data-dependent branching (running pool only when some
        # sample needs it) makes ranks issue different all-gather / reduce-
        # scatter collectives -> NCCL deadlock/timeout. Always-compute-both +
        # masked-combine keeps the collective pattern identical across ranks and
        # still lets gradients reach both stems every step (0 where unused — the
        # backward traverses the masked branch even when its weight is 0).
        rgb_tokens = self.rgb(rgb_in)            # (N, H', W', D)
        pool_tokens = self.pool(x, valid_mask)   # (N, H', W', D)
        w = is_rgb.view(N, 1, 1, 1).to(rgb_tokens.dtype)
        return w * rgb_tokens + (1.0 - w) * pool_tokens

    def reset_parameters(self):
        self.rgb.reset_parameters()
        self.pool.reset_parameters()
