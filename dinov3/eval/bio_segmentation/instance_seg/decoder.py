"""
CellViT / UNETR-style HoVerNet decoder.

The decoder turns a set of intermediate ViT feature maps (all at patch
resolution Hp = H / patch_size) plus the raw image into three dense outputs:

    np : [B, 2, H, W]          nucleus-pixel logits
    hv : [B, 2, H, W]          horizontal/vertical distance regression (raw)
    tp : [B, num_types, H, W]  nucleus-type logits (multi-class datasets only)

Flexible layer taps
-------------------
The decoder's upsampling depth is fixed by geometry: ``n_up = log2(patch_size)``
(= 4 for patch16). But the number of ViT layers you *tap* is free. A fusion
front-end splits the K tapped features into 4 contiguous buckets and projects
each bucket (concat → 1×1 conv) to a common ``embed_proj`` dim, yielding the 4
UNETR skip inputs. So:
    - K = 4 (even4)      → 1 tap per bucket  (CellViT-exact)
    - K = 8 (more layers) → 2 taps per bucket
    - K < 4              → shallow buckets share the available taps
This is what makes "even4 is not fixed, more layers are fine" a config knob.

Reference: Hatamizadeh et al., "UNETR" (WACV 2022); Hörst et al., "CellViT".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basic building blocks (instance-norm + leaky-relu, UNETR style)
# ---------------------------------------------------------------------------

def _conv3x3(ic: int, oc: int) -> nn.Conv2d:
    return nn.Conv2d(ic, oc, kernel_size=3, padding=1, bias=False)


class _BasicBlock(nn.Module):
    """Two 3×3 conv-norm-act layers with a residual projection."""

    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.conv1 = _conv3x3(ic, oc)
        self.norm1 = nn.InstanceNorm2d(oc, affine=True)
        self.conv2 = _conv3x3(oc, oc)
        self.norm2 = nn.InstanceNorm2d(oc, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
        self.res = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if ic != oc else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.res(x))


class _PrUpBlock(nn.Module):
    """Project a patch-resolution ViT skip up by 2^(1+num_layer) via transpose convs."""

    def __init__(self, ic: int, oc: int, num_layer: int):
        super().__init__()
        self.init_up = nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(oc, oc, kernel_size=2, stride=2),
                _BasicBlock(oc, oc),
            )
            for _ in range(num_layer)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_up(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _UpBlock(nn.Module):
    """Upsample ×2, concat the skip, fuse."""

    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2)
        self.block = _BasicBlock(oc * 2, oc)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class _UNETRBranch(nn.Module):
    """One full UNETR decoder pathway (image stem + 4 ViT skips → dense logits)."""

    def __init__(self, embed_proj: int, feature: int, out_ch: int, image_ch: int = 3):
        super().__init__()
        self.encoder1 = _BasicBlock(image_ch, feature)            # full res, from image
        self.encoder2 = _PrUpBlock(embed_proj, feature * 2, num_layer=2)   # → H/2
        self.encoder3 = _PrUpBlock(embed_proj, feature * 4, num_layer=1)   # → H/4
        self.encoder4 = _PrUpBlock(embed_proj, feature * 8, num_layer=0)   # → H/8
        self.decoder5 = _UpBlock(embed_proj, feature * 8)         # bottleneck → H/8
        self.decoder4 = _UpBlock(feature * 8, feature * 4)        # → H/4
        self.decoder3 = _UpBlock(feature * 4, feature * 2)        # → H/2
        self.decoder2 = _UpBlock(feature * 2, feature)           # → H
        self.out = nn.Conv2d(feature, out_ch, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder1(image)   # H
        e2 = self.encoder2(z0)      # H/2  (shallowest bucket → finest skip)
        e3 = self.encoder3(z1)      # H/4
        e4 = self.encoder4(z2)      # H/8
        d = self.decoder5(z3, e4)   # H/8  (deepest bucket is the bottleneck)
        d = self.decoder4(d, e3)    # H/4
        d = self.decoder3(d, e2)    # H/2
        d = self.decoder2(d, e1)    # H
        return self.out(d)


# ---------------------------------------------------------------------------
# Tap → bucket assignment
# ---------------------------------------------------------------------------

def assign_buckets(num_taps: int, n_buckets: int = 4) -> List[List[int]]:
    """Map K tapped layers onto n_buckets contiguous buckets (shallow → deep).

    K >= n_buckets: contiguous split (extra taps concatenated within a bucket).
    K <  n_buckets: shallow buckets reuse the nearest available tap.
    """
    if num_taps < 1:
        raise ValueError("num_taps must be >= 1")
    if num_taps >= n_buckets:
        return [list(map(int, g)) for g in np.array_split(np.arange(num_taps), n_buckets)]
    # Fewer taps than buckets: bucket i uses tap min(i, K-1).
    return [[min(i, num_taps - 1)] for i in range(n_buckets)]


class HoVerNetDecoder(nn.Module):
    """Shared fusion front-end + per-branch UNETR decoders (NP / HV / TP)."""

    def __init__(
        self,
        tap_dims: List[int],
        num_types: int = 0,
        feature_size: int = 32,
        embed_proj: int = 384,
        image_ch: int = 3,
        patch_size: int = 16,
    ):
        super().__init__()
        n_up = int(round(math.log2(patch_size)))
        if 2 ** n_up != patch_size:
            raise ValueError(f"patch_size must be a power of 2, got {patch_size}")
        if n_up != 4:
            # The UNETR pathway above hard-codes 4 upsample stages.
            raise ValueError(
                f"This decoder assumes patch_size=16 (4 upsample stages); got patch_size={patch_size}."
            )

        self.buckets = assign_buckets(len(tap_dims), n_buckets=4)
        self.num_types = num_types

        # One 1×1 fusion conv per bucket: concat taps in the bucket → embed_proj.
        self.fuse = nn.ModuleList(
            nn.Conv2d(sum(tap_dims[i] for i in bucket), embed_proj, kernel_size=1)
            for bucket in self.buckets
        )

        self.np_branch = _UNETRBranch(embed_proj, feature_size, out_ch=2, image_ch=image_ch)
        self.hv_branch = _UNETRBranch(embed_proj, feature_size, out_ch=2, image_ch=image_ch)
        self.tp_branch = (
            _UNETRBranch(embed_proj, feature_size, out_ch=num_types, image_ch=image_ch)
            if num_types and num_types > 0
            else None
        )

    def forward(self, image: torch.Tensor, taps: List[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        zs: List[torch.Tensor] = []
        for bucket, conv in zip(self.buckets, self.fuse):
            feat = taps[bucket[0]] if len(bucket) == 1 else torch.cat([taps[i] for i in bucket], dim=1)
            zs.append(conv(feat))
        z0, z1, z2, z3 = zs

        out: Dict[str, Optional[torch.Tensor]] = {
            "np": self.np_branch(image, z0, z1, z2, z3),
            "hv": self.hv_branch(image, z0, z1, z2, z3),   # raw regression (no activation)
            "tp": None,
        }
        if self.tp_branch is not None:
            out["tp"] = self.tp_branch(image, z0, z1, z2, z3)
        return out
