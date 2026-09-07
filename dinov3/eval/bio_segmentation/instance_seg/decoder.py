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


class _DepthwiseDownsample(nn.Module):
    """Lightweight stride-2 local feature stage."""

    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic, ic, kernel_size=3, stride=2, padding=1, groups=ic, bias=False),
            nn.Conv2d(ic, oc, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, oc), oc),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNSpatialAdapter(nn.Module):
    """Image pyramid that preserves local cues at 1/4, 1/8 and 1/16 scales."""

    def __init__(self, feature: int, embed_proj: int, width: int = 32, image_ch: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(image_ch, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
        )
        self.to_quarter = _DepthwiseDownsample(width, feature * 4)
        self.to_eighth = _DepthwiseDownsample(feature * 4, feature * 8)
        self.to_sixteenth = _DepthwiseDownsample(feature * 8, embed_proj)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(image)
        quarter = self.to_quarter(x)
        eighth = self.to_eighth(quarter)
        sixteenth = self.to_sixteenth(eighth)
        return {"quarter": quarter, "eighth": eighth, "sixteenth": sixteenth}


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
        local_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        e1 = self.encoder1(image)   # H
        e2 = self.encoder2(z0)      # H/2  (shallowest bucket → finest skip)
        e3 = self.encoder3(z1)      # H/4
        e4 = self.encoder4(z2)      # H/8
        if local_features is not None:
            e3 = e3 + local_features["quarter"]
            e4 = e4 + local_features["eighth"]
        d = self.decoder5(z3, e4)   # H/8  (deepest bucket is the bottleneck)
        d = self.decoder4(d, e3)    # H/4
        d = self.decoder3(d, e2)    # H/2
        d = self.decoder2(d, e1)    # H
        return self.out(d)


class _BucketFPNFusion(nn.Module):
    """Spatial FPN over bucket projections while preserving the four UNETR inputs."""

    def __init__(self, tap_dims: List[int], buckets: List[List[int]], embed_proj: int):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(sum(tap_dims[i] for i in bucket), embed_proj, kernel_size=1)
            for bucket in buckets
        )
        self.lateral = nn.ModuleList(
            nn.Conv2d(embed_proj, embed_proj, kernel_size=3, padding=1)
            for _ in buckets
        )

    def forward(self, taps: List[torch.Tensor], buckets: List[List[int]]) -> List[torch.Tensor]:
        projected = []
        for bucket, projection in zip(buckets, self.projections):
            x = taps[bucket[0]] if len(bucket) == 1 else torch.cat([taps[i] for i in bucket], dim=1)
            projected.append(projection(x))
        # Build real spatial scales from patch-resolution taps, then fuse top-down.
        h, w = projected[0].shape[-2:]
        pyramids = [projected[i] if i == 0 else torch.nn.functional.avg_pool2d(
            projected[i], kernel_size=2 ** min(i, 3), stride=2 ** min(i, 3)
        ) for i in range(len(projected))]
        topdown = [None] * len(pyramids)
        topdown[-1] = self.lateral[-1](pyramids[-1])
        for i in range(len(pyramids) - 2, -1, -1):
            up = torch.nn.functional.interpolate(topdown[i + 1], size=pyramids[i].shape[-2:], mode="nearest")
            topdown[i] = self.lateral[i](pyramids[i] + up)
        return [torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) for x in topdown]


class _MultiLayerFPNFusion(nn.Module):
    """Independent per-tap projections followed by high-resolution skip FPN fusion."""

    def __init__(self, tap_dims: List[int], buckets: List[List[int]], embed_proj: int):
        super().__init__()
        self.tap_projections = nn.ModuleList(
            nn.Conv2d(dim, embed_proj, kernel_size=1) for dim in tap_dims
        )
        self.fuse = nn.ModuleList(
            nn.Conv2d(embed_proj, embed_proj, kernel_size=3, padding=1) for _ in buckets
        )

    def forward(self, taps: List[torch.Tensor], buckets: List[List[int]]) -> List[torch.Tensor]:
        projected = [proj(tap) for proj, tap in zip(self.tap_projections, taps)]
        h, w = projected[0].shape[-2:]
        # Each bucket receives an independent projection; the shallowest map is
        # retained as the high-resolution skip in the top-down path.
        bucket_maps = []
        for bucket in buckets:
            bucket_maps.append(torch.stack([projected[i] for i in bucket], dim=0).mean(dim=0))
        topdown = [None] * len(bucket_maps)
        topdown[-1] = self.fuse[-1](bucket_maps[-1])
        for i in range(len(bucket_maps) - 2, -1, -1):
            up = torch.nn.functional.interpolate(topdown[i + 1], size=bucket_maps[i].shape[-2:], mode="nearest")
            topdown[i] = self.fuse[i](bucket_maps[i] + up)
        return [torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) for x in topdown]


class _UNetScaleFusion(nn.Module):
    """逐级上采样 path with explicit skip projections for the four tapped features."""

    def __init__(self, tap_dims: List[int], buckets: List[List[int]], embed_proj: int):
        super().__init__()
        self.tap_projections = nn.ModuleList(
            nn.Conv2d(dim, embed_proj, kernel_size=1) for dim in tap_dims
        )
        self.fuse = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(embed_proj * 2, embed_proj, kernel_size=3, padding=1),
                nn.InstanceNorm2d(embed_proj, affine=True),
                nn.GELU(),
            ) for _ in buckets
        )

    def forward(self, taps: List[torch.Tensor], buckets: List[List[int]]) -> List[torch.Tensor]:
        projected = [proj(tap) for proj, tap in zip(self.tap_projections, taps)]
        h, w = projected[0].shape[-2:]
        skips = []
        for bucket_index, bucket in enumerate(buckets):
            skip = torch.stack([projected[i] for i in bucket], dim=0).mean(dim=0)
            if bucket_index:
                scale = 2 ** min(bucket_index, 3)
                skip = torch.nn.functional.avg_pool2d(skip, kernel_size=scale, stride=scale)
            skips.append(skip)
        path = skips[-1]
        outputs = [None] * len(skips)
        outputs[-1] = path
        for i in range(len(skips) - 2, -1, -1):
            path = torch.nn.functional.interpolate(path, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            path = self.fuse[i](torch.cat([path, skips[i]], dim=1))
            outputs[i] = path
        return [torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) for x in outputs]


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
        fusion_mode: str = "bucket_concat",
        decoder_variant: str = "current",
        spatial_adapter: bool = False,
        spatial_adapter_width: int = 32,
        hv_auxiliary: bool = False,
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
        self.fusion_mode = fusion_mode
        self.decoder_variant = decoder_variant
        self.spatial_adapter_enabled = bool(spatial_adapter)
        self.spatial_adapter_fusion = "additive" if self.spatial_adapter_enabled else "none"
        self.hv_auxiliary_enabled = bool(hv_auxiliary)

        if decoder_variant not in {"current", "fpn", "unet", "multi_layer_fpn"}:
            raise ValueError(f"Unsupported decoder_variant={decoder_variant!r}")
        if decoder_variant == "fpn":
            self.variant_fusion = _BucketFPNFusion(tap_dims, self.buckets, embed_proj)
            self.fuse = None
            self.weight_logits = None
        elif decoder_variant == "unet":
            self.variant_fusion = _UNetScaleFusion(tap_dims, self.buckets, embed_proj)
            self.fuse = None
            self.weight_logits = None
        elif decoder_variant == "multi_layer_fpn":
            self.variant_fusion = _MultiLayerFPNFusion(tap_dims, self.buckets, embed_proj)
            self.fuse = None
            self.weight_logits = None
        elif fusion_mode == "bucket_concat":
            # One 1×1 fusion conv per bucket: concat taps in the bucket → embed_proj.
            self.fuse = nn.ModuleList(
                nn.Conv2d(sum(tap_dims[i] for i in bucket), embed_proj, kernel_size=1)
                for bucket in self.buckets
            )
            self.weight_logits = None
        elif fusion_mode == "weighted_sum":
            # Project each tap independently, then learn a softmax-weighted sum
            # inside each skip bucket. This keeps branch input width fixed.
            self.fuse = nn.ModuleList(
                nn.ModuleList(
                    nn.Conv2d(tap_dims[i], embed_proj, kernel_size=1)
                    for i in bucket
                )
                for bucket in self.buckets
            )
            self.weight_logits = nn.ParameterList(
                nn.Parameter(torch.zeros(len(bucket))) for bucket in self.buckets
            )
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")

        self.spatial_adapter = (
            CNNSpatialAdapter(feature_size, embed_proj, width=spatial_adapter_width, image_ch=image_ch)
            if self.spatial_adapter_enabled
            else None
        )

        self.np_branch = _UNETRBranch(embed_proj, feature_size, out_ch=2, image_ch=image_ch)
        self.hv_branch = _UNETRBranch(embed_proj, feature_size, out_ch=2, image_ch=image_ch)
        # This is deliberately independent from the historical HV branch.  The
        # primary HV prediction remains the only input to instance postprocess.
        self.hv_aux_branch = (
            _UNETRBranch(embed_proj, feature_size, out_ch=2, image_ch=image_ch)
            if self.hv_auxiliary_enabled
            else None
        )
        self.tp_branch = (
            _UNETRBranch(embed_proj, feature_size, out_ch=num_types, image_ch=image_ch)
            if num_types and num_types > 0
            else None
        )

    def forward(self, image: torch.Tensor, taps: List[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        local_features = self.spatial_adapter(image) if self.spatial_adapter is not None else None
        zs: List[torch.Tensor] = []
        if self.decoder_variant != "current":
            zs = self.variant_fusion(taps, self.buckets)
        elif self.fusion_mode == "bucket_concat":
            for bucket, conv in zip(self.buckets, self.fuse):
                feat = taps[bucket[0]] if len(bucket) == 1 else torch.cat([taps[i] for i in bucket], dim=1)
                zs.append(conv(feat))
        else:
            assert self.weight_logits is not None
            for bucket, projections, logits in zip(self.buckets, self.fuse, self.weight_logits):
                projected = [proj(taps[i]) for i, proj in zip(bucket, projections)]
                weights = torch.softmax(logits.float(), dim=0).to(projected[0].dtype)
                fused = sum(w * feat for w, feat in zip(weights, projected))
                zs.append(fused)
        z0, z1, z2, z3 = zs
        if local_features is not None:
            z3 = z3 + local_features["sixteenth"]

        out: Dict[str, Optional[torch.Tensor]] = {
            "np": self.np_branch(image, z0, z1, z2, z3, local_features),
            "hv": self.hv_branch(image, z0, z1, z2, z3, local_features),   # raw regression (no activation)
            "hv_aux": None,
            "tp": None,
        }
        if self.hv_aux_branch is not None:
            out["hv_aux"] = self.hv_aux_branch(image, z0, z1, z2, z3, local_features)
        if self.tp_branch is not None:
            out["tp"] = self.tp_branch(image, z0, z1, z2, z3, local_features)
        return out
