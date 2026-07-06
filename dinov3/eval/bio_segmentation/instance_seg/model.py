"""
DINOHoVerNet: DINOv3 backbone (frozen or fine-tuned) + HoVerNet decoder.

The backbone is the *only* thing that changes between comparison rows
(bio-DINOv3 vs generic DINOv3 vs other FMs); the decoder, data, and metrics are
held fixed so a score delta is attributable to the backbone.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .decoder import HoVerNetDecoder

logger = logging.getLogger("bio_seg.instance_seg.model")


class DINOHoVerNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        layers: List[int],
        num_types: int = 0,
        freeze_backbone: bool = True,
        feature_size: int = 32,
        embed_proj: int = 384,
    ):
        super().__init__()
        self.backbone = backbone
        # Always tap in ascending (shallow→deep) order; get_intermediate_layers
        # returns features in block order regardless, so keep the list sorted.
        self.layers = sorted(int(i) for i in layers)
        self.freeze_backbone = freeze_backbone
        self.num_types = num_types

        embed_dim = int(backbone.embed_dim)
        patch_size = int(backbone.patch_size)
        tap_dims = [embed_dim] * len(self.layers)

        self.decoder = HoVerNetDecoder(
            tap_dims=tap_dims,
            num_types=num_types,
            feature_size=feature_size,
            embed_proj=embed_proj,
            image_ch=3,
            patch_size=patch_size,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # Cache backbone compute dtype (vit_7b is loaded in bf16 when frozen).
        try:
            self._bb_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            self._bb_dtype = torch.float32

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep a frozen backbone in eval mode regardless of the module's mode.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone.get_intermediate_layers(
            x.to(self._bb_dtype),
            n=self.layers,
            reshape=True,
            return_class_token=False,
        )
        return [f.float() for f in feats]

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.freeze_backbone:
            with torch.no_grad():
                taps = self._extract(x)
        else:
            taps = self._extract(x)
        return self.decoder(x.float(), taps)

    def trainable_parameters(self):
        if self.freeze_backbone:
            return self.decoder.parameters()
        return self.parameters()


def build_dino_hovernet(
    checkpoint: str,
    train_config: str,
    layers: List[int],
    num_types: int = 0,
    freeze_backbone: bool = True,
    feature_size: int = 32,
    embed_proj: int = 384,
    device: torch.device = torch.device("cuda"),
) -> DINOHoVerNet:
    """Load a DINOv3 backbone and wrap it with the HoVerNet decoder."""
    from ..model_utils import load_dinov3_backbone

    backbone = load_dinov3_backbone(
        checkpoint, train_config_path=train_config, device=device, freeze=freeze_backbone
    )
    model = DINOHoVerNet(
        backbone,
        layers=layers,
        num_types=num_types,
        freeze_backbone=freeze_backbone,
        feature_size=feature_size,
        embed_proj=embed_proj,
    ).to(device)
    logger.info(
        "DINOHoVerNet ready: layers=%s num_types=%s freeze=%s feature_size=%s embed_proj=%s",
        model.layers, num_types, freeze_backbone, feature_size, embed_proj,
    )
    return model
