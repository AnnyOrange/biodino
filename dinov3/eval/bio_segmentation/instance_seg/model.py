"""
DINOHoVerNet: DINOv3 backbone (frozen or fine-tuned) + HoVerNet decoder.

The backbone is the *only* thing that changes between comparison rows
(bio-DINOv3 vs generic DINOv3 vs other FMs); the decoder, data, and metrics are
held fixed so a score delta is attributable to the backbone.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

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
        trainable_backbone_blocks: Optional[int] = None,
        feature_size: int = 32,
        embed_proj: int = 384,
    ):
        super().__init__()
        self.backbone = backbone
        # Always tap in ascending (shallow→deep) order; get_intermediate_layers
        # returns features in block order regardless, so keep the list sorted.
        self.layers = sorted(int(i) for i in layers)
        self.freeze_backbone = freeze_backbone
        self.trainable_backbone_blocks = trainable_backbone_blocks
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

        self._configure_backbone_trainability()

        # Cache backbone compute dtype (vit_7b is loaded in bf16 when frozen).
        try:
            self._bb_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            self._bb_dtype = torch.float32

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen prefixes stay deterministic while the selected tail blocks train.
        if self.freeze_backbone:
            self.backbone.eval()
        elif self.trainable_backbone_blocks is not None:
            self.backbone.eval()
            for block in self.backbone.blocks[-self.trainable_backbone_blocks :]:
                block.train(mode)
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                norm.train(mode)
        return self

    def _configure_backbone_trainability(self) -> None:
        if self.freeze_backbone and self.trainable_backbone_blocks is not None:
            raise ValueError("trainable_backbone_blocks requires freeze_backbone=False")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()
            return

        if self.trainable_backbone_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad_(True)
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError("Partial fine-tuning requires backbone.blocks")
        n_blocks = int(self.trainable_backbone_blocks)
        if not 1 <= n_blocks <= len(blocks):
            raise ValueError(f"trainable_backbone_blocks must be in [1, {len(blocks)}], got {n_blocks}")

        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad_(True)
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for param in norm.parameters():
                param.requires_grad_(True)

    @property
    def backbone_mode(self) -> str:
        if self.freeze_backbone:
            return "frozen"
        if self.trainable_backbone_blocks is not None:
            return f"last{self.trainable_backbone_blocks}"
        return "finetune"

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
        return (param for param in self.parameters() if param.requires_grad)

    def trainable_backbone_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.backbone.parameters() if param.requires_grad)


def build_dino_hovernet(
    checkpoint: str,
    train_config: str,
    layers: List[int],
    num_types: int = 0,
    freeze_backbone: bool = True,
    trainable_backbone_blocks: Optional[int] = None,
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
        trainable_backbone_blocks=trainable_backbone_blocks,
        feature_size=feature_size,
        embed_proj=embed_proj,
    ).to(device)
    logger.info(
        "DINOHoVerNet ready: layers=%s num_types=%s backbone_mode=%s feature_size=%s embed_proj=%s",
        model.layers, num_types, model.backbone_mode, feature_size, embed_proj,
    )
    return model
