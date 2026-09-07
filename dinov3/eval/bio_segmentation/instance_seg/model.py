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
        lora_rank: Optional[int] = None,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        adapter: bool = False,
        adapter_dim: int = 128,
        feature_size: int = 32,
        embed_proj: int = 384,
        fusion_mode: str = "bucket_concat",
        decoder_variant: str = "current",
        spatial_adapter: bool = False,
        spatial_adapter_width: int = 32,
        hv_auxiliary: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        # Always tap in ascending (shallow→deep) order; get_intermediate_layers
        # returns features in block order regardless, so keep the list sorted.
        self.layers = sorted(int(i) for i in layers)
        self.freeze_backbone = freeze_backbone
        self.trainable_backbone_blocks = trainable_backbone_blocks
        self.lora_rank = lora_rank
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.adapter_enabled = bool(adapter)
        self.adapter_dim = int(adapter_dim)
        self.num_types = num_types

        if self.lora_rank is not None:
            if not self.freeze_backbone:
                raise ValueError("LoRA mode expects freeze_backbone=True; only LoRA parameters are trainable.")
            from dinov3.utils.lora_utils import apply_lora_to_vit_backbone

            self.backbone = apply_lora_to_vit_backbone(
                self.backbone,
                r=int(self.lora_rank),
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=[
                    r"blocks\.\d+\.attn\.qkv",
                    r"blocks\.\d+\.attn\.proj",
                ],
            )

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
            fusion_mode=fusion_mode,
            decoder_variant=decoder_variant,
            spatial_adapter=spatial_adapter,
            spatial_adapter_width=spatial_adapter_width,
            hv_auxiliary=hv_auxiliary,
        )
        if self.adapter_enabled:
            self.feature_adapters = nn.ModuleList([
                nn.Sequential(nn.Conv2d(embed_dim, self.adapter_dim, 1), nn.GELU(), nn.Conv2d(self.adapter_dim, embed_dim, 1))
                for _ in self.layers
            ])
            for module in self.feature_adapters:
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)
        else:
            self.feature_adapters = nn.ModuleList()

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
            if self.lora_rank is not None:
                for module in self.backbone.modules():
                    if any("lora_" in name for name, _ in module.named_parameters(recurse=False)):
                        module.train(mode)
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
            if self.lora_rank is not None:
                for name, param in self.backbone.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
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
        if self.lora_rank is not None:
            return f"lora{self.lora_rank}"
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
        feats = [f.float() for f in feats]
        if self.adapter_enabled:
            feats = [f + adapter(f) for f, adapter in zip(feats, self.feature_adapters)]
        return feats

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.freeze_backbone and self.lora_rank is None and not self.adapter_enabled:
            with torch.no_grad():
                taps = self._extract(x)
        else:
            taps = self._extract(x)
        return self.decoder(x.float(), taps)

    def trainable_parameters(self):
        if self.freeze_backbone:
            if self.adapter_enabled:
                return (param for param in self.parameters() if param.requires_grad)
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
    lora_rank: Optional[int] = None,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    adapter: bool = False,
    adapter_dim: int = 128,
    feature_size: int = 32,
    embed_proj: int = 384,
    fusion_mode: str = "bucket_concat",
    decoder_variant: str = "current",
    spatial_adapter: bool = False,
    spatial_adapter_width: int = 32,
    hv_auxiliary: bool = False,
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
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter=adapter,
        adapter_dim=adapter_dim,
        feature_size=feature_size,
        embed_proj=embed_proj,
        fusion_mode=fusion_mode,
        decoder_variant=decoder_variant,
        spatial_adapter=spatial_adapter,
        spatial_adapter_width=spatial_adapter_width,
        hv_auxiliary=hv_auxiliary,
    ).to(device)
    logger.info(
        "DINOHoVerNet ready: layers=%s num_types=%s backbone_mode=%s feature_size=%s embed_proj=%s fusion_mode=%s decoder_variant=%s spatial_adapter=%s hv_auxiliary=%s trainable_params=%d trainable_backbone_params=%d",
        model.layers,
        num_types,
        model.backbone_mode,
        feature_size,
        embed_proj,
        fusion_mode,
        decoder_variant,
        spatial_adapter,
        hv_auxiliary,
        sum(param.numel() for param in model.parameters() if param.requires_grad),
        sum(param.numel() for param in model.trainable_backbone_parameters()),
    )
    return model
