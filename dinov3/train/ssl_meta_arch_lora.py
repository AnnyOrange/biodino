# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
SSL Meta Architecture with LoRA support for efficient fine-tuning.

This module extends SSLMetaArch to support LoRA (Low-Rank Adaptation) training,
allowing efficient fine-tuning on custom datasets while keeping most parameters frozen.
"""

import logging
from typing import Optional

import torch
from torch import nn

from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.utils.lora_utils import (
    apply_lora_to_vit_backbone,
    freeze_non_lora_params,
    get_trainable_params_info,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
from dinov3.train.param_groups import get_params_groups_with_decay_fsdp, fuse_params_groups

logger = logging.getLogger("dinov3")


class SSLMetaArchLoRA(SSLMetaArch):
    """
    SSL Meta Architecture with LoRA support.
    
    This class extends SSLMetaArch to enable LoRA fine-tuning:
    - Applies LoRA to student and teacher backbones
    - Freezes non-LoRA parameters
    - Provides methods to save/load LoRA checkpoints
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # LoRA configuration from cfg
        self.lora_cfg = cfg.get("lora", {})
        self.lora_r = self.lora_cfg.get("r", 8)
        self.lora_alpha = self.lora_cfg.get("alpha", 16.0)
        self.lora_dropout = self.lora_cfg.get("dropout", 0.0)
        self.lora_target_modules = self.lora_cfg.get("target_modules", None)
        self.lora_enable_q = self.lora_cfg.get("enable_q", True)
        self.lora_enable_k = self.lora_cfg.get("enable_k", True)
        self.lora_enable_v = self.lora_cfg.get("enable_v", True)
        self.lora_train_bias = self.lora_cfg.get("train_bias", False)
        self.lora_train_heads = self.lora_cfg.get("train_heads", True)  # Whether to train DINO/IBOT heads
        
        self._lora_applied = False
    
    def _apply_lora_to_backbones(self):
        """Apply LoRA to student backbone."""
        if self._lora_applied:
            logger.warning("LoRA already applied, skipping...")
            return
        
        logger.info(f"Applying LoRA with r={self.lora_r}, alpha={self.lora_alpha}")
        
        # Check if ChannelViT is enabled
        enable_channelvit = getattr(self.student.backbone, 'enable_channelvit', False)
        if enable_channelvit:
            logger.info("ChannelViT is enabled - will keep channel_embed trainable")
        
        # Apply LoRA to student backbone
        apply_lora_to_vit_backbone(
            self.student.backbone,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            enable_lora_q=self.lora_enable_q,
            enable_lora_k=self.lora_enable_k,
            enable_lora_v=self.lora_enable_v,
        )
        
        # Freeze non-LoRA parameters in backbone
        freeze_non_lora_params(self.student.backbone, train_bias=self.lora_train_bias)
        
        # If ChannelViT is enabled, ensure channel_embed is trainable
        if enable_channelvit and hasattr(self.student.backbone, 'channel_embed'):
            if self.student.backbone.channel_embed is not None:
                self.student.backbone.channel_embed.requires_grad = True
                logger.info("ChannelViT channel_embed is set to trainable")
        
        # Optionally train the heads
        if self.lora_train_heads:
            for param in self.student.dino_head.parameters():
                param.requires_grad = True
            for param in self.student.ibot_head.parameters():
                param.requires_grad = True
            logger.info("DINO and IBOT heads are trainable")
        else:
            for param in self.student.dino_head.parameters():
                param.requires_grad = False
            for param in self.student.ibot_head.parameters():
                param.requires_grad = False
            logger.info("DINO and IBOT heads are frozen")
        
        self._lora_applied = True
        
        # Log trainable parameters info
        info = get_trainable_params_info(self.student)
        logger.info(f"Total parameters: {info['total_params']:,}")
        logger.info(f"Trainable parameters: {info['trainable_params']:,}")
        logger.info(f"Trainable ratio: {info['trainable_ratio']:.4%}")
    
    def init_weights(self) -> None:
        """Initialize weights and apply LoRA."""
        super().init_weights()
        
        # Apply LoRA after loading pretrained weights
        self._apply_lora_to_backbones()
    
    def get_params_groups(self):
        """
        Get parameter groups for optimizer.
        
        For LoRA training, we only include trainable parameters.
        """
        all_params_groups = []
        
        for name, m in self.student.items():
            logger.info(f"Getting parameter groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        
        return all_params_groups
    
    def save_lora_weights(self, path: str, extra_state: Optional[dict] = None):
        """Save LoRA weights to a checkpoint file."""
        save_lora_checkpoint(
            self.student.backbone,
            path,
            bias="all" if self.lora_train_bias else "none",
            extra_state=extra_state,
        )
    
    def load_lora_weights(self, path: str, strict: bool = False):
        """Load LoRA weights from a checkpoint file."""
        load_lora_checkpoint(
            self.student.backbone,
            path,
            strict=strict,
        )
        # Sync to teacher
        self.model_ema.load_state_dict(self.student.state_dict())


def get_lora_params_groups_with_decay_fsdp(
    model,
    lr_decay_rate=1.0,
    patch_embed_lr_mult=1.0,
    dino_head_wd_multiplier=1.0,
    lora_lr_multiplier=1.0,
):
    """
    Get parameter groups with special handling for LoRA parameters.
    
    Args:
        model: The model with LoRA layers
        lr_decay_rate: Base learning rate decay rate
        patch_embed_lr_mult: Learning rate multiplier for patch embedding
        dino_head_wd_multiplier: Weight decay multiplier for DINO head
        lora_lr_multiplier: Learning rate multiplier specifically for LoRA parameters
    """
    base_groups = get_params_groups_with_decay_fsdp(
        model,
        lr_decay_rate=lr_decay_rate,
        patch_embed_lr_mult=patch_embed_lr_mult,
        dino_head_wd_multiplier=dino_head_wd_multiplier,
    )
    
    # Adjust LoRA parameters
    for group in base_groups:
        if "lora_" in group.get("name", ""):
            group["lr_multiplier"] *= lora_lr_multiplier
            # LoRA parameters typically don't need weight decay
            group["wd_multiplier"] = 0.0
    
    return base_groups

