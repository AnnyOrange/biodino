# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Utility functions for applying LoRA to DINOv3 models.
"""

import logging
import re
from typing import List, Optional, Set, Union

import torch
import torch.nn as nn

from dinov3.layers.lora import LoRALinear, LoRAQKVLinear, mark_only_lora_as_trainable, lora_state_dict

logger = logging.getLogger("dinov3")


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    enable_lora_q: bool = True,
    enable_lora_k: bool = True,
    enable_lora_v: bool = True,
) -> nn.Module:
    """
    Apply LoRA to specified modules in the model.
    
    Args:
        model: The model to apply LoRA to
        r: Rank of LoRA matrices
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate for LoRA
        target_modules: List of module name patterns to apply LoRA to.
                       Supports regex patterns. If None, defaults to ["qkv", "proj"]
        exclude_modules: List of module name patterns to exclude
        enable_lora_q: Apply LoRA to Q in QKV layers
        enable_lora_k: Apply LoRA to K in QKV layers
        enable_lora_v: Apply LoRA to V in QKV layers
    
    Returns:
        The modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    # Compile patterns
    target_patterns = [re.compile(p) for p in target_modules]
    exclude_patterns = [re.compile(p) for p in exclude_modules]
    
    def should_apply_lora(name: str) -> bool:
        """Check if LoRA should be applied to this module."""
        # Check exclusions first
        for pattern in exclude_patterns:
            if pattern.search(name):
                return False
        # Check if matches target
        for pattern in target_patterns:
            if pattern.search(name):
                return True
        return False
    
    def is_qkv_layer(name: str) -> bool:
        """Check if this is a QKV projection layer."""
        return "qkv" in name.lower()
    
    # Find and replace target modules
    replaced_count = 0
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            modules_to_replace.append((name, module))
    
    for name, module in modules_to_replace:
        # Navigate to parent module
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr_name = name
        else:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        
        # Create LoRA wrapper
        if is_qkv_layer(name):
            lora_layer = LoRAQKVLinear(
                original_layer=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora_q=enable_lora_q,
                enable_lora_k=enable_lora_k,
                enable_lora_v=enable_lora_v,
            )
        else:
            lora_layer = LoRALinear(
                original_layer=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        
        # Replace module
        setattr(parent, attr_name, lora_layer)
        replaced_count += 1
        logger.info(f"Applied LoRA to: {name}")
    
    logger.info(f"Total LoRA layers added: {replaced_count}")
    
    return model


def apply_lora_to_vit_backbone(
    backbone: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    enable_lora_q: bool = True,
    enable_lora_k: bool = True,
    enable_lora_v: bool = True,
) -> nn.Module:
    """
    Apply LoRA specifically to ViT backbone in DINOv3.
    
    This targets:
    - QKV projections in self-attention (blocks.*.attn.qkv)
    - Output projections in self-attention (blocks.*.attn.proj)
    - MLP layers (blocks.*.mlp.fc1, blocks.*.mlp.fc2 or blocks.*.mlp.w1, blocks.*.mlp.w2, blocks.*.mlp.w3)
    
    Args:
        backbone: The ViT backbone model
        r: Rank of LoRA matrices
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate
        target_modules: Custom list of target modules. If None, uses default ViT targets
        enable_lora_q/k/v: Control which parts of QKV get LoRA
    
    Returns:
        Modified backbone with LoRA
    """
    if target_modules is None:
        # Default targets for ViT
        target_modules = [
            r"blocks\.\d+\.attn\.qkv",      # QKV projection
            r"blocks\.\d+\.attn\.proj",     # Output projection
            # For SwiGLU FFN
            r"blocks\.\d+\.mlp\.w1",
            r"blocks\.\d+\.mlp\.w2", 
            r"blocks\.\d+\.mlp\.w3",
            # For standard MLP
            r"blocks\.\d+\.mlp\.fc1",
            r"blocks\.\d+\.mlp\.fc2",
        ]
    
    return apply_lora_to_model(
        backbone,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        enable_lora_q=enable_lora_q,
        enable_lora_k=enable_lora_k,
        enable_lora_v=enable_lora_v,
    )


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from the model."""
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            params.append(param)
    return params


def get_trainable_params_info(model: nn.Module) -> dict:
    """
    Get information about trainable parameters in the model.
    
    Returns:
        Dict with 'total_params', 'trainable_params', 'trainable_ratio'
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "frozen_params": total_params - trainable_params,
    }


def freeze_non_lora_params(model: nn.Module, train_bias: bool = False) -> None:
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: The model with LoRA layers
        train_bias: Whether to keep bias parameters trainable
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif train_bias and "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA weights into original layers for faster inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerge all LoRA weights from original layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    bias: str = "none",
    extra_state: Optional[dict] = None,
) -> None:
    """
    Save only the LoRA parameters to a checkpoint.
    
    Args:
        model: The model with LoRA layers
        path: Path to save the checkpoint
        bias: How to handle biases ("none", "all", "lora_only")
        extra_state: Additional state to save (e.g., optimizer state)
    """
    state = {
        "lora_state_dict": lora_state_dict(model, bias=bias),
    }
    if extra_state:
        state.update(extra_state)
    
    torch.save(state, path)
    logger.info(f"Saved LoRA checkpoint to {path}")


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = False,
) -> dict:
    """
    Load LoRA parameters from a checkpoint.
    
    Args:
        model: The model with LoRA layers
        path: Path to the checkpoint
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        The loaded checkpoint dict
    """
    checkpoint = torch.load(path, map_location="cpu")
    
    if "lora_state_dict" in checkpoint:
        lora_state = checkpoint["lora_state_dict"]
    else:
        # Assume the whole checkpoint is LoRA state
        lora_state = checkpoint
    
    # Load LoRA parameters
    missing_keys, unexpected_keys = [], []
    model_state = model.state_dict()
    
    for k, v in lora_state.items():
        if k in model_state:
            model_state[k] = v
        else:
            unexpected_keys.append(k)
    
    for k in model_state:
        if "lora_" in k and k not in lora_state:
            missing_keys.append(k)
    
    model.load_state_dict(model_state, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing LoRA keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected LoRA keys: {unexpected_keys}")
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Strict loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}")
    
    logger.info(f"Loaded LoRA checkpoint from {path}")
    
    return checkpoint

