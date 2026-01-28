# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
LoRA (Low-Rank Adaptation) module for DINOv3.

LoRA allows efficient fine-tuning by injecting trainable low-rank matrices into 
the model while keeping the original weights frozen.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA layer that wraps a Linear layer with low-rank adaptation.
    
    The forward pass computes: output = W @ x + (B @ A) @ x * scaling
    where W is the frozen original weight, and A, B are trainable low-rank matrices.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Args:
            original_layer: The original nn.Linear layer to wrap
            r: Rank of the low-rank matrices
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout probability for LoRA layers
            merge_weights: Whether to merge weights during inference
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        # Dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA matrices: A with Kaiming, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into the original layer for faster inference."""
        if not self.merged:
            self.original_layer.weight.data += (
                self.lora_B @ self.lora_A
            ).to(self.original_layer.weight.dtype) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from the original layer."""
        if self.merged:
            self.original_layer.weight.data -= (
                self.lora_B @ self.lora_A
            ).to(self.original_layer.weight.dtype) * self.scaling
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        
        # Original forward
        result = self.original_layer(x)
        
        # LoRA forward: (B @ A) @ x * scaling
        lora_output = self.lora_dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # [*, r]
        lora_output = F.linear(lora_output, self.lora_B)  # [*, out_features]
        
        return result + lora_output * self.scaling
    
    @property
    def weight(self):
        """Return the weight for compatibility."""
        return self.original_layer.weight
    
    @property
    def bias(self):
        """Return the bias for compatibility."""
        return self.original_layer.bias
    
    @property
    def in_features(self):
        return self.original_layer.in_features
    
    @property
    def out_features(self):
        return self.original_layer.out_features


class LoRAQKVLinear(nn.Module):
    """
    LoRA layer specifically designed for QKV projection in attention.
    
    This applies LoRA separately to Q, K, V projections which share a single weight matrix.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        enable_lora_q: bool = True,
        enable_lora_k: bool = True,
        enable_lora_v: bool = True,
    ):
        """
        Args:
            original_layer: The original QKV nn.Linear layer
            r: Rank of the low-rank matrices
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout probability
            enable_lora_q: Whether to apply LoRA to Q
            enable_lora_k: Whether to apply LoRA to K
            enable_lora_v: Whether to apply LoRA to V
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        assert out_features % 3 == 0, "QKV layer output must be divisible by 3"
        self.qkv_dim = out_features // 3
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA parameters for Q, K, V separately
        self.enable_lora_q = enable_lora_q
        self.enable_lora_k = enable_lora_k
        self.enable_lora_v = enable_lora_v
        
        if enable_lora_q:
            self.lora_A_q = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_q = nn.Parameter(torch.empty(self.qkv_dim, r))
        
        if enable_lora_k:
            self.lora_A_k = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_k = nn.Parameter(torch.empty(self.qkv_dim, r))
        
        if enable_lora_v:
            self.lora_A_v = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_v = nn.Parameter(torch.empty(self.qkv_dim, r))
        
        # Dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA matrices."""
        if self.enable_lora_q:
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
        if self.enable_lora_k:
            nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k)
        if self.enable_lora_v:
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_layer(x)
        
        # Split result into Q, K, V
        q, k, v = result.chunk(3, dim=-1)
        
        # Apply LoRA
        x_dropped = self.lora_dropout(x)
        
        if self.enable_lora_q:
            lora_q = F.linear(F.linear(x_dropped, self.lora_A_q), self.lora_B_q)
            q = q + lora_q * self.scaling
        
        if self.enable_lora_k:
            lora_k = F.linear(F.linear(x_dropped, self.lora_A_k), self.lora_B_k)
            k = k + lora_k * self.scaling
        
        if self.enable_lora_v:
            lora_v = F.linear(F.linear(x_dropped, self.lora_A_v), self.lora_B_v)
            v = v + lora_v * self.scaling
        
        return torch.cat([q, k, v], dim=-1)
    
    @property
    def weight(self):
        return self.original_layer.weight
    
    @property
    def bias(self):
        return self.original_layer.bias
    
    @property
    def in_features(self):
        return self.original_layer.in_features
    
    @property
    def out_features(self):
        return self.original_layer.out_features


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Mark only LoRA parameters as trainable in the model.
    
    Args:
        model: The model containing LoRA layers
        bias: How to handle bias parameters:
            - "none": No bias is trainable
            - "all": All biases are trainable
            - "lora_only": Only biases in LoRA layers are trainable
    """
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, (LoRALinear, LoRAQKVLinear)) and m.original_layer.bias is not None:
                m.original_layer.bias.requires_grad = True


def lora_state_dict(model: nn.Module, bias: str = "none") -> dict:
    """
    Return only the LoRA parameters state dict.
    
    Args:
        model: The model containing LoRA layers
        bias: How to handle bias parameters (same as mark_only_lora_as_trainable)
    
    Returns:
        State dict containing only LoRA parameters
    """
    state_dict = model.state_dict()
    
    if bias == "none":
        return {k: v for k, v in state_dict.items() if "lora_" in k}
    elif bias == "all":
        return {k: v for k, v in state_dict.items() if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # This requires more complex logic to identify biases in LoRA layers
        lora_keys = {k for k in state_dict.keys() if "lora_" in k}
        # Include biases that are adjacent to lora parameters
        bias_keys = set()
        for k in lora_keys:
            # Try to find associated bias
            parts = k.rsplit(".", 1)
            if len(parts) > 1:
                bias_key = parts[0].replace("lora_A", "original_layer").replace("lora_B", "original_layer") + ".bias"
                if bias_key in state_dict:
                    bias_keys.add(bias_key)
        return {k: v for k, v in state_dict.items() if k in lora_keys or k in bias_keys}
    
    return {k: v for k, v in state_dict.items() if "lora_" in k}

