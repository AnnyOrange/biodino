# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class PatchEmbedPerChannel(nn.Module):
    """
    ChannelViT-style patch embedding: treats multi-channel images as volumetric data.
    
    This module processes each channel independently using Conv3d, allowing for
    channel-specific embeddings while maintaining spatial relationships.
    
    Args:
        img_size: Image size (assumed square).
        patch_size: Patch token size (assumed square).
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        flatten_embedding: Whether to flatten the output embedding.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten_embedding: bool = False,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * in_chans
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        # ChannelViT uses Conv3d: (1, embed_dim, kernel=(1, P, P))
        # Input: (B, 1, C, H, W) -> Output: (B, embed_dim, C, H', W')
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_HW[0], patch_HW[1]),
            stride=(1, patch_HW[0], patch_HW[1]),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ChannelViT patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            If flatten_embedding=False: (B, embed_dim, C, H', W')
            If flatten_embedding=True: (B, C*H'*W', embed_dim)
        """
        B, C, H, W = x.shape
        
        # Add channel dimension for Conv3d: (B, C, H, W) -> (B, 1, C, H, W)
        x = x.unsqueeze(1)
        
        # Apply Conv3d: (B, 1, C, H, W) -> (B, embed_dim, C, H', W')
        x = self.proj(x)
        
        if self.flatten_embedding:
            # Flatten order: Channel -> H -> W (ChannelViT key ordering)
            # (B, embed_dim, C, H', W') -> (B, embed_dim, C*H'*W')
            x = x.flatten(2)
            # Transpose to (B, C*H'*W', embed_dim)
            x = x.transpose(1, 2)
        
        return x

    def reset_parameters(self):
        """Initialize Conv3d weights using Kaiming uniform initialization."""
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
