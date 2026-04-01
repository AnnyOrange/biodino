"""
DINOv3 backbone loading utilities for bio-image evaluation.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_dinov3_backbone(
    checkpoint_path: str,
    model_size: str = 'l',
    device: torch.device = torch.device('cuda'),
    freeze: bool = True,
) -> nn.Module:
    """
    Load a pretrained DINOv3 backbone from a checkpoint file.

    Handles checkpoints saved with various wrapper prefixes
    ('module.', 'backbone.', 'model', 'state_dict').

    Args:
        checkpoint_path: path to the .pth checkpoint file.
        model_size: 'l' (ViT-L/16) or '7b' (ViT-7B/16).
        device: target device.
        freeze: if True, freeze all backbone parameters.

    Returns:
        Loaded and frozen backbone in eval mode.
    """
    from dinov3.hub.backbones import dinov3_vitl16, dinov3_vit7b16

    logger.info(f"Loading DINOv3 {model_size.upper()} backbone...")
    if model_size.lower() == 'l':
        model = dinov3_vitl16(pretrained=False)
    elif model_size.lower() == '7b':
        model = dinov3_vit7b16(pretrained=False)
    else:
        raise ValueError(f"Unsupported model size: {model_size}. Choose 'l' or '7b'.")

    logger.info(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    # MARK(lxy): keep the frozen 7B backbone in bf16 on GPU to reduce VRAM.
    if freeze and model_size.lower() == '7b':
        model = model.to(dtype=torch.bfloat16)
    model.eval()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    logger.info(f"Backbone ready: embed_dim={model.embed_dim}, patch_size={model.patch_size}")
    return model
