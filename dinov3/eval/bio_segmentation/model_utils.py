"""
DINOv3 backbone loading for bio-image evaluation — uses the same local ViT
definition as training (`build_model_from_cfg` / `build_model_for_eval`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dinov3.configs import get_default_config
from dinov3.models import build_model_for_eval

logger = logging.getLogger(__name__)


def _peek_consolidated_checkpoint_key(path: Path) -> str | None:
    """Return which top-level key to use for init_model_from_checkpoint_for_evals, or None for a flat state dict."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        return None
    if "teacher" in sd:
        return "teacher"
    if "model" in sd:
        return "model"
    if "state_dict" in sd:
        return "state_dict"
    return None


def load_dinov3_backbone(
    checkpoint_path: str,
    train_config_path: str,
    device: torch.device = torch.device("cuda"),
    freeze: bool = True,
):
    """
    Load the teacher backbone with the same architecture as training.

    ``checkpoint_path`` may be:
      - a **DCP checkpoint directory** (e.g. ``.../ckpt/1024``), or
      - a **consolidated** ``.pth`` (``teacher`` / ``model`` / ``state_dict`` / flat state dict).

    ``train_config_path`` is merged on top of ``ssl_default_config`` and must match
    the training run (especially ``student.*`` used by ``build_model_from_cfg``).
    """
    default_cfg = get_default_config()
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(train_config_path))

    ck = Path(checkpoint_path)
    consolidated_key: str | None = "teacher"
    if ck.is_file():
        consolidated_key = _peek_consolidated_checkpoint_key(ck)

    logger.info(
        "Loading train-compatible backbone (config=%s, ckpt=%s, consolidated_key=%s)",
        train_config_path,
        checkpoint_path,
        consolidated_key,
    )
    model = build_model_for_eval(
        cfg,
        pretrained_weights=checkpoint_path,
        shard_unsharded_model=False,
        consolidated_checkpoint_key=consolidated_key,
    )

    model = model.to(device)
    if freeze and str(getattr(cfg.student, "arch", "")) == "vit_7b":
        model = model.to(dtype=torch.bfloat16)

    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    logger.info("Backbone ready: embed_dim=%s, patch_size=%s", model.embed_dim, model.patch_size)
    return model
