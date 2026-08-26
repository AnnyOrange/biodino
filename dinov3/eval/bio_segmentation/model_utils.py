"""
DINOv3 backbone loading for bio-image evaluation — uses the same local ViT
definition as training (`build_model_from_cfg` / `build_model_for_eval`).
"""

from __future__ import annotations

import logging
import math
import pickle
from functools import partial
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from dinov3.configs import get_default_config
from dinov3.models import build_model_for_eval, build_model_from_cfg

logger = logging.getLogger(__name__)


class _ChAdaTokenLearner(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _ChAdaTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if return_attention:
            x = self.norm1(src) if self.norm_first else src
            _, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            return attn_weights
        return super().forward(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )


class _EvalChAdaViT(nn.Module):
    """Minimal ChAda-ViT backbone wrapper with DINO-style feature access."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 2,
        max_number_channels: int = 10,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_channels = max_number_channels
        self.enable_channelvit = True

        self.token_learner = _ChAdaTokenLearner(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.token_learner.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_token = nn.Parameter(torch.zeros(1, self.max_channels, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList(
            [
                _ChAdaTransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=2048,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

    def _patch_pos_embed(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        num_patches = x.shape[2]
        base_patches = self.pos_embed.shape[2] - 1
        if num_patches == base_patches and width == height:
            return self.pos_embed[:, :, 1:]

        dim = x.shape[-1]
        grid = int(math.sqrt(base_patches))
        width0 = width // self.patch_size + 0.1
        height0 = height // self.patch_size + 0.1
        patch_pos = F.interpolate(
            self.pos_embed[:, :, 1:].reshape(1, grid, grid, dim).permute(0, 3, 1, 2),
            scale_factor=(width0 / grid, height0 / grid),
            mode="bicubic",
        )
        if int(width0) != patch_pos.shape[-2] or int(height0) != patch_pos.shape[-1]:
            raise RuntimeError("ChAda-ViT positional embedding interpolation failed.")
        return patch_pos.permute(0, 2, 3, 1).view(1, 1, -1, dim)

    def _tokenize(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        batch_size, channels, width, height = imgs.shape
        if channels > self.max_channels:
            raise ValueError(
                f"ChAda-ViT checkpoint supports at most {self.max_channels} channels, got {channels}."
            )

        flat = imgs.reshape(batch_size * channels, 1, width, height)
        tokens_per_channel = self.token_learner(flat)
        spatial_tokens = tokens_per_channel.shape[1]
        tokens = tokens_per_channel.reshape(batch_size, channels, spatial_tokens, self.embed_dim)
        channel_mask = torch.zeros(
            batch_size,
            channels,
            spatial_tokens,
            dtype=torch.bool,
            device=imgs.device,
        )

        tokens = tokens + self._patch_pos_embed(tokens, width, height)
        tokens = tokens + self.channel_token[:, :channels].expand(
            batch_size,
            -1,
            spatial_tokens,
            -1,
        )

        embeddings = tokens.reshape(batch_size, channels * spatial_tokens, self.embed_dim)
        cls = self.cls_token.expand(batch_size, -1, -1) + self.pos_embed[:, :, 0]
        embeddings = torch.cat([cls, embeddings], dim=1)

        channel_mask = channel_mask.reshape(batch_size, channels * spatial_tokens)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=imgs.device)
        channel_mask = torch.cat([cls_mask, channel_mask], dim=1)
        return self.pos_drop(embeddings), channel_mask, channels, spatial_tokens

    def _select_layers(self, n: int | Sequence[int]) -> List[int]:
        depth = len(self.blocks)
        if isinstance(n, int):
            if n <= 0:
                raise ValueError("n must be positive.")
            return list(range(max(0, depth - n), depth))
        return [int(i) for i in n]

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
    ):
        x, channel_mask, channels, spatial_tokens = self._tokenize(x)
        selected = set(self._select_layers(n))
        outputs = []
        h_patch = int(math.sqrt(spatial_tokens))
        w_patch = spatial_tokens // h_patch

        for i, blk in enumerate(self.blocks):
            x = blk(x, src_key_padding_mask=channel_mask)
            if i in selected:
                y = self.norm(x)
                patch_tokens = y[:, 1:].reshape(
                    y.shape[0],
                    channels,
                    spatial_tokens,
                    self.embed_dim,
                )
                patch_tokens = patch_tokens.reshape(y.shape[0], channels * spatial_tokens, self.embed_dim)
                if reshape:
                    patch_tokens = patch_tokens.reshape(
                        y.shape[0],
                        channels,
                        h_patch,
                        w_patch,
                        self.embed_dim,
                    ).mean(dim=1)
                    patch_tokens = patch_tokens.permute(0, 3, 1, 2).contiguous()
                outputs.append((patch_tokens, y[:, 0]) if return_class_token else patch_tokens)
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_intermediate_layers(x, n=1, reshape=False)[0].mean(dim=1)


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


def _extract_state_dict(ckpt: object) -> dict[str, torch.Tensor] | None:
    if not isinstance(ckpt, dict):
        return None
    for key in ("state_dict", "model", "teacher"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # type: ignore[return-value]
    return None


def _is_chadavit_state(state_dict: dict[str, torch.Tensor] | None) -> bool:
    if not state_dict:
        return False
    return any(k.endswith("token_learner.proj.weight") for k in state_dict)


def _load_chadavit_backbone(
    checkpoint_path: str,
    device: torch.device,
    freeze: bool,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(ckpt)
    if not _is_chadavit_state(state):
        raise ValueError(f"Not a ChAda-ViT checkpoint: {checkpoint_path}")

    assert state is not None
    clean_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            clean_state[key[len("backbone.") :]] = value
        elif key.startswith("encoder."):
            clean_state[key[len("encoder.") :]] = value

    token_weight = clean_state["token_learner.proj.weight"]
    patch_size = int(token_weight.shape[-1])
    embed_dim = int(token_weight.shape[0])
    max_channels = int(clean_state["channel_token"].shape[1])
    pos_tokens = int(clean_state["pos_embed"].shape[2] - 1)
    img_size = int(math.sqrt(pos_tokens)) * patch_size

    model = _EvalChAdaViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=12,
        num_heads=2,
        max_number_channels=max_channels,
    )
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.warning("ChAda-ViT missing keys: %s", missing)
    if unexpected:
        logger.info("ChAda-ViT ignored non-backbone keys: %s", unexpected[:20])

    model = model.to(device)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)

    logger.info(
        "ChAda-ViT backbone ready: embed_dim=%s, patch_size=%s, max_channels=%s",
        model.embed_dim,
        model.patch_size,
        model.max_channels,
    )
    return model


def _load_dcp_backbone_for_eval(
    checkpoint_dir: Path,
    cfg,
    device: torch.device,
    freeze: bool,
):
    """Load a training DCP directory into the bare teacher backbone for eval.

    Some local training runs saved the full SSL wrapper in DCP with keys such as
    ``model.teacher.backbone.*``.  The generic eval loader expects
    ``model.backbone.*`` and cannot map those keys, so we read the teacher
    backbone tensors directly and then load them into an unsharded eval model.
    """
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint.filesystem as dcpfs

    metadata_path = checkpoint_dir / ".metadata"
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)
    checkpoint_keys = set(metadata.state_dict_metadata.keys())

    prefix_candidates = (
        "model.teacher.backbone.",
        "model.model_ema.backbone.",
        "model.student.backbone.",
    )
    prefix = max(prefix_candidates, key=lambda p: sum(k.startswith(p) for k in checkpoint_keys))
    if not any(k.startswith(prefix) for k in checkpoint_keys):
        raise KeyError(f"No backbone tensors found in DCP checkpoint: {checkpoint_dir}")

    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    template_state = model.state_dict()
    tensors_to_load = {}
    missing = []
    for key, tensor in template_state.items():
        checkpoint_key = prefix + key
        if checkpoint_key in checkpoint_keys:
            tensors_to_load[checkpoint_key] = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device="cpu",
            )
        else:
            missing.append(key)
    if missing:
        raise KeyError(
            f"DCP checkpoint is missing {len(missing)} eval backbone keys; "
            f"examples={missing[:10]}"
        )

    dcp.load(
        tensors_to_load,
        storage_reader=dcpfs.FileSystemReader(checkpoint_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=False),
    )

    # The model was built on meta; allocate real tensors before loading.
    model.to_empty(device="cpu")
    state_dict = {key[len(prefix) :]: value for key, value in tensors_to_load.items()}
    msg = model.load_state_dict(state_dict, strict=True)
    logger.info(
        "Loaded DCP eval backbone from %s using prefix=%s with msg: %s",
        checkpoint_dir,
        prefix,
        msg,
    )

    if freeze:
        # Cast while still on CPU so reduced-precision Frozen backbones never
        # materialize a transient fp32 parameter set on the target GPU.
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


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
    ck = Path(checkpoint_path)
    if ck.is_file():
        try:
            ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception:
            logger.exception("Failed to read checkpoint: %s", checkpoint_path)
            raise
        if _is_chadavit_state(_extract_state_dict(ckpt)):
            return _load_chadavit_backbone(checkpoint_path, device=device, freeze=freeze)

    default_cfg = get_default_config()
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(train_config_path))

    if ck.is_dir() and (ck / ".metadata").exists():
        model = _load_dcp_backbone_for_eval(ck, cfg, device=device, freeze=freeze)
        logger.info("Backbone ready: embed_dim=%s, patch_size=%s", model.embed_dim, model.patch_size)
        return model

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
