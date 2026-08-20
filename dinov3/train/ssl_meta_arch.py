# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import copy
import gc
import logging
import math
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import (
    AcquisitionOrbitDeflationLoss,
    acquisition_tangent_fraction,
    apply_acquisition_tangent_gradient_projection,
    build_acquisition_tangent_basis,
    ConditionalEdgeGraphPredictor,
    ConditionalMorphologyGraphLoss,
    ConditionalMorphologyGraphWeights,
    ConditionalFeaturePredictor,
    DINOLoss,
    DistributedSIGReg,
    GramLoss,
    KoLeoLoss,
    KoLeoLossDistributed,
    NestedChannelInnovationLoss,
    NestedChannelInnovationWeights,
    ScoutKernelDeltaTransportLoss,
    conditional_innovation_residual,
    centered_cosine_kernel,
    cross_view_stable_kernel_delta,
    iBOTPatchLoss,
    martingale_increment_orthogonality,
    project_onto_acquisition_tangent,
    rank_matched_random_tangent_basis,
)
from dinov3.models import build_model_from_cfg
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters

logger = logging.getLogger("dinov3")


def _configure_partial_backbone(
    backbone: nn.Module,
    *,
    trainable_last_blocks: int,
    trainable_extra_stem: bool,
) -> tuple[int, int]:
    """Freeze a ViT except its final blocks/norm and optional new channel stem."""
    if trainable_last_blocks < 0:
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        return trainable, total
    if not hasattr(backbone, "blocks"):
        raise ValueError("optim.trainable_last_blocks requires a ViT-style backbone with .blocks")

    depth = len(backbone.blocks)
    if trainable_last_blocks > depth:
        raise ValueError(
            f"optim.trainable_last_blocks={trainable_last_blocks} exceeds backbone depth={depth}"
        )

    backbone.requires_grad_(False)
    if trainable_last_blocks:
        for block in backbone.blocks[-trainable_last_blocks:]:
            block.requires_grad_(True)
    for name in ("norm", "cls_norm", "local_cls_norm"):
        module = getattr(backbone, name, None)
        if module is not None:
            module.requires_grad_(True)

    if trainable_extra_stem:
        stem = getattr(backbone, "patch_embed", None)
        if stem is None:
            raise ValueError("optim.trainable_extra_stem requires backbone.patch_embed")
        found_extra = False
        for name in ("extra", "extra_scale", "pool"):
            component = getattr(stem, name, None)
            if component is not None:
                component.requires_grad_(True)
                found_extra = True
        channel_embed = getattr(backbone, "channel_embed", None)
        if channel_embed is not None:
            channel_embed.requires_grad_(True)
            found_extra = True
        if not found_extra:
            raise ValueError(
                "optim.trainable_extra_stem=true, but no residual/pool/channel embedding was found"
            )

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return trainable, total


def _sample_channel_subset_mask(
    valid_mask: Tensor,
    *,
    min_channels: int,
    max_channels: int,
    require_omission: bool = False,
) -> Tensor:
    """Sample one non-empty subset for each sample in a padded channel batch.

    When ``require_omission`` is true, every sample with at least two valid
    channels loses at least one channel. Single-channel samples remain unchanged
    and are marked inactive by the caller.
    """
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected a [B, C] channel mask, got {tuple(valid_mask.shape)}")
    if min_channels <= 0 or max_channels < min_channels:
        raise ValueError(
            f"Invalid channel subset range: min_channels={min_channels}, max_channels={max_channels}"
        )

    valid_mask = valid_mask.to(dtype=torch.bool)
    subset = torch.zeros_like(valid_mask)
    for row in range(valid_mask.shape[0]):
        available = valid_mask[row].nonzero(as_tuple=False).flatten()
        if available.numel() == 0:
            raise ValueError(f"Sample {row} has no valid channels")
        available_count = int(available.numel())
        omission_cap = available_count - 1 if require_omission and available_count > 1 else available_count
        upper = min(max_channels, omission_cap)
        lower = min(min_channels, upper)
        count = int(torch.randint(lower, upper + 1, (1,), device=valid_mask.device).item())
        chosen = available[torch.randperm(available.numel(), device=valid_mask.device)[:count]]
        subset[row, chosen] = True
    return subset


def _sample_nested_channel_masks(valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Sample S subset M subset F for channel-count-aware martingale losses.

    The midpoint is approximately half the available channels.  This keeps the
    two increments meaningful for the common 3-channel case (1 -> 2 -> 3)
    while still providing a balanced split for rarer high-channel fields.
    """
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected a [B, C] channel mask, got {tuple(valid_mask.shape)}")

    valid_mask = valid_mask.to(dtype=torch.bool)
    middle = torch.zeros_like(valid_mask)
    lower = torch.zeros_like(valid_mask)
    for row in range(valid_mask.shape[0]):
        available = valid_mask[row].nonzero(as_tuple=False).flatten()
        available_count = int(available.numel())
        if available_count == 0:
            raise ValueError(f"Sample {row} has no valid channels")
        middle_count = max(1, (available_count + 1) // 2)
        middle_channels = available[
            torch.randperm(available_count, device=valid_mask.device)[:middle_count]
        ]
        middle[row, middle_channels] = True
        if middle_count == 1:
            lower[row, middle_channels] = True
            continue
        lower_count = int(torch.randint(1, middle_count, (1,), device=valid_mask.device).item())
        lower_channels = middle_channels[
            torch.randperm(middle_count, device=valid_mask.device)[:lower_count]
        ]
        lower[row, lower_channels] = True
    return middle, lower


def _require_rgb_backbone_for_nri(backbone) -> None:
    """NRI downsamples the same RGB image; it must not ride a channel stem."""
    stem_type = getattr(backbone, "stem_type", None)
    if stem_type in ("", "auto"):
        stem_type = None
    if stem_type is not None:
        raise ValueError(
            "nested_resolution_innovation requires the standard RGB PatchEmbed "
            f"(stem_type=null, in_chans=3), got stem_type={stem_type!r}. "
            "Residual-MC / dualroute change the input carrier, so a gain cannot "
            "be attributed to NRI."
        )
    if bool(getattr(backbone, "enable_channelvit", False)):
        raise ValueError("nested_resolution_innovation requires enable_channelvit=false")
    in_chans = getattr(backbone, "in_chans", None)
    if in_chans is not None and int(in_chans) != 3:
        raise ValueError(
            "nested_resolution_innovation requires in_chans=3, "
            f"got in_chans={in_chans}"
        )


def _make_low_resolution_observation(images: Tensor, downsample_factor: int) -> Tensor:
    """Remove fine spatial detail while retaining the exact field of view."""
    if images.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] images, got {tuple(images.shape)}")
    if downsample_factor <= 1:
        raise ValueError(f"downsample_factor must be greater than one, got {downsample_factor}")
    height, width = images.shape[-2:]
    low_size = (height // downsample_factor, width // downsample_factor)
    if min(low_size) < 1:
        raise ValueError(
            f"downsample_factor={downsample_factor} is too large for spatial size {(height, width)}"
        )
    work = images.float()
    low = F.interpolate(work, size=low_size, mode="bilinear", align_corners=False, antialias=True)
    restored = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return restored.to(dtype=images.dtype)


def _make_acquisition_orbit_views(
    images: Tensor,
    *,
    contrast_scale: float,
    background_scale: float,
    blur_mix: float,
    num_perturbations: int,
) -> Tensor:
    """Construct deterministic, label-preserving acquisition perturbations."""
    if images.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] images, got {tuple(images.shape)}")
    if not 1 <= num_perturbations <= 3:
        raise ValueError(f"num_perturbations must be in [1, 3], got {num_perturbations}")
    if contrast_scale <= 0:
        raise ValueError(f"contrast_scale must be positive, got {contrast_scale}")
    if background_scale < 0 or not 0 <= blur_mix <= 1:
        raise ValueError("background_scale must be non-negative and blur_mix must be in [0, 1]")

    work = images.float()
    spatial_mean = work.mean(dim=(-2, -1), keepdim=True)
    spatial_std = work.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    contrast = spatial_mean + contrast_scale * (work - spatial_mean)
    background = work + background_scale * spatial_std
    blurred = F.avg_pool2d(F.pad(work, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    psf_blur = (1.0 - blur_mix) * work + blur_mix * blurred
    return torch.stack((contrast, background, psf_blur), dim=1)[:, :num_perturbations].to(images.dtype)


class SSLMetaArch(nn.Module):
    """
    Modified version of SSLMetaArchCompilable including gram loss:
    - Gram loss is used only if gram.use_loss is set to true
    """

    def __init__(self, cfg):
        super().__init__()

        # assert cfg.multidistillation.enabled is False
        assert cfg.crops.local_crops_number > 0
        assert cfg.ibot.separate_head is True
        assert cfg.train.centering == "sinkhorn_knopp"

        self.distributed_mode = str(getattr(cfg.compute_precision, "distributed_mode", "fsdp")).lower()
        if self.distributed_mode not in {"fsdp", "ddp"}:
            raise ValueError(
                "compute_precision.distributed_mode must be 'fsdp' or 'ddp', got "
                f"{self.distributed_mode!r}"
            )
        # The FSDP implementation currently supports SHARD_GRAD_OP only.
        if self.distributed_mode == "fsdp":
            assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"
        self._ddp_wrapped = False

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        # Only build the gram backbone when gram loss is actually enabled,
        # otherwise it wastes ~14 GB of GPU memory for nothing.
        if cfg.gram.use_loss or cfg.gram.compute_stats:
            gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
            torch.cuda.empty_cache()
            gc.collect()
            gram_model_dict["backbone"] = gram_backbone
            logger.info("Gram backbone built (gram.use_loss or compute_stats is enabled)")
        else:
            logger.info("Gram backbone skipped (gram.use_loss=False, compute_stats=False)")
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {cfg.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {cfg.dino.koleo_loss_distributed}")
        if cfg.dino.koleo_loss_distributed:
            logger.info(f"OPTIONS -- KOLEO -- topk: {cfg.dino.koleo_topk}")
            logger.info(
                f"OPTIONS -- KOLEO -- distributed_loss_group_size: {cfg.dino.koleo_distributed_loss_group_size}"
            )
            assert cfg.dino.koleo_distributed_replicas == 0, (
                "Option `dino.koleo_distributed_replicas` is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=cfg.dino.koleo_topk,
                loss_group_size=cfg.dino.koleo_distributed_loss_group_size,
            )
        else:
            assert cfg.dino.koleo_topk == 1, "Non-distributed KoLeo loss only supports `dino.koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        sigreg_cfg = getattr(cfg, "sigreg", None)
        self.sigreg_enabled = sigreg_cfg is not None and sigreg_cfg.enabled
        self.sigreg_weight_schedule_enabled = False
        if self.sigreg_enabled:
            assert sigreg_cfg.mode == "bottleneck", (
                f"Only sigreg.mode='bottleneck' is supported (got '{sigreg_cfg.mode}')."
            )
            self.sigreg_mode = sigreg_cfg.mode
            self.sigreg_loss = DistributedSIGReg(
                num_slices=sigreg_cfg.num_slices,
                range_max=sigreg_cfg.range_max,
                n_knots=sigreg_cfg.n_knots,
            )
            self.sigreg_loss_weight = sigreg_cfg.loss_weight
            self.sigreg_koleo_too = sigreg_cfg.koleo_too
            schedule_cfg = sigreg_cfg.get("weight_schedule")
            if schedule_cfg is not None and schedule_cfg.enabled:
                schedule_type = str(schedule_cfg.type).lower()
                if schedule_type not in {"cosine", "step"}:
                    raise ValueError(f"Unsupported SIGReg weight schedule type: {schedule_type}")
                total_updates = int(cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs)
                start_update = int(schedule_cfg.start_update)
                end_update = int(schedule_cfg.end_update)
                if end_update < 0:
                    end_update = total_updates - 1
                if not 0 <= start_update < total_updates:
                    raise ValueError(f"SIGReg schedule start_update must be in [0, {total_updates}), got {start_update}")
                if end_update < start_update or end_update >= total_updates:
                    raise ValueError(
                        f"SIGReg schedule end_update must be in [{start_update}, {total_updates}), got {end_update}"
                    )
                final_weight = float(schedule_cfg.final_weight)
                if final_weight < 0:
                    raise ValueError(f"SIGReg schedule final_weight must be non-negative, got {final_weight}")
                self.sigreg_weight_schedule_enabled = True
                self.sigreg_weight_schedule_type = schedule_type
                self.sigreg_weight_schedule_start = start_update
                self.sigreg_weight_schedule_end = end_update
                self.sigreg_weight_schedule_final = final_weight
                logger.info(
                    "OPTIONS -- SIGREG weight schedule: type=%s, start=%d, end=%d, weight=%s->%s",
                    schedule_type,
                    start_update,
                    end_update,
                    self.sigreg_loss_weight,
                    final_weight,
                )
            logger.info(
                "OPTIONS -- SIGREG: enabled, mode=%s, weight=%s, num_slices=%s, koleo_too=%s",
                self.sigreg_mode,
                self.sigreg_loss_weight,
                sigreg_cfg.num_slices,
                self.sigreg_koleo_too,
            )

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        assert 0 <= cfg.ibot.mask_ratio_min_max[0] < cfg.ibot.mask_ratio_min_max[1] <= 1, (
            "provide a valid cfg.ibot.mask_ratio_min_max"
        )
        assert 0 <= cfg.ibot.mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
        logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_norm_last_layer: {cfg.ibot.head_norm_last_layer}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes, compile_sinkhorn=cfg.train.compile)

        nci_cfg = cfg.nested_channel_innovation
        self.nci_enabled = bool(nci_cfg.enabled)
        self.nci_loss_weight = float(nci_cfg.loss_weight)
        self.nci_observation_protocol = str(nci_cfg.observation_protocol).lower()
        self.nci_min_channels = int(nci_cfg.min_channels)
        self.nci_max_channels = int(nci_cfg.max_channels)
        self.nci_predictor_lr_multiplier = float(nci_cfg.predictor_lr_multiplier)
        self.nci_checkpoint_subset_forward = bool(nci_cfg.checkpoint_subset_forward)
        self.nci_checkpoint_full_forward = bool(nci_cfg.checkpoint_full_forward)
        self.nci_martingale_enabled = bool(nci_cfg.martingale_enabled)
        self.nci_martingale_lower_loss_weight = float(nci_cfg.martingale_lower_loss_weight)
        self.nci_martingale_cross_orthogonality_weight = float(
            nci_cfg.martingale_cross_orthogonality_weight
        )
        self.nci_martingale_checkpoint_middle_forward = bool(
            nci_cfg.martingale_checkpoint_middle_forward
        )
        if self.nci_enabled:
            if self.nci_loss_weight < 0:
                raise ValueError(
                    f"nested_channel_innovation.loss_weight must be non-negative, got {self.nci_loss_weight}"
                )
            if self.nci_observation_protocol not in {
                "unmasked_shared",
                "masked_shared",
                "legacy_mask_mismatch",
            }:
                raise ValueError(
                    "nested_channel_innovation.observation_protocol must be one of "
                    "{'unmasked_shared', 'masked_shared', 'legacy_mask_mismatch'}, got "
                    f"{self.nci_observation_protocol!r}"
                )
            if self.nci_predictor_lr_multiplier <= 0:
                raise ValueError(
                    "nested_channel_innovation.predictor_lr_multiplier must be positive, got "
                    f"{self.nci_predictor_lr_multiplier}"
                )
            if self.nci_martingale_enabled and self.nci_observation_protocol != "masked_shared":
                raise ValueError(
                    "nested_channel_innovation.martingale_enabled requires observation_protocol='masked_shared'"
                )
            if self.nci_martingale_lower_loss_weight < 0:
                raise ValueError(
                    "nested_channel_innovation.martingale_lower_loss_weight must be non-negative, got "
                    f"{self.nci_martingale_lower_loss_weight}"
                )
            if self.nci_martingale_cross_orthogonality_weight < 0:
                raise ValueError(
                    "nested_channel_innovation.martingale_cross_orthogonality_weight must be non-negative, got "
                    f"{self.nci_martingale_cross_orthogonality_weight}"
                )
            predictor_hidden_dim = int(nci_cfg.predictor_hidden_dim)
            student_model_dict["nci_predictor"] = ConditionalFeaturePredictor(
                embed_dim,
                hidden_dim=predictor_hidden_dim,
            )
            teacher_model_dict["nci_predictor"] = ConditionalFeaturePredictor(
                embed_dim,
                hidden_dim=predictor_hidden_dim,
            )
            self.nci_loss = NestedChannelInnovationLoss(
                min_std=float(nci_cfg.min_std),
                stop_gradient=bool(nci_cfg.stop_gradient),
                weights=NestedChannelInnovationWeights(
                    predictor=float(nci_cfg.predictor_loss_weight),
                    invariance=float(nci_cfg.invariance_loss_weight),
                    variance=float(nci_cfg.variance_loss_weight),
                    orthogonality=float(nci_cfg.orthogonality_loss_weight),
                ),
            )
            if self.nci_martingale_enabled:
                # The two conditional maps represent distinct filtration
                # steps: S -> M and M -> F. Sharing them would collapse the
                # martingale construction into an ordinary one-step adapter.
                student_model_dict["nci_mid_predictor"] = ConditionalFeaturePredictor(
                    embed_dim,
                    hidden_dim=predictor_hidden_dim,
                )
                teacher_model_dict["nci_mid_predictor"] = ConditionalFeaturePredictor(
                    embed_dim,
                    hidden_dim=predictor_hidden_dim,
                )
                self.nci_mid_loss = NestedChannelInnovationLoss(
                    min_std=float(nci_cfg.min_std),
                    stop_gradient=bool(nci_cfg.stop_gradient),
                    metric_prefix="nci_mid",
                    weights=NestedChannelInnovationWeights(
                        predictor=float(nci_cfg.predictor_loss_weight),
                        invariance=float(nci_cfg.invariance_loss_weight),
                        variance=float(nci_cfg.variance_loss_weight),
                        orthogonality=float(nci_cfg.orthogonality_loss_weight),
                    ),
                )

        cmgi_cfg = cfg.conditional_morphology_graph
        self.cmgi_enabled = bool(cmgi_cfg.enabled)
        self.cmgi_loss_weight = float(cmgi_cfg.loss_weight)
        self.cmgi_min_channels = int(cmgi_cfg.min_channels)
        self.cmgi_max_channels = int(cmgi_cfg.max_channels)
        self.cmgi_predictor_lr_multiplier = float(cmgi_cfg.predictor_lr_multiplier)
        self.cmgi_condition_source = str(cmgi_cfg.condition_source).lower()
        self.cmgi_predictor_mode = str(cmgi_cfg.predictor_mode).lower()
        self.cmgi_edge_predictor_dim = int(cmgi_cfg.edge_predictor_dim)
        if self.cmgi_enabled:
            if self.cmgi_loss_weight < 0:
                raise ValueError(
                    "conditional_morphology_graph.loss_weight must be non-negative, got "
                    f"{self.cmgi_loss_weight}"
                )
            if self.cmgi_predictor_lr_multiplier <= 0:
                raise ValueError(
                    "conditional_morphology_graph.predictor_lr_multiplier must be positive, got "
                    f"{self.cmgi_predictor_lr_multiplier}"
                )
            if self.cmgi_condition_source not in {"teacher", "student"}:
                raise ValueError(
                    "conditional_morphology_graph.condition_source must be 'teacher' or 'student', got "
                    f"{self.cmgi_condition_source!r}"
                )
            if self.cmgi_predictor_mode not in {"feature", "edge"}:
                raise ValueError(
                    "conditional_morphology_graph.predictor_mode must be 'feature' or 'edge', got "
                    f"{self.cmgi_predictor_mode!r}"
                )
            if self.cmgi_edge_predictor_dim <= 0:
                raise ValueError(
                    "conditional_morphology_graph.edge_predictor_dim must be positive, got "
                    f"{self.cmgi_edge_predictor_dim}"
                )
            predictor_hidden_dim = int(cmgi_cfg.predictor_hidden_dim)
            if self.cmgi_predictor_mode == "feature":
                student_model_dict["cmgi_predictor"] = ConditionalFeaturePredictor(
                    embed_dim,
                    hidden_dim=predictor_hidden_dim,
                )
                teacher_model_dict["cmgi_predictor"] = ConditionalFeaturePredictor(
                    embed_dim,
                    hidden_dim=predictor_hidden_dim,
                )
            else:
                student_model_dict["cmgi_predictor"] = ConditionalEdgeGraphPredictor(
                    embed_dim,
                    edge_dim=self.cmgi_edge_predictor_dim,
                    hidden_dim=predictor_hidden_dim,
                )
                teacher_model_dict["cmgi_predictor"] = ConditionalEdgeGraphPredictor(
                    embed_dim,
                    edge_dim=self.cmgi_edge_predictor_dim,
                    hidden_dim=predictor_hidden_dim,
                )
            self.cmgi_loss = ConditionalMorphologyGraphLoss(
                local_radius=int(cmgi_cfg.local_radius),
                min_innovation=float(cmgi_cfg.min_innovation),
                selection_fraction=float(cmgi_cfg.selection_fraction),
                max_edge_weight=float(cmgi_cfg.max_edge_weight),
                huber_beta=float(cmgi_cfg.huber_beta),
                gate_mode=str(cmgi_cfg.gate_mode),
                predictor_mode=self.cmgi_predictor_mode,
                stop_gradient=bool(cmgi_cfg.stop_gradient),
                weights=ConditionalMorphologyGraphWeights(
                    predictor=float(cmgi_cfg.predictor_loss_weight),
                    graph=float(cmgi_cfg.graph_loss_weight),
                ),
            )

        nri_cfg = cfg.nested_resolution_innovation
        self.nri_enabled = bool(nri_cfg.enabled)
        self.nri_loss_weight = float(nri_cfg.loss_weight)
        self.nri_downsample_factor = int(nri_cfg.downsample_factor)
        self.nri_feature_mode = str(nri_cfg.feature_mode)
        self.nri_predictor_lr_multiplier = float(nri_cfg.predictor_lr_multiplier)
        if self.nri_enabled:
            if self.nri_loss_weight < 0:
                raise ValueError(
                    "nested_resolution_innovation.loss_weight must be non-negative, got "
                    f"{self.nri_loss_weight}"
                )
            if self.nri_downsample_factor <= 1:
                raise ValueError(
                    "nested_resolution_innovation.downsample_factor must be greater than one, got "
                    f"{self.nri_downsample_factor}"
                )
            if self.nri_feature_mode not in {"cls", "cls_patch_mean"}:
                raise ValueError(
                    "nested_resolution_innovation.feature_mode must be 'cls' or 'cls_patch_mean', got "
                    f"{self.nri_feature_mode!r}"
                )
            if self.nri_predictor_lr_multiplier <= 0:
                raise ValueError(
                    "nested_resolution_innovation.predictor_lr_multiplier must be positive, got "
                    f"{self.nri_predictor_lr_multiplier}"
                )
            predictor_hidden_dim = int(nri_cfg.predictor_hidden_dim)
            student_model_dict["nri_predictor"] = ConditionalFeaturePredictor(
                embed_dim,
                hidden_dim=predictor_hidden_dim,
            )
            teacher_model_dict["nri_predictor"] = ConditionalFeaturePredictor(
                embed_dim,
                hidden_dim=predictor_hidden_dim,
            )
            self.nri_loss = NestedChannelInnovationLoss(
                min_std=float(nri_cfg.min_std),
                stop_gradient=bool(nri_cfg.stop_gradient),
                metric_prefix="nri",
                weights=NestedChannelInnovationWeights(
                    predictor=float(nri_cfg.predictor_loss_weight),
                    invariance=float(nri_cfg.invariance_loss_weight),
                    variance=float(nri_cfg.variance_loss_weight),
                    orthogonality=float(nri_cfg.orthogonality_loss_weight),
                ),
            )

        acq_cfg = cfg.acquisition_orbit_deflation
        self.acq_deflation_enabled = bool(acq_cfg.enabled)
        self.acq_deflation_mode = str(acq_cfg.mode)
        self.acq_deflation_loss_weight = float(acq_cfg.loss_weight)
        self.acq_projection_strength = float(getattr(acq_cfg, "projection_strength", 1.0))
        self.acq_projection_scope = str(getattr(acq_cfg, "projection_scope", "cls")).lower()
        self.acq_num_perturbations = int(acq_cfg.num_perturbations)
        self.acq_contrast_scale = float(acq_cfg.contrast_scale)
        self.acq_background_scale = float(acq_cfg.background_scale)
        self.acq_blur_mix = float(acq_cfg.blur_mix)
        self._acquisition_anchor_backbone: nn.Module | None = None
        self.acq_deflation_loss: AcquisitionOrbitDeflationLoss | None = None
        if self.acq_deflation_enabled:
            if self.distributed_mode != "ddp":
                raise ValueError("acquisition_orbit_deflation currently requires distributed_mode=ddp")
            if cfg.distillation.enabled or cfg.multidistillation.enabled:
                raise ValueError("acquisition_orbit_deflation does not support distillation meta-architectures")
            if self.acq_deflation_loss_weight < 0:
                raise ValueError(
                    "acquisition_orbit_deflation.loss_weight must be non-negative, got "
                    f"{self.acq_deflation_loss_weight}"
                )
            if self.acq_deflation_mode not in {
                "deflate",
                "random_tangent",
                "direct_consistency",
                "gradient_projection",
                "random_gradient_projection",
            }:
                raise ValueError(
                    "acquisition_orbit_deflation.mode must be deflate, random_tangent, direct_consistency, "
                    "gradient_projection, or random_gradient_projection, got "
                    f"{self.acq_deflation_mode!r}"
                )
            if self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"} and not (
                0.0 <= self.acq_projection_strength <= 1.0
            ):
                raise ValueError(
                    "acquisition_orbit_deflation.projection_strength must be in [0, 1], got "
                    f"{self.acq_projection_strength}"
                )
            if self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"} and (
                self.acq_projection_scope not in {"cls", "cls_patch"}
            ):
                raise ValueError(
                    "acquisition_orbit_deflation.projection_scope must be 'cls' or 'cls_patch', got "
                    f"{self.acq_projection_scope!r}"
                )
            if self.acq_deflation_mode in {"deflate", "random_tangent"}:
                self.acq_deflation_loss = AcquisitionOrbitDeflationLoss(
                    min_singular_value=float(acq_cfg.min_singular_value),
                    relative_singular_value=float(acq_cfg.relative_singular_value),
                )
            logger.info(
                "Acquisition orbit mode=%s perturbations=%d weight=%s projection_strength=%s scope=%s",
                self.acq_deflation_mode,
                self.acq_num_perturbations,
                self.acq_deflation_loss_weight,
                self.acq_projection_strength,
                self.acq_projection_scope,
            )

        scout_cfg = cfg.scout_kernel_transport
        self.scout_transport_enabled = bool(scout_cfg.enabled)
        self.scout_transport_loss_weight = float(scout_cfg.loss_weight)
        self.scout_transport_target_mode = str(scout_cfg.target_mode).lower()
        self.scout_transport_current_feature_protocol = str(
            getattr(scout_cfg, "current_feature_protocol", "masked_student")
        ).lower()
        self.scout_transport_directional_damping = float(
            getattr(scout_cfg, "directional_damping", 0.1)
        )
        self.scout_transport_displacement_budget_ratio = float(
            getattr(scout_cfg, "displacement_budget_ratio", 0.0)
        )
        self.scout_stable_relative_eigenvalue = float(
            getattr(scout_cfg, "stable_relative_eigenvalue", 0.05)
        )
        self.scout_stable_min_eigenvalue = float(getattr(scout_cfg, "stable_min_eigenvalue", 1.0e-6))
        self.scout_config_path = scout_cfg.scout_config_path
        self.scout_anchor_checkpoint = scout_cfg.scout_anchor_checkpoint
        self.scout_adapted_checkpoint = scout_cfg.scout_adapted_checkpoint
        self._scout_large_anchor_backbone: nn.Module | None = None
        self._scout_anchor_backbone: nn.Module | None = None
        self._scout_adapted_backbone: nn.Module | None = None
        # The shuffled-target control must not consume the global CUDA RNG.
        # Otherwise it would also perturb future masks/augmentations and cease
        # to isolate the semantic effect of the scout target.
        self._scout_shuffled_target_step = 0
        if self.scout_transport_enabled:
            if self.distributed_mode != "ddp":
                raise ValueError("scout_kernel_transport currently requires distributed_mode=ddp")
            if cfg.distillation.enabled or cfg.multidistillation.enabled:
                raise ValueError("scout_kernel_transport does not support distillation meta-architectures")
            if self.scout_transport_loss_weight < 0:
                raise ValueError(
                    "scout_kernel_transport.loss_weight must be non-negative, got "
                    f"{self.scout_transport_loss_weight}"
                )
            if self.scout_transport_target_mode not in {
                "delta",
                "stable_delta",
                "shuffled_delta",
                "shuffled_stable_delta",
                "final_kernel",
            }:
                raise ValueError(
                    "scout_kernel_transport.target_mode must be one of "
                    "{'delta', 'stable_delta', 'shuffled_delta', "
                    "'shuffled_stable_delta', 'final_kernel'}, got "
                    f"{self.scout_transport_target_mode!r}"
                )
            if self.scout_transport_current_feature_protocol not in {
                "masked_student",
                "anchor_consistent",
                "mask_matched",
            }:
                raise ValueError(
                    "scout_kernel_transport.current_feature_protocol must be one of "
                    "{'masked_student', 'anchor_consistent', 'mask_matched'}, got "
                    f"{self.scout_transport_current_feature_protocol!r}"
                )
            if self.scout_transport_directional_damping <= 0:
                raise ValueError(
                    "scout_kernel_transport.directional_damping must be positive, got "
                    f"{self.scout_transport_directional_damping}"
                )
            if self.scout_transport_displacement_budget_ratio < 0:
                raise ValueError(
                    "scout_kernel_transport.displacement_budget_ratio must be non-negative, got "
                    f"{self.scout_transport_displacement_budget_ratio}"
                )
            if self.scout_stable_relative_eigenvalue < 0 or self.scout_stable_min_eigenvalue < 0:
                raise ValueError("scout stable eigenvalue thresholds must be non-negative")
            for name, path in (
                ("scout_config_path", self.scout_config_path),
                ("scout_anchor_checkpoint", self.scout_anchor_checkpoint),
                ("scout_adapted_checkpoint", self.scout_adapted_checkpoint),
            ):
                if not path:
                    raise ValueError(f"scout_kernel_transport.{name} must be set when enabled")
            self.scout_transport_loss = ScoutKernelDeltaTransportLoss(
                directional_damping=self.scout_transport_directional_damping,
                displacement_budget_ratio=self.scout_transport_displacement_budget_ratio,
            )
            logger.info(
                "Scout kernel transport enabled: weight=%s target=%s current_protocol=%s "
                "directional_damping=%s displacement_budget_ratio=%s stable_relative_eigenvalue=%s",
                self.scout_transport_loss_weight,
                self.scout_transport_target_mode,
                self.scout_transport_current_feature_protocol,
                self.scout_transport_directional_damping,
                self.scout_transport_displacement_budget_ratio,
                self.scout_stable_relative_eigenvalue,
            )

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        self.trainable_last_blocks = int(getattr(cfg.optim, "trainable_last_blocks", -1))
        self.trainable_extra_stem = bool(getattr(cfg.optim, "trainable_extra_stem", False))
        trainable_backbone, total_backbone = _configure_partial_backbone(
            self.student.backbone,
            trainable_last_blocks=self.trainable_last_blocks,
            trainable_extra_stem=self.trainable_extra_stem,
        )
        logger.info(
            "Backbone trainability: last_blocks=%d extra_stem=%s trainable=%d/%d (%.2f%%)",
            self.trainable_last_blocks,
            self.trainable_extra_stem,
            trainable_backbone,
            total_backbone,
            100.0 * trainable_backbone / max(1, total_backbone),
        )

        subset_cfg = cfg.channel_subset
        self.channel_subset_enabled = bool(subset_cfg.enabled)
        self.channel_subset_min = int(subset_cfg.min_channels)
        self.channel_subset_max = int(subset_cfg.max_channels)
        if (self.nci_enabled or self.cmgi_enabled) and self.channel_subset_enabled:
            raise ValueError(
                "conditional channel objectives and legacy channel_subset cannot be enabled together: "
                "they keep the main SSL path full-channel symmetric"
            )
        if self.nci_enabled and self.cmgi_enabled:
            raise ValueError("nested_channel_innovation and conditional_morphology_graph cannot be enabled together")
        if self.channel_subset_enabled:
            if self.channel_subset_min <= 0 or self.channel_subset_max < self.channel_subset_min:
                raise ValueError(
                    "channel_subset requires 0 < min_channels <= max_channels, got "
                    f"{self.channel_subset_min}, {self.channel_subset_max}"
                )
            supports_channel_masks = bool(getattr(self.student.backbone, "enable_channelvit", False)) or (
                getattr(self.student.backbone, "stem_type", None) is not None
            )
            if not supports_channel_masks:
                raise ValueError(
                    "channel_subset.enabled=true requires ChannelViT or a channel-aware stem_type"
                )
            logger.info(
                "Full-channel teacher -> subset-channel student enabled: student channels=%d..%d",
                self.channel_subset_min,
                self.channel_subset_max,
            )
        if self.nci_enabled:
            if self.nci_min_channels <= 0 or self.nci_max_channels < self.nci_min_channels:
                raise ValueError(
                    "nested_channel_innovation requires 0 < min_channels <= max_channels, got "
                    f"{self.nci_min_channels}, {self.nci_max_channels}"
                )
            supports_channel_masks = bool(getattr(self.student.backbone, "enable_channelvit", False)) or (
                getattr(self.student.backbone, "stem_type", None) is not None
            )
            if not supports_channel_masks:
                raise ValueError(
                    "nested_channel_innovation.enabled=true requires ChannelViT or a channel-aware stem_type"
                )
            if cfg.distillation.enabled or cfg.multidistillation.enabled:
                raise ValueError("nested_channel_innovation does not yet support distillation meta-architectures")
            logger.info(
                "Nested channel innovation enabled: subset channels=%d..%d overall_weight=%s protocol=%s "
                "subset_checkpoint=%s full_checkpoint=%s martingale=%s",
                self.nci_min_channels,
                self.nci_max_channels,
                self.nci_loss_weight,
                self.nci_observation_protocol,
                self.nci_checkpoint_subset_forward,
                self.nci_checkpoint_full_forward,
                self.nci_martingale_enabled,
            )
        if self.cmgi_enabled:
            if self.cmgi_min_channels <= 0 or self.cmgi_max_channels < self.cmgi_min_channels:
                raise ValueError(
                    "conditional_morphology_graph requires 0 < min_channels <= max_channels, got "
                    f"{self.cmgi_min_channels}, {self.cmgi_max_channels}"
                )
            if bool(getattr(self.student.backbone, "enable_channelvit", False)):
                raise ValueError(
                    "conditional_morphology_graph currently requires a fixed-grid channel-aware stem, not ChannelViT"
                )
            if getattr(self.student.backbone, "stem_type", None) is None:
                raise ValueError(
                    "conditional_morphology_graph.enabled=true requires a channel-aware stem_type"
                )
            if cfg.distillation.enabled or cfg.multidistillation.enabled:
                raise ValueError("conditional_morphology_graph does not yet support distillation meta-architectures")
            logger.info(
                "Conditional morphology graph enabled: subset channels=%d..%d radius=%d overall_weight=%s",
                self.cmgi_min_channels,
                self.cmgi_max_channels,
                self.cmgi_loss.local_radius,
                self.cmgi_loss_weight,
            )
        if self.nri_enabled:
            if cfg.distillation.enabled or cfg.multidistillation.enabled:
                raise ValueError("nested_resolution_innovation does not yet support distillation meta-architectures")
            _require_rgb_backbone_for_nri(self.student.backbone)
            logger.info(
                "Nested resolution innovation enabled on RGB PatchEmbed: downsample=%dx "
                "feature=%s overall_weight=%s in_chans=%s stem_type=%s",
                self.nri_downsample_factor,
                self.nri_feature_mode,
                self.nri_loss_weight,
                getattr(self.student.backbone, "in_chans", None),
                getattr(self.student.backbone, "stem_type", None),
            )

        # getting config params fixed:
        self.n_local_crops = self.cfg.crops.local_crops_number
        self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )

        # Gram
        self.gram_use_loss = self.cfg.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            # Gram regularization
            self.gram_loss = GramLoss(
                apply_norm=self.cfg.gram.normalized,
                remove_only_teacher_neg=self.cfg.gram.remove_only_teacher_neg,
                remove_neg=self.cfg.gram.remove_neg,
            )
            # Construct gram teacher
            self.has_gram_teacher = True if not cfg.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")
            else:
                self.gram_teacher = None

            self.gram_loss_weight = self.cfg.gram.loss_weight
            if self.cfg.gram.get("loss_weight_schedule"):
                iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * cfg.optim.epochs
                schedule_cfg = self.cfg.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_cfg.start,
                    peak=schedule_cfg.peak,
                    end=schedule_cfg.end,
                    warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                    ),
                )
                logger.info(f"Applying gram loss weight schedule instead of `cfg.gram.loss_weight`: {schedule_cfg}")
            else:
                self.gram_loss_schedule = None
            self.gram_ema_teacher = self.cfg.gram.ema_teacher  # If true use the EMA_teacher as gram_teacher
            self.gram_ckpt = self.cfg.gram.ckpt  # Checkpoint to the first gram teacher model
            self.gram_img_level = self.cfg.gram.img_level  # Apply the loss on the image, if false on the batch
            self.gram_tokens_used = self.cfg.gram.tokens_used  # Any value in ["all", "masked", "unmasked"]
            # Update the teacher frequently
            self.gram_rep_update = self.cfg.gram.rep_update  # bool, if yes the gram teacher will be updated at the freq
            self.gram_update_frequency = self.cfg.gram.update_frequency  # defined by this var update_frequency
            self.gram_it_first_update = self.cfg.gram.it_first_update  # after iteration it_first_update is passed.
            self.gram_it_load_ema_teacher = (
                self.cfg.gram.it_load_ema_teacher
            )  # after iteration it_load_ema the ema teacher is loaded into the gram teacher
            self.gram_compute_stats = self.cfg.gram.compute_stats  # whether to compute auxiliary stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {cfg.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {cfg.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {cfg.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {cfg.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}")

            if cfg.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size must be set to use gram loss")
            if cfg.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size shoud be None when gram.ema_teacher=True")

            self.student_crop_size = cfg.crops.global_crops_size
            self.gram_global_teacher_resize_method = cfg.gram.global_teacher_resize_method
            self.gram_global_teacher_resize_antialias = cfg.gram.global_teacher_resize_antialias
            logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher size: {cfg.crops.gram_teacher_crops_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {cfg.gram.global_teacher_resize_method}")
            logger.info(
                f"OPTIONS -- global crops GRAM teacher resize antialias: {cfg.gram.global_teacher_resize_antialias}"
            )

    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes
        assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes
        assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        if self.nci_enabled:
            self.student.nci_predictor.reset_parameters()
            if self.nci_martingale_enabled:
                self.student.nci_mid_predictor.reset_parameters()
        if self.cmgi_enabled:
            self.student.cmgi_predictor.reset_parameters()
        if self.nri_enabled:
            self.student.nri_predictor.reset_parameters()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        if self.sigreg_enabled:
            self.sigreg_loss.reset_buffers()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                init_fsdp_model_from_checkpoint(
                    self.gram_teacher,
                    self.gram_ckpt,
                    skip_load_keys=[
                        "dino_head",
                        "ibot_head",
                        "dino_loss.center",
                        "ibot_patch_loss.center",
                    ],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
                self.gram_teacher_initialized = True
            else:
                raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )
            self.model_ema.load_state_dict(self.student.state_dict())
            # Spot-check: verify student and teacher backbone are identical after weight sync
            if distributed.is_main_process():
                teacher_params = dict(self.teacher.named_parameters())
                checked = 0
                for name, s_param in self.student.named_parameters():
                    t_param = teacher_params.get(name)
                    if t_param is None or "backbone." not in name:
                        continue
                    try:
                        s_data = s_param._local_tensor if hasattr(s_param, "_local_tensor") else s_param.data
                        t_data = t_param._local_tensor if hasattr(t_param, "_local_tensor") else t_param.data
                        match = torch.allclose(s_data.float(), t_data.float(), atol=0)
                        logger.info(f"[INIT CHECK] {name}: student == teacher: {match}")
                    except Exception as e:
                        logger.info(f"[INIT CHECK] skip {name}: {e}")
                    checked += 1
                    if checked >= 2:
                        break
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")
        if self.acq_deflation_enabled and self.acq_deflation_mode != "direct_consistency":
            # Do not register the anchor as a trainable/checkpoint module. It
            # is a fixed calibration artifact copied after released weights load.
            anchor_backbone = copy.deepcopy(self.teacher.backbone)
            anchor_backbone.requires_grad_(False)
            anchor_backbone.eval()
            object.__setattr__(self, "_acquisition_anchor_backbone", anchor_backbone)
        if self.scout_transport_enabled:
            large_anchor = copy.deepcopy(self.teacher.backbone)
            large_anchor.requires_grad_(False)
            large_anchor.eval()
            object.__setattr__(self, "_scout_large_anchor_backbone", large_anchor)
            object.__setattr__(
                self,
                "_scout_anchor_backbone",
                self._build_frozen_scout_backbone(str(self.scout_anchor_checkpoint)),
            )
            object.__setattr__(
                self,
                "_scout_adapted_backbone",
                self._build_frozen_scout_backbone(
                    str(self.scout_adapted_checkpoint),
                    from_training_checkpoint=True,
                ),
            )

    def _build_frozen_scout_backbone(
        self,
        checkpoint_path: str,
        *,
        from_training_checkpoint: bool = False,
    ) -> nn.Module:
        """Build a frozen scout from either released or continued-training weights."""
        default_cfg = get_default_config()
        scout_cfg = OmegaConf.merge(default_cfg, OmegaConf.load(str(self.scout_config_path)))
        backbone, _ = build_model_from_cfg(scout_cfg, only_teacher=True)
        dtype_by_name = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        backbone.to_empty(device="cuda")
        backbone.to(dtype=dtype_by_name[self.cfg.compute_precision.param_dtype])
        holder = nn.ModuleDict({"backbone": backbone})
        init_fsdp_model_from_checkpoint(
            holder,
            checkpoint_path,
            skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
            keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
            process_group=distributed.get_process_subgroup(),
            checkpoint_state_prefix="teacher.backbone." if from_training_checkpoint else None,
        )
        backbone.requires_grad_(False)
        backbone.eval()
        return backbone

    @torch.no_grad()
    def _get_acquisition_anchor_features(
        self,
        *,
        images: Tensor,
        channel_ids: Tensor | None,
        channel_valid_mask: Tensor | None,
        return_patches: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return frozen CLS features and optionally aligned dense token fields."""
        anchor_backbone = self._acquisition_anchor_backbone
        if anchor_backbone is None:
            raise RuntimeError("Acquisition anchor was not initialized")
        anchor_output = anchor_backbone(
            images,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )
        anchor_features = anchor_output["x_norm_clstoken"]
        orbit_images = _make_acquisition_orbit_views(
            images,
            contrast_scale=self.acq_contrast_scale,
            background_scale=self.acq_background_scale,
            blur_mix=self.acq_blur_mix,
            num_perturbations=self.acq_num_perturbations,
        )
        batch_size, num_views = orbit_images.shape[:2]
        orbit_images = orbit_images.flatten(0, 1)
        if channel_ids is not None:
            channel_ids = channel_ids.unsqueeze(1).expand(-1, num_views, -1).flatten(0, 1)
        if channel_valid_mask is not None:
            channel_valid_mask = channel_valid_mask.unsqueeze(1).expand(-1, num_views, -1).flatten(0, 1)
        orbit_output = anchor_backbone(
            orbit_images,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )
        orbit_features = orbit_output["x_norm_clstoken"].unflatten(0, (batch_size, num_views))
        if not return_patches:
            return anchor_features, orbit_features
        anchor_patches = anchor_output["x_norm_patchtokens"]
        orbit_patches = orbit_output["x_norm_patchtokens"].unflatten(0, (batch_size, num_views))
        return anchor_features, orbit_features, anchor_patches, orbit_patches

    def _get_acquisition_student_features(
        self,
        *,
        orbit_images: Tensor,
        channel_ids: Tensor | None,
        channel_valid_mask: Tensor | None,
    ) -> Tensor:
        """Return differentiable student features for a physical nuisance orbit."""
        batch_size, num_views = orbit_images.shape[:2]
        images = orbit_images.flatten(0, 1)
        if channel_ids is not None:
            channel_ids = channel_ids.unsqueeze(1).expand(-1, num_views, -1).flatten(0, 1)
        if channel_valid_mask is not None:
            channel_valid_mask = channel_valid_mask.unsqueeze(1).expand(-1, num_views, -1).flatten(0, 1)
        features = self.student.backbone(
            images,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )["x_norm_clstoken"]
        return features.unflatten(0, (batch_size, num_views))

    @torch.no_grad()
    def _get_scout_transport_artifacts(
        self,
        *,
        images: Tensor,
        masks: Tensor | None,
        channel_ids: Tensor | None,
        channel_valid_mask: Tensor | None,
        include_large_anchor: bool = True,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Return an optional L anchor and a width-agnostic scout kernel delta."""
        large_anchor = self._scout_large_anchor_backbone
        scout_anchor = self._scout_anchor_backbone
        scout_adapted = self._scout_adapted_backbone
        if scout_anchor is None or scout_adapted is None or (include_large_anchor and large_anchor is None):
            raise RuntimeError("Scout transport artifacts were not initialized")
        large_features = None
        if include_large_anchor:
            assert large_anchor is not None
            large_features = large_anchor(
                images,
                masks=masks,
                channel_ids=channel_ids,
                channel_valid_mask=channel_valid_mask,
                is_training=True,
            )["x_norm_clstoken"]
        scout_anchor_features = scout_anchor(
            images,
            masks=masks,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )["x_norm_clstoken"]
        scout_adapted_features = scout_adapted(
            images,
            masks=masks,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )["x_norm_clstoken"]
        scout_adapted_kernel = centered_cosine_kernel(scout_adapted_features)
        scout_delta = scout_adapted_kernel - centered_cosine_kernel(scout_anchor_features)
        return large_features, scout_delta, scout_adapted_kernel

    @staticmethod
    def _is_channelvit_backbone(backbone: nn.Module) -> bool:
        if isinstance(backbone, DistributedDataParallel):
            backbone = backbone.module
        return bool(getattr(backbone, "enable_channelvit", False))

    @staticmethod
    def _expand_channelvit_masks(
        masks: Tensor,
        n_channels: int,
        channel_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Expand spatial iBOT masks to ChannelViT's C * H * W token layout."""
        if masks.ndim != 2:
            raise ValueError(f"Expected masks to be 2D, got shape={tuple(masks.shape)}")
        if n_channels <= 0:
            raise ValueError(f"Expected a positive channel count, got {n_channels}")

        expanded_masks = masks.unsqueeze(1).expand(-1, n_channels, -1)
        if channel_valid_mask is not None:
            channel_valid_mask = channel_valid_mask.to(device=masks.device, dtype=torch.bool)
            if channel_valid_mask.shape != (masks.shape[0], n_channels):
                raise ValueError(
                    "channel_valid_mask must have shape "
                    f"{(masks.shape[0], n_channels)}, got {tuple(channel_valid_mask.shape)}"
                )
            expanded_masks = expanded_masks & channel_valid_mask.unsqueeze(-1)
        expanded_masks = expanded_masks.reshape(masks.shape[0], -1)
        mask_indices_list = expanded_masks.flatten().nonzero().flatten()
        masks_weight = (
            (1 / expanded_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(expanded_masks)[expanded_masks]
        )
        n_masked_patches = torch.full(
            (1,),
            fill_value=mask_indices_list.shape[0],
            dtype=torch.long,
            device=expanded_masks.device,
        )
        return expanded_masks, mask_indices_list, masks_weight, n_masked_patches

    def forward_backward(
        self,
        data,
        *,
        teacher_temp,
        iteration=0,
        loss_divisor: float = 1.0,
        **ignored_kwargs,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        global_channel_ids = data.get("collated_global_channel_ids")
        local_channel_ids = data.get("collated_local_channel_ids")
        global_channel_valid_mask = data.get("collated_global_channel_valid_mask")
        local_channel_valid_mask = data.get("collated_local_channel_valid_mask")
        if global_channel_ids is not None:
            global_channel_ids = global_channel_ids.cuda(non_blocking=True)
        if local_channel_ids is not None:
            local_channel_ids = local_channel_ids.cuda(non_blocking=True)
        if global_channel_valid_mask is not None:
            global_channel_valid_mask = global_channel_valid_mask.cuda(non_blocking=True)
        if local_channel_valid_mask is not None:
            local_channel_valid_mask = local_channel_valid_mask.cuda(non_blocking=True)

        student_global_channel_valid_mask = global_channel_valid_mask
        student_local_channel_valid_mask = local_channel_valid_mask
        nci_global_channel_valid_mask = None
        nci_active_samples = None
        nci_lower_channel_valid_mask = None
        nci_lower_active_samples = None
        cmgi_global_channel_valid_mask = None
        cmgi_active_samples = None
        if self.channel_subset_enabled:
            if global_channel_valid_mask is None or local_channel_valid_mask is None:
                raise ValueError(
                    "channel_subset.enabled=true requires channel ids/masks from a packwds_chvit dataset"
                )
            base_full_mask = global_channel_valid_mask[:B]
            base_subset_mask = _sample_channel_subset_mask(
                base_full_mask,
                min_channels=self.channel_subset_min,
                max_channels=self.channel_subset_max,
            )
            # Collation is crop-major, so reuse the same subset for every view of
            # a sample while the teacher retains the original full-channel mask.
            student_global_channel_valid_mask = base_subset_mask.repeat(n_global_crops, 1)
            student_local_channel_valid_mask = base_subset_mask.repeat(n_local_crops, 1)
            metrics_dict["teacher_channels_per_sample"] = base_full_mask.sum(dim=1).float().mean()
            metrics_dict["student_channels_per_sample"] = base_subset_mask.sum(dim=1).float().mean()
        elif self.nci_enabled:
            if global_channel_valid_mask is None or local_channel_valid_mask is None:
                raise ValueError(
                    "nested_channel_innovation.enabled=true requires channel ids/masks from a "
                    "packwds_chvit dataset"
            )
            base_full_mask = global_channel_valid_mask[:B]
            if self.nci_martingale_enabled:
                base_middle_mask, base_lower_mask = _sample_nested_channel_masks(base_full_mask)
                nci_global_channel_valid_mask = base_middle_mask.repeat(n_global_crops, 1)
                nci_lower_channel_valid_mask = base_lower_mask.repeat(n_global_crops, 1)
                nci_active_samples = base_full_mask.sum(dim=1) > base_middle_mask.sum(dim=1)
                nci_lower_active_samples = base_middle_mask.sum(dim=1) > base_lower_mask.sum(dim=1)
                metrics_dict["nci_middle_channels_per_sample"] = (
                    base_middle_mask.sum(dim=1).float().mean()
                )
                metrics_dict["nci_lower_channels_per_sample"] = (
                    base_lower_mask.sum(dim=1).float().mean()
                )
            else:
                base_subset_mask = _sample_channel_subset_mask(
                    base_full_mask,
                    min_channels=self.nci_min_channels,
                    max_channels=self.nci_max_channels,
                    require_omission=True,
                )
                nci_global_channel_valid_mask = base_subset_mask.repeat(n_global_crops, 1)
                nci_active_samples = base_full_mask.sum(dim=1) > base_subset_mask.sum(dim=1)
            metrics_dict["nci_full_channels_per_sample"] = base_full_mask.sum(dim=1).float().mean()
            metrics_dict["nci_subset_channels_per_sample"] = (
                nci_global_channel_valid_mask[:B].sum(dim=1).float().mean()
            )
        elif self.cmgi_enabled:
            if global_channel_valid_mask is None or local_channel_valid_mask is None:
                raise ValueError(
                    "conditional_morphology_graph.enabled=true requires channel ids/masks from a "
                    "packwds_chvit dataset"
                )
            base_full_mask = global_channel_valid_mask[:B]
            base_subset_mask = _sample_channel_subset_mask(
                base_full_mask,
                min_channels=self.cmgi_min_channels,
                max_channels=self.cmgi_max_channels,
                require_omission=True,
            )
            cmgi_global_channel_valid_mask = base_subset_mask.repeat(n_global_crops, 1)
            cmgi_active_samples = base_full_mask.sum(dim=1) > base_subset_mask.sum(dim=1)
            metrics_dict["cmgi_full_channels_per_sample"] = base_full_mask.sum(dim=1).float().mean()
            metrics_dict["cmgi_subset_channels_per_sample"] = base_subset_mask.sum(dim=1).float().mean()
            metrics_dict["cmgi_condition_source_teacher"] = float(self.cmgi_condition_source == "teacher")
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        if bool(getattr(self.cfg.student, "enable_channelvit", False)) or self._is_channelvit_backbone(
            self.student.backbone
        ):
            masks, mask_indices_list, masks_weight, n_masked_patches_tensor = self._expand_channelvit_masks(
                masks,
                n_channels=global_crops.shape[1],
                channel_valid_mask=student_global_channel_valid_mask,
            )

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
            gram_teacher_channel_ids = data.get("collated_gram_teacher_channel_ids")
            gram_teacher_channel_valid_mask = data.get("collated_gram_teacher_channel_valid_mask")
            if gram_teacher_channel_ids is not None:
                gram_teacher_channel_ids = gram_teacher_channel_ids.cuda(non_blocking=True)
            if gram_teacher_channel_valid_mask is not None:
                gram_teacher_channel_valid_mask = gram_teacher_channel_valid_mask.cuda(non_blocking=True)
        else:
            gram_teacher_crops = None
            gram_teacher_channel_ids = None
            gram_teacher_channel_valid_mask = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            channel_ids=global_channel_ids.unflatten(0, (n_global_crops, B))
            if global_channel_ids is not None
            else None,
            channel_valid_mask=global_channel_valid_mask.unflatten(0, (n_global_crops, B))
            if global_channel_valid_mask is not None
            else None,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Build the frozen local nuisance geometry before the student forward.
        # Gradient projection is an identity in the forward pass, so this does
        # not alter teacher targets or the current student activations.
        acq_anchor_features = None
        acq_orbit_features = None
        acq_tangent_basis = None
        acq_tangent_active = None
        acq_tangent_metrics: dict[str, Tensor] = {}
        acq_patch_tangent_basis = None
        acq_patch_tangent_active = None
        acq_patch_tangent_metrics: dict[str, Tensor] = {}
        acq_gradient_metrics: dict[str, Tensor] = {}
        anchor_channel_ids = None
        anchor_channel_mask = None
        if self.acq_deflation_enabled:
            anchor_images = global_crops.unflatten(0, (n_global_crops, B))[0]
            anchor_channel_ids = (
                global_channel_ids.unflatten(0, (n_global_crops, B))[0]
                if global_channel_ids is not None
                else None
            )
            anchor_channel_mask = (
                global_channel_valid_mask.unflatten(0, (n_global_crops, B))[0]
                if global_channel_valid_mask is not None
                else None
            )
            if self.acq_deflation_mode != "direct_consistency":
                use_patch_projection = (
                    self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"}
                    and self.acq_projection_scope == "cls_patch"
                )
                anchor_artifacts = self._get_acquisition_anchor_features(
                    images=anchor_images,
                    channel_ids=anchor_channel_ids,
                    channel_valid_mask=anchor_channel_mask,
                    return_patches=use_patch_projection,
                )
                if use_patch_projection:
                    (
                        acq_anchor_features,
                        acq_orbit_features,
                        acq_anchor_patches,
                        acq_orbit_patches,
                    ) = anchor_artifacts
                else:
                    acq_anchor_features, acq_orbit_features = anchor_artifacts
                if self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"}:
                    acq_tangent_basis, acq_tangent_active, acq_tangent_metrics = (
                        build_acquisition_tangent_basis(
                            anchor_features=acq_anchor_features,
                            perturbed_anchor_features=acq_orbit_features,
                            min_singular_value=float(self.cfg.acquisition_orbit_deflation.min_singular_value),
                            relative_singular_value=float(
                                self.cfg.acquisition_orbit_deflation.relative_singular_value
                            ),
                        )
                    )
                    if self.acq_deflation_mode == "random_gradient_projection":
                        acq_tangent_basis = rank_matched_random_tangent_basis(acq_tangent_basis)
                    if use_patch_projection:
                        acq_patch_tangent_basis, acq_patch_tangent_active, patch_metrics = (
                            build_acquisition_tangent_basis(
                                anchor_features=acq_anchor_patches.flatten(0, 1),
                                perturbed_anchor_features=acq_orbit_patches.permute(0, 2, 1, 3).flatten(0, 1),
                                min_singular_value=float(
                                    self.cfg.acquisition_orbit_deflation.min_singular_value
                                ),
                                relative_singular_value=float(
                                    self.cfg.acquisition_orbit_deflation.relative_singular_value
                                ),
                            )
                        )
                        acq_patch_tangent_metrics = {
                            f"acq_patch_{name[4:]}": value for name, value in patch_metrics.items()
                        }
                        if self.acq_deflation_mode == "random_gradient_projection":
                            acq_patch_tangent_basis = rank_matched_random_tangent_basis(
                                acq_patch_tangent_basis
                            )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            global_channel_ids=global_channel_ids.unflatten(0, (n_global_crops, B))
            if global_channel_ids is not None
            else None,
            local_channel_ids=local_channel_ids.unflatten(0, (n_local_crops, B))
            if local_channel_ids is not None
            else None,
            global_channel_valid_mask=student_global_channel_valid_mask.unflatten(0, (n_global_crops, B))
            if student_global_channel_valid_mask is not None
            else None,
            local_channel_valid_mask=student_local_channel_valid_mask.unflatten(0, (n_local_crops, B))
            if student_local_channel_valid_mask is not None
            else None,
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
            global_tangent_basis=acq_tangent_basis,
            global_tangent_active=acq_tangent_active,
            global_patch_tangent_basis=acq_patch_tangent_basis,
            global_patch_tangent_active=acq_patch_tangent_active,
            tangent_projection_strength=self.acq_projection_strength,
            tangent_gradient_metrics=acq_gradient_metrics,
        )

        if self.nci_enabled:
            nci_global_crops = global_crops.unflatten(0, (n_global_crops, B))
            nci_lower_cls = None
            nci_global_channel_ids = (
                global_channel_ids.unflatten(0, (n_global_crops, B))
                if global_channel_ids is not None
                else None
            )
            if self.nci_observation_protocol == "unmasked_shared":
                # The residual r_F = z_F - E[z_F | z_S] is a conditional
                # channel quantity, not an iBOT-mask quantity.  Keep the
                # normal masked main path intact, but make both auxiliary
                # observations unmasked and otherwise identical.
                nci_full_cls = self.get_nci_full_output(
                    global_crops=nci_global_crops,
                    global_channel_ids=nci_global_channel_ids,
                    global_channel_valid_mask=global_channel_valid_mask.unflatten(0, (n_global_crops, B)),
                )
                nci_subset_cls = self.get_nci_subset_output(
                    global_crops=nci_global_crops,
                    global_channel_ids=nci_global_channel_ids,
                    global_channel_valid_mask=nci_global_channel_valid_mask.unflatten(0, (n_global_crops, B)),
                    requires_grad=not self.nci_loss.stop_gradient,
                )
            elif self.nci_observation_protocol == "masked_shared":
                # Hold the exact iBOT observation fixed on both sides.  This
                # isolates channel-conditional innovation from the legacy
                # full-masked/subset-unmasked mismatch.
                nci_full_cls = student_global["cls_pre_head"]
                nci_subset_cls = self.get_nci_subset_output(
                    global_crops=nci_global_crops,
                    masks=masks,
                    global_channel_ids=nci_global_channel_ids,
                    global_channel_valid_mask=nci_global_channel_valid_mask.unflatten(0, (n_global_crops, B)),
                    # M must be differentiable so the lower martingale
                    # increment can shape the shared backbone. The F|M loss
                    # still detaches M, preserving the NCI firewall.
                    requires_grad=self.nci_martingale_enabled or not self.nci_loss.stop_gradient,
                    checkpoint_backbone=self.nci_martingale_enabled
                    and self.nci_martingale_checkpoint_middle_forward,
                )
                if self.nci_martingale_enabled:
                    if nci_lower_channel_valid_mask is None:
                        raise RuntimeError("Martingale NCI lower channel masks were not initialized")
                    nci_lower_cls = self.get_nci_subset_output(
                        global_crops=nci_global_crops,
                        masks=masks,
                        global_channel_ids=nci_global_channel_ids,
                        global_channel_valid_mask=nci_lower_channel_valid_mask.unflatten(
                            0, (n_global_crops, B)
                        ),
                        requires_grad=not self.nci_loss.stop_gradient,
                    )
            else:
                # Reproduce pre-correction screens only.  This mixes the
                # masked main-path feature with an unmasked subset feature.
                nci_full_cls = student_global["cls_pre_head"]
                nci_subset_cls = self.get_nci_subset_output(
                    global_crops=nci_global_crops,
                    global_channel_ids=nci_global_channel_ids,
                    global_channel_valid_mask=nci_global_channel_valid_mask.unflatten(0, (n_global_crops, B)),
                    requires_grad=True,
                )
        else:
            nci_full_cls = None
            nci_subset_cls = None
            nci_lower_cls = None

        if self.cmgi_enabled:
            cmgi_subset_patches = self.get_cmgi_subset_output(
                global_crops=global_crops.unflatten(0, (n_global_crops, B)),
                global_channel_ids=global_channel_ids.unflatten(0, (n_global_crops, B))
                if global_channel_ids is not None
                else None,
                global_channel_valid_mask=cmgi_global_channel_valid_mask.unflatten(0, (n_global_crops, B)),
            )
        else:
            cmgi_subset_patches = None

        if self.nri_enabled:
            nri_low_features = self.get_nri_low_resolution_output(
                global_crops=global_crops.unflatten(0, (n_global_crops, B)),
                masks=masks,
                global_channel_ids=global_channel_ids.unflatten(0, (n_global_crops, B))
                if global_channel_ids is not None
                else None,
                global_channel_valid_mask=global_channel_valid_mask.unflatten(0, (n_global_crops, B))
                if global_channel_valid_mask is not None
                else None,
            )
        else:
            nri_low_features = None

        if self.acq_deflation_enabled and self.acq_deflation_mode == "direct_consistency":
            acq_orbit_images = _make_acquisition_orbit_views(
                anchor_images,
                contrast_scale=self.acq_contrast_scale,
                background_scale=self.acq_background_scale,
                blur_mix=self.acq_blur_mix,
                num_perturbations=self.acq_num_perturbations,
            )
            acq_orbit_features = self._get_acquisition_student_features(
                orbit_images=acq_orbit_images,
                channel_ids=anchor_channel_ids,
                channel_valid_mask=anchor_channel_mask,
            )

        if self.scout_transport_enabled:
            scout_global_crops = global_crops.unflatten(0, (n_global_crops, B))
            scout_global_masks = masks.unflatten(0, (n_global_crops, B))
            scout_global_channel_ids = (
                global_channel_ids.unflatten(0, (n_global_crops, B))
                if global_channel_ids is not None
                else None
            )
            scout_global_channel_mask = (
                global_channel_valid_mask.unflatten(0, (n_global_crops, B))
                if global_channel_valid_mask is not None
                else None
            )
            scout_first_mask = (
                scout_global_masks[0]
                if self.scout_transport_current_feature_protocol == "mask_matched"
                else None
            )
            (
                scout_large_anchor_features,
                scout_delta,
                scout_final_kernel,
            ) = self._get_scout_transport_artifacts(
                images=scout_global_crops[0],
                masks=scout_first_mask,
                channel_ids=scout_global_channel_ids[0] if scout_global_channel_ids is not None else None,
                channel_valid_mask=scout_global_channel_mask[0]
                if scout_global_channel_mask is not None
                else None,
            )
            if self.scout_transport_target_mode == "final_kernel":
                scout_target = scout_final_kernel
                scout_stability_metrics = {}
            elif self.scout_transport_target_mode in {"stable_delta", "shuffled_stable_delta"}:
                _, scout_second_delta, _ = self._get_scout_transport_artifacts(
                    images=scout_global_crops[1],
                    masks=scout_global_masks[1]
                    if self.scout_transport_current_feature_protocol == "mask_matched"
                    else None,
                    channel_ids=scout_global_channel_ids[1] if scout_global_channel_ids is not None else None,
                    channel_valid_mask=scout_global_channel_mask[1]
                    if scout_global_channel_mask is not None
                    else None,
                    include_large_anchor=False,
                )
                scout_target, scout_stability_metrics = cross_view_stable_kernel_delta(
                    scout_delta,
                    scout_second_delta,
                    relative_eigenvalue=self.scout_stable_relative_eigenvalue,
                    min_eigenvalue=self.scout_stable_min_eigenvalue,
                )
            else:
                scout_target = scout_delta
                scout_stability_metrics = {}
            if self.scout_transport_target_mode in {"shuffled_delta", "shuffled_stable_delta"}:
                # Preserve target energy and spectrum while breaking the
                # sample correspondence. This isolates the proposed scout
                # relation direction from generic auxiliary-loss strength.
                shuffle_generator = torch.Generator(device=scout_target.device)
                shuffle_generator.manual_seed(17_291 + self._scout_shuffled_target_step)
                self._scout_shuffled_target_step += 1
                permutation = torch.randperm(
                    B,
                    device=scout_target.device,
                    generator=shuffle_generator,
                )
                scout_target = scout_target[permutation][:, permutation]
        else:
            scout_large_anchor_features = None
            scout_target = None
            scout_stability_metrics = {}

        # Gram output
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                channel_ids=gram_teacher_channel_ids.unflatten(0, (n_global_crops, B))
                if gram_teacher_channel_ids is not None
                else None,
                channel_valid_mask=gram_teacher_channel_valid_mask.unflatten(0, (n_global_crops, B))
                if gram_teacher_channel_valid_mask is not None
                else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )
        if self.nci_enabled:
            if nci_full_cls is None or nci_subset_cls is None:
                raise RuntimeError("Nested channel innovation features were not initialized")
            nci_loss, nci_metrics = self.nci_loss(
                full_features=nci_full_cls,
                subset_features=nci_subset_cls,
                active_samples=nci_active_samples,
                predictor=self.student.nci_predictor,
            )
            if self.nci_martingale_enabled:
                if nci_lower_cls is None or nci_lower_active_samples is None:
                    raise RuntimeError("Martingale NCI features were not initialized")
                nci_lower_loss, nci_lower_metrics = self.nci_mid_loss(
                    full_features=nci_subset_cls,
                    subset_features=nci_lower_cls,
                    active_samples=nci_lower_active_samples,
                    predictor=self.student.nci_mid_predictor,
                )
                upper_increment = conditional_innovation_residual(
                    full_features=nci_full_cls,
                    subset_features=nci_subset_cls,
                    predictor=self.student.nci_predictor,
                    stop_gradient=self.nci_loss.stop_gradient,
                )
                lower_increment = conditional_innovation_residual(
                    full_features=nci_subset_cls,
                    subset_features=nci_lower_cls,
                    predictor=self.student.nci_mid_predictor,
                    stop_gradient=self.nci_mid_loss.stop_gradient,
                )
                nci_cross_loss, nci_cross_metrics = martingale_increment_orthogonality(
                    upper_increment=upper_increment,
                    lower_increment=lower_increment,
                    active_samples=nci_lower_active_samples,
                )
                # Average the two same-scale conditional objectives so MCI is
                # not merely a stronger auxiliary-loss baseline than NCI.
                nci_loss = (
                    nci_loss + self.nci_martingale_lower_loss_weight * nci_lower_loss
                ) / (1.0 + self.nci_martingale_lower_loss_weight)
                nci_loss = nci_loss + self.nci_martingale_cross_orthogonality_weight * nci_cross_loss
                loss_dict["nci_martingale_lower_loss"] = nci_lower_loss.detach()
                loss_dict["nci_martingale_lower_loss_weight"] = self.nci_martingale_lower_loss_weight
                loss_dict["nci_martingale_cross_loss_weight"] = (
                    self.nci_martingale_cross_orthogonality_weight
                )
                loss_dict.update(nci_lower_metrics)
                loss_dict.update(nci_cross_metrics)
            loss_accumulator += self.nci_loss_weight * nci_loss
            loss_dict["nci_loss"] = nci_loss.detach()
            loss_dict["nci_loss_weight"] = self.nci_loss_weight
            loss_dict["nci_martingale_enabled"] = float(self.nci_martingale_enabled)
            loss_dict["nci_unmasked_shared_observation"] = float(
                self.nci_observation_protocol == "unmasked_shared"
            )
            loss_dict["nci_masked_shared_observation"] = float(
                self.nci_observation_protocol == "masked_shared"
            )
            loss_dict.update(nci_metrics)
        if self.cmgi_enabled:
            cmgi_loss, cmgi_metrics = self.cmgi_loss(
                full_features=student_global["patch_pre_head"],
                teacher_features=teacher_global["patch_pre_head"],
                subset_features=cmgi_subset_patches,
                active_samples=cmgi_active_samples,
                predictor=self.student.cmgi_predictor,
                masks=masks.unflatten(0, (n_global_crops, B)),
            )
            loss_accumulator += self.cmgi_loss_weight * cmgi_loss
            loss_dict["cmgi_loss"] = cmgi_loss.detach()
            loss_dict["cmgi_loss_weight"] = self.cmgi_loss_weight
            loss_dict.update(cmgi_metrics)
        if self.nri_enabled:
            nri_full_features = self._select_nri_features(
                student_global["cls_pre_head"],
                student_global["patch_pre_head"],
            )
            nri_loss, nri_metrics = self.nri_loss(
                full_features=nri_full_features,
                subset_features=nri_low_features,
                active_samples=torch.ones(B, dtype=torch.bool, device=nri_full_features.device),
                predictor=self.student.nri_predictor,
            )
            loss_accumulator += self.nri_loss_weight * nri_loss
            loss_dict["nri_loss"] = nri_loss.detach()
            loss_dict["nri_loss_weight"] = self.nri_loss_weight
            loss_dict.update(nri_metrics)
        if self.acq_deflation_enabled:
            current_features = student_global["cls_pre_head"][0]
            if self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"}:
                if acq_tangent_basis is None or acq_tangent_active is None or acq_anchor_features is None:
                    raise RuntimeError("Acquisition tangent projection was not initialized")
                acq_loss = current_features.new_zeros(())
                acq_metrics = {
                    **acq_tangent_metrics,
                    **acq_patch_tangent_metrics,
                    "acq_projection_prehead_tangent_fraction": acquisition_tangent_fraction(
                        current_features.detach() - acq_anchor_features,
                        tangent_basis=acq_tangent_basis,
                        active_rows=acq_tangent_active,
                    ).detach(),
                    "acq_gradient_projection": current_features.new_tensor(1.0),
                    "acq_projection_strength": current_features.new_tensor(self.acq_projection_strength),
                }
            elif self.acq_deflation_mode == "direct_consistency":
                positive = F.cosine_similarity(
                    current_features.unsqueeze(1), acq_orbit_features, dim=-1
                )
                acq_loss = (1.0 - positive).mean()
                acq_metrics = {
                    "acq_direct_consistency": acq_loss.detach(),
                    "acq_direct_positive_cosine": positive.mean().detach(),
                }
            else:
                if self.acq_deflation_mode == "random_tangent":
                    acq_orbit_features = acq_anchor_features.unsqueeze(1) + torch.randn_like(acq_orbit_features)
                if self.acq_deflation_loss is None:
                    raise RuntimeError("Acquisition orbit deflation loss was not initialized")
                acq_loss, _, acq_metrics = self.acq_deflation_loss(
                    current_features=current_features,
                    anchor_features=acq_anchor_features,
                    perturbed_anchor_features=acq_orbit_features,
                )
            if self.acq_deflation_mode not in {"gradient_projection", "random_gradient_projection"}:
                loss_accumulator += self.acq_deflation_loss_weight * acq_loss
            loss_dict["acq_deflation_loss"] = acq_loss.detach()
            loss_dict["acq_deflation_loss_weight"] = (
                0.0
                if self.acq_deflation_mode in {"gradient_projection", "random_gradient_projection"}
                else self.acq_deflation_loss_weight
            )
            loss_dict["acq_random_tangent"] = float(
                self.acq_deflation_mode in {"random_tangent", "random_gradient_projection"}
            )
            loss_dict.update(acq_metrics)
        if self.scout_transport_enabled:
            if scout_large_anchor_features is None or scout_target is None:
                raise RuntimeError("Scout transport artifacts were not initialized")
            scout_current_features = student_global["cls_pre_head"][0]
            if self.scout_transport_current_feature_protocol == "anchor_consistent":
                # The frozen L anchor saw an unmasked global crop. Re-evaluate
                # the trainable L on that exact observation so its kernel delta
                # represents adaptation rather than the fixed iBOT mask effect.
                scout_current_features = self.student.backbone(
                    scout_global_crops[0],
                    channel_ids=scout_global_channel_ids[0]
                    if scout_global_channel_ids is not None
                    else None,
                    channel_valid_mask=scout_global_channel_mask[0]
                    if scout_global_channel_mask is not None
                    else None,
                    is_training=True,
                )["x_norm_clstoken"]
            scout_loss, scout_metrics = self.scout_transport_loss(
                current_features=scout_current_features,
                anchor_features=scout_large_anchor_features,
                scout_delta=scout_target,
            )
            loss_accumulator += self.scout_transport_loss_weight * scout_loss
            loss_dict["scout_kernel_transport_loss"] = scout_loss.detach()
            loss_dict["scout_kernel_transport_loss_weight"] = self.scout_transport_loss_weight
            loss_dict["scout_kernel_transport_displacement_budget_ratio"] = (
                self.scout_transport_displacement_budget_ratio
            )
            loss_dict["scout_kernel_transport_is_delta"] = float(
                self.scout_transport_target_mode
                in {"delta", "stable_delta", "shuffled_delta", "shuffled_stable_delta"}
            )
            loss_dict["scout_kernel_transport_anchor_consistent"] = float(
                self.scout_transport_current_feature_protocol in {"anchor_consistent", "mask_matched"}
            )
            loss_dict["scout_kernel_transport_mask_matched"] = float(
                self.scout_transport_current_feature_protocol == "mask_matched"
            )
            loss_dict.update(scout_metrics)
            loss_dict.update(scout_stability_metrics)

        scaled_loss = loss_accumulator / float(loss_divisor)
        self.backprop_loss(scaled_loss)
        if acq_gradient_metrics:
            loss_dict.update(acq_gradient_metrics)

        # Log loss finite check at iteration 0 to catch degenerate initialization early
        if iteration == 0 and distributed.is_main_process():
            try:
                loss_val = loss_accumulator.item()
            except Exception:
                loss_val = float(loss_accumulator)
            import math
            if math.isfinite(loss_val):
                logger.info(f"[LOSS CHECK] iter 0 loss = {loss_val:.4f} (finite OK)")
            else:
                logger.warning(f"[LOSS CHECK] iter 0 loss = {loss_val} — NON-FINITE, check init")

        # Return unscaled loss for logging; backward used scaled_loss
        return loss_accumulator.detach(), metrics_dict | loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
        channel_ids=None,
        channel_valid_mask=None,
    ):
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)
        if channel_ids is not None:
            channel_ids = channel_ids.flatten(0, 1)
        if channel_valid_mask is not None:
            channel_valid_mask = channel_valid_mask.flatten(0, 1)

        backbone_out = self.teacher.backbone(
            images,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
            is_training=True,
        )
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(
        self,
        images,
        *,
        masks,
        teacher_global,
        student_global,
        student_global_crops_size,
        channel_ids=None,
        channel_valid_mask=None,
    ):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]
            if channel_ids is not None:
                channel_ids = channel_ids.flatten(0, 1)
            if channel_valid_mask is not None:
                channel_valid_mask = channel_valid_mask.flatten(0, 1)

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(
                    images,
                    channel_ids=channel_ids,
                    channel_valid_mask=channel_valid_mask,
                    is_training=True,
                )
            teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.cfg.student.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.cfg.student.patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(
        self,
        *,
        global_crops,
        local_crops,
        upperbound,
        masks,
        mask_indices_list,
        global_channel_ids=None,
        local_channel_ids=None,
        global_channel_valid_mask=None,
        local_channel_valid_mask=None,
        global_tangent_basis: Tensor | None = None,
        global_tangent_active: Tensor | None = None,
        global_patch_tangent_basis: Tensor | None = None,
        global_patch_tangent_active: Tensor | None = None,
        tangent_projection_strength: float = 1.0,
        tangent_gradient_metrics: dict[str, Tensor] | None = None,
    ):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)
        if global_channel_ids is not None:
            global_channel_ids = global_channel_ids.flatten(0, 1)
        if local_channel_ids is not None:
            local_channel_ids = local_channel_ids.flatten(0, 1)
        if global_channel_valid_mask is not None:
            global_channel_valid_mask = global_channel_valid_mask.flatten(0, 1)
        if local_channel_valid_mask is not None:
            local_channel_valid_mask = local_channel_valid_mask.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            channel_ids=[global_channel_ids, local_channel_ids],
            channel_valid_mask=[global_channel_valid_mask, local_channel_valid_mask],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        if global_tangent_basis is not None or global_tangent_active is not None:
            if global_tangent_basis is None or global_tangent_active is None:
                raise ValueError("global_tangent_basis and global_tangent_active must be provided together")
            if global_tangent_basis.shape[0] != B:
                raise ValueError(
                    "The acquisition tangent must describe the first global crop with batch size "
                    f"{B}, got {tuple(global_tangent_basis.shape)}"
                )
            first_global_cls = apply_acquisition_tangent_gradient_projection(
                g_cls[:B],
                tangent_basis=global_tangent_basis,
                active_rows=global_tangent_active,
                strength=tangent_projection_strength,
            )
            if tangent_gradient_metrics is not None:
                def _record_tangent_gradient(gradient: Tensor) -> Tensor:
                    with torch.no_grad():
                        tangent_part = project_onto_acquisition_tangent(
                            gradient,
                            tangent_basis=global_tangent_basis,
                            active_rows=global_tangent_active,
                        )
                        filtered = gradient - tangent_projection_strength * tangent_part
                        gradient_energy = gradient.float().square().sum(dim=-1).mean()
                        tangent_gradient_metrics["acq_gradient_tangent_fraction_before"] = (
                            tangent_part.float().square().sum(dim=-1).mean()
                            / gradient_energy.clamp_min(1.0e-12)
                        ).detach()
                        tangent_gradient_metrics["acq_gradient_tangent_fraction_after"] = (
                            acquisition_tangent_fraction(
                                filtered,
                                tangent_basis=global_tangent_basis,
                                active_rows=global_tangent_active,
                            ).detach()
                        )
                        tangent_gradient_metrics["acq_gradient_removed_energy_fraction"] = (
                            (tangent_projection_strength * tangent_part).float().square().sum(dim=-1).mean()
                            / gradient_energy.clamp_min(1.0e-12)
                        ).detach()
                    return gradient

                first_global_cls.register_hook(_record_tangent_gradient)
            # The first global crop is the calibrated physical view. Other
            # DINO crops retain their ordinary gradients as a stable control.
            g_cls = torch.cat((first_global_cls, g_cls[B:]), dim=0)

        if global_patch_tangent_basis is not None or global_patch_tangent_active is not None:
            if global_patch_tangent_basis is None or global_patch_tangent_active is None:
                raise ValueError(
                    "global_patch_tangent_basis and global_patch_tangent_active must be provided together"
                )
            patch_count = g_patch.shape[1]
            if global_patch_tangent_basis.shape[0] != B * patch_count:
                raise ValueError(
                    "The dense acquisition tangent must describe B * P tokens from the first global crop, "
                    f"got {tuple(global_patch_tangent_basis.shape)} for B={B}, P={patch_count}"
                )
            first_global_patches = apply_acquisition_tangent_gradient_projection(
                g_patch[:B].flatten(0, 1),
                tangent_basis=global_patch_tangent_basis,
                active_rows=global_patch_tangent_active,
                strength=tangent_projection_strength,
            )
            if tangent_gradient_metrics is not None:
                def _record_patch_tangent_gradient(gradient: Tensor) -> Tensor:
                    with torch.no_grad():
                        tangent_part = project_onto_acquisition_tangent(
                            gradient,
                            tangent_basis=global_patch_tangent_basis,
                            active_rows=global_patch_tangent_active,
                        )
                        filtered = gradient - tangent_projection_strength * tangent_part
                        gradient_energy = gradient.float().square().sum(dim=-1).mean()
                        tangent_gradient_metrics["acq_patch_gradient_tangent_fraction_before"] = (
                            tangent_part.float().square().sum(dim=-1).mean()
                            / gradient_energy.clamp_min(1.0e-12)
                        ).detach()
                        tangent_gradient_metrics["acq_patch_gradient_tangent_fraction_after"] = (
                            acquisition_tangent_fraction(
                                filtered,
                                tangent_basis=global_patch_tangent_basis,
                                active_rows=global_patch_tangent_active,
                            ).detach()
                        )
                        tangent_gradient_metrics["acq_patch_gradient_removed_energy_fraction"] = (
                            (tangent_projection_strength * tangent_part).float().square().sum(dim=-1).mean()
                            / gradient_energy.clamp_min(1.0e-12)
                        ).detach()
                    return gradient

                first_global_patches.register_hook(_record_patch_tangent_gradient)
            g_patch = torch.cat(
                (first_global_patches.unflatten(0, (B, patch_count)), g_patch[B:]),
                dim=0,
            )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        bottleneck = None
        if self.sigreg_enabled and self.sigreg_mode == "bottleneck":
            buffer, bottleneck = self.student.dino_head(buffer, return_bottleneck=True)
            bottleneck = torch.split_with_sizes(bottleneck, sizes, dim=0)
        else:
            buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        if bottleneck is not None:
            global_out["bottleneck_pre_norm"] = bottleneck[0].unflatten(0, [n_global_crops, B])
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def _get_nci_channel_output(
        self,
        *,
        global_crops,
        masks: Tensor | None = None,
        global_channel_ids=None,
        global_channel_valid_mask,
        requires_grad: bool,
        checkpoint_backbone: bool = False,
    ) -> Tensor:
        """Encode an unmasked full/subset channel view for conditional NCI."""
        n_global_crops, batch_size, _, _, _ = global_crops.shape
        images = global_crops.flatten(0, 1)
        channel_ids = global_channel_ids.flatten(0, 1) if global_channel_ids is not None else None
        channel_valid_mask = global_channel_valid_mask.flatten(0, 1)
        if checkpoint_backbone:
            if not requires_grad:
                raise ValueError("NCI activation checkpointing requires a differentiable subset forward")

            def encode(images: Tensor) -> Tensor:
                backbone_out = self.student.backbone(
                    images,
                    masks=masks,
                    channel_ids=channel_ids,
                    channel_valid_mask=channel_valid_mask,
                    is_training=True,
                )
                return backbone_out["x_norm_clstoken"]

            # Non-reentrant checkpointing retains parameter gradients even when
            # the image tensor itself is not a differentiation target.
            cls_tokens = activation_checkpoint(encode, images, use_reentrant=False)
        else:
            grad_context = nullcontext() if requires_grad else torch.no_grad()
            with grad_context:
                backbone_out = self.student.backbone(
                    images,
                    masks=masks,
                    channel_ids=channel_ids,
                    channel_valid_mask=channel_valid_mask,
                    is_training=True,
                )
            cls_tokens = backbone_out["x_norm_clstoken"]
        return cls_tokens.unflatten(0, (n_global_crops, batch_size))

    def get_nci_full_output(
        self,
        *,
        global_crops,
        global_channel_ids=None,
        global_channel_valid_mask,
    ) -> Tensor:
        """Return the differentiable full-channel side of the shared observation."""
        return self._get_nci_channel_output(
            global_crops=global_crops,
            global_channel_ids=global_channel_ids,
            global_channel_valid_mask=global_channel_valid_mask,
            requires_grad=True,
            checkpoint_backbone=self.nci_checkpoint_full_forward,
        )

    def get_nci_subset_output(
        self,
        *,
        global_crops,
        masks: Tensor | None = None,
        global_channel_ids=None,
        global_channel_valid_mask,
        requires_grad: bool,
        checkpoint_backbone: bool | None = None,
    ) -> Tensor:
        """Return the subset side, optionally behind the stop-gradient firewall."""
        return self._get_nci_channel_output(
            global_crops=global_crops,
            masks=masks,
            global_channel_ids=global_channel_ids,
            global_channel_valid_mask=global_channel_valid_mask,
            requires_grad=requires_grad,
            checkpoint_backbone=requires_grad
            and (
                self.nci_checkpoint_subset_forward
                if checkpoint_backbone is None
                else checkpoint_backbone
            ),
        )

    def get_cmgi_subset_output(
        self,
        *,
        global_crops,
        global_channel_ids=None,
        global_channel_valid_mask,
    ) -> Tensor:
        """Encode S in the same feature space used to define CMGI innovation."""
        n_global_crops, batch_size, _, _, _ = global_crops.shape
        images = global_crops.flatten(0, 1)
        channel_ids = global_channel_ids.flatten(0, 1) if global_channel_ids is not None else None
        channel_valid_mask = global_channel_valid_mask.flatten(0, 1)
        # The population target is E[T(C) | T(S)].  Reading S from the EMA
        # teacher therefore keeps the gate's conditional quantity stable as
        # the full student adapts. The student option remains useful for an
        # explicit ablation of this design choice.
        if self.cmgi_condition_source == "teacher":
            source_backbone = self.teacher.backbone
            grad_context = torch.no_grad()
        else:
            source_backbone = self.student.backbone
            grad_context = torch.no_grad() if self.cmgi_loss.stop_gradient else nullcontext()
        with grad_context:
            backbone_out = source_backbone(
                images,
                channel_ids=channel_ids,
                channel_valid_mask=channel_valid_mask,
                is_training=True,
            )
        return backbone_out["x_norm_patchtokens"].unflatten(0, (n_global_crops, batch_size))

    def _select_nri_features(self, cls_features: Tensor, patch_features: Tensor) -> Tensor:
        if self.nri_feature_mode == "cls":
            return cls_features
        return 0.5 * (cls_features + patch_features.mean(dim=-2))

    def get_nri_low_resolution_output(
        self,
        *,
        global_crops,
        masks,
        global_channel_ids=None,
        global_channel_valid_mask=None,
    ) -> Tensor:
        """Encode a nested low-pass view without exposing predictor gradients."""
        n_global_crops, batch_size, _, _, _ = global_crops.shape
        images = _make_low_resolution_observation(
            global_crops.flatten(0, 1),
            self.nri_downsample_factor,
        )
        channel_ids = global_channel_ids.flatten(0, 1) if global_channel_ids is not None else None
        channel_valid_mask = (
            global_channel_valid_mask.flatten(0, 1) if global_channel_valid_mask is not None else None
        )
        grad_context = torch.no_grad() if self.nri_loss.stop_gradient else nullcontext()
        with grad_context:
            backbone_out = self.student.backbone(
                images,
                masks=masks,
                channel_ids=channel_ids,
                channel_valid_mask=channel_valid_mask,
                is_training=True,
            )
        cls_features = backbone_out["x_norm_clstoken"].unflatten(0, (n_global_crops, batch_size))
        patch_features = backbone_out["x_norm_patchtokens"].unflatten(0, (n_global_crops, batch_size))
        return self._select_nri_features(cls_features, patch_features)

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # SIGReg replaces KoLeo by default, following the official FINO recipe.
        if self.sigreg_enabled:
            sigreg_features = student_global["bottleneck_pre_norm"]
            sigreg_loss = sum(self.sigreg_loss(x, seed_step=iteration) for x in sigreg_features) / n_global_crops
            sigreg_loss_weight = self.get_sigreg_loss_weight(iteration)
            loss_dict["sigreg_loss"] = sigreg_loss
            loss_dict["sigreg_loss_weight"] = sigreg_loss_weight
            loss_accumulator += sigreg_loss_weight * koleo_scale * sigreg_loss

            if self.sigreg_koleo_too:
                koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
                loss_dict["koleo_loss"] = koleo_loss
                loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss
        else:
            koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
            loss_dict["koleo_loss"] = koleo_loss
            loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    def get_sigreg_loss_weight(self, iteration: int) -> float:
        if not self.sigreg_weight_schedule_enabled:
            return float(self.sigreg_loss_weight)

        start = self.sigreg_weight_schedule_start
        end = self.sigreg_weight_schedule_end
        final = self.sigreg_weight_schedule_final
        if iteration < start:
            return float(self.sigreg_loss_weight)
        if self.sigreg_weight_schedule_type == "step" or iteration >= end or end == start:
            return float(final)

        progress = (iteration - start) / (end - start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(final + (self.sigreg_loss_weight - final) * cosine)

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self):
        super().train()
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg):
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
            float_input=getattr(cfg.crops, "float_input", False),
            augmentation_policy=getattr(cfg.crops, "augmentation_policy", "dinov3"),
            paired_global_geometry=getattr(cfg.crops, "paired_global_geometry", False),
        )

    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            params_groups = list(self.get_maybe_fused_params_for_submodel(m))
            for group in params_groups:
                group["is_backbone"] = name == "backbone"
                if name in {"nci_predictor", "nci_mid_predictor"}:
                    group["lr_multiplier"] *= self.nci_predictor_lr_multiplier
                elif name == "cmgi_predictor":
                    group["lr_multiplier"] *= self.cmgi_predictor_lr_multiplier
                elif name == "nri_predictor":
                    group["lr_multiplier"] *= self.nri_predictor_lr_multiplier
            all_params_groups += params_groups
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        if self.distributed_mode == "ddp":
            # Keep checkpoint keys plain until initialization has loaded the
            # released backbone; DDP wrapping happens immediately afterward.
            dtype_by_name = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            param_dtype = dtype_by_name[self.cfg.compute_precision.param_dtype]
            for model in (self.student, self.model_ema, getattr(self, "gram_teacher", None)):
                if model is not None:
                    model.to_empty(device="cuda")
                    model.to(dtype=param_dtype)
            logger.info("DISTRIBUTED DDP -- materialized full model replicas")
            return

        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
            inference_only_models_process_groups.append(default_process_group)
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        # Keep DDP usable on legacy PyTorch installations that predate FSDP2.
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize

        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def finish_distributed_training_setup(self) -> None:
        """Wrap trainable submodules after plain-checkpoint initialization."""
        if self.distributed_mode != "ddp" or self._ddp_wrapped:
            return
        device_id = torch.cuda.current_device()
        for name, module in list(self.student.items()):
            self.student[name] = DistributedDataParallel(
                module,
                device_ids=[device_id],
                output_device=device_id,
                broadcast_buffers=False,
                # Only conditional innovation predictors can be inactive on a
                # rank. Avoid DDP's graph traversal for ordinary full-model,
                # DBI, and Scout runs.
                find_unused_parameters=bool(self.nci_enabled or self.cmgi_enabled or self.nri_enabled),
                gradient_as_bucket_view=True,
            )
        self._ddp_wrapped = True
        logger.info("DISTRIBUTED DDP -- wrapped %d trainable student modules", len(self.student))

    def broadcast_to_subgroups(self, tensor, over_dim, global_batch_size=None):
        """
        This is an operation that takes a tensor from the default process group, gathers it, stacks it, then scatters it within a smaller process subgroup
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
