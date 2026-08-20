# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    register_dont_save_hooks,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.data import (
    DeterministicDataStream,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.ssl_meta_arch_lora import SSLMetaArchLoRA

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

logger = logging.getLogger("dinov3")


def _summarize_nan_tensor(name, tensor, sample_keys=None, max_extreme=8):
    if not torch.is_tensor(tensor):
        return f"{name}: not a tensor"

    with torch.no_grad():
        t = tensor.detach()
        finite = torch.isfinite(t)
        finite_count = int(finite.sum().item())
        total_count = t.numel()
        parts = [
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite={finite_count}/{total_count}"
        ]
        if finite_count:
            vals = t[finite].float()
            parts.append(
                "min={:.6g} max={:.6g} mean={:.6g} std={:.6g}".format(
                    float(vals.min().item()),
                    float(vals.max().item()),
                    float(vals.mean().item()),
                    float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
                )
            )
        else:
            parts.append("all values are non-finite")

        if sample_keys:
            batch_size = len(sample_keys)
            if t.ndim >= 4 and batch_size > 0 and t.shape[0] % batch_size == 0:
                n_crops = t.shape[0] // batch_size
                per_sample = t.reshape(n_crops, batch_size, *t.shape[1:])
                per_sample_flat = per_sample.flatten(2)
                per_sample_finite = torch.isfinite(per_sample_flat).all(dim=2).all(dim=0)
                per_sample_abs = torch.nan_to_num(
                    per_sample_flat.float().abs(),
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=float("inf"),
                ).amax(dim=2).amax(dim=0)
                topk = min(max_extreme, batch_size)
                top_indices = torch.topk(per_sample_abs, k=topk).indices.cpu().tolist()
                extreme = []
                for idx in top_indices:
                    key = sample_keys[idx] if idx < len(sample_keys) else str(idx)
                    extreme.append(
                        f"{idx}:{float(per_sample_abs[idx].item()):.6g}:"
                        f"{'finite' if bool(per_sample_finite[idx].item()) else 'NONFINITE'}:{key}"
                    )
                parts.append(f"top_abs_by_sample={extreme}")
    return " ".join(parts)


def _log_nan_debug_batch(data, logger):
    sample_keys = data.get("sample_keys") or []
    sample_channel_ids = data.get("sample_channel_ids")
    if sample_keys:
        logger.warning("NaN debug sample_keys (%d): %s", len(sample_keys), sample_keys)
    if sample_channel_ids is not None:
        logger.warning("NaN debug sample_channel_ids: %s", sample_channel_ids)
    for name in ("collated_global_crops", "collated_local_crops", "collated_gram_teacher_crops"):
        if name in data:
            logger.warning("NaN debug %s", _summarize_nan_tensor(name, data[name], sample_keys=sample_keys))


def _to_jsonable_scalar(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 1:
            return value.item()
        return value.cpu().tolist()
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_int_list(value):
    value = _to_jsonable_scalar(value)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _estimate_patch_tokens_per_image(cfg):
    try:
        patch_size = int(cfg.student.patch_size)
        global_crop_sizes = _as_int_list(cfg.crops.global_crops_size)
        local_crop_sizes = _as_int_list(cfg.crops.local_crops_size)
        if len(global_crop_sizes) != 1 or len(local_crop_sizes) != 1:
            return None
        global_tokens = 2 * (global_crop_sizes[0] / patch_size) ** 2
        local_tokens = int(cfg.crops.local_crops_number) * (local_crop_sizes[0] / patch_size) ** 2
        tokens = global_tokens + local_tokens
        return int(tokens) if float(tokens).is_integer() else tokens
    except Exception:
        return None


def _build_raw_loss_static_fields(cfg, accum_steps, real_global_batch_size, effective_global_batch_size):
    dataset_path = str(cfg.train.dataset_path)
    patch_tokens_per_image = _estimate_patch_tokens_per_image(cfg)
    fields = {
        "output_dir": str(cfg.train.output_dir),
        "dataset_path": dataset_path,
        "dataset_type": dataset_path.split(":", 1)[0],
        "augmentation_policy": str(getattr(cfg.crops, "augmentation_policy", "dinov3")),
        "decoder_rgb_mean": _to_jsonable_scalar(cfg.crops.rgb_mean),
        "decoder_rgb_std": _to_jsonable_scalar(cfg.crops.rgb_std),
        "arch": str(cfg.student.arch),
        "patch_size": int(cfg.student.patch_size),
        "global_crops_size": _to_jsonable_scalar(cfg.crops.global_crops_size),
        "local_crops_size": _to_jsonable_scalar(cfg.crops.local_crops_size),
        "local_crops_number": int(cfg.crops.local_crops_number),
        "accum_steps": int(accum_steps),
        "real_global_batch_size": int(real_global_batch_size),
        "effective_global_batch_size": int(effective_global_batch_size),
        "freeze_backbone_updates": int(getattr(cfg.optim, "freeze_backbone_updates", 0)),
        "trainable_last_blocks": int(getattr(cfg.optim, "trainable_last_blocks", -1)),
        "trainable_extra_stem": bool(getattr(cfg.optim, "trainable_extra_stem", False)),
        "channel_subset_enabled": bool(getattr(cfg.channel_subset, "enabled", False)),
        "channel_subset_min": int(getattr(cfg.channel_subset, "min_channels", 1)),
        "channel_subset_max": int(getattr(cfg.channel_subset, "max_channels", 3)),
        "official_epoch_length": int(cfg.train.OFFICIAL_EPOCH_LENGTH),
        "patch_tokens_per_image_estimate": patch_tokens_per_image,
        "controlled_data_stream": bool(getattr(cfg.train, "wds_deterministic_resampling", False)),
    }
    return fields


def _batch_sample_key_digest(data) -> str | None:
    """Compact audit trail proving compared arms consumed the same samples."""
    sample_keys = data.get("sample_keys")
    if not sample_keys:
        return None
    payload = "\n".join(map(str, sample_keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _append_raw_loss_metrics(
    raw_metrics_file,
    static_fields,
    cfg,
    iteration,
    micro_steps_completed_in_run,
    lr,
    wd,
    mom,
    last_layer_lr,
    backbone_lr,
    total_loss,
    metrics_dict,
    batch_sample_key_digest=None,
):
    if not distributed.is_main_process():
        return

    optimizer_updates_completed = int(iteration + 1)
    effective_global_batch_size = int(static_fields["effective_global_batch_size"])
    samples_seen = optimizer_updates_completed * effective_global_batch_size
    tokens_per_image = static_fields.get("patch_tokens_per_image_estimate")
    tokens_seen = samples_seen * tokens_per_image if tokens_per_image is not None else None
    official_epoch_length = int(cfg.train.OFFICIAL_EPOCH_LENGTH)

    row = dict(static_fields)
    row.update(
        {
            "time_unix": time.time(),
            "optimizer_update": int(iteration),
            "optimizer_updates_completed": optimizer_updates_completed,
            "micro_steps_completed_in_run": int(micro_steps_completed_in_run),
            "micro_steps_completed_estimate": optimizer_updates_completed * int(static_fields["accum_steps"]),
            "epoch_float": optimizer_updates_completed / official_epoch_length,
            "epoch_index": int(iteration // official_epoch_length),
            "samples_seen": samples_seen,
            "image_visits": samples_seen,
            "patch_tokens_seen_estimate": tokens_seen,
            "lr": _to_jsonable_scalar(lr),
            "wd": _to_jsonable_scalar(wd),
            "mom": _to_jsonable_scalar(mom),
            "last_layer_lr": _to_jsonable_scalar(last_layer_lr),
            "backbone_lr": _to_jsonable_scalar(backbone_lr),
            "total_loss": _to_jsonable_scalar(total_loss),
            "batch_sample_key_digest": batch_sample_key_digest,
        }
    )
    row.update({k: _to_jsonable_scalar(v) for k, v in metrics_dict.items()})

    with open(raw_metrics_file, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_dino",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules (use effective global batch including gradient accumulation)
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    accum_steps = max(1, int(getattr(cfg.optim, "gradient_accumulation_steps", 1)))
    effective_global_batch = cfg.train.batch_size_per_gpu * distributed.get_world_size() * accum_steps
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= effective_global_batch / 256.0
        lr_end *= effective_global_batch / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(effective_global_batch / 1024.0)
        lr_end *= 4 * math.sqrt(effective_global_batch / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, freeze_backbone=False):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if freeze_backbone and param_group.get("is_backbone", False):
            param_group["lr"] = 0.0
        elif is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def do_test(cfg, model, iteration, process_group, do_low_freq=False):
    checkpoint_pg = distributed.get_checkpoint_process_group()
    # dump a sharded checkpoint
    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    if distributed.is_subgroup_main_process():
        eval_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.sharded_eval_checkpoint:
        ckpt_path = eval_dir / "sharded_teacher_checkpoint"
        if distributed.is_subgroup_main_process():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        teacher_backbone = model.model_ema
        save_checkpoint(
            ckpt_dir=ckpt_path,
            iteration=iteration,
            model=teacher_backbone,
            overwrite=True,
            sharded=True,
            process_group=checkpoint_pg,
        )
        if not distributed.is_subgroup_main_process():
            return
    else:
        new_state_dict = model.model_ema.state_dict()
        for k, tensor in list(new_state_dict.items()):
            if isinstance(tensor, DTensor):
                new_state_dict[k] = tensor.full_tensor()
        if not distributed.is_subgroup_main_process():
            return
        # save teacher checkpoint
        ckpt_path = eval_dir / "teacher_checkpoint.pth"
        torch.save({"teacher": new_state_dict}, ckpt_path)
        logger.info("Saved eval checkpoint: %s", ckpt_path)


def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    if cfg.multidistillation.enabled:
        assert cfg.multidistillation.global_batch_size % distributed.get_subgroup_size() == 0
        local_batch_size = cfg.multidistillation.global_batch_size // distributed.get_subgroup_size()
        dataloader_batch_size_per_gpu = (
            cfg.multidistillation.global_batch_size + (distributed.get_world_size() - 1)
        ) // distributed.get_world_size()
    else:
        local_batch_size = None  # will default to the standard local batch size matching the data batch size
        dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

    accum_steps = max(1, int(getattr(cfg.optim, "gradient_accumulation_steps", 1)))

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )
    batch_size = dataloader_batch_size_per_gpu
    num_workers = cfg.train.num_workers
    dataset_path = cfg.train.dataset_path
    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=model.build_data_augmentation_dino(cfg),
        target_transform=lambda _: (),
        target_channels=cfg.student.in_chans,
        wds_shuffle_buffer=getattr(cfg.train, "wds_shuffle_buffer", 1000),
        wds_resample_seed=cfg.train.seed + start_iter + 1,
        wds_deterministic_resampling=bool(
            getattr(cfg.train, "wds_deterministic_resampling", False)
        ),
    )

    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if cfg.train.cache_dataset else SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * accum_steps * dataloader_batch_size_per_gpu,
        drop_last=True,
        pin_memory=getattr(cfg.train, "pin_memory", True),
        prefetch_factor=getattr(cfg.train, "prefetch_factor", None),
        collate_fn=collate_fn,
    )
    if bool(getattr(cfg.train, "wds_deterministic_resampling", False)):
        if num_workers != 0:
            raise ValueError(
                "train.wds_deterministic_resampling=true requires train.num_workers=0 "
                "so augmentation and masking remain comparable across method arms"
            )
        data_loader = DeterministicDataStream(
            data_loader,
            seed=cfg.train.seed + 1_000_003 * distributed.get_rank(),
            start_fetch_index=start_iter * accum_steps,
        )
        logger.info(
            "WebDataset controlled stream enabled: fixed shard/sample order and isolated per-batch augmentation RNG"
        )
    return data_loader


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size] if isinstance(cfg.crops.global_crops_size, int) else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size] if isinstance(cfg.crops.local_crops_size, int) else cfg.crops.local_crops_size
    )
    gram_teacher_crops_sizes = (
        [cfg.crops.gram_teacher_crops_size]
        if cfg.crops.gram_teacher_crops_size is None or isinstance(cfg.crops.gram_teacher_crops_size, int)
        else cfg.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (global_crops_size_i, local_crops_size_i, gram_teacher_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes)
    ):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
        cfg_i.train.seed = cfg.train.seed + increment + 1
        loaders.append(build_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))

    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=seed,
            name="MultiResDL",
        )
    return data_loader


def do_train(cfg, model, resume=False):
    process_subgroup = distributed.get_process_subgroup()
    checkpoint_pg = distributed.get_checkpoint_process_group()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Plain checkpoints use unwrapped module names. DDP wrapping is deferred
    # until after initialization; FSDP is already wrapped and this is a no-op.
    model.init_weights()
    model.finish_distributed_training_setup()
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=checkpoint_pg,
            )
            + 1
        )
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    accum_steps = max(1, int(getattr(cfg.optim, "gradient_accumulation_steps", 1)))
    freeze_backbone_updates = max(0, int(getattr(cfg.optim, "freeze_backbone_updates", 0)))

    if cfg.multidistillation.enabled:
        real_global_batch_size = cfg.multidistillation.global_batch_size
    else:
        real_global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    effective_global_batch_size = real_global_batch_size * accum_steps

    logger.info(
        "Gradient accumulation: %d | real global batch: %d | effective optimizer batch: %d",
        accum_steps,
        real_global_batch_size,
        effective_global_batch_size,
    )
    if freeze_backbone_updates:
        logger.info(
            "FINO-style heads warmup: backbone LR is zero for the first %d optimizer updates",
            freeze_backbone_updates,
        )

    # Build data loader
    data_loader = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    raw_metrics_file = os.path.join(cfg.train.output_dir, "raw_loss_metrics.jsonl")
    raw_loss_static_fields = _build_raw_loss_static_fields(
        cfg=cfg,
        accum_steps=accum_steps,
        real_global_batch_size=real_global_batch_size,
        effective_global_batch_size=effective_global_batch_size,
    )
    if distributed.is_main_process():
        logger.info("Raw per-update loss metrics will be written to %s", raw_metrics_file)
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    consecutive_nan_count = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    accum_loss_for_log = None
    accum_metrics = None

    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="Training",
        n_iterations=max_iter * accum_steps,
        start_iteration=start_iter * accum_steps,
    ):
        if iteration >= max_iter:
            break

        it = iteration
        batch_sample_key_digest = _batch_sample_key_digest(data)
        data["global_batch_size"] = real_global_batch_size

        micro_step_in_cycle = micro_step % accum_steps
        is_cycle_start = micro_step_in_cycle == 0
        is_update_step = (micro_step_in_cycle + 1) == accum_steps

        if accum_steps > 1:
            # FSDP2 otherwise reduce-scatters gradients after every micro-step.
            # DDP exposes the same intent through ``require_backward_grad_sync``.
            for student_module in student.values():
                if hasattr(student_module, "set_requires_gradient_sync"):
                    student_module.set_requires_gradient_sync(is_update_step, recurse=True)
                elif isinstance(student_module, torch.nn.parallel.DistributedDataParallel):
                    student_module.require_backward_grad_sync = is_update_step

        if is_cycle_start:
            if (iteration + 1) % 150 == 0:
                logger.info("Garbage collection")
                gc.collect()

            if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
                logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
                model.gram_load_ema_teacher()

            lr = lr_schedule[it]
            wd = wd_schedule[it]
            mom = momentum_schedule[it]
            teacher_temp = teacher_temp_schedule[it]
            last_layer_lr = last_layer_lr_schedule[it]
            backbone_is_frozen = it < freeze_backbone_updates
            if it == freeze_backbone_updates and freeze_backbone_updates > 0:
                logger.info("FINO-style heads warmup complete; unfreezing backbone at optimizer update %d", it)
            apply_optim_scheduler(
                optimizer,
                lr,
                wd,
                last_layer_lr,
                freeze_backbone=backbone_is_frozen,
            )
            backbone_lr = 0.0 if backbone_is_frozen else lr

            accum_loss_for_log = None
            accum_metrics = {}

        total_loss, metrics_dict = model.forward_backward(
            data,
            teacher_temp=teacher_temp,
            iteration=it,
            loss_divisor=accum_steps,
        )

        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()

        metrics_values = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
        )
        torch.distributed.all_reduce(
            metrics_values,
            op=torch.distributed.ReduceOp.AVG,
            group=distributed.get_process_subgroup(),
        )
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            _log_nan_debug_batch(data, logger)
            if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0

        if accum_loss_for_log is None:
            accum_loss_for_log = total_loss.detach()
        else:
            accum_loss_for_log = accum_loss_for_log + total_loss.detach()

        assert accum_metrics is not None
        for k, v in metrics_dict.items():
            v = v.detach() if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32, device=total_loss.device)
            accum_metrics[k] = accum_metrics.get(k, torch.zeros_like(v)) + v

        micro_step += 1
        if not is_update_step:
            continue

        total_loss = accum_loss_for_log / accum_steps
        metrics_dict = {k: v / accum_steps for k, v in accum_metrics.items()}

        if cfg.optim.clip_grad:
            for k, v in student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm.full_tensor().item()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm.item()
                )

        optimizer.step()
        model.update_ema(mom)

        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(backbone_lr=backbone_lr)
        metric_logger.update(real_global_batch_size=real_global_batch_size)
        metric_logger.update(effective_global_batch_size=effective_global_batch_size)
        metric_logger.update(total_loss=total_loss, **metrics_dict)
        _append_raw_loss_metrics(
            raw_metrics_file=raw_metrics_file,
            static_fields=raw_loss_static_fields,
            cfg=cfg,
            iteration=iteration,
            micro_steps_completed_in_run=micro_step,
            lr=lr,
            wd=wd,
            mom=mom,
            last_layer_lr=last_layer_lr,
            backbone_lr=backbone_lr,
            total_loss=total_loss,
            metrics_dict=metrics_dict,
            batch_sample_key_digest=batch_sample_key_digest,
        )

        if (
            cfg.evaluation.eval_period_iterations > 0
            and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
        ):
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()

        if (iteration + 1) % cfg.checkpointing.period == 0:
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                sharded=bool(getattr(cfg.checkpointing, "sharded", False)),
                process_group=checkpoint_pg,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))

        iteration = iteration + 1
        optimizer.zero_grad(set_to_none=True)
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    if args.multi_distillation:
        print("performing multidistillation run")
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
        logger.info("setup_multidistillation done")
        assert cfg.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(OmegaConf.to_yaml(cfg))
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "SSLMetaArchLoRA": SSLMetaArchLoRA,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    logger.info(f"Making meta arch {meta_arch.__name__}")
    with torch.device("meta"):
        model = meta_arch(cfg)
    model.prepare_for_distributed_training()
    # Fill all values with `nans` so that we identify
    # non-initialized values
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        model.init_weights()
        model.finish_distributed_training_setup()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}", distributed.get_checkpoint_process_group())
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
