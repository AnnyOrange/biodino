"""
Export EMA teacher (and optionally student) backbone weights from a training DCP
checkpoint directory to single-file .pth checkpoints for bio_segmentation eval.

Must be launched like training, e.g. torchrun --nproc_per_node=K -m ...
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import find_latest_checkpoint, load_checkpoint
from dinov3.configs import exit_job, setup_config, setup_job, setup_multidistillation
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.ssl_meta_arch_lora import SSLMetaArchLoRA
from dinov3.train.train import get_args_parser

logger = logging.getLogger("dinov3")


def _materialize_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if isinstance(v, DTensor):
            v = v.full_tensor()
        out[k] = v.detach().cpu()
    return out


def _pick_meta_arch(cfg):
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "SSLMetaArchLoRA": SSLMetaArchLoRA,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    return meta_arch


def _build_model_like_train(cfg):
    meta_arch = _pick_meta_arch(cfg)

    with torch.device("meta"):
        model = meta_arch(cfg)

    model.prepare_for_distributed_training()

    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    return model


def main(argv: list[str] | None = None) -> None:
    parser = get_args_parser(add_help=True)
    parser.add_argument(
        "--ckpt-iter",
        default="latest",
        type=str,
        help='Checkpoint iteration under output_dir/ckpt, e.g. "299" or "latest".',
    )
    parser.add_argument(
        "--export-student",
        action="store_true",
        help="Also export student backbone to student_backbone_evalstyle.pth.",
    )
    args = parser.parse_args(argv)

    if args.multi_distillation:
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)

    process_subgroup = distributed.get_process_subgroup()
    ckpt_root = Path(cfg.train.output_dir) / "ckpt"

    if args.ckpt_iter == "latest":
        ckpt_dir = find_latest_checkpoint(ckpt_root)
        if ckpt_dir is None:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_root}")
    else:
        ckpt_dir = ckpt_root / str(args.ckpt_iter)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    model = _build_model_like_train(cfg)
    model.init_weights()

    load_checkpoint(
        ckpt_dir=ckpt_dir,
        model=model,
        optimizer=None,
        strict_loading=False,
        process_group=process_subgroup,
    )

    export_dir = Path(cfg.train.output_dir) / "eval" / f"export_{ckpt_dir.name}"
    if distributed.is_subgroup_main_process():
        export_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    teacher_full_sd = _materialize_state_dict(model.model_ema.state_dict())
    teacher_backbone_sd = {k[len("backbone.") :]: v for k, v in teacher_full_sd.items() if k.startswith("backbone.")}

    if distributed.is_subgroup_main_process():
        torch.save(
            {"teacher": teacher_full_sd},
            export_dir / "teacher_checkpoint_trainstyle.pth",
        )
        torch.save(
            {"model": teacher_backbone_sd},
            export_dir / "teacher_backbone_evalstyle.pth",
        )

    if args.export_student:
        student_full_sd = _materialize_state_dict(model.student.state_dict())
        student_backbone_sd = {
            k[len("backbone.") :]: v for k, v in student_full_sd.items() if k.startswith("backbone.")
        }
        if distributed.is_subgroup_main_process():
            torch.save(
                {"model": student_backbone_sd},
                export_dir / "student_backbone_evalstyle.pth",
            )

    torch.distributed.barrier()

    if distributed.is_subgroup_main_process():
        logger.info("Export done: %s", export_dir)

    exit_job()


if __name__ == "__main__":
    main()
