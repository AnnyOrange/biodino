#!/usr/bin/env python
"""Evaluate Cellpose-trained frozen-encoder HoVerNet heads on benchmark seg tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet
from dinov3.eval.bio_segmentation.instance_seg.train import _build_instance_dataset, evaluate


DATA_ROOTS = {
    "pannuke": "/mnt/huawei_deepcad/benchmark/segmentation/pannuke/extracted",
    "tissuenet": "/mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted",
    "conic": "/mnt/huawei_deepcad/benchmark/segmentation/conic/extracted",
    "bbbc038": "/mnt/huawei_deepcad/benchmark/segmentation/bbbc038/extracted",
    "livecell": "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell/LIVECell_dataset_2021",
    "monuseg": "/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted",
}


MODELS = {
    "vitl_hires_gram_3074": (
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/ckpt/3074/checkpoint.pth",
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/config.yaml",
        [5, 11, 17, 23],
    ),
    "vitl_hires_gram_2049": (
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/ckpt/2049/checkpoint.pth",
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/config.yaml",
        [5, 11, 17, 23],
    ),
    "vitl_hires_gram_1024": (
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/ckpt/1024/checkpoint.pth",
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/config.yaml",
        [5, 11, 17, 23],
    ),
    "vitl_oep1025_11274": (
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt/11274/checkpoint.pth",
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/config.yaml",
        [5, 11, 17, 23],
    ),
    "vitl_oep1025_14349": (
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt/14349/checkpoint.pth",
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/config.yaml",
        [5, 11, 17, 23],
    ),
    "vitl_oep1025_13324": (
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt/13324/checkpoint.pth",
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/config.yaml",
        [5, 11, 17, 23],
    ),
    "hplus_rgb3_9224": (
        "ckpt/9224/checkpoint.pth",
        "outputs/bio_continue_rgb3_vith16plus/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_rgb3_8199": (
        "ckpt/8199/checkpoint.pth",
        "outputs/bio_continue_rgb3_vith16plus/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_rgb3_12299": (
        "ckpt/12299/checkpoint.pth",
        "outputs/bio_continue_rgb3_vith16plus/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_rgb3_10249_bpq": (
        "ckpt/10249/checkpoint.pth",
        "outputs/bio_continue_rgb3_vith16plus/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_5tb_14819": (
        "outputs/5tb_hplus_packwds_ep15_b1024/ckpt/14819/checkpoint.pth",
        "outputs/5tb_hplus_packwds_ep15_b1024/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_5tb_9879": (
        "outputs/5tb_hplus_packwds_ep15_b1024/ckpt/9879/checkpoint.pth",
        "outputs/5tb_hplus_packwds_ep15_b1024/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_5tb_12349": (
        "outputs/5tb_hplus_packwds_ep15_b1024/ckpt/12349/checkpoint.pth",
        "outputs/5tb_hplus_packwds_ep15_b1024/config.yaml",
        [7, 15, 23, 31],
    ),
    "hplus_5tb_32109_bpq": (
        "outputs/5tb_hplus_packwds_ep15_b1024/ckpt/32109/checkpoint.pth",
        "outputs/5tb_hplus_packwds_ep15_b1024/config.yaml",
        [7, 15, 23, 31],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DATA_ROOTS), choices=sorted(DATA_ROOTS))
    parser.add_argument("--head-root", default="outputs/instance_seg/cellpose_decoder_finetune_topckpts_local8")
    parser.add_argument("--output-root", default="outputs/instance_seg/cellpose_decoder_finetune_topckpts_local8_benchmark")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--embed-proj", type=int, default=512)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--fg-thresh", type=float, default=0.5)
    parser.add_argument("--energy-thresh", type=float, default=0.4)
    args = parser.parse_args()

    checkpoint, train_config, layers = MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is not visible; run this on the local GPU machine.")

    model = build_dino_hovernet(
        checkpoint=checkpoint,
        train_config=train_config,
        layers=layers,
        num_types=0,
        freeze_backbone=True,
        feature_size=args.feature_size,
        embed_proj=args.embed_proj,
        device=device,
    )
    head_path = Path(args.head_root) / args.model / "best_head.pth"
    if not head_path.exists():
        raise FileNotFoundError(f"Missing trained decoder head: {head_path}")
    model.decoder.load_state_dict(torch.load(head_path, map_location=device))
    patch_size = int(model.backbone.patch_size)

    for dataset in args.datasets:
        out_dir = Path(args.output_root) / args.model / dataset
        out_json = out_dir / "results.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_args = SimpleNamespace(dataset=dataset, data_root=DATA_ROOTS[dataset])
        base_test = _build_instance_dataset(ds_args, "test", do_normalize=True)
        metrics = evaluate(
            model,
            base_test,
            device,
            num_types=0,
            crop_size=args.crop_size,
            stride=args.stride,
            patch_size=patch_size,
            max_images=args.max_eval_images,
            tta=args.tta,
            fg_thresh=args.fg_thresh,
            energy_thresh=args.energy_thresh,
        )
        result = {
            "test": metrics,
            "_meta": {
                "model": args.model,
                "dataset": dataset,
                "head_path": str(head_path),
                "backbone_checkpoint": checkpoint,
                "train_config": train_config,
                "layers": layers,
                "num_types": 0,
                "source": "cellpose_decoder_finetune",
            },
        }
        with out_json.open("w") as f:
            json.dump(result, f, indent=2)
        print(dataset, {k: round(v, 4) for k, v in metrics.items()}, "->", out_json, flush=True)


if __name__ == "__main__":
    main()
