#!/usr/bin/env python3
"""Encode packed microscopy samples in a selected frozen mature-HS6 readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinov3.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    make_classification_eval_transform,
)
from dinov3.eval.bio_classification.common import create_linear_input
from dinov3.eval.bio_frozen_eval.encoder import _load_multichannel_stats
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone

try:
    from scripts.build_expert_feature_bank import decode_sample, iter_raw_samples, sample_metadata
except ModuleNotFoundError:
    from build_expert_feature_bank import decode_sample, iter_raw_samples, sample_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--resize-size", type=int, default=288)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument(
        "--feature-protocol",
        choices=("final_cls", "nlb2_cls", "nlb2_avg"),
        default="final_cls",
    )
    parser.add_argument(
        "--normalization-protocol",
        choices=("train", "eval_imagenet"),
        default="train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if (
        args.batch_size <= 0
        or args.torch_num_threads <= 0
        or args.resize_size < args.crop_size
        or args.crop_size <= 0
    ):
        raise ValueError("Expected positive batch/crop and resize-size >= crop-size")
    if not 0 <= args.p_low < args.p_high <= 100:
        raise ValueError("Expected 0 <= p-low < p-high <= 100")
    for path in [*args.shards, args.checkpoint, args.train_config]:
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(min(4, args.torch_num_threads))
    device = torch.device(args.device)
    backbone = load_dinov3_backbone(
        str(args.checkpoint),
        str(args.train_config),
        device=device,
        freeze=True,
    ).eval()
    mean, std = _load_multichannel_stats(args.train_config)
    if args.normalization_protocol == "eval_imagenet":
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    transform = make_classification_eval_transform(
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        mean=mean,
        std=std,
    )
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.amp_dtype]

    keys: list[str] = []
    features: list[np.ndarray] = []
    domains: list[str] = []
    organisms: list[str] = []
    acquisitions: list[str] = []
    sample_types: list[str] = []
    image_batch = []
    pending: list[tuple[str, dict[str, str]]] = []

    @torch.inference_mode()
    def flush() -> None:
        if not image_batch:
            return
        images = torch.stack([transform(image) for image in image_batch]).to(
            device,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and args.amp_dtype != "fp32",
        ):
            if args.feature_protocol == "final_cls":
                encoded = backbone(images, is_training=True)["x_norm_clstoken"]
            else:
                tokens = backbone.get_intermediate_layers(
                    images,
                    n=2,
                    reshape=False,
                    return_class_token=True,
                )
                encoded = create_linear_input(
                    tokens,
                    use_n_blocks=2,
                    use_avgpool=args.feature_protocol == "nlb2_avg",
                )
        encoded = F.normalize(encoded.float(), dim=-1).cpu().numpy().astype(np.float16)
        features.append(encoded)
        for key, metadata in pending:
            keys.append(key)
            domains.append(metadata["domain"])
            organisms.append(metadata["organism"])
            acquisitions.append(metadata["acquisition_family"])
            sample_types.append(metadata["sample_type"])
        image_batch.clear()
        pending.clear()
        if len(keys) % (10 * args.batch_size) == 0:
            print(f"[anchor-bank] {len(keys)} samples", flush=True)

    for shard in args.shards:
        for key, raw in iter_raw_samples(shard):
            decoded = decode_sample(raw, args.p_low, args.p_high)
            if decoded is None:
                continue
            image, meta = decoded
            image_batch.append(image)
            pending.append(
                (
                    f"{shard.name}::{key}",
                    sample_metadata(meta, {}),
                )
            )
            if len(image_batch) >= args.batch_size:
                flush()
    flush()
    if not keys:
        raise RuntimeError("No packed samples were encoded")
    feature_array = np.concatenate(features, axis=0)
    if len(set(keys)) != len(keys) or feature_array.shape[0] != len(keys):
        raise RuntimeError("Anchor feature keys are duplicated or misaligned")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=np.asarray(keys),
        features=feature_array,
        reliability=np.ones(len(keys), dtype=np.float32),
        domain=np.asarray(domains),
        organism=np.asarray(organisms),
        acquisition_family=np.asarray(acquisitions),
        sample_type=np.asarray(sample_types),
        model=np.asarray(
            f"mature_hs6_anchor_{args.feature_protocol}_{args.normalization_protocol}"
        ),
        expert_role=np.asarray("general"),
        feature_protocol=np.asarray(args.feature_protocol),
        n_last_blocks=np.asarray(1 if args.feature_protocol == "final_cls" else 2),
        use_avgpool=np.asarray(args.feature_protocol == "nlb2_avg"),
        normalization_protocol=np.asarray(args.normalization_protocol),
        source_shards=np.asarray([str(path) for path in args.shards]),
        normalization_percentiles=np.asarray([args.p_low, args.p_high], dtype=np.float32),
        transform_resize_crop=np.asarray([args.resize_size, args.crop_size], dtype=np.int32),
        transform_mean=np.asarray(mean, dtype=np.float32),
        transform_std=np.asarray(std, dtype=np.float32),
        checkpoint=np.asarray(str(args.checkpoint)),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(keys),
                "feature_dim": int(feature_array.shape[1]),
                "feature_protocol": args.feature_protocol,
                "normalization_protocol": args.normalization_protocol,
                "checkpoint": str(args.checkpoint),
                "resize_crop": [args.resize_size, args.crop_size],
                "mean": list(mean),
                "std": list(std),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
