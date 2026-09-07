#!/usr/bin/env python3
"""Extract model-native BBBC048 frozen features for Fig. 3 Panel D.

External foundation-model caches already exist under ``outputs/02_eval_runs``.
This utility supplies the missing ImageNet and official DINOv3 feature caches
using the same committed BBBC048 sample ordering as the benchmark protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_L_WEIGHTS = REPO_ROOT / "outputs/torch_cache/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# Shared location is visible to the dedicated single-3090 evaluation host.
DEFAULT_7B_WEIGHTS = REPO_ROOT / "outputs/torch_cache/hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"


class PILFrozenEncoder:
    """Give native image models the ``encode_images`` API used by the harness."""

    def __init__(self, model, transform, device: str):
        self.model = model
        self.transform = transform
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode_images(self, images: list) -> np.ndarray:
        x = torch.stack([self.transform(image) for image in images]).to(self.device, non_blocking=True)
        features = self.model(x)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", next(iter(features.values())))
        if features.ndim > 2:
            features = features.flatten(1)
        features = torch.nn.functional.normalize(features.float(), dim=1)
        return features.cpu().numpy().astype(np.float16)


def make_encoder(model_key: str, device: str, l_weights: Path, vit7b_weights: Path) -> PILFrozenEncoder:
    import torch
    from torch import nn
    from torchvision import models

    from dinov3.data.transforms import make_classification_eval_transform
    from dinov3.eval.bio_classification.common import LinearFeatureModel
    from dinov3.hub import backbones

    if model_key == "imagenet_resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        transform = weights.transforms(crop_size=224, resize_size=256)
    elif model_key == "dinov3_official_vitl16":
        if not l_weights.exists():
            raise FileNotFoundError(f"Original DINOv3-L weights are missing: {l_weights}")
        backbone = backbones.dinov3_vitl16(pretrained=True, weights=str(l_weights), check_hash=False)
        model = LinearFeatureModel(backbone, n_last_blocks=1, use_avgpool=True, autocast_dtype=torch.bfloat16)
        transform = make_classification_eval_transform(resize_size=256, crop_size=224)
    elif model_key == "dinov3_official_vit7b16":
        if not vit7b_weights.exists():
            raise FileNotFoundError(f"Original DINOv3-7B weights are missing: {vit7b_weights}")
        backbone = backbones.dinov3_vit7b16(pretrained=True, weights=str(vit7b_weights), check_hash=False)
        # The 7B model needs bf16 parameters to stay inside a 24 GB RTX 3090.
        backbone = backbone.to(dtype=torch.bfloat16)
        model = LinearFeatureModel(backbone, n_last_blocks=1, use_avgpool=True, autocast_dtype=torch.bfloat16)
        transform = make_classification_eval_transform(resize_size=256, crop_size=224)
    else:
        raise ValueError(f"Unknown model: {model_key}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return PILFrozenEncoder(model=model, transform=transform, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--models", default="imagenet_resnet50,dinov3_official_vitl16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--l-weights", type=Path, default=DEFAULT_L_WEIGHTS)
    parser.add_argument("--vit7b-weights", type=Path, default=DEFAULT_7B_WEIGHTS)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based contiguous sample shard to encode (requires --num-shards).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of contiguous sample shards; shard caches retain original indices for strict merging.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from dinov3.eval.bio_frozen_eval.encoder import extract_features
    from dinov3.eval.bio_frozen_eval.registry import build_dataset

    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    allowed = {"imagenet_resnet50", "dinov3_official_vitl16", "dinov3_official_vit7b16"}
    unknown = sorted(set(model_keys) - allowed)
    if unknown:
        raise ValueError(f"Unknown --models values: {unknown}; expected {sorted(allowed)}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require --num-shards > 0 and 0 <= --shard-index < --num-shards.")
    out_dir = args.output_dir / "panel_d_feature_cache/bbbc048-cellcycle"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, task = build_dataset(
        "bbbc048-cellcycle", "train", None, None, benchmark_root=args.benchmark_root
    )
    if task != "classification":
        raise RuntimeError(f"Expected classification task, got {task!r}")
    total_samples = len(dataset)
    start = total_samples * args.shard_index // args.num_shards
    stop = total_samples * (args.shard_index + 1) // args.num_shards
    indices = np.arange(start, stop, dtype=np.int64)
    if args.num_shards > 1:
        dataset = Subset(dataset, indices.tolist())
        print(
            f"[panel-d-features] shard {args.shard_index + 1}/{args.num_shards}: "
            f"global indices [{start}, {stop})",
            flush=True,
        )
    for model_key in model_keys:
        suffix = "" if args.num_shards == 1 else f".part{args.shard_index:02d}of{args.num_shards:02d}"
        feature_path = out_dir / f"{model_key}{suffix}.npz"
        print(f"[panel-d-features] loading {model_key}", flush=True)
        encoder = make_encoder(model_key, args.device, args.l_weights, args.vit7b_weights)
        features, labels = extract_features(
            dataset,
            encoder,
            feature_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            model_name=model_key,
            save_features=True,
            save_paths=False,
        )
        if args.num_shards > 1:
            # Keep global ordering explicit so shards can be merged without relying
            # on launch order or individual DataLoader completion order.
            np.savez(
                feature_path,
                features=features.astype(np.float16),
                labels=labels,
                indices=indices,
                model=model_key,
            )
        print(
            f"[panel-d-features] completed {model_key}: features={features.shape} labels={labels.shape} -> {feature_path}",
            flush=True,
        )
        del encoder, features, labels
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
