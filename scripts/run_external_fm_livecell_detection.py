#!/usr/bin/env python3
"""Run the H+ LIVECell center-to-patch probe with an external frozen encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path("/mnt/huawei_deepcad/dinov3")
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT), str(BENCHMARK_MODEL_ROOT), str(BENCHMARK_MODEL_ROOT / "_vendor")]
sys.path.append("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311")

from dinov3.eval.bio_detection.center_probe import (  # noqa: E402
    LiveCellCenterDataset,
    _estimate_pos_weight,
    _eval,
)
from run_dense_probe_benchmark import DenseFeatureExtractor  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


class ExternalPatchFeatureModel(nn.Module):
    """Convert model-native dense maps to the fixed 14x14 H+ probe grid."""

    def __init__(self, model_name: str):
        super().__init__()
        self.extractor = DenseFeatureExtractor(model_name, "cuda", canonical=False)

    @torch.inference_mode()
    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(normalized_images.device)
        std = IMAGENET_STD.to(normalized_images.device)
        images_01 = (normalized_images * std + mean).clamp_(0, 1)
        feature_map = self.extractor(images_01)
        if feature_map.shape[-2:] != (14, 14):
            feature_map = torch.nn.functional.interpolate(
                feature_map.float(), size=(14, 14), mode="bilinear", align_corners=False
            )
        return feature_map.flatten(2).transpose(1, 2).float().contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    result_path = output / "results_bio_detection.json"
    if result_path.exists():
        print(f"[skip] {result_path}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.benchmark_root) / "segmentation/LIVECell"
    datasets = {
        split: LiveCellCenterDataset(
            str(data_root), split, image_size=224, patch_size=16,
            max_samples=args.max_samples_per_split, seed=args.seed,
        )
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, generator=generator,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        ),
    }
    feature_model = ExternalPatchFeatureModel(args.model).cuda().eval()
    sample_images, sample_labels = next(iter(loaders["train"]))
    sample_features = feature_model(sample_images.cuda(non_blocking=True)).clone()
    if sample_features.shape[1] != sample_labels.shape[1]:
        raise RuntimeError(
            f"Feature grid {sample_features.shape[1]} does not match labels {sample_labels.shape[1]}"
        )
    torch.manual_seed(args.seed)
    head = nn.Linear(int(sample_features.shape[-1]), 1).cuda()
    pos_weight = torch.tensor([_estimate_pos_weight(loaders["train"])], device="cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        head.train()
        for images, labels in loaders["train"]:
            logits = head(feature_model(images.cuda(non_blocking=True)).clone()).squeeze(-1)
            loss = criterion(logits, labels.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch + 1}/{args.epochs}] val={_eval(feature_model, head, loaders['val'])}", flush=True)

    val_metrics = _eval(feature_model, head, loaders["val"])
    test_metrics = _eval(feature_model, head, loaders["test"])
    result = {
        "model": args.model,
        "dataset": "livecell",
        "probe": "livecell_center_patch_linear",
        "feature_grid": "14x14",
        "encoder_preprocess": "model-native",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pos_weight": float(pos_weight.item()),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, result_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
