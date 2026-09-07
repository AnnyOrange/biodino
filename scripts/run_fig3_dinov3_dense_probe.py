#!/usr/bin/env python3
"""Run an H+/16 checkpoint in the existing external-FM dense probe protocol.

This deliberately reuses dataset loading, feature-cache format, and the frozen
linear head from ``benchmark_model/run_dense_probe_benchmark.py``.  Therefore
the two DINOv3 H+/16 rows use the same 20-epoch, one-last-layer protocol as the
14 external foundation models rather than the stronger, separate BioDINO dense
configuration used in earlier drafts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument(
        "--normalization",
        choices=("config", "imagenet"),
        default="config",
        help="Use the checkpoint training RGB stats or the official ImageNet DINOv3 stats.",
    )
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--max-feature-side", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    parser.add_argument("--skip-summary", action="store_true", help="Do not append the shared summary.csv (safe for parallel jobs).")
    parser.add_argument("--out-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import run_dense_probe_benchmark as dense
    from omegaconf import OmegaConf
    from dinov3.eval.bio_segmentation.feature_extractor import _prepare_input_channels
    from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone

    train_cfg = OmegaConf.load(args.train_config)
    if args.normalization == "imagenet":
        rgb_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        rgb_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    else:
        rgb_mean = torch.tensor(train_cfg.crops.rgb_mean, dtype=torch.float32).view(1, 3, 1, 1)
        rgb_std = torch.tensor(train_cfg.crops.rgb_std, dtype=torch.float32).view(1, 3, 1, 1)

    class Dinov3DenseFeatureExtractor:
        # Keep one frozen backbone alive for every split in this process.
        _backbone: torch.nn.Module | None = None
        input_size = args.input_size
        patch_size = 16
        mean = rgb_mean
        std = rgb_std

        def __init__(self, _model_name: str, device: str, canonical: bool = False):
            del canonical
            self.device = torch.device(device)
            if type(self)._backbone is None:
                type(self)._backbone = load_dinov3_backbone(
                    str(args.checkpoint),
                    str(args.train_config),
                    device=self.device,
                    freeze=True,
                )
            self.backbone = type(self)._backbone
            patch_size = getattr(self.backbone, "patch_size", 16)
            self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)

        @torch.inference_mode()
        def __call__(self, images_01: torch.Tensor) -> torch.Tensor:
            if images_01.shape[-2:] != (self.input_size, self.input_size):
                images_01 = F.interpolate(
                    images_01,
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            # Match each DINOv3 checkpoint's pretraining normalization, just as
            # every external encoder applies its own model-native normalization.
            images_01 = (images_01 - self.mean.to(images_01.device)) / self.std.to(images_01.device)
            images = _prepare_input_channels(images_01.to(self.device), self.backbone)
            with torch.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.float16):
                return self.backbone.get_intermediate_layers(
                    images, n=1, reshape=True, return_class_token=False
                )[0].float()

    for dataset in args.datasets:
        if dataset not in dense.DATASET_CONFIGS:
            raise KeyError(f"Unknown dense-probe dataset: {dataset}")

    dense.OUT_ROOT = args.out_root
    dense.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dense.DenseFeatureExtractor = Dinov3DenseFeatureExtractor
    # ``extract_cache`` reads these fields from the original parser namespace.
    args.img_size = 0
    args.feature_canonical = False
    # The 3090 node's evaluation environment includes the required metrics.
    os.environ.setdefault("DENSE_PROBE_METRIC_PYTHON", sys.executable)

    for dataset in args.datasets:
        print(f"[run] DINOv3 dense probe model={args.model_label} dataset={dataset}", flush=True)
        caches = {
            split: dense.extract_cache(args, args.model_label, dataset, split)
            for split in args.splits
        }
        result_path = dense.run_linear_probe(args, args.model_label, dataset, caches)
        if not args.skip_summary:
            dense.append_summary(args.model_label, dataset, result_path)
        print(f"[done] {args.model_label} {dataset}: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
