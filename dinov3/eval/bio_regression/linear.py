from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

import dinov3.distributed as distributed
from dinov3.eval.bio_classification.common import (
    LinearFeatureModel,
    build_classification_transform,
    checkpoint_stem,
    extract_features_simple,
    load_backbone,
    parse_autocast_dtype,
)
from dinov3.eval.helpers import write_results
from dinov3.utils.bio_io import read_bio_image_as_numpy

logger = logging.getLogger("dinov3.bio_regression.linear")
RESULTS_FILENAME = "results-regression.csv"


class BBBC013Dataset(Dataset):
    """BBBC013 dose regression from well images and official plate map."""

    def __init__(self, root: str, split: str, transform=None, target_transform: str = "log1p", seed: int = 0, max_samples: int = 0):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        samples = self._collect_samples()
        samples = self._split(samples, split=split, seed=seed)
        if max_samples > 0 and len(samples) > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(samples), size=max_samples, replace=False))
            samples = [samples[i] for i in idx]
        self.samples = samples
        if not self.samples:
            raise ValueError(f"BBBC013 split={split} is empty under {root}")
        logger.info("BBBC013 split=%s size=%d", split, len(self.samples))

    def _read_plate_map(self) -> List[float]:
        path = self.root / "BBBC013_v1_platemap_all.txt"
        if not path.is_file():
            path = self.root / "BBBC013_v1_platemap_wortmannin.txt"
        vals: List[float] = []
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.upper().startswith("DESCRIPTION"):
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
        if len(vals) < 96:
            raise ValueError(f"Expected at least 96 plate-map values in {path}, got {len(vals)}")
        return vals[:96]

    def _collect_samples(self) -> List[Tuple[str, float]]:
        targets = self._read_plate_map()
        image_dir = self.root / "BBBC013_v1_images_bmp"
        samples: List[Tuple[str, float]] = []
        for path in sorted(image_dir.glob("*.BMP")):
            match = re.search(r"Channel\d+-(\d+)-", path.name)
            if not match:
                continue
            well_idx = int(match.group(1)) - 1
            if 0 <= well_idx < len(targets):
                y = float(targets[well_idx])
                if self.target_transform == "log1p":
                    y = math.log1p(y)
                samples.append((str(path), y))
        if not samples:
            raise FileNotFoundError(f"No BBBC013 BMP samples found in {image_dir}")
        return samples

    @staticmethod
    def _split(samples: Sequence[Tuple[str, float]], split: str, seed: int) -> List[Tuple[str, float]]:
        split = "val" if split == "valid" else split
        rng = np.random.default_rng(seed)
        order = np.arange(len(samples))
        rng.shuffle(order)
        n = len(order)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        parts = {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
        }
        return [samples[i] for i in parts[split]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = read_bio_image_as_numpy(path, target_channels=3, normalize=True)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image, torch.tensor(float(target), dtype=torch.float32)


def _metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(np.float64)
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 0 else float("nan")
    pearson = float(np.corrcoef(pred, target)[0, 1]) if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}


def _fit_ridge(train_x, train_y, val_x, val_y, alphas: Iterable[float]):
    from sklearn.linear_model import Ridge

    best = None
    best_score = float("inf")
    best_metrics = {}
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        metrics = _metrics(pred, val_y)
        logger.info("alpha=%s val=%s", alpha, metrics)
        if metrics["rmse"] < best_score:
            best = model
            best_score = metrics["rmse"]
            best_metrics = metrics
    return best, best_metrics


def run_bio_regression_eval(
    *,
    arch: str,
    weights: str,
    checkpoint: str | None,
    train_config: str | None,
    benchmark_root: str,
    output_dir: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    resize_size: int,
    crop_size: int,
    n_last_blocks: int,
    use_avgpool: bool,
    autocast_dtype: torch.dtype,
    alphas: Tuple[float, ...],
    max_samples_per_split: int,
    seed: int,
    target_transform: str,
) -> Dict[str, float | str | int | bool]:
    os.makedirs(output_dir, exist_ok=True)
    if dataset.lower() != "bbbc013":
        raise ValueError("Currently supported bio regression datasets: bbbc013")
    data_root = Path(benchmark_root) / "Regression" / "BBBC013"
    backbone = load_backbone(repo_dir=".", arch=arch, weights=weights, checkpoint=checkpoint, train_config=train_config)
    feature_model = LinearFeatureModel(backbone, n_last_blocks=n_last_blocks, use_avgpool=use_avgpool, autocast_dtype=autocast_dtype)
    eval_transform = build_classification_transform(resize_size=resize_size, crop_size=crop_size)
    train_ds = BBBC013Dataset(str(data_root), "train", transform=eval_transform, seed=seed, max_samples=max_samples_per_split, target_transform=target_transform)
    val_ds = BBBC013Dataset(str(data_root), "val", transform=eval_transform, seed=seed, max_samples=max_samples_per_split, target_transform=target_transform)
    test_ds = BBBC013Dataset(str(data_root), "test", transform=eval_transform, seed=seed, max_samples=max_samples_per_split, target_transform=target_transform)
    train_x, train_y = extract_features_simple(feature_model, train_ds, batch_size=batch_size, num_workers=num_workers, desc="train")
    val_x, val_y = extract_features_simple(feature_model, val_ds, batch_size=batch_size, num_workers=num_workers, desc="val")
    test_x, test_y = extract_features_simple(feature_model, test_ds, batch_size=batch_size, num_workers=num_workers, desc="test")
    model, val_metrics = _fit_ridge(train_x.numpy(), train_y.numpy(), val_x.numpy(), val_y.numpy(), alphas)
    test_pred = model.predict(test_x.numpy())
    test_metrics = _metrics(test_pred, test_y.numpy())
    results: Dict[str, float | str | int | bool] = {
        "dataset": dataset,
        "checkpoint": checkpoint_stem(checkpoint, weights),
        "feature_dim": int(train_x.shape[1]),
        "n_last_blocks": n_last_blocks,
        "use_avgpool": use_avgpool,
        "target_transform": target_transform,
        "best_alpha": float(model.alpha),
    }
    results.update({f"val_{k}": v for k, v in val_metrics.items()})
    results.update({f"test_{k}": v for k, v in test_metrics.items()})
    with open(os.path.join(output_dir, "results_bio_regression.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_results(results, output_dir, RESULTS_FILENAME)
    logger.info("Bio regression results: %s", results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bio-image regression linear/ridge probe.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", default="dinov3_vitb16")
    parser.add_argument("--weights", default="")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--dataset", default="bbbc013", choices=["bbbc013"])
    parser.add_argument("--output-dir", default="outputs/bio_regression/bbbc013")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--no-avgpool", action="store_true")
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-transform", default="log1p", choices=["none", "log1p"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    distributed.enable(set_cuda_current_device=True, overwrite=True)
    run_bio_regression_eval(
        arch=args.arch,
        weights=args.weights,
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
        alphas=tuple(args.alphas),
        max_samples_per_split=args.max_samples_per_split,
        seed=args.seed,
        target_transform=args.target_transform,
    )


if __name__ == "__main__":
    main()
