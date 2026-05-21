"""
Bio-image linear evaluation for CamelyonPatch, aligned with DINOv3 classification eval.

This script keeps the bio-image I/O corrosion layer, but follows DINOv3's
classification conventions more closely:
  - deterministic eval transform for pre-extracted features by default;
  - DINOv3 `create_linear_input` for CLS/avgpool feature construction;
  - DINOv3 LR scaling rule `lr * global_batch / 256`;
  - explicit top-1 / balanced top-1 / macro-F1 / macro-AUROC metrics;
  - `results-linear.csv` plus a detailed JSON result file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import dinov3.distributed as distributed
from dinov3.eval.bio_classification.common import (
    LinearFeatureModel,
    build_camelyonpatch_dataset,
    build_classification_transform,
    checkpoint_stem,
    compute_metrics_from_logits,
    ensure_main_process_write,
    extract_features_simple,
    get_num_classes_from_dataset,
    load_backbone,
    parse_autocast_dtype,
)
from dinov3.eval.helpers import write_results
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3.bio_classification.linear")
RESULTS_FILENAME = "results-linear.csv"


def scale_lr(learning_rate: float, batch_size: int) -> float:
    return float(learning_rate) * (batch_size * distributed.get_world_size()) / 256.0


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _init_linear(feature_dim: int, num_classes: int) -> nn.Linear:
    classifier = nn.Linear(feature_dim, num_classes)
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()
    return classifier.cuda()


def extract_split_features(
    *,
    feature_model: nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info("Extracting %s features", split_name)
    features, labels = extract_features_simple(
        feature_model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        desc=split_name,
    )
    features = features.float().cpu()
    labels = labels.long().cpu().view(-1)
    logger.info("%s features=%s labels=%s", split_name, tuple(features.shape), tuple(labels.shape))
    return features, labels


def train_linear_on_features(
    *,
    classifier: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> nn.Module:
    dataset = TensorDataset(train_features, train_labels)
    sampler = None
    shuffle = True
    if distributed.is_enabled() and distributed.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        shuffle = False
        classifier = DistributedDataParallel(classifier, device_ids=[torch.cuda.current_device()])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    if len(loader) == 0:
        raise ValueError("Training feature loader is empty; reduce --batch-size.")

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=0.0)
    criterion = nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")

    classifier.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for feats, targets in metric_logger.log_every(loader, 50, f"Linear train epoch {epoch + 1}/{epochs}"):
            feats = feats.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            loss = criterion(classifier(feats), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric_logger.update(loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

    classifier.eval()
    return _unwrap(classifier)


@torch.inference_mode()
def predict_logits(classifier: nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    outputs = []
    classifier.eval()
    for (feats,) in loader:
        outputs.append(classifier(feats.cuda(non_blocking=True)).float().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_classifier(
    *,
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
) -> Dict[str, float]:
    logits = predict_logits(classifier, features, batch_size=batch_size)
    return compute_metrics_from_logits(logits, labels, num_classes=num_classes, batch_size=batch_size)


def grid_search_lr(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    feature_dim: int,
    num_classes: int,
    learning_rates: Iterable[float],
    epochs: int,
    batch_size: int,
    num_workers: int,
    selection_metric: str,
) -> Tuple[float, nn.Module, Dict[str, float]]:
    best_lr = None
    best_score = float("-inf")
    best_classifier = None
    best_metrics: Dict[str, float] = {}

    for raw_lr in learning_rates:
        lr = scale_lr(float(raw_lr), batch_size)
        logger.info("Training linear classifier raw_lr=%g scaled_lr=%g", raw_lr, lr)
        classifier = _init_linear(feature_dim, num_classes)
        classifier = train_linear_on_features(
            classifier=classifier,
            train_features=train_features,
            train_labels=train_labels,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_metrics = evaluate_classifier(
            classifier=classifier,
            features=val_features,
            labels=val_labels,
            num_classes=num_classes,
            batch_size=batch_size,
        )
        logger.info("raw_lr=%g val_metrics=%s", raw_lr, val_metrics)
        if selection_metric not in val_metrics:
            raise KeyError(f"Selection metric {selection_metric!r} not in metrics: {sorted(val_metrics)}")
        if val_metrics[selection_metric] > best_score:
            best_score = val_metrics[selection_metric]
            best_lr = float(raw_lr)
            best_classifier = classifier
            best_metrics = val_metrics

    assert best_lr is not None and best_classifier is not None
    logger.info("Best raw_lr=%g %s=%.4f", best_lr, selection_metric, best_score)
    return best_lr, best_classifier, best_metrics


def run_bio_linear_eval(
    *,
    repo_dir: str,
    arch: str,
    weights: str,
    checkpoint: str | None,
    train_config: str | None,
    data_root: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    n_last_blocks: int,
    use_avgpool: bool,
    learning_rates: Tuple[float, ...],
    train_split: str,
    val_split: str,
    resize_size: int,
    crop_size: int,
    train_augmentation: bool,
    autocast_dtype: torch.dtype,
    selection_metric: str,
    dataset: str = "camelyonpatch",
    benchmark_root: str = "/mnt/huawei_deepcad/benchmark",
    max_samples_per_split: int = 0,
    seed: int = 0,
) -> Dict[str, float | str | int | bool]:
    cudnn.benchmark = True
    os.makedirs(output_dir, exist_ok=True)

    backbone = load_backbone(
        repo_dir=repo_dir,
        arch=arch,
        weights=weights,
        checkpoint=checkpoint,
        train_config=train_config,
    )
    feature_model = LinearFeatureModel(backbone, n_last_blocks=n_last_blocks, use_avgpool=use_avgpool, autocast_dtype=autocast_dtype)

    if dataset == "camelyonpatch":
        if not data_root:
            raise ValueError("--data-root is required when --dataset camelyonpatch.")
        train_dataset = build_camelyonpatch_dataset(
            data_root=data_root,
            split=train_split,
            train_transform=train_augmentation,
            resize_size=resize_size,
            crop_size=crop_size,
        )
        val_dataset = build_camelyonpatch_dataset(data_root=data_root, split=val_split, resize_size=resize_size, crop_size=crop_size)
        test_dataset = build_camelyonpatch_dataset(data_root=data_root, split="test", resize_size=resize_size, crop_size=crop_size)
    else:
        from dinov3.eval.bio_classification.datasets.benchmark import build_bio_classification_dataset

        train_dataset = build_bio_classification_dataset(
            dataset,
            benchmark_root,
            train_split,
            transform=build_classification_transform(train_transform=train_augmentation, resize_size=resize_size, crop_size=crop_size),
            max_samples=max_samples_per_split,
            seed=seed,
        )
        val_dataset = build_bio_classification_dataset(
            dataset,
            benchmark_root,
            val_split,
            transform=build_classification_transform(resize_size=resize_size, crop_size=crop_size),
            max_samples=max_samples_per_split,
            seed=seed,
        )
        test_dataset = build_bio_classification_dataset(
            dataset,
            benchmark_root,
            "test",
            transform=build_classification_transform(resize_size=resize_size, crop_size=crop_size),
            max_samples=max_samples_per_split,
            seed=seed,
        )
    num_classes = get_num_classes_from_dataset(train_dataset)

    train_features, train_labels = extract_split_features(
        feature_model=feature_model, dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, split_name=train_split
    )
    val_features, val_labels = extract_split_features(
        feature_model=feature_model, dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, split_name=val_split
    )
    test_features, test_labels = extract_split_features(
        feature_model=feature_model, dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, split_name="test"
    )

    feature_dim = int(train_features.shape[1])
    best_lr, best_classifier, best_val_metrics = grid_search_lr(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        feature_dim=feature_dim,
        num_classes=num_classes,
        learning_rates=learning_rates,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=max(0, min(num_workers, 4)),
        selection_metric=selection_metric,
    )
    test_metrics = evaluate_classifier(
        classifier=best_classifier,
        features=test_features,
        labels=test_labels,
        num_classes=num_classes,
        batch_size=batch_size,
    )

    results: Dict[str, float | str | int | bool] = {
        "dataset": dataset,
        "checkpoint": checkpoint_stem(checkpoint, weights),
        "best_lr": best_lr,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "n_last_blocks": n_last_blocks,
        "use_avgpool": use_avgpool,
        "epochs": epochs,
        "batch_size": batch_size,
        "selection_metric": selection_metric,
    }
    results.update({f"val_{k}": v for k, v in best_val_metrics.items()})
    results.update({f"test_{k}": v for k, v in test_metrics.items()})

    if ensure_main_process_write():
        with open(os.path.join(output_dir, "results_bio_linear.json"), "w") as f:
            json.dump(results, f, indent=2)
        torch.save(best_classifier.state_dict(), os.path.join(output_dir, "best_linear_classifier.pth"))
        write_results(results, output_dir, RESULTS_FILENAME)
    logger.info("Bio linear results: %s", results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CamelyonPatch linear evaluation aligned with DINOv3 classification eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-dir", type=str, default=".")
    parser.add_argument("--arch", type=str, default="dinov3_vitb16")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None, help="Train/eval checkpoint; requires --train-config.")
    parser.add_argument("--train-config", type=str, default=None, help="Training config matching --checkpoint architecture.")
    parser.add_argument("--dataset", type=str, default="camelyonpatch", help="camelyonpatch or benchmark datasets: bloodmnist, bbbc048, cyclops, midog25")
    parser.add_argument("--benchmark-root", type=str, default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--data-root", type=str, default="", help="Required only for --dataset camelyonpatch.")
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "valid", "val"])
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "valid", "val"])
    parser.add_argument("--output-dir", type=str, default="./outputs/camelyonpatch_linear")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--no-avgpool", action="store_true")
    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true", default=True, help="Use random train augmentation for train split feature extraction.")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false", help="Disable train split random augmentation.")
    parser.add_argument("--selection-metric", type=str, default="accuracy_top1")
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="Debug/smoke cap; 0 means use all samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autocast-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    distributed.enable(set_cuda_current_device=True, overwrite=True)
    run_bio_linear_eval(
        repo_dir=args.repo_dir,
        arch=args.arch,
        weights=args.weights,
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        learning_rates=tuple(args.learning_rates),
        train_split=args.train_split,
        val_split=args.val_split,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        train_augmentation=args.train_augmentation,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
        selection_metric=args.selection_metric,
        dataset=args.dataset,
        benchmark_root=args.benchmark_root,
        max_samples_per_split=args.max_samples_per_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
