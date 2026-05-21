"""Bio-image k-NN evaluation for CamelyonPatch, aligned with DINOv3 k-NN conventions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, Tuple

import torch
import torch.backends.cudnn as cudnn

import dinov3.distributed as distributed
from dinov3.eval.bio_classification.common import (
    build_camelyonpatch_dataset,
    checkpoint_stem,
    ensure_main_process_write,
    extract_features_simple,
    get_num_classes_from_dataset,
    load_backbone,
    parse_autocast_dtype,
    topk_accuracy_from_scores,
)
from dinov3.eval.helpers import write_results

logger = logging.getLogger("dinov3.bio_classification.knn")
RESULTS_FILENAME = "results-knn.csv"


@torch.inference_mode()
def knn_predict_scores(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    num_classes: int,
    k: int,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Weighted k-NN classifier following DINOv3's cosine-similarity convention."""
    train_features = train_features.float().cuda(non_blocking=True)
    train_labels = train_labels.long().cuda(non_blocking=True).view(-1)
    test_features = test_features.float().cuda(non_blocking=True)
    scores_all = []
    kk = min(int(k), train_features.shape[0])
    train_t = train_features.T.contiguous()

    for start in range(0, test_features.shape[0], chunk_size):
        feats = test_features[start : start + chunk_size]
        sims = feats @ train_t
        top_sims, top_idx = sims.topk(kk, dim=1, largest=True, sorted=True)
        top_labels = train_labels[top_idx]
        weights = torch.softmax(top_sims / temperature, dim=1)
        scores = torch.zeros(feats.shape[0], num_classes, device=feats.device)
        scores.scatter_add_(1, top_labels, weights)
        scores_all.append(scores.cpu())
    return torch.cat(scores_all, dim=0)


def run_bio_knn_eval(
    *,
    repo_dir: str,
    arch: str,
    weights: str,
    checkpoint: str | None,
    train_config: str | None,
    data_root: str,
    output_dir: str,
    ks: Tuple[int, ...],
    temperature: float,
    batch_size: int,
    num_workers: int,
    resize_size: int,
    crop_size: int,
    train_split: str,
    metric_type: str,
    autocast_dtype: torch.dtype,
) -> Dict[str, float | str | int]:
    del autocast_dtype  # The backbone forward controls autocast internally when needed.
    cudnn.benchmark = True
    os.makedirs(output_dir, exist_ok=True)

    model = load_backbone(
        repo_dir=repo_dir,
        arch=arch,
        weights=weights,
        checkpoint=checkpoint,
        train_config=train_config,
        normalize=True,
    )
    train_dataset = build_camelyonpatch_dataset(data_root=data_root, split=train_split, resize_size=resize_size, crop_size=crop_size)
    test_dataset = build_camelyonpatch_dataset(data_root=data_root, split="test", resize_size=resize_size, crop_size=crop_size)
    num_classes = get_num_classes_from_dataset(train_dataset)

    train_features, train_labels = extract_features_simple(
        model, train_dataset, batch_size=batch_size, num_workers=num_workers, desc=train_split
    )
    test_features, test_labels = extract_features_simple(
        model, test_dataset, batch_size=batch_size, num_workers=num_workers, desc="test"
    )

    average = "macro" if metric_type == "mean_per_class_accuracy" else "micro"
    metric_topks = (1, 5) if num_classes >= 5 else (1,)
    results: Dict[str, float | str | int] = {
        "dataset": "camelyonpatch",
        "checkpoint": checkpoint_stem(checkpoint, weights),
        "metric_type": metric_type,
        "num_classes": num_classes,
    }

    for k in sorted(set(ks)):
        scores = knn_predict_scores(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            num_classes=num_classes,
            k=k,
            temperature=temperature,
            chunk_size=batch_size,
        )
        accs = topk_accuracy_from_scores(
            scores,
            test_labels,
            topks=metric_topks,
            average=average,
            num_classes=num_classes,
        )
        for metric_name, value in accs.items():
            pretty = metric_name.replace("top-", "Top ")
            results[f"{k} {pretty}"] = value
        logger.info("%s-NN results: %s", k, accs)

    if ensure_main_process_write():
        with open(os.path.join(output_dir, "results_bio_knn.json"), "w") as f:
            json.dump(results, f, indent=2)
        write_results(results, output_dir, RESULTS_FILENAME)
    logger.info("Bio k-NN results: %s", results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CamelyonPatch k-NN evaluation aligned with DINOv3 classification eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-dir", type=str, default=".")
    parser.add_argument("--arch", type=str, default="dinov3_vitb16")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None, help="Train/eval checkpoint; requires --train-config.")
    parser.add_argument("--train-config", type=str, default=None, help="Training config matching --checkpoint architecture.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "valid", "val"])
    parser.add_argument("--output-dir", type=str, default="./outputs/camelyonpatch_knn")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 100, 200])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--metric-type", type=str, default="mean_accuracy", choices=["mean_accuracy", "mean_per_class_accuracy"])
    parser.add_argument("--autocast-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    distributed.enable(set_cuda_current_device=True, overwrite=True)
    run_bio_knn_eval(
        repo_dir=args.repo_dir,
        arch=args.arch,
        weights=args.weights,
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        ks=tuple(args.ks),
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        train_split=args.train_split,
        metric_type=args.metric_type,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
    )


if __name__ == "__main__":
    main()
