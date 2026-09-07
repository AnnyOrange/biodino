#!/usr/bin/env python3
"""Evaluate frozen cross-species organelle retrieval between HPA and Cyclops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from dinov3.eval.bio_frozen_eval.datasets import load_image
from dinov3.eval.bio_frozen_eval.encoder import (
    Dinov3CkptEncoder,
    extract_features,
    parse_autocast_dtype,
)
from dinov3.eval.bio_frozen_eval.retrieval_clustering import query_gallery_metrics


class CrossSpeciesManifestDataset(Dataset):
    def __init__(self, benchmark_root: Path, manifest: Path, role: str):
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["role"] == role]
        if not self.rows:
            raise ValueError(f"No {role} rows in {manifest}")
        self.benchmark_root = benchmark_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.benchmark_root / row["image_path"]
        return load_image(path), int(row["label"]), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-last-blocks", type=int, default=2)
    parser.add_argument("--no-avgpool", action="store_true")
    parser.add_argument("--autocast-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-result", action="store_true")
    return parser.parse_args()


def macro_query_gallery_metrics(
    gallery_x: np.ndarray,
    gallery_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    metric_device: str,
) -> dict[str, float]:
    per_class = [
        query_gallery_metrics(
            gallery_x,
            gallery_y,
            query_x[query_y == label],
            query_y[query_y == label],
            metric_device=metric_device,
        )
        for label in sorted(np.unique(query_y))
    ]
    return {
        metric: float(np.mean([values[metric] for values in per_class]))
        for metric in per_class[0]
    }


def direction_metrics(
    gallery_x: np.ndarray,
    gallery_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    metric_device: str,
) -> dict:
    return {
        "micro": query_gallery_metrics(
            gallery_x,
            gallery_y,
            query_x,
            query_y,
            metric_device=metric_device,
        ),
        "macro_class": macro_query_gallery_metrics(
            gallery_x,
            gallery_y,
            query_x,
            query_y,
            metric_device=metric_device,
        ),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists() and not args.overwrite_result:
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    encoder = Dinov3CkptEncoder(
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        device=args.device,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
    )
    features = {}
    labels = {}
    feature_files = {}
    for role in ("human", "yeast"):
        dataset = CrossSpeciesManifestDataset(args.benchmark_root, args.manifest, role)
        feature_file = args.output_dir / "features" / f"{args.model_name}_{role}.npz"
        x, y = extract_features(
            dataset,
            encoder,
            feature_file,
            args.batch_size,
            args.num_workers,
            args.overwrite_features,
            args.model_name,
            save_features=True,
            save_paths=True,
        )
        features[role] = x.astype(np.float32)
        labels[role] = np.asarray(y).astype(np.int64)
        feature_files[role] = str(feature_file)

    directions = {
        "human_query_yeast_gallery": direction_metrics(
            features["yeast"],
            labels["yeast"],
            features["human"],
            labels["human"],
            metric_device=args.metric_device,
        ),
        "yeast_query_human_gallery": direction_metrics(
            features["human"],
            labels["human"],
            features["yeast"],
            labels["yeast"],
            metric_device=args.metric_device,
        ),
    }
    bidirectional_macro = {
        metric: float(
            np.mean(
                [values["macro_class"][metric] for values in directions.values()]
            )
        )
        for metric in next(iter(directions.values()))["macro_class"]
    }
    payload = {
        "model": args.model_name,
        "protocol": "hpa-cyclops-shared-organelle-v1",
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "manifest": str(args.manifest),
        "n_human": int(len(labels["human"])),
        "n_yeast": int(len(labels["yeast"])),
        "n_classes": int(len(np.unique(labels["human"]))),
        "readout": {
            "n_last_blocks": args.n_last_blocks,
            "avgpool": not args.no_avgpool,
        },
        "feature_files": feature_files,
        "directions": directions,
        "bidirectional_macro": bidirectional_macro,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
