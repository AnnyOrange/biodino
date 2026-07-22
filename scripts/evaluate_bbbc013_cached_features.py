#!/usr/bin/env python3
"""Run the compound-aware BBBC013 protocol on existing frozen-feature caches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dinov3.eval.bio_frozen_eval.datasets import BBBC013RegressionDataset
from dinov3.eval.bio_frozen_eval.probes import run_bbbc013_compound_oof_probe
from dinov3.eval.bio_frozen_eval.run_classification import BBBC013_SPLIT_PROTOCOL


FIELDS = [
    "candidate",
    "feature_file",
    "split",
    "target_transform",
    "fold_protocol",
    "r2",
    "spearman",
    "mae",
    "wortmannin_r2",
    "wortmannin_spearman",
    "wortmannin_mae",
    "ly294002_r2",
    "ly294002_spearman",
    "ly294002_mae",
    "n_train",
    "n_test",
]


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use LABEL=/path/to/features.npz")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate label and path must be non-empty")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-eval-root", type=Path)
    args = parser.parse_args()

    dataset = BBBC013RegressionDataset(Path(args.benchmark_root) / "Regression" / "BBBC013")
    expected_targets = np.asarray([sample.target for sample in dataset.samples], dtype=float)
    sample_paths = [sample.image_path for sample in dataset.samples]
    rows = []
    for label, feature_path in args.candidate:
        with np.load(feature_path) as cache:
            features = cache["features"]
            targets = cache["labels"].astype(float)
        if targets.shape != expected_targets.shape or not np.allclose(targets, expected_targets):
            raise ValueError(f"{label}: cached BBBC013 targets do not match the current plate map")
        result = run_bbbc013_compound_oof_probe(features, targets, sample_paths)
        row = {
            "candidate": label,
            "feature_file": str(feature_path),
            "split": BBBC013_SPLIT_PROTOCOL,
            "target_transform": "log1p",
            "fold_protocol": "leave-one-replicate-row-out",
            "n_train": result.n_train,
            "n_test": result.n_test,
            **result.metrics,
        }
        rows.append(row)
        if args.output_eval_root:
            source_candidates = [
                feature_path.parents[2] / "last_result.json",
                feature_path.parents[2] / "bbbc013" / "last_result.json",
            ]
            source_result_path = next((path for path in source_candidates if path.is_file()), None)
            if source_result_path is None:
                raise FileNotFoundError(
                    f"{label}: cannot find source last_result.json near {feature_path}"
                )
            source = json.loads(source_result_path.read_text())
            checkpoint = str(source.get("checkpoint", ""))
            checkpoint_id = Path(checkpoint).parent.name if checkpoint else str(source.get("model", "unknown"))
            eval_row = {
                **source,
                "dataset": "bbbc013",
                "task": "regression",
                "split": BBBC013_SPLIT_PROTOCOL,
                "feature_file": str(feature_path),
                "target_transform": "log1p",
                "fold_protocol": "leave-one-replicate-row-out",
                **result.to_dict(),
            }
            eval_dir = args.output_eval_root / label / "bio_regression" / "bbbc013" / checkpoint_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "last_result.json").write_text(json.dumps(eval_row, indent=2))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
