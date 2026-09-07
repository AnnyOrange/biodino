#!/usr/bin/env python3
"""Strictly merge contiguous Fig. 3 feature shards into one protocol cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    from dinov3.eval.bio_frozen_eval.registry import build_dataset

    dataset, task = build_dataset(
        "bbbc048-cellcycle", "train", None, None, benchmark_root=args.benchmark_root
    )
    if task != "classification":
        raise RuntimeError(f"Expected classification task, got {task!r}")
    expected_labels = np.asarray([int(label) for _, label in dataset.samples], dtype=np.int64)
    cache_dir = args.output_dir / "panel_d_feature_cache/bbbc048-cellcycle"
    output_path = cache_dir / f"{args.model}.npz"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {output_path}; pass --overwrite after verification.")

    loaded = []
    for shard_index in range(args.num_shards):
        path = cache_dir / f"{args.model}.part{shard_index:02d}of{args.num_shards:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard {path}")
        with np.load(path, allow_pickle=False) as pack:
            missing = {"features", "labels", "indices"} - set(pack.files)
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}")
            loaded.append(
                (
                    np.asarray(pack["indices"], dtype=np.int64),
                    np.asarray(pack["features"]),
                    np.asarray(pack["labels"], dtype=np.int64),
                )
            )

    indices = np.concatenate([item[0] for item in loaded])
    features = np.concatenate([item[1] for item in loaded])
    labels = np.concatenate([item[2] for item in loaded])
    expected_indices = np.arange(len(expected_labels), dtype=np.int64)
    order = np.argsort(indices)
    if not np.array_equal(indices[order], expected_indices):
        raise ValueError("Shard indices are incomplete, duplicate, or outside the BBBC048 sample range.")
    features = features[order]
    labels = labels[order]
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Merged feature labels do not match the canonical BBBC048 ordering.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez(temporary, features=features, labels=labels, model=args.model)
    temporary.replace(output_path)
    print(
        f"[panel-d-merge] verified {args.num_shards} shards, features={features.shape}, wrote {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
