#!/usr/bin/env python3
"""Add Cellpose and Multimodal CellSeg to the external dense-probe protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path("/mnt/huawei_deepcad/dinov3")
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT), str(BENCHMARK_MODEL_ROOT), str(BENCHMARK_MODEL_ROOT / "_vendor")]
sys.path.append("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311")

import run_dense_probe_benchmark as dense  # noqa: E402


DATASETS = {
    "cellpose": Path("/mnt/huawei_deepcad/benchmark/segmentation/Cellpose"),
    "multimodal_cellseg": Path(
        "/mnt/huawei_deepcad/benchmark/segmentation/Multimodal_CellSeg/neurips22_cellseg"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-feature-side", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dense.OUT_ROOT = Path(args.out_root)
    dense.DATASET_ROOTS.update(DATASETS)
    dense.DATASET_IMG_SIZE.update({"cellpose": 512, "multimodal_cellseg": 512})
    dense.DATASET_CONFIGS.update({
        "cellpose": {"num_classes": 2, "class_names": ["background", "cell"]},
        "multimodal_cellseg": {"num_classes": 2, "class_names": ["background", "cell"]},
    })
    result_path = dense.OUT_ROOT / "linear_probe" / args.dataset / args.model / "results.json"
    if result_path.exists():
        print(f"[skip] {result_path}")
        return 0

    run_args = SimpleNamespace(
        img_size=0,
        overwrite_cache=False,
        feature_canonical=False,
        device="cuda",
        extract_batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        max_feature_side=args.max_feature_side,
        overwrite_probe=False,
        epochs=args.epochs,
        lr=1e-3,
        probe_batch_size=64,
        probe_num_workers=0,
        weight_decay=1e-4,
        dropout=0.1,
        eval_every=args.eval_every,
        train_samples=None,
        train_fraction=None,
        seed=args.seed,
    )
    caches = {
        split: dense.extract_cache(run_args, args.model, args.dataset, split)
        for split in ("train", "val", "test")
    }
    result_path = dense.run_linear_probe(run_args, args.model, args.dataset, caches)
    print(f"[done] {args.model}/{args.dataset}: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
