#!/usr/bin/env python3
"""Re-evaluate the only random-split v4 FM14 linear dataset with three seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCH)]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))

import run_external_fm_linear_probe as probe  # noqa: E402
from run_fm_rules import StandardExternalEncoder  # noqa: E402
from dinov3.eval.bio_frozen_eval.probes import run_classification_probe  # noqa: E402
from dinov3.eval.bio_frozen_eval.registry import build_dataset  # noqa: E402
from dinov3.eval.bio_frozen_eval.run_classification import (  # noqa: E402
    resolve_dataset_resize_size, resolve_image_size,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(probe.MODEL_REGISTRY))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/fm14_lc25000_multiseed_20261016")
    args = parser.parse_args()
    output = args.output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    if all((output / f"seed{seed}.json").exists() for seed in range(5)):
        print(f"[skip] {args.model}", flush=True)
        return 0
    torch.manual_seed(0)
    np.random.seed(0)
    size = resolve_image_size("lc25000", "best", 224)
    resize = resolve_dataset_resize_size("lc25000", size, 0)
    encoder = StandardExternalEncoder(args.model, size, resize)
    dataset, task = build_dataset("lc25000", "train", None, None, "/mnt/huawei_deepcad/benchmark")
    if task != "classification":
        raise ValueError(f"Unexpected LC25000 task: {task}")
    settings = SimpleNamespace(
        model=args.model, batch_size=64, num_workers=2, overwrite_features=False,
        no_save_features=False, save_paths=False,
    )
    feature_file = output / f"{args.model}.npz"
    features, labels = probe._extract(dataset, encoder, feature_file, settings)
    for seed in range(5):
        if (output / f"seed{seed}.json").exists():
            continue
        result = run_classification_probe(features, labels, train_fraction=0.8, seed=seed)
        record = dict(
            model=args.model, dataset="lc25000", split="internal-80-20",
            seed=seed, **result.to_dict(),
            feature_file=str(feature_file), crop_size=size, resize_size=resize,
            protocol_id="bio-eval-union-v4", status="OBSERVATIONAL",
        )
        (output / f"seed{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[complete] {args.model}/lc25000/seed{seed}: {result.metrics['macro_f1']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
