#!/usr/bin/env python3
"""Reprobe cached new-model LC25000 features over three split seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from dinov3.eval.bio_frozen_eval.probes import run_classification_probe
from run_new_fm_v4_linear_20261016 import MODELS, ROOT


FIELDS = ("model", "seed0", "seed1", "seed2", "seed3", "seed4", "mean_macro_f1", "sd_macro_f1", "original_seed0",
          "seed0_reproduction_delta", "feature_file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    root = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / args.model / "linear"
    original_file = root / "lc25000.json"
    if not original_file.is_file():
        raise FileNotFoundError(f"LC25000 linear probe is incomplete: {original_file}")
    original = json.loads(original_file.read_text())
    feature_file = Path(original["feature_file"])
    if not feature_file.is_file() or not original_file.is_file():
        raise FileNotFoundError(f"LC25000 linear probe is incomplete: {feature_file}, {original_file}")
    with np.load(feature_file) as cached:
        features, labels = cached["features"].astype(np.float32), cached["labels"]
    scores = {}
    for seed in range(5):
        target = root / f"lc25000_seed{seed}.json"
        if target.is_file():
            scores[seed] = float(json.loads(target.read_text())["macro_f1"])
            continue
        result = run_classification_probe(features, labels, train_fraction=0.8, seed=seed)
        scores[seed] = result.metrics["macro_f1"]
        target.write_text(json.dumps(dict(
            model=args.model, dataset="lc25000", seed=seed, split="internal-80-20",
            feature_file=str(feature_file), **result.to_dict(),
        ), indent=2) + "\n")
    delta = scores[0] - float(original["macro_f1"])
    if abs(delta) > 1e-6:
        raise ValueError(f"LC25000 seed0 did not reproduce: delta={delta}")
    row = dict(
        model=args.model, **{f"seed{seed}": f"{scores[seed]:.8f}" for seed in range(5)},
        mean_macro_f1=f"{statistics.mean(scores.values()):.8f}",
        sd_macro_f1=f"{statistics.stdev(scores.values()):.8f}",
        original_seed0=f"{original['macro_f1']:.8f}",
        seed0_reproduction_delta=f"{delta:.8f}", feature_file=str(feature_file),
    )
    with (root / "lc25000_seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
