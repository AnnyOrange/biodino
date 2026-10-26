#!/usr/bin/env python3
"""Extend new-FM v4 clustering to five seeds from existing feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_fm14_cached_clustering_five_seed_20261019 import DATASETS, evaluate


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"
MODELS = ("distillpath", "morphem", "genbio_pathfm", "openpath")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--verify-seed2", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        cell = RUN / args.model / "retrieval_exact" / dataset
        reference = json.loads((cell / "seed2.json").read_text())
        for seed in ((2,) if args.verify_seed2 else (3, 4)):
            target = cell / f"seed{seed}.json"
            if target.is_file() and not args.verify_seed2:
                print(f"[skip] {args.model}/{dataset}/seed{seed}", flush=True)
                continue
            result = evaluate(reference, dataset, seed)
            result["multiseed_protocol"] = "v4-clustering-seed-repeat"
            if args.verify_seed2:
                old_rows = reference.get("rows", [reference])
                new_rows = result.get("rows", [result])
                for old, new in zip(old_rows, new_rows):
                    if "cluster_accuracy" in old:
                        for metric in ("cluster_accuracy", "ari", "nmi", "silhouette_cosine"):
                            if not np.isclose(old[metric], new[metric], atol=1e-6):
                                raise ValueError(f"Mismatch in {dataset}/{metric}")
                print(f"[verified] {args.model}/{dataset}/seed2", flush=True)
                continue
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(target)
            print(f"[complete] {args.model}/{dataset}/seed{seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
