#!/usr/bin/env python3
"""Summarize v4 clustering variability for the new frozen encoders."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from run_new_fm_v4_clustering_multiseed_20261016 import DATASETS
from run_new_fm_v4_linear_20261016 import MODELS, ROOT
from summarize_fm14_retrieval_multiseed_20261016 import clustering_row


RUN = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"
FIELDS = ("model", "dataset", "seed0_nmi", "seed1_nmi", "seed2_nmi", "seed3_nmi", "seed4_nmi",
          "n_seeds", "mean_nmi", "sd_nmi", "result_dir")


def main() -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            directory = RUN / model / "retrieval_exact" / dataset
            scores = {}
            for seed in range(5):
                file = directory / f"seed{seed}.json"
                if not file.is_file():
                    continue
                payload = json.loads(file.read_text())
                if (payload.get("model"), payload.get("dataset"), payload.get("seed")) != (model, dataset, seed):
                    raise ValueError(f"Mismatched result: {file}")
                scores[seed] = float(clustering_row(payload, dataset)["nmi"])
            if not scores:
                continue
            values = list(scores.values())
            rows.append(dict(
                model=model, dataset=dataset,
                **{f"seed{seed}_nmi": f"{scores[seed]:.8f}" if seed in scores else "" for seed in range(5)},
                n_seeds=len(values), mean_nmi=f"{statistics.mean(values):.8f}",
                sd_nmi=f"{statistics.stdev(values):.8f}" if len(values) > 1 else "",
                result_dir=str(directory),
            ))
    with (RUN / "clustering_seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete={sum(row['n_seeds'] == 5 for row in rows)}/{len(MODELS) * len(DATASETS)}")


if __name__ == "__main__":
    main()
