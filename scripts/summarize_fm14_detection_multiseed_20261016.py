#!/usr/bin/env python3
"""Summarize matched FM14 detection-proxy seed repeats."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_models"
RUN = ROOT / "outputs/02_eval_runs/fm14_detection_multiseed_20261016"
FIELDS = ("model", "dataset", "seed0", "seed1", "seed2", "seed3", "seed4", "n_seeds", "mean_f1", "sd_f1", "min_f1", "max_f1", "seed0_source")


def main() -> None:
    rows = []
    for path in sorted(REPORT.glob("fm_*.json")):
        record = json.loads(path.read_text())
        model = path.stem.removeprefix("fm_")
        cells = record["checkpoints"]["pretrained"]["cells"]["detection_proxy"]
        for dataset in ("bbbc038", "conic", "livecell"):
            original = cells[dataset]["preferred"]
            seed0 = float(original["value"]) * 100
            scores = {0: seed0}
            for seed in range(1, 5):
                result = RUN / model / dataset / f"seed{seed}" / "component_result.json"
                if result.is_file():
                    data = json.loads(result.read_text())
                    if data["seed"] != seed or data["dataset"] != dataset or data["model"] != model:
                        raise ValueError(f"Mismatched result: {result}")
                    scores[seed] = float(data["test_patch_f1"])
            values = list(scores.values())
            rows.append(dict(
                model=model, dataset=dataset,
                **{f"seed{seed}": f"{scores[seed]:.6f}" if seed in scores else "" for seed in range(5)},
                n_seeds=len(values), mean_f1=f"{statistics.mean(values):.6f}",
                sd_f1=f"{statistics.stdev(values):.6f}" if len(values) > 1 else "",
                min_f1=f"{min(values):.6f}", max_f1=f"{max(values):.6f}",
                seed0_source=original["source"],
            ))
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["n_seeds"] == 5 for row in rows)
    print(f"complete={complete}/{len(rows)} new_seed_results={sum(row['n_seeds'] - 1 for row in rows)}")


if __name__ == "__main__":
    main()
