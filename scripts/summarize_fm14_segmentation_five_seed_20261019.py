#!/usr/bin/env python3
"""Validate five-seed FM14 dense probes against the audited v4 source cells."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path

from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/02_eval_runs/fm14_segmentation_five_seed_20261019"
FIELDS = ("model", "dataset", "split_protocol", "budget", "source_root",
          "seed0", "seed1", "seed2", "seed3", "seed4", "n_seeds",
          "mean_mDice", "sd_mDice")


def main() -> None:
    cells = json.loads((RUN / "audit.json").read_text())["cells"]
    rows = []
    for cell in cells:
        if cell["status"] != "READY_CACHED":
            raise ValueError(f"Unready cell: {cell}")
        for budget in (20, 50):
            scores = {}
            for seed in range(5):
                base = (Path(cell["source_root"]) if seed < 3 else
                        RUN / "results" / cell["model"] / cell["dataset"] / cell["split_protocol"])
                path = base / f"E{budget}" / f"seed{seed}" / "results.json"
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                meta = payload["_meta"]
                if int(meta["seed"]) != seed or int(meta["probe_epochs"]) != budget:
                    raise ValueError(f"Wrong seed or budget: {path}")
                validate_best_head(meta)
                score = float(payload["test"]["mDice"])
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite mDice: {path}")
                scores[seed] = score
            values = list(scores.values())
            rows.append({"model": cell["model"], "dataset": cell["dataset"],
                         "split_protocol": cell["split_protocol"], "budget": budget,
                         "source_root": cell["source_root"],
                         **{f"seed{seed}": scores.get(seed, "") for seed in range(5)},
                         "n_seeds": len(scores), "mean_mDice": statistics.mean(values),
                         "sd_mDice": statistics.stdev(values) if len(values) > 1 else ""})
    target = RUN / "seed_summary.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    print(f"complete={sum(row['n_seeds'] == 5 for row in rows)}/{len(rows)} "
          f"additional_results={sum(row['n_seeds'] - 3 for row in rows)}")


if __name__ == "__main__":
    main()
