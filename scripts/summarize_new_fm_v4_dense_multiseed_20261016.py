#!/usr/bin/env python3
"""Aggregate new-FM v4 detection and segmentation seed repeats."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from run_new_fm_v4_linear_20261016 import ROOT


RUN = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(grouped: dict[tuple, dict[int, float]], labels: tuple[str, ...]) -> list[dict]:
    rows = []
    for key, scores in sorted(grouped.items()):
        values = list(scores.values())
        rows.append(dict(
            zip(labels, key),
            **{f"seed{seed}": f"{scores[seed]:.8f}" if seed in scores else "" for seed in range(5)},
            n_seeds=len(values), mean=f"{statistics.mean(values):.8f}",
            sd=f"{statistics.stdev(values):.8f}" if len(values) > 1 else "",
        ))
    return rows


def main() -> None:
    detection = defaultdict(dict)
    for path in RUN.glob("*/detection/*/seed*/component_result.json"):
        model, _, dataset, seed_dir, _ = path.relative_to(RUN).parts
        seed = int(seed_dir.removeprefix("seed"))
        payload = json.loads(path.read_text())
        if (payload.get("model"), payload.get("dataset"), payload.get("probe_seed")) != (model, dataset, seed):
            raise ValueError(f"Mismatched detection result: {path}")
        detection[(model, dataset)][seed] = float(payload["test_patch_f1"])
    det_fields = ("model", "dataset", "seed0", "seed1", "seed2", "seed3", "seed4", "n_seeds", "mean", "sd")
    det_rows = summarize(detection, ("model", "dataset"))
    write_csv(RUN / "detection_seed_summary.csv", det_fields, det_rows)

    segmentation = defaultdict(dict)
    for path in RUN.glob("*/segmentation/*/*/E*/seed*/results.json"):
        model, _, dataset, split, budget_dir, seed_dir, _ = path.relative_to(RUN).parts
        seed = int(seed_dir.removeprefix("seed"))
        budget = int(budget_dir.removeprefix("E"))
        payload = json.loads(path.read_text())
        meta = payload["_meta"]
        if int(meta["seed"]) != seed:
            raise ValueError(f"Mismatched segmentation seed: {path}")
        segmentation[(model, dataset, split, budget)][seed] = float(payload["test"]["mDice"])
    seg_fields = ("model", "dataset", "split", "budget", "seed0", "seed1", "seed2", "seed3", "seed4", "n_seeds", "mean", "sd")
    seg_rows = summarize(segmentation, ("model", "dataset", "split", "budget"))
    write_csv(RUN / "segmentation_seed_summary.csv", seg_fields, seg_rows)
    print(f"detection={sum(r['n_seeds'] == 5 for r in det_rows)}/{len(det_rows)} "
          f"segmentation={sum(r['n_seeds'] == 5 for r in seg_rows)}/{len(seg_rows)}")


if __name__ == "__main__":
    main()
