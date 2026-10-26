#!/usr/bin/env python3
"""Summarize FM14 LC25000 split-seed sensitivity against original v4 values."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_models"
RUN = ROOT / "outputs/02_eval_runs/fm14_lc25000_multiseed_20261016"
LOCAL_5090 = ROOT / "outputs/02_eval_runs/fm14_lc25000_siglip2_local5090_verify_20261016"
LOCAL_VERIFY = ROOT / "outputs/02_eval_runs/fm14_lc25000_local5090_verify_20261016"
FIELDS = ("model", "seed0", "seed1", "seed2", "seed3", "seed4", "n_seeds", "mean_macro_f1", "sd_macro_f1",
          "original_seed0", "seed0_reproduction_delta", "repeat_source", "original_source")


def main() -> None:
    rows = []
    for path in sorted(REPORT.glob("fm_*.json")):
        model = path.stem.removeprefix("fm_")
        cell = json.loads(path.read_text())["checkpoints"]["pretrained"]["cells"]["classification"]["lc25000"]
        preferred = cell["preferred"]
        original = json.loads(Path(preferred["source"]).read_text())
        values = {}
        repeat_root = LOCAL_5090 if model == "siglip2" else RUN
        local_seed0 = LOCAL_VERIFY / model / "seed0.json"
        if all((LOCAL_VERIFY / model / f"seed{seed}.json").is_file() for seed in (0, 1, 2)):
            if abs(json.loads(local_seed0.read_text())["macro_f1"] - float(original["macro_f1"])) < 1e-6:
                repeat_root = LOCAL_VERIFY
        for seed in range(5):
            source_root = RUN if seed >= 3 else repeat_root
            result = source_root / model / f"seed{seed}.json"
            if result.is_file():
                data = json.loads(result.read_text())
                if data["model"] != model or data["seed"] != seed or data["dataset"] != "lc25000":
                    raise ValueError(f"Mismatched result: {result}")
                values[seed] = float(data["macro_f1"])
        scores = list(values.values())
        delta = values.get(0, float("nan")) - float(original["macro_f1"])
        rows.append(dict(
            model=model, **{f"seed{seed}": f"{values[seed]:.8f}" if seed in values else "" for seed in range(5)},
            n_seeds=len(scores), mean_macro_f1=f"{statistics.mean(scores):.8f}" if scores else "",
            sd_macro_f1=f"{statistics.stdev(scores):.8f}" if len(scores) > 1 else "",
            original_seed0=f"{original['macro_f1']:.8f}",
            seed0_reproduction_delta=f"{delta:.8f}" if 0 in values else "",
            repeat_source=str(repeat_root / model),
            original_source=preferred["source"],
        ))
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete={sum(row['n_seeds'] == 5 for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
