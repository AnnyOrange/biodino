#!/usr/bin/env python3
"""Summarize FM14 v4 clustering NMI over three KMeans seeds."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_models"
RUN = ROOT / "outputs/02_eval_runs/fm14_retrieval_multiseed_20261016"
LOCAL_5090 = ROOT / "outputs/02_eval_runs/fm14_retrieval_local5090_verify_20261016"
DATASETS = ("nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k",
            "hpa-subcellular", "rxrx1-cross", "lc25000")
FIELDS = ("model", "dataset", "seed0_nmi", "seed1_nmi", "seed2_nmi", "seed3_nmi", "seed4_nmi", "n_seeds",
          "mean_nmi", "sd_nmi", "original_seed0_nmi", "seed0_reproduction_delta",
          "repeat_source", "original_source")


def clustering_row(payload: dict, dataset: str) -> dict:
    if "rows" not in payload:
        return payload
    aggregation = "location" if dataset == "hpa-subcellular" else "global-perturbation"
    rows = [row for row in payload["rows"] if row.get("task") == "clustering"
            and row.get("aggregation") == aggregation]
    if dataset == "hpa-subcellular":
        rows = [row for row in rows if row.get("protocol") == "custom-v1-single-location-all41"]
    if len(rows) != 1:
        raise ValueError(f"Expected one {dataset}/{aggregation} clustering row; got {len(rows)}")
    return rows[0]


def main() -> None:
    rows = []
    for path in sorted(REPORT.glob("fm_*.json")):
        model = path.stem.removeprefix("fm_")
        cells = json.loads(path.read_text())["checkpoints"]["pretrained"]["cells"]["clustering"]
        for dataset in DATASETS:
            preferred = cells[dataset]["preferred"]
            if preferred is None:
                continue
            if preferred["metric"] != "nmi":
                raise ValueError(f"Unexpected clustering metric for {model}/{dataset}")
            repeat_root = RUN
            local_seed0 = LOCAL_5090 / model / dataset / "seed0.json"
            if all((LOCAL_5090 / model / dataset / f"seed{seed}.json").is_file() for seed in (0, 1, 2)):
                local_payload = json.loads(local_seed0.read_text())
                if abs(float(clustering_row(local_payload, dataset)["nmi"]) - float(preferred["value"])) < 1e-6:
                    repeat_root = LOCAL_5090
            values = {}
            for seed in range(5):
                source_root = RUN if seed >= 3 else repeat_root
                result = source_root / model / dataset / f"seed{seed}.json"
                if result.is_file():
                    payload = json.loads(result.read_text())
                    if payload["model"] != model or payload["dataset"] != dataset or payload["seed"] != seed:
                        raise ValueError(f"Mismatched result: {result}")
                    values[seed] = float(clustering_row(payload, dataset)["nmi"])
            scores = list(values.values())
            delta = values.get(0, float("nan")) - float(preferred["value"])
            rows.append(dict(
                model=model, dataset=dataset,
                **{f"seed{seed}_nmi": f"{values[seed]:.8f}" if seed in values else "" for seed in range(5)},
                n_seeds=len(scores), mean_nmi=f"{statistics.mean(scores):.8f}" if scores else "",
                sd_nmi=f"{statistics.stdev(scores):.8f}" if len(scores) > 1 else "",
                original_seed0_nmi=f"{preferred['value']:.8f}",
                seed0_reproduction_delta=f"{delta:.8f}" if 0 in values else "",
                repeat_source=str(repeat_root / model / dataset),
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
