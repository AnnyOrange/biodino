#!/usr/bin/env python3
"""Build a live completion matrix for the 14-model external FM benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


MODELS = [
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
]
PROTOCOLS = ["hpa", "nct-cross", "lc25000-diagnostic", "rxrx1-cross"]
SEG_DATASETS = ["bbbc038", "conic", "livecell", "monuseg", "pannuke", "tissuenet"]


def successful_datasets(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("error") and row.get("dataset"):
                rows.add(row["dataset"])
    return rows


def successful_ood_models(path: Path) -> set[str]:
    if not path.exists():
        return set()
    models: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("error") and row.get("model"):
                models.add(row["model"])
    return models


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="outputs/02_eval_runs/external_fm_fair_protocol_20260721",
    )
    parser.add_argument(
        "--ood-summary",
        default="/mnt/huawei_deepcad/benchmark_model/benchmark_runs/external_ood_protocol_20260707/summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/00_reports/external_foundation_gapfill_20260723",
    )
    args = parser.parse_args()

    # Import here so this status tool still reports partial files if the main
    # training environment is temporarily unavailable.
    from dinov3.eval.bio_frozen_eval.registry import ALL_DATASETS

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    ood_models = successful_ood_models(Path(args.ood_summary))
    rows = []
    for model in MODELS:
        class_done = successful_datasets(run_root / "classification" / model / "summary.csv")
        retrieval_done = [
            protocol
            for protocol in PROTOCOLS
            if (
                run_root / "retrieval_clustering" / model / "done" / model / f"{protocol}.json"
            ).exists()
        ]
        segmentation_done = [
            dataset
            for dataset in SEG_DATASETS
            if (
                run_root / "segmentation" / "linear_probe" / dataset / model / "results.json"
            ).exists()
        ]
        rows.append({
            "model": model,
            "classification_done": len(class_done),
            "classification_total": len(ALL_DATASETS),
            "classification_missing": ";".join(d for d in ALL_DATASETS if d not in class_done),
            "retrieval_done": len(retrieval_done),
            "retrieval_total": len(PROTOCOLS),
            "retrieval_protocols": ";".join(retrieval_done),
            "segmentation_done": len(segmentation_done),
            "segmentation_total": len(SEG_DATASETS),
            "ood_done": int(model in ood_models),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    csv_tmp = output_dir / "completion_matrix.csv.tmp"
    with csv_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, output_dir / "completion_matrix.csv")

    class_done = sum(int(row["classification_done"]) for row in rows)
    class_total = sum(int(row["classification_total"]) for row in rows)
    retr_done = sum(int(row["retrieval_done"]) for row in rows)
    retr_total = sum(int(row["retrieval_total"]) for row in rows)
    seg_done = sum(int(row["segmentation_done"]) for row in rows)
    seg_total = sum(int(row["segmentation_total"]) for row in rows)
    ood_done = sum(int(row["ood_done"]) for row in rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# External foundation model gap-fill status",
        "",
        f"Updated: {timestamp}",
        "",
        f"- Classification: {class_done}/{class_total} model-dataset results",
        f"- Retrieval/clustering: {retr_done}/{retr_total} model-protocol results",
        f"- Dense segmentation: {seg_done}/{seg_total} model-dataset results",
        f"- External OOD: {ood_done}/{len(MODELS)} models",
        "",
        "| Model | Classification | Retrieval/clustering | Segmentation | OOD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['classification_done']}/{row['classification_total']} "
            f"| {row['retrieval_done']}/{row['retrieval_total']} "
            f"| {row['segmentation_done']}/{row['segmentation_total']} | {row['ood_done']}/1 |"
        )
    atomic_write(output_dir / "README.md", "\n".join(lines) + "\n")
    atomic_write(
        output_dir / "status.json",
        json.dumps({
            "updated": timestamp,
            "classification": [class_done, class_total],
            "retrieval_clustering": [retr_done, retr_total],
            "segmentation": [seg_done, seg_total],
            "ood": [ood_done, len(MODELS)],
        }, indent=2) + "\n",
    )
    print(lines[4])
    print(lines[5])
    print(lines[6])
    print(lines[7])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
