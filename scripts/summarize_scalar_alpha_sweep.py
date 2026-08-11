#!/usr/bin/env python3
"""Rank alpha checkpoints using the complete scalar-task suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from finalize_splus_checkpoint_ab import CLASSIFICATION_DATASETS, RETRIEVAL_DATASETS, load_json


def checkpoint_ids(root: Path) -> list[str]:
    base = root / "bio_classification" / "bloodmnist"
    return sorted(
        (path.name for path in base.iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path, checkpoint: str) -> dict:
    classification = [
        float(load_json(root / "bio_classification" / name / checkpoint / "last_result.json")["macro_f1"])
        for name in CLASSIFICATION_DATASETS
    ]
    regression = float(
        load_json(root / "bio_regression" / "bbbc005" / checkpoint / "last_result.json")["r2"]
    )
    retrieval = []
    clustering = []
    for name in RETRIEVAL_DATASETS:
        result = load_json(root / "bio_retrieval" / name / checkpoint / "last_result.json")
        retrieval.append(float(result["map_at_5"]))
        clustering.append(float(result["nmi"]))
    families = {
        "c25_macro_f1": mean(classification),
        "bbbc005_r2": regression,
        "retrieval4_map_at_5": mean(retrieval),
        "clustering4_nmi": mean(clustering),
    }
    return {
        "alpha_checkpoint": int(checkpoint),
        "alpha": int(checkpoint) / 100.0,
        **families,
        "family4_equal_mean": mean(families.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+")
    args = parser.parse_args()

    checkpoints = args.checkpoints or checkpoint_ids(args.eval_root)
    rows = [collect(args.eval_root, checkpoint) for checkpoint in checkpoints]
    rows.sort(key=lambda row: (-row["family4_equal_mean"], row["alpha_checkpoint"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "best.json").write_text(json.dumps(rows[0], indent=2) + "\n")
    (args.output_dir / "ranked.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
