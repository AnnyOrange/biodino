#!/usr/bin/env python3
"""Summarize mature HS6-L retrieval/clustering readout variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    summary = []
    for path in sorted(args.input_root.glob("*/summary.csv")):
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows = [row for row in rows if row.get("dataset") in DATASETS and not row.get("error")]
        if {row["dataset"] for row in rows} != set(DATASETS):
            continue
        tag = path.parent.name
        summary.append(
            {
                "readout": tag,
                "macro_map_at_5": mean(rows, "map_at_5"),
                "macro_recall_at_1": mean(rows, "recall_at_1"),
                "macro_nmi": mean(rows, "nmi"),
                "macro_ari": mean(rows, "ari"),
                "datasets": {
                    row["dataset"]: {
                        "map_at_5": float(row["map_at_5"]),
                        "recall_at_1": float(row["recall_at_1"]),
                        "nmi": float(row["nmi"]),
                        "ari": float(row["ari"]),
                    }
                    for row in rows
                },
            }
        )
    summary.sort(key=lambda row: (row["macro_nmi"], row["macro_map_at_5"]), reverse=True)
    payload = {
        "input_root": str(args.input_root),
        "num_complete_readouts": len(summary),
        "ranking": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(
            f"{row['readout']:10s} mAP@5={row['macro_map_at_5']:.6f} "
            f"R@1={row['macro_recall_at_1']:.6f} NMI={row['macro_nmi']:.6f} "
            f"ARI={row['macro_ari']:.6f}"
        )


if __name__ == "__main__":
    main()
