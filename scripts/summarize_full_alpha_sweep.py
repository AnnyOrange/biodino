#!/usr/bin/env python3
"""Rank fully evaluated alpha checkpoints with equal family weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from finalize_splus_checkpoint_ab import SUMMARY_METRICS, collect_candidate


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    details = []
    for checkpoint in args.checkpoints:
        candidate = collect_candidate(
            f"alpha_{int(checkpoint) / 100:.2f}",
            args.core_root,
            args.dense_root,
            checkpoint,
        )
        score = mean(candidate["summary"][key] for key in SUMMARY_METRICS)
        rows.append(
            {
                "alpha_checkpoint": int(checkpoint),
                "alpha": int(checkpoint) / 100.0,
                **candidate["summary"],
                "family6_equal_mean": score,
            }
        )
        details.append(candidate)
    rows.sort(key=lambda row: (-row["family6_equal_mean"], row["alpha_checkpoint"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "best.json").write_text(json.dumps(rows[0], indent=2) + "\n")
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
