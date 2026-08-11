#!/usr/bin/env python3
"""Rank raw checkpoints whose scalar and dense results live in separate roots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from finalize_splus_checkpoint_ab import SUMMARY_METRICS, collect_candidate


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
            f"ckpt_{checkpoint}", args.core_root, args.dense_root, checkpoint
        )
        score = mean(candidate["summary"][metric] for metric in SUMMARY_METRICS)
        rows.append(
            {
                "checkpoint": int(checkpoint),
                **candidate["summary"],
                "family6_equal_mean": score,
            }
        )
        details.append(candidate)
    rows.sort(key=lambda row: (-row["family6_equal_mean"], row["checkpoint"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary_by_checkpoint.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "best.json").write_text(json.dumps(rows[0], indent=2) + "\n")
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
