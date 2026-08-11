#!/usr/bin/env python3
"""Rank scalar-suite candidates stored in separate evaluation roots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_scalar_alpha_sweep import collect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "EVAL_ROOT", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label, eval_root, checkpoint in args.candidate:
        row = collect(Path(eval_root), checkpoint)
        row.pop("alpha_checkpoint", None)
        row.pop("alpha", None)
        rows.append({"label": label, "checkpoint": int(checkpoint), **row})
    rows.sort(key=lambda row: (-row["family4_equal_mean"], row["label"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "ranked.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "best.json").write_text(json.dumps(rows[0], indent=2) + "\n")
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
