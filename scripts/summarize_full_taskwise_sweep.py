#!/usr/bin/env python3
"""Rank a complete taskwise checkpoint sweep with equal family weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from finalize_splus_checkpoint_ab import SUMMARY_METRICS, collect_candidate


def checkpoint_ids(root: Path) -> list[str]:
    dataset_root = root / "bio_classification" / "bloodmnist"
    checkpoints = [path.name for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(checkpoints, key=int)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+")
    args = parser.parse_args()

    checkpoints = args.checkpoints or checkpoint_ids(args.eval_root)
    if not checkpoints:
        raise ValueError(f"No checkpoint results found under {args.eval_root}")

    rows = []
    details = []
    for checkpoint in checkpoints:
        candidate = collect_candidate(
            f"ckpt_{checkpoint}", args.eval_root, args.eval_root, checkpoint
        )
        family_mean = mean(candidate["summary"][key] for key in SUMMARY_METRICS)
        rows.append(
            {
                "checkpoint": int(checkpoint),
                **candidate["summary"],
                "family6_equal_mean": family_mean,
            }
        )
        details.append(candidate)

    ranked = sorted(rows, key=lambda row: (-row["family6_equal_mean"], row["checkpoint"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_by_checkpoint.csv", ranked)
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.output_dir / "best.json").write_text(json.dumps(ranked[0], indent=2) + "\n")

    lines = [
        "# Full taskwise checkpoint sweep",
        "",
        "Selection uses an equal mean over six families: Classification-25 macro-F1, "
        "BBBC005 R2, Retrieval-4 mAP@5, Clustering-4 NMI, Segmentation-8 mDice, "
        "and LIVECell detection F1.",
        "",
        "| rank | checkpoint | family6 | C25 | Reg | Ret | Clust | Seg | Det |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['rank']} | {row['checkpoint']} | {row['family6_equal_mean']:.6f} | "
            f"{row['c25_macro_f1']:.6f} | {row['bbbc005_r2']:.6f} | "
            f"{row['retrieval4_map_at_5']:.6f} | {row['clustering4_nmi']:.6f} | "
            f"{row['segmentation8_mdice']:.6f} | {row['livecell_detection_f1']:.6f} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(ranked[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
