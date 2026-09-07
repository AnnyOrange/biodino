#!/usr/bin/env python3
"""Audit a deterministic Acquisition-Orbit Deflation training matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


COMMON_METRICS = (
    "total_loss",
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "ibot_loss",
    "koleo_loss",
    "backbone_grad_norm",
)
DBI_METRICS = (
    "acq_deflation_loss",
    "acq_deflation_loss_weight",
    "acq_random_tangent",
    "acq_shuffled_tangent",
    "acq_tangent_rank",
    "acq_tangent_singular_rms",
    "acq_patch_tangent_rank",
    "acq_patch_tangent_singular_rms",
    "acq_projection_prehead_tangent_fraction",
    "acq_gradient_projection",
    "acq_projection_strength",
    "acq_gradient_tangent_fraction_before",
    "acq_gradient_tangent_fraction_after",
    "acq_gradient_removed_energy_fraction",
    "acq_patch_gradient_tangent_fraction_before",
    "acq_patch_gradient_tangent_fraction_after",
    "acq_patch_gradient_removed_energy_fraction",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean_metrics(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
        result[metric] = fmean(values) if values else None
    return result


def _first_digest_mismatch(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left.get("batch_sample_key_digest") != right.get("batch_sample_key_digest"):
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline", default="full_full")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--tail", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.tail <= 0:
        raise ValueError("--tail must be positive")

    names = [args.baseline, *args.arms]
    rows_by_arm = {}
    for name in names:
        root = args.baseline_root if name == args.baseline and args.baseline_root else args.root
        rows_by_arm[name] = _read_rows(root / name / "raw_loss_metrics.jsonl")
    if any(not rows for rows in rows_by_arm.values()):
        raise RuntimeError("No raw metrics for: " + ", ".join(name for name, rows in rows_by_arm.items() if not rows))

    reference = rows_by_arm[args.baseline]
    common_updates = min(len(rows) for rows in rows_by_arm.values())
    if any(any(row.get("batch_sample_key_digest") is None for row in rows) for rows in rows_by_arm.values()):
        raise RuntimeError("Raw metrics lack controlled-stream sample digests")
    mismatch = {
        name: _first_digest_mismatch(reference[:common_updates], rows[:common_updates])
        for name, rows in rows_by_arm.items()
        if name != args.baseline
    }
    mismatch = {name: index for name, index in mismatch.items() if index is not None}
    if mismatch:
        raise RuntimeError(f"Sample-sequence mismatch: {mismatch}")

    complete = len({len(rows) for rows in rows_by_arm.values()}) == 1
    if not complete and not args.allow_partial:
        raise RuntimeError(f"Arms have incomplete unequal lengths: { {name: len(rows) for name, rows in rows_by_arm.items()} }")

    report: dict[str, Any] = {
        "root": str(args.root),
        "baseline_root": str(args.baseline_root or args.root),
        "baseline": args.baseline,
        "updates_by_arm": {name: len(rows) for name, rows in rows_by_arm.items()},
        "common_aligned_updates": common_updates,
        "all_arms_complete_and_aligned": complete,
        "arms": {},
    }
    for name, rows in rows_by_arm.items():
        tail_rows = rows[-min(args.tail, len(rows)) :]
        report["arms"][name] = {
            "last_update": int(rows[-1]["optimizer_update"]),
            "tail_updates": len(tail_rows),
            "tail_common_metrics": _mean_metrics(tail_rows, COMMON_METRICS),
            "tail_dbi_metrics": _mean_metrics(tail_rows, DBI_METRICS),
        }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
