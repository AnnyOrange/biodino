#!/usr/bin/env python3
"""Audit a deterministic Scout-Guided Plasticity Transport screen.

The target-strength-preserving shuffled control is only interpretable when all
arms receive the exact same sample sequence. This tool verifies that contract
and summarizes the stable-kernel transport diagnostics without treating them
as downstream evidence.
"""

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
SCOUT_METRICS = (
    "scout_kernel_transport_loss",
    "scout_kernel_transport_loss_weight",
    "scout_kernel_transport_displacement_budget_ratio",
    "scout_kernel_transport_is_delta",
    "scout_kernel_transport_anchor_consistent",
    "scout_kernel_transport_mask_matched",
    "skdt_loss",
    "skdt_alignment",
    "skdt_current_delta_norm",
    "skdt_scout_delta_norm",
    "skdt_relative_displacement",
    "skdt_budget_gate",
    "skdt_active",
    "skdt_cross_view_alignment",
    "skdt_stable_rank",
    "skdt_stable_energy_ratio",
    "skdt_stable_target_norm",
    "skdt_stable_active",
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
    parser.add_argument("--baseline", default="full_full")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Optional root containing the baseline arm; defaults to --root.",
    )
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--tail", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.tail <= 0:
        raise ValueError("--tail must be positive")

    baseline_root = args.baseline_root or args.root
    rows_by_arm = {
        args.baseline: _read_rows(baseline_root / args.baseline / "raw_loss_metrics.jsonl"),
        **{name: _read_rows(args.root / name / "raw_loss_metrics.jsonl") for name in args.arms},
    }
    if any(not rows for rows in rows_by_arm.values()):
        raise RuntimeError("No raw metrics for: " + ", ".join(name for name, rows in rows_by_arm.items() if not rows))
    if any(any(row.get("batch_sample_key_digest") is None for row in rows) for rows in rows_by_arm.values()):
        raise RuntimeError("Raw metrics lack controlled-stream sample digests")

    reference = rows_by_arm[args.baseline]
    common_updates = min(len(rows) for rows in rows_by_arm.values())
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
        lengths = {name: len(rows) for name, rows in rows_by_arm.items()}
        raise RuntimeError(f"Arms have incomplete unequal lengths: {lengths}")

    report: dict[str, Any] = {
        "root": str(args.root),
        "baseline": args.baseline,
        "baseline_root": str(baseline_root),
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
            "tail_scout_metrics": _mean_metrics(tail_rows, SCOUT_METRICS),
        }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
