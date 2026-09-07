#!/usr/bin/env python3
"""Audit a controlled Nested Channel Innovation training matrix.

The script treats matching per-update sample digests as a prerequisite for
comparing arms.  It reports the common data prefix and tail means of the
conditional-residual diagnostics without claiming a downstream result.
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
)
NCI_METRICS = (
    "nci_loss",
    "nci_predictor_loss",
    "nci_invariance_loss",
    "nci_variance_loss",
    "nci_orthogonality_loss",
    "nci_residual_rms",
    "nci_residual_std",
    "nci_full_subset_cosine",
    "nci_positive_cosine",
    "nci_alignment_margin",
    "nci_predictor_r2",
    "nci_active_fraction",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean_metrics(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, float | None]:
    return {
        metric: (fmean(values) if (values := _finite_values(rows, metric)) else None)
        for metric in metrics
    }


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
    parser.add_argument("--tail", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Report the aligned common prefix instead of requiring all arms to finish.",
    )
    args = parser.parse_args()
    if args.tail <= 0:
        raise ValueError("--tail must be positive")

    baseline_root = args.baseline_root or args.root
    rows_by_arm = {
        args.baseline: _read_rows(baseline_root / args.baseline / "raw_loss_metrics.jsonl"),
        **{name: _read_rows(args.root / name / "raw_loss_metrics.jsonl") for name in args.arms},
    }
    if any(not rows for rows in rows_by_arm.values()):
        empty = [name for name, rows in rows_by_arm.items() if not rows]
        raise RuntimeError(f"No raw metrics for: {', '.join(empty)}")

    reference = rows_by_arm[args.baseline]
    digest_missing = [
        name
        for name, rows in rows_by_arm.items()
        if any(row.get("batch_sample_key_digest") is None for row in rows)
    ]
    if digest_missing:
        raise RuntimeError("Raw metrics lack sample digests for: " + ", ".join(digest_missing))

    common_updates = min(len(rows) for rows in rows_by_arm.values())
    mismatch = {
        name: _first_digest_mismatch(reference[:common_updates], rows[:common_updates])
        for name, rows in rows_by_arm.items()
        if name != args.baseline
    }
    mismatch = {name: index for name, index in mismatch.items() if index is not None}
    complete = len({len(rows) for rows in rows_by_arm.values()}) == 1
    if mismatch:
        raise RuntimeError(f"Sample-sequence mismatch: {mismatch}")
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
            "tail_nci_metrics": _mean_metrics(tail_rows, NCI_METRICS),
        }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
