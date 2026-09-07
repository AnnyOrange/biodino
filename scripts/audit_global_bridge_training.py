#!/usr/bin/env python3
"""Audit the matched full-bank bridge training arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("baseline", "true", "shuffled")
GBT_METRICS = (
    "gbt_bank_coverage",
    "gbt_bank_samples",
    "gbt_active_rows",
    "gbt_valid_direction_fraction",
    "gbt_target_angle",
    "gbt_anchor_target_cosine",
    "gbt_student_target_cosine",
    "global_bridge_transport_loss",
    "backbone_grad_norm",
    "total_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--experiment-tag", default="globalbridge_k20")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_metric(rows: list[dict], metric: str) -> dict[str, float] | None:
    values = [float(row[metric]) for row in rows if metric in row]
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def run_path(args: argparse.Namespace, arm: str) -> Path:
    if arm == "baseline":
        root = args.baseline_run_root or args.run_root
        run_name = f"baseline_seed{args.seed}_u{args.updates}"
    else:
        root = args.run_root
        run_name = f"{args.experiment_tag}_{arm}_seed{args.seed}_u{args.updates}"
    return root / run_name / "raw_loss_metrics.jsonl"


def main() -> None:
    args = parse_args()
    rows_by_arm = {}
    for arm in ARMS:
        path = run_path(args, arm)
        rows = load_rows(path)
        if len(rows) != args.updates:
            raise ValueError(f"{arm} has {len(rows)} updates, expected {args.updates}: {path}")
        rows_by_arm[arm] = rows

    digests = {
        arm: [str(row.get("batch_sample_key_digest", "")) for row in rows]
        for arm, rows in rows_by_arm.items()
    }
    reference = digests["baseline"]
    stream_matches = {arm: value == reference for arm, value in digests.items()}
    first_mismatch = {
        arm: next(
            (
                index
                for index, (left, right) in enumerate(zip(reference, value))
                if left != right
            ),
            None,
        )
        for arm, value in digests.items()
    }
    nonfinite = {
        arm: [
            {
                "update": index,
                "metric": metric,
                "value": row[metric],
            }
            for index, row in enumerate(rows)
            for metric in GBT_METRICS
            if metric in row and not math.isfinite(float(row[metric]))
        ]
        for arm, rows in rows_by_arm.items()
    }
    activity = {
        arm: {
            metric: summary
            for metric in GBT_METRICS
            if (summary := summarize_metric(rows, metric)) is not None
        }
        for arm, rows in rows_by_arm.items()
    }
    payload = {
        "updates": args.updates,
        "stream_matches_baseline": stream_matches,
        "first_stream_mismatch_update": first_mismatch,
        "all_streams_match": all(stream_matches.values()),
        "nonfinite_metrics": nonfinite,
        "all_metrics_finite": not any(nonfinite.values()),
        "activity": activity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_streams_match"] or not payload["all_metrics_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
