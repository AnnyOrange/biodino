#!/usr/bin/env python3
"""Audit matched sample streams and summarize expert-consensus activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("baseline", "single", "consensus", "shuffled")
ECR_METRICS = (
    "ecr_bank_coverage",
    "ecr_relation_samples",
    "ecr_candidate_edges",
    "ecr_expert_mutual_edges_mean",
    "ecr_consensus_edges",
    "ecr_selected_edges",
    "ecr_active_rows",
    "ecr_selected_fraction",
    "ecr_residual_mean",
    "expert_consensus_residual_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument(
        "--experiment-tag",
        default="",
        help="Prefix non-baseline run directories, for example oidroute.",
    )
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


def main() -> None:
    args = parse_args()
    rows_by_arm = {}
    for arm in ARMS:
        run_name = f"{arm}_seed{args.seed}_u{args.updates}"
        if arm != "baseline" and args.experiment_tag:
            run_name = f"{args.experiment_tag}_{run_name}"
        path = (
            args.run_root
            / run_name
            / "raw_loss_metrics.jsonl"
        )
        rows = load_rows(path)
        if len(rows) != args.updates:
            raise ValueError(f"{arm} has {len(rows)} updates, expected {args.updates}: {path}")
        rows_by_arm[arm] = rows

    digest_by_arm = {
        arm: [str(row.get("batch_sample_key_digest", "")) for row in rows]
        for arm, rows in rows_by_arm.items()
    }
    reference = digest_by_arm["baseline"]
    stream_matches = {arm: digests == reference for arm, digests in digest_by_arm.items()}
    first_mismatch = {}
    for arm, digests in digest_by_arm.items():
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(reference, digests)) if left != right),
            None,
        )
        first_mismatch[arm] = mismatch

    activity = {
        arm: {
            metric: summary
            for metric in ECR_METRICS
            if (summary := summarize_metric(rows, metric)) is not None
        }
        for arm, rows in rows_by_arm.items()
    }
    payload = {
        "updates": args.updates,
        "stream_matches_baseline": stream_matches,
        "first_stream_mismatch_update": first_mismatch,
        "all_streams_match": all(stream_matches.values()),
        "activity": activity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_streams_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
