#!/usr/bin/env python3
"""Compare S+ objective ablations with dataset-wise ranks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


PRIMARY_METRICS = {
    "classification": "balanced_accuracy",
    "regression": "r2",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=EVAL_DIR")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=EVAL_DIR")
    return label, Path(path)


def parse_checkpoint_selection(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=CHECKPOINT")
    label, checkpoint = value.split("=", 1)
    try:
        return label, int(checkpoint)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CHECKPOINT must be an integer") from exc


def checkpoint_id(result: dict, result_path: Path) -> int:
    checkpoint = str(result.get("checkpoint", ""))
    match = re.search(r"/ckpt/(\d+)(?:/checkpoint\.pth)?$", checkpoint)
    if match:
        return int(match.group(1))
    model = str(result.get("model", ""))
    match = re.search(r"(\d+)$", model)
    if match:
        return int(match.group(1))
    for part in reversed(result_path.parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer checkpoint from {result_path}")


def load_rows(runs: list[tuple[str, Path]], retrieval_metric: str) -> list[dict]:
    rows = []
    for label, root in runs:
        for path in sorted(root.rglob("last_result.json")):
            result = json.loads(path.read_text())
            task = str(result.get("task", ""))
            metric = retrieval_metric if task == "retrieval_clustering" else PRIMARY_METRICS.get(task)
            if metric is None or metric not in result:
                continue
            rows.append(
                {
                    "candidate": label,
                    "checkpoint": checkpoint_id(result, path),
                    "task": task,
                    "dataset": str(result["dataset"]),
                    "metric": metric,
                    "value": float(result[metric]),
                    "result_path": str(path),
                }
            )
    return rows


def average_ranks(values: list[tuple[tuple[str, int], float]]) -> dict[tuple[str, int], float]:
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    ranks = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for candidate, _ in ordered[index:end]:
            ranks[candidate] = rank
        index = end
    return ranks


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=EVAL_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-datasets", type=int, default=9)
    parser.add_argument("--reference-label", default="official")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--select-checkpoint",
        action="append",
        type=parse_checkpoint_selection,
        default=[],
        help="Restrict a run to one checkpoint, for example baseline=255.",
    )
    parser.add_argument(
        "--retrieval-metric",
        default="nmi",
        choices=("recall_at_1", "map_at_10", "cluster_accuracy", "ari", "nmi"),
        help="LC25000 recall@1 saturates in this proxy, so clustering NMI is the default discriminator.",
    )
    args = parser.parse_args()

    rows = load_rows(args.run, args.retrieval_metric)
    selected_checkpoints = dict(args.select_checkpoint)
    rows = [
        row
        for row in rows
        if row["candidate"] not in selected_checkpoints
        or row["checkpoint"] == selected_checkpoints[row["candidate"]]
    ]
    reference = {}
    baseline = {}
    for row in rows:
        if row["candidate"] == args.reference_label:
            reference[(row["task"], row["dataset"])] = row["value"]
        if row["candidate"] == args.baseline_label:
            baseline[(row["checkpoint"], row["task"], row["dataset"])] = row["value"]
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[(row["task"], row["dataset"])].append(
            ((row["candidate"], row["checkpoint"]), row["value"])
        )

    ranks = {}
    for dataset, values in by_dataset.items():
        for candidate, rank in average_ranks(values).items():
            ranks[(candidate, dataset)] = rank

    aggregate = defaultdict(
        lambda: {
            "rank_sum": 0.0,
            "datasets": 0,
            "wins": 0,
            "ref_wins": 0,
            "ref_ties": 0,
            "ref_losses": 0,
            "base_wins": 0,
            "base_ties": 0,
            "base_losses": 0,
        }
    )
    for row in rows:
        candidate = (row["candidate"], row["checkpoint"])
        dataset = (row["task"], row["dataset"])
        rank = ranks[(candidate, dataset)]
        row["rank"] = rank
        reference_value = reference.get(dataset)
        baseline_value = baseline.get((row["checkpoint"], *dataset))
        row["delta_vs_reference"] = "" if reference_value is None else row["value"] - reference_value
        row["delta_vs_baseline"] = "" if baseline_value is None else row["value"] - baseline_value
        aggregate[candidate]["rank_sum"] += rank
        aggregate[candidate]["datasets"] += 1
        aggregate[candidate]["wins"] += int(rank == 1.0)
        if reference_value is not None:
            delta = row["value"] - reference_value
            aggregate[candidate]["ref_wins"] += int(delta > 0)
            aggregate[candidate]["ref_ties"] += int(delta == 0)
            aggregate[candidate]["ref_losses"] += int(delta < 0)
        if baseline_value is not None:
            delta = row["value"] - baseline_value
            aggregate[candidate]["base_wins"] += int(delta > 0)
            aggregate[candidate]["base_ties"] += int(delta == 0)
            aggregate[candidate]["base_losses"] += int(delta < 0)

    summaries = []
    for (label, checkpoint), stats in aggregate.items():
        covered = stats["datasets"]
        summaries.append(
            {
                "candidate": label,
                "checkpoint": checkpoint,
                "datasets_covered": covered,
                "complete": int(covered >= args.expected_datasets),
                "mean_rank": stats["rank_sum"] / covered,
                "wins": stats["wins"],
                "wins_vs_reference": stats["ref_wins"],
                "ties_vs_reference": stats["ref_ties"],
                "losses_vs_reference": stats["ref_losses"],
                "wins_vs_baseline": stats["base_wins"],
                "ties_vs_baseline": stats["base_ties"],
                "losses_vs_baseline": stats["base_losses"],
            }
        )
    summaries.sort(key=lambda row: (-row["complete"], row["mean_rank"], -row["wins"]))
    rows.sort(key=lambda row: (row["task"], row["dataset"], row["rank"], row["candidate"]))

    write_csv(
        args.output_dir / "details.csv",
        rows,
        [
            "candidate",
            "checkpoint",
            "task",
            "dataset",
            "metric",
            "value",
            "delta_vs_reference",
            "delta_vs_baseline",
            "rank",
            "result_path",
        ],
    )
    write_csv(
        args.output_dir / "summary.csv",
        summaries,
        [
            "candidate",
            "checkpoint",
            "datasets_covered",
            "complete",
            "mean_rank",
            "wins",
            "wins_vs_reference",
            "ties_vs_reference",
            "losses_vs_reference",
            "wins_vs_baseline",
            "ties_vs_baseline",
            "losses_vs_baseline",
        ],
    )
    for row in summaries:
        print(
            f"{row['candidate']}@{row['checkpoint']}: covered={row['datasets_covered']} "
            f"complete={row['complete']} mean_rank={row['mean_rank']:.3f} wins={row['wins']} "
            f"vs_ref={row['wins_vs_reference']}/{row['ties_vs_reference']}/{row['losses_vs_reference']} "
            f"vs_base={row['wins_vs_baseline']}/{row['ties_vs_baseline']}/{row['losses_vs_baseline']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
