#!/usr/bin/env python3
"""Wait for a C3 checkpoint sweep, select its winner, and run the full core benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DATASETS = {"bloodmnist", "bbbc048-cellcycle", "cyclops-protein-loc"}


def load_c3(eval_root: Path, checkpoints: set[int]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for path in eval_root.glob("bio_classification/*/*/summary.csv"):
        try:
            checkpoint = int(path.parent.name)
        except ValueError:
            continue
        if checkpoint not in checkpoints:
            continue
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        dataset = str(row.get("dataset", ""))
        if dataset not in EXPECTED_DATASETS or row.get("error"):
            continue
        try:
            rows[checkpoint][dataset] = (
                float(row["balanced_accuracy"]),
                float(row["macro_f1"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

    complete = {}
    for checkpoint, datasets in rows.items():
        if set(datasets) != EXPECTED_DATASETS:
            continue
        complete[checkpoint] = {
            "mean_balanced_accuracy": sum(value[0] for value in datasets.values()) / len(datasets),
            "mean_macro_f1": sum(value[1] for value in datasets.values()) / len(datasets),
        }
    return complete


def wait_for_results(
    eval_root: Path,
    checkpoints: set[int],
    poll_seconds: int,
    timeout_hours: float,
) -> dict[int, dict[str, float]]:
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        complete = load_c3(eval_root, checkpoints)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"complete C3 checkpoints={sorted(complete)} expected={sorted(checkpoints)}",
            flush=True,
        )
        if set(complete) == checkpoints:
            return complete
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for C3 results under {eval_root}")
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--c3-eval-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--output-parent", type=Path, default=Path("outputs/02_eval_runs"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    checkpoints = set(args.checkpoints)
    scores = wait_for_results(
        args.c3_eval_root,
        checkpoints,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    selected = max(
        scores,
        key=lambda checkpoint: (
            scores[checkpoint]["mean_balanced_accuracy"],
            scores[checkpoint]["mean_macro_f1"],
            -checkpoint,
        ),
    )
    selection = {
        "selected_checkpoint": selected,
        "selection_metric": "C3 mean balanced accuracy; macro-F1 then earlier checkpoint as tie-breakers",
        "scores": {str(key): value for key, value in sorted(scores.items())},
    }
    args.c3_eval_root.mkdir(parents=True, exist_ok=True)
    (args.c3_eval_root / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)
    if args.select_only:
        return 0

    host = socket.gethostname().split(".")[0]
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_dir = args.output_parent / (
        f"{args.train_root.name}__full_auto_selected{selected}_{host}_b128_{date}"
    )
    command = [
        sys.executable,
        "-m",
        "dinov3.eval.bio_benchmark",
        "--checkpoints-dir",
        str(args.train_root / "ckpt"),
        "--checkpoint-iters",
        str(selected),
        "--train-config",
        str(args.train_root / "config.yaml"),
        "--benchmark-root",
        str(args.benchmark_root),
        "--output-dir",
        str(output_dir),
        "--tasks",
        "classification",
        "regression",
        "retrieval",
        "--gpus",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "--jobs-per-gpu",
        "1",
        "--max-concurrent-jobs",
        "8",
        "--max-cpu-jobs",
        "8",
        "--frozen-datasets-per-job",
        "1",
        "--frozen-batch-size",
        "128",
        "--frozen-n-last-blocks",
        "1",
        "--autocast-dtype",
        "fp16",
        "--frozen-channel-policy",
        "auto",
        "--classification-resolution-protocol",
        "best",
        "--classification-image-size",
        "224",
        "--num-workers",
        "2",
        "--seed",
        "0",
        "--train-fraction",
        "0.8",
    ]
    print("Launching full core benchmark:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    (output_dir / "auto_eval_complete.json").write_text(
        json.dumps({"selected_checkpoint": selected, "selection": str(args.c3_eval_root)}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
