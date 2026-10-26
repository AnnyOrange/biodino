#!/usr/bin/env python3
"""Run disjoint v4 linear-probe shards and merge their completed results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from run_new_fm_v4_linear_20261016 import MODELS, ROOT, V4_LINEAR_DATASETS


RUN = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"
SHARDS = (
    ("pcam",),
    ("tissuemnist", "lc25000", "organcmnist", "livecell-cell-count"),
    ("bbbc048-cellcycle", "chammi-allen-task2", "chammi-cp-task3", "organsmnist", "bbbc005", "retinamnist"),
    ("nct-crc-he", "chammi-cp-task2", "chammi-cp-task1", "chammi-allen-task1", "dermamnist", "bbbc013"),
    ("chestmnist", "cyclops-protein-loc", "organamnist", "chammi-hpa-task2", "midog25-atypical", "conic-cell-count"),
    ("octmnist", "pathmnist", "chammi-hpa-task1", "bloodmnist", "pneumoniamnist"),
)


def shard_root(model: str, index: int) -> Path:
    return RUN / "shards" / f"{model}_s{index}"


def merge(model: str) -> None:
    destination = RUN / model / "linear"
    destination.mkdir(parents=True, exist_ok=True)
    rows = {}
    header = None
    sources = [destination] + [shard_root(model, i) / model / "linear" for i in range(len(SHARDS))]
    failures = []
    for source in sources:
        summary = source / "summary.csv"
        if summary.is_file():
            with summary.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if header is None:
                    header = reader.fieldnames
                elif reader.fieldnames != header:
                    raise ValueError(f"Inconsistent summary columns: {summary}")
                for row in reader:
                    if row["dataset"] in V4_LINEAR_DATASETS:
                        rows[row["dataset"]] = row
        failed = source / "failures.json"
        if failed.is_file():
            failures.extend(json.loads(failed.read_text()))
        if source == destination:
            continue
        for result in source.glob("*.json"):
            if result.stem not in V4_LINEAR_DATASETS:
                continue
            target = destination / result.name
            if target.is_file() and json.loads(target.read_text()) != json.loads(result.read_text()):
                raise ValueError(f"Conflicting shard result: {target}")
            shutil.copy2(result, target)
    missing = [dataset for dataset in V4_LINEAR_DATASETS if not (destination / f"{dataset}.json").is_file()]
    if missing or failures:
        raise RuntimeError(f"Linear merge incomplete for {model}: missing={missing}, failures={failures}")
    if set(rows) != set(V4_LINEAR_DATASETS) or header is None:
        raise ValueError(f"Summary rows do not match v4 datasets for {model}")
    temporary = destination / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows[dataset] for dataset in V4_LINEAR_DATASETS)
    temporary.replace(destination / "summary.csv")
    (destination / "failures.json").write_text("[]\n")
    print(f"[merged] {model}: {len(rows)}/{len(V4_LINEAR_DATASETS)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", type=int, choices=range(len(SHARDS)))
    group.add_argument("--merge", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.merge:
        merge(args.model)
        return
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_new_fm_v4_linear_20261016.py"),
        "--model", args.model, "--output-root", str(shard_root(args.model, args.shard)),
        "--batch-size", str(args.batch_size), "--num-workers", "2",
        "--datasets", *SHARDS[args.shard],
    ], check=True)


if __name__ == "__main__":
    main()
