#!/usr/bin/env python3
"""Admit the five four-GPU replay checkpoints as a distinct evaluation arm."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

import watch_rolling_1pb_eval_20261019 as evaluation


RUN = evaluation.REPO / "outputs/01_training_runs/HS6_L_random1pb_replay_missing_4x3090qi_20261020"
ARM = "rolling1pb_replay4"
STEPS = {975, 1463, 1951, 2439, 2927}


def scan() -> None:
    manifest_path = evaluation.OUT / "campaign_manifest.json"
    with (evaluation.OUT / "_state/manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        roots = manifest.setdefault("training_roots", {})
        if roots.get(ARM) != str(RUN):
            roots[ARM] = str(RUN)
            manifest["replay_comparison"] = {
                "arm": ARM,
                "baseline_step": 487,
                "world_size": 4,
                "global_batch_size": 1024,
                "source_batches": [f"batch-{number:06d}" for number in range(1, 6)],
                "exact_original_weight_recovery": False,
            }
            evaluation.atomic_json(manifest_path, manifest)
    for record in evaluation.full_checkpoint_records(RUN):
        if record["step"] not in STEPS:
            continue
        exported = evaluation.export_teacher(
            record, evaluation.OUT / "teachers_replay4")
        if exported is not None:
            evaluation.admit(ARM, record["step"], *exported)
    evaluation.prioritize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("once", "watch"))
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            scan()
        except Exception as error:
            print(f"REPLAY_SCAN_ERROR {type(error).__name__}: {error}", flush=True)
            if args.mode == "once":
                raise
        if args.mode == "once":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
