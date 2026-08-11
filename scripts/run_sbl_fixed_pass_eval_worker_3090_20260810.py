#!/usr/bin/env python3
"""Claim and evaluate ready S/B/L fixed-pass points on one shared-storage GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path


MODELS = ("S", "B", "L")
LABELS = ("10", "20", "50", "100")
PASS_BUDGETS = ("8", "15")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def point_key(model: str, label: str, pass_budget: str) -> str:
    return f"{model}_random{label}_pass{pass_budget}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/huawei_deepcad/dinov3"))
    parser.add_argument("--python-bin", type=Path, default=Path("/home/inspur/anaconda3/envs/dinov3/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--ready-age-seconds", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    points_root = repo / "outputs/01_training_runs/SBL_splus_datafp_alpha075_20260810"
    eval_root = repo / "outputs/02_eval_runs/SBL_splus_datafp_alpha075_full_20260810"
    report_root = repo / "outputs/00_reports/splus_bl_fixed_pass_data_scaling_20260810"
    state_root = report_root / "eval_3090_fleet"
    claim_root = state_root / "claims"
    failure_root = state_root / "failures"
    log_root = repo / "outputs/auto_eval_logs/splus_bl_data_fp_20260810/3090_fleet"
    for path in (claim_root, failure_root, log_root, eval_root):
        path.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    worker = f"{host}-gpu{args.gpu}-pid{os.getpid()}"
    worker_status = state_root / "workers" / f"{host}_gpu{args.gpu}.json"
    tasks = "classification regression retrieval segmentation detection"
    points = [
        (model, label, pass_budget)
        for model in MODELS
        for label in LABELS
        for pass_budget in PASS_BUDGETS
    ]

    while True:
        complete = 0
        ready = 0
        claimed = 0
        progressed = False
        for model, label, pass_budget in points:
            key = point_key(model, label, pass_budget)
            point = points_root / model / f"random_{label}" / f"pass_{pass_budget}"
            checkpoint = point / "ckpt/75/checkpoint.pth"
            config = point / "config.yaml"
            interpolation_manifest = point / "interpolation_manifest.json"
            output = eval_root / model / f"random_{label}" / f"pass_{pass_budget}"
            done_marker = output / "_online_status/ckpt_75.done"
            failed_marker = output / "_online_status/ckpt_75.failed"
            if done_marker.is_file() and not failed_marker.exists():
                complete += 1
                continue
            required = (checkpoint, config, interpolation_manifest)
            if not all(path.is_file() and path.stat().st_size > 0 for path in required):
                continue
            if time.time() - max(path.stat().st_mtime for path in required) < args.ready_age_seconds:
                continue
            ready += 1

            failures = sorted(failure_root.glob(f"{key}.attempt*.json"))
            if len(failures) >= args.max_attempts:
                continue
            claim = claim_root / f"{key}.lock"
            try:
                claim.mkdir()
            except FileExistsError:
                claimed += 1
                continue

            attempt = len(failures) + 1
            owner = {
                "attempt": attempt,
                "checkpoint": str(checkpoint),
                "claimed_at_utc": utc_now(),
                "gpu": args.gpu,
                "host": host,
                "pid": os.getpid(),
                "worker": worker,
            }
            atomic_json(claim / "owner.json", owner)
            atomic_json(worker_status, {**owner, "state": "running", "point": key})
            point_log = log_root / f"{key}.{host}.attempt{attempt}.log"
            frozen_batch = {"S": "64", "B": "32", "L": "16"}[model]
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": str(args.python_bin),
                    "CHECKPOINT_ITERS": "75",
                    "GPUS": args.gpu,
                    "JOBS_PER_GPU": str(args.jobs_per_gpu),
                    "MAX_CONCURRENT_JOBS": str(args.jobs_per_gpu),
                    "MAX_CPU_JOBS": str(args.jobs_per_gpu),
                    "CONCURRENT_TASK_GROUPS": "0",
                    "FROZEN_DATASETS_PER_JOB": "1",
                    "SEGMENTATION_DATASETS_PER_JOB": "1",
                    "TASKS": tasks,
                    "FROZEN_BATCH_SIZE": frozen_batch,
                    "FROZEN_CHANNEL_POLICY": "auto",
                    "FROZEN_CHANNEL_TTA_SAMPLES": "8",
                    "AUTOCAST_DTYPE": "bf16",
                    "NUM_WORKERS": "2",
                    "SEG_FEATURE_BATCH_SIZE": "32",
                    "SEG_FEATURE_NUM_WORKERS": "4",
                    "SEG_PROBE_BATCH_SIZE": "32",
                    "SEG_PROBE_NUM_WORKERS": "4",
                    "DET_BATCH_SIZE": "8",
                    "EVAL_BLAS_THREADS": "1",
                    "DRY_RUN": "0",
                }
            )
            command = [
                "bash",
                str(repo / "scripts/run_bio_benchmark_all.sh"),
                str(point / "ckpt"),
                str(config),
                str(output),
                "/mnt/huawei_deepcad/benchmark",
            ]
            started = time.time()
            with point_log.open("a", buffering=1) as log:
                print(f"[{utc_now()}] worker={worker} attempt={attempt} command={' '.join(command)}", file=log)
                result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"[{utc_now()}] benchmark_rc={result.returncode}", file=log)
                if result.returncode == 0:
                    reconcile = subprocess.run(
                        [
                            str(args.python_bin),
                            str(repo / "scripts/reconcile_complete_bio_eval_checkpoints.py"),
                            "--eval-root",
                            str(output),
                            "--checkpoints",
                            "75",
                            "--min-result-age-seconds",
                            "0",
                        ],
                        cwd=repo,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    print(f"[{utc_now()}] reconcile_rc={reconcile.returncode}", file=log)
                if done_marker.is_file() and not failed_marker.exists():
                    cleanup = subprocess.run(
                        [
                            str(args.python_bin),
                            str(repo / "scripts/prune_completed_bio_eval_cache.py"),
                            "--eval-root",
                            str(output),
                            "--checkpoint",
                            "75",
                            "--tasks",
                            *tasks.split(),
                            "--delete",
                        ],
                        cwd=repo,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    print(f"[{utc_now()}] cleanup_rc={cleanup.returncode}", file=log)

            elapsed = time.time() - started
            if done_marker.is_file() and not failed_marker.exists():
                atomic_json(
                    state_root / "done" / f"{key}.json",
                    {**owner, "completed_at_utc": utc_now(), "elapsed_seconds": elapsed},
                )
                atomic_json(worker_status, {**owner, "state": "completed", "point": key})
                progressed = True
                # Keep the claim as an immutable record; the done marker makes it non-blocking.
                break

            atomic_json(
                failure_root / f"{key}.attempt{attempt}.json",
                {
                    **owner,
                    "benchmark_returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                    "failed_at_utc": utc_now(),
                    "log": str(point_log),
                },
            )
            shutil.rmtree(claim, ignore_errors=True)
            atomic_json(worker_status, {**owner, "state": "retry_wait", "point": key})
            time.sleep(min(300, 60 * attempt))
            progressed = True
            break

        atomic_json(
            worker_status,
            {
                "complete_points": complete,
                "host": host,
                "pid": os.getpid(),
                "ready_points": ready,
                "state": "idle" if not progressed else "scanning",
                "updated_at_utc": utc_now(),
                "worker": worker,
            },
        )
        if complete == len(points):
            (state_root / "all_points.complete").write_text(utc_now() + "\n")
            return 0
        if not progressed:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
