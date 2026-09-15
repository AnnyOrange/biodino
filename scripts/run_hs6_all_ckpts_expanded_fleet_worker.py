#!/usr/bin/env python3
"""Claim HS6 checkpoint evaluations from a shared single-GPU 3090 queue."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
DEFAULT_OUTPUT_ROOT = (
    REPO
    / "outputs/02_eval_runs/hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909"
)
BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")

REGRESSION_DATASETS = (
    "bbbc013",
    "bbbc005",
    "conic-cell-count",
    "livecell-cell-count",
)
RETRIEVAL_DATASETS = (
    "lc25000",
    "nct-crc-he-100",
    "nct-crc-he-1k",
    "crc-val-he-7k",
    "hpa-subcellular",
    "rxrx1-cross",
)
DETECTION_DATASETS = ("livecell", "bbbc038", "conic")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    train_run: Path
    frozen_batch_size: int
    detection_batch_size: int


@dataclass(frozen=True)
class Job:
    model: ModelSpec
    checkpoint_id: int
    checkpoint: Path

    @property
    def key(self) -> str:
        return f"{self.model.name}__ckpt_{self.checkpoint_id}"


# Longest/most memory-intensive models come first so the tail of the fleet is short.
MODEL_SPECS = (
    ModelSpec(
        "hs6_hplus",
        REPO
        / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818",
        4,
        2,
    ),
    ModelSpec(
        "hs6_l",
        REPO
        / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818",
        16,
        4,
    ),
    ModelSpec(
        "hs6_b",
        REPO
        / "outputs/01_training_runs/HS6_B_robust_biosafe256_gb1024_lr1p5e4_wu3_tw30_nosig_e15_seed0_8x5090hxw_20260818",
        32,
        8,
    ),
    ModelSpec(
        "hs6_splus",
        REPO
        / "outputs/01_training_runs/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b",
        64,
        8,
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def discover_jobs() -> list[Job]:
    jobs: list[Job] = []
    for spec in MODEL_SPECS:
        config = spec.train_run / "config.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"missing config: {config}")
        checkpoints: list[tuple[int, Path]] = []
        for checkpoint in (spec.train_run / "ckpt").glob("*/checkpoint.pth"):
            try:
                checkpoint_id = int(checkpoint.parent.name)
            except ValueError:
                continue
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                continue
            if not zipfile.is_zipfile(checkpoint):
                print(f"[invalid-checkpoint] not a readable zip archive: {checkpoint}", file=sys.stderr)
                continue
            checkpoints.append((checkpoint_id, checkpoint))
        if not checkpoints:
            print(
                f"[invalid-model] no readable checkpoints: {spec.train_run / 'ckpt'}",
                file=sys.stderr,
            )
            continue
        jobs.extend(Job(spec, checkpoint_id, path) for checkpoint_id, path in sorted(checkpoints))
    return jobs


def summary_valid(path: Path, expected_tasks: set[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        observed_tasks = {row.get("task", "") for row in rows}
        return (
            bool(rows)
            and not any(row.get("error") for row in rows)
            and bool(observed_tasks & expected_tasks)
            and observed_tasks <= expected_tasks
        )
    except (OSError, csv.Error):
        return False


def json_result_valid(path: Path, dataset: str, expected_tasks: set[str]) -> bool:
    """Accept the canonical JSON result emitted by newer bio evaluators.

    Some retrieval datasets emit only ``last_result.json`` while older datasets
    also emit ``summary.csv``.  Both are final result artifacts and should be
    treated equivalently by the fleet validator.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text())
        if payload.get("error") or payload.get("dataset") != dataset:
            return False
        rows = payload.get("rows")
        if rows is None:
            rows = [payload]
        if not isinstance(rows, list) or not rows:
            return False
        if any(not isinstance(row, dict) or row.get("error") for row in rows):
            return False
        observed_tasks = {str(row.get("task", "")) for row in rows}
        return bool(observed_tasks & expected_tasks)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return False


def detection_valid(path: Path, dataset: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text())
        return payload.get("dataset") == dataset and "test_patch_f1" in payload
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def validation(job: Job, output_root: Path) -> tuple[bool, list[str]]:
    root = output_root / job.model.name
    checkpoint_id = str(job.checkpoint_id)
    missing: list[str] = []
    for dataset in REGRESSION_DATASETS:
        result = root / "bio_regression" / dataset / checkpoint_id / "summary.csv"
        json_result = result.with_name("last_result.json")
        if not (
            summary_valid(result, {"regression"})
            or json_result_valid(json_result, dataset, {"regression"})
        ):
            missing.append(str(result))
    for dataset in RETRIEVAL_DATASETS:
        result = root / "bio_retrieval" / dataset / checkpoint_id / "summary.csv"
        expected_tasks = {"retrieval", "clustering", "retrieval_clustering"}
        json_result = result.with_name("last_result.json")
        if not (
            summary_valid(result, expected_tasks)
            or json_result_valid(json_result, dataset, expected_tasks)
        ):
            missing.append(str(result))
    for dataset in DETECTION_DATASETS:
        result = (
            root
            / "bio_detection"
            / dataset
            / checkpoint_id
            / "results_bio_detection.json"
        )
        if not detection_valid(result, dataset):
            missing.append(str(result))
    return not missing, missing


def failure_count(failure_root: Path, key: str) -> int:
    return len(list(failure_root.glob(f"{key}.attempt*.json")))


def clear_stale_claim(claim: Path, stale_seconds: float) -> bool:
    try:
        owner = json.loads((claim / "owner.json").read_text())
        age = time.time() - float(owner["claimed_at_unix"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        age = time.time() - claim.stat().st_mtime
    if age < stale_seconds:
        return False
    shutil.rmtree(claim, ignore_errors=True)
    return not claim.exists()


def try_claim(
    claim_root: Path,
    job: Job,
    worker: str,
    stale_seconds: float,
) -> Path | None:
    claim = claim_root / f"{job.key}.lock"
    try:
        claim.mkdir()
    except FileExistsError:
        if not clear_stale_claim(claim, stale_seconds):
            return None
        try:
            claim.mkdir()
        except FileExistsError:
            return None
    atomic_json(
        claim / "owner.json",
        {
            "job": job.key,
            "worker": worker,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "claimed_at_unix": time.time(),
            "claimed_at_utc": utc_now(),
            "checkpoint": str(job.checkpoint),
        },
    )
    return claim


def command(job: Job, args: argparse.Namespace) -> list[str]:
    return [
        "bash",
        str(args.repo / "scripts/run_bio_benchmark_all.sh"),
        str(job.model.train_run / "ckpt"),
        str(job.model.train_run / "config.yaml"),
        str(args.output_root / job.model.name),
        str(args.benchmark_root),
    ]


def environment(job: Job, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "PYTHONPATH": f"{args.repo}:{env.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "CHECKPOINT_ITERS": str(job.checkpoint_id),
            "GPUS": "0",
            "TASKS": "regression retrieval detection",
            "REGRESSION_DATASETS": " ".join(REGRESSION_DATASETS),
            "RETRIEVAL_DATASETS": " ".join(RETRIEVAL_DATASETS),
            "DETECTION_DATASETS": " ".join(DETECTION_DATASETS),
            "JOBS_PER_GPU": "1",
            "MAX_CONCURRENT_JOBS": "1",
            "MAX_CPU_JOBS": "1",
            "FROZEN_DATASETS_PER_JOB": "1",
            "CONCURRENT_TASK_GROUPS": "0",
            "FROZEN_BATCH_SIZE": str(job.model.frozen_batch_size),
            "DET_BATCH_SIZE": str(job.model.detection_batch_size),
            "NUM_WORKERS": "2",
            "EVAL_BLAS_THREADS": "4",
            "AUTOCAST_DTYPE": "bf16",
            "FROZEN_SPLIT_PROTOCOL": "current",
            "REGRESSION_RESOLUTION_PROTOCOL": "best",
            "RXRX1_FULL": "0",
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker", default=f"{socket.gethostname()}-gpu0")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--claim-stale-seconds", type=float, default=48 * 3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = discover_jobs()
    if args.dry_run:
        for job in jobs:
            print(
                f"{job.key}\t{job.checkpoint}\t"
                f"frozen_bs={job.model.frozen_batch_size}\tdet_bs={job.model.detection_batch_size}"
            )
        print(f"jobs={len(jobs)}")
        return 0

    state_root = args.output_root / "_state"
    claim_root = state_root / "claims"
    done_root = state_root / "done"
    failure_root = state_root / "failures"
    log_root = state_root / "job_logs"
    worker_root = state_root / "workers"
    for path in (claim_root, done_root, failure_root, log_root, worker_root):
        path.mkdir(parents=True, exist_ok=True)
    atomic_json(
        worker_root / f"{args.worker}.json",
        {
            "worker": args.worker,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu": args.gpu,
            "started_at_utc": utc_now(),
            "jobs_discovered": len(jobs),
        },
    )

    completed = skipped = failed = 0
    while True:
        claimed_any = False
        for job in jobs:
            valid, _ = validation(job, args.output_root)
            if valid:
                skipped += 1
                continue
            if failure_count(failure_root, job.key) >= args.max_attempts:
                continue
            claim = try_claim(claim_root, job, args.worker, args.claim_stale_seconds)
            if claim is None:
                continue
            claimed_any = True
            started = time.time()
            cmd = command(job, args)
            log_path = log_root / f"{job.key}.log"
            print(f"[claim] {args.worker}: {job.key}", flush=True)
            print(f"[command] {shlex.join(cmd)}", flush=True)
            with log_path.open("a") as log:
                log.write(f"\n[{utc_now()}] $ {shlex.join(cmd)}\n")
                log.flush()
                process = subprocess.run(
                    cmd,
                    cwd=args.repo,
                    env=environment(job, args),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            valid, missing = validation(job, args.output_root)
            payload: dict[str, object] = {
                "job": job.key,
                "model": job.model.name,
                "checkpoint_id": job.checkpoint_id,
                "checkpoint": str(job.checkpoint),
                "worker": args.worker,
                "returncode": process.returncode,
                "elapsed_seconds": time.time() - started,
                "result_valid": valid,
                "missing_or_invalid": missing,
                "finished_at_utc": utc_now(),
                "log": str(log_path),
            }
            if process.returncode == 0 and valid:
                completed += 1
                atomic_json(done_root / f"{job.key}.json", payload)
                print(f"[done] {job.key}", flush=True)
            else:
                failed += 1
                attempt = failure_count(failure_root, job.key) + 1
                atomic_json(failure_root / f"{job.key}.attempt{attempt}.json", payload)
                print(
                    f"[failed] {job.key} rc={process.returncode} invalid={len(missing)}",
                    flush=True,
                )
            shutil.rmtree(claim, ignore_errors=True)
            break
        if not claimed_any:
            break

    print(
        f"[queue-idle] worker={args.worker} completed={completed} "
        f"failed={failed} skipped_scans={skipped}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
