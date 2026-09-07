#!/usr/bin/env python3
"""Claim and evaluate fractional-epoch 5M checkpoints on one 3090 worker."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path


OFFICIAL_EPOCH_LENGTH = 4856
SNAPSHOT_PERIOD = 488
TOTAL_UPDATES = 15 * OFFICIAL_EPOCH_LENGTH
EXPECTED_SNAPSHOTS = TOTAL_UPDATES // SNAPSHOT_PERIOD

CLASSIFICATION_DATASETS = (
    "bbbc048-cellcycle bloodmnist breastmnist chestmnist cyclops-protein-loc "
    "dermamnist midog25-atypical octmnist organamnist organcmnist organsmnist "
    "pathmnist pneumoniamnist retinamnist tissuemnist"
)
RETRIEVAL_DATASETS = "lc25000 nct-crc-he-1k crc-val-he-7k"
STANDARD_SEGMENTATION_DATASETS = "bbbc038 cellpose conic livecell pannuke tissuenet"

# SEGMENTATION_PROTOCOL=best resolves these settings inside DINOv3.
SEGMENTATION_PROTOCOL = {
    "bbbc038": {"size": 512, "resize": "pad", "layers": "even4"},
    "cellpose": {"size": 512, "resize": "pad", "layers": "last1"},
    "conic": {"size": 256, "resize": "stretch", "layers": "even4"},
    "livecell": {"size": 512, "resize": "pad", "layers": "even4"},
    "monuseg": {"size": 768, "resize": "pad", "layers": "last1"},
    "pannuke": {"size": 256, "resize": "stretch", "layers": "even4"},
    "tissuenet": {"size": 256, "resize": "stretch", "layers": "last1"},
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/huawei_deepcad/dinov3"))
    parser.add_argument(
        "--train-run",
        type=Path,
        default=Path(
            "/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/"
            "L_s0packwds_final512_5m_e15_gb1024_seed0_bs32_8x5090zxr_20260817"
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/dinov3/outputs/02_eval_inputs/final5m_l_s0_fractional_20260817"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/"
            "final5m_l_s0_fractional_s0protocol_3090fleet_20260817"
        ),
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--python-bin", type=Path, default=Path("/home/inspur/anaconda3/envs/dinov3/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--core-jobs-per-gpu", type=int, default=3)
    parser.add_argument("--seg-jobs-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--ready-age-seconds", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def gpu_is_idle(gpu: str) -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def discover_snapshots(train_run: Path, ready_age_seconds: float) -> list[tuple[int, Path]]:
    now = time.time()
    snapshots: list[tuple[int, Path]] = []
    for directory in (train_run / "eval").glob("training_*"):
        try:
            checkpoint_id = int(directory.name.removeprefix("training_"))
        except ValueError:
            continue
        if (checkpoint_id + 1) % SNAPSHOT_PERIOD:
            continue
        checkpoint = directory / "teacher_checkpoint.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            continue
        if now - checkpoint.stat().st_mtime < ready_age_seconds:
            continue
        snapshots.append((checkpoint_id, checkpoint))
    return sorted(snapshots)


def prepare_adapter(input_root: Path, checkpoint_id: int, source: Path) -> None:
    input_root.mkdir(parents=True, exist_ok=True)
    with (input_root / ".manifest.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        adapter_dir = input_root / str(checkpoint_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter = adapter_dir / "checkpoint.pth"
        if adapter.is_symlink() or adapter.exists():
            if adapter.resolve() != source.resolve():
                raise RuntimeError(f"adapter conflict: {adapter}")
        else:
            adapter.symlink_to(source)

        manifest = input_root / "checkpoint_curve.tsv"
        if not manifest.exists():
            manifest.write_text("checkpoint_id\timage_visits\tepoch_float\tkind\tsource\n")
        existing = {
            line.split("\t", 1)[0]
            for line in manifest.read_text().splitlines()[1:]
            if line.strip()
        }
        if str(checkpoint_id) not in existing:
            updates = checkpoint_id + 1
            with manifest.open("a") as handle:
                handle.write(
                    f"{checkpoint_id}\t{updates * 1024}\t"
                    f"{updates / OFFICIAL_EPOCH_LENGTH:.8f}\tteacher\t{source}\n"
                )


def base_env(args: argparse.Namespace, checkpoint_id: int, jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "CHECKPOINT_ITERS": str(checkpoint_id),
            "GPUS": args.gpu,
            "JOBS_PER_GPU": str(jobs),
            "MAX_CONCURRENT_JOBS": str(jobs),
            "MAX_CPU_JOBS": str(jobs),
            "CONCURRENT_TASK_GROUPS": "0",
            "FROZEN_BATCH_SIZE": "32",
            "FROZEN_CHANNEL_POLICY": "auto",
            "FROZEN_CHANNEL_TTA_SAMPLES": "8",
            "FROZEN_CHANNEL_POLICY_SEED": "0",
            "FROZEN_SPLIT_PROTOCOL": "s0-internal",
            "AUTOCAST_DTYPE": "bf16",
            "CLASSIFICATION_RESOLUTION_PROTOCOL": "manual",
            "CLASSIFICATION_IMAGE_SIZE": "224",
            "CLASSIFICATION_RESIZE_SIZE": "0",
            "REGRESSION_RESOLUTION_PROTOCOL": "manual",
            "REGRESSION_IMAGE_SIZE": "224",
            "REGRESSION_RESIZE_SIZE": "0",
            "TRAIN_FRACTION": "0.8",
            "SEGMENTATION_PROTOCOL": "best",
            "SEGMENTATION_DATASETS_PER_JOB": "1",
            "SEGMENTATION_CHANNEL_POLICY": "auto",
            "SEGMENTATION_CHANNEL_TTA_SAMPLES": "8",
            "SEGMENTATION_CHANNEL_POLICY_SEED": "0",
            "SEG_FEATURE_BATCH_SIZE": "32",
            "SEG_FEATURE_NUM_WORKERS": "2",
            "SEG_PROBE_EPOCHS": "50",
            "SEG_PROBE_BATCH_SIZE": "32",
            "SEG_PROBE_NUM_WORKERS": "2",
            "DET_EPOCHS": "5",
            "DET_BATCH_SIZE": "8",
            "DETECTION_CHANNEL_POLICY": "auto",
            "NUM_WORKERS": "2",
            "EVAL_BLAS_THREADS": "1",
            "SEED": "0",
            "DRY_RUN": "0",
        }
    )
    return env


def run_phase(
    *,
    args: argparse.Namespace,
    checkpoint_id: int,
    phase: str,
    tasks: str,
    jobs: int,
    extra_env: dict[str, str],
    log,
) -> int:
    output = args.output_root / f"point_{checkpoint_id}" / phase
    env = base_env(args, checkpoint_id, jobs)
    env.update(extra_env)
    command = [
        "bash",
        str(args.repo / "scripts/run_bio_benchmark_all.sh"),
        str(args.input_root),
        str(args.train_run / "config.yaml"),
        str(output),
        str(args.benchmark_root),
    ]
    env["TASKS"] = tasks
    print(f"[{utc_now()}] phase={phase} jobs={jobs} command={' '.join(command)}", file=log, flush=True)
    result = subprocess.run(
        command,
        cwd=args.repo,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    print(f"[{utc_now()}] phase={phase} rc={result.returncode}", file=log, flush=True)
    return result.returncode


def evaluate_point(args: argparse.Namespace, checkpoint_id: int, point_log: Path) -> int:
    point_log.parent.mkdir(parents=True, exist_ok=True)
    with point_log.open("a", buffering=1) as log:
        core_rc = run_phase(
            args=args,
            checkpoint_id=checkpoint_id,
            phase="core",
            tasks="classification regression retrieval detection",
            jobs=args.core_jobs_per_gpu,
            extra_env={
                "FROZEN_DATASETS_PER_JOB": "15",
                "CLASSIFICATION_DATASETS": CLASSIFICATION_DATASETS,
                "REGRESSION_DATASETS": "bbbc005",
                "RETRIEVAL_DATASETS": RETRIEVAL_DATASETS,
                "DETECTION_DATASETS": "livecell",
            },
            log=log,
        )
        if core_rc:
            return core_rc

        standard_seg_rc = run_phase(
            args=args,
            checkpoint_id=checkpoint_id,
            phase="segmentation_standard",
            tasks="segmentation",
            jobs=args.seg_jobs_per_gpu,
            extra_env={"SEGMENTATION_DATASETS": STANDARD_SEGMENTATION_DATASETS},
            log=log,
        )
        if standard_seg_rc:
            return standard_seg_rc

        return run_phase(
            args=args,
            checkpoint_id=checkpoint_id,
            phase="segmentation_monuseg",
            tasks="segmentation",
            jobs=1,
            extra_env={"SEGMENTATION_DATASETS": "monuseg"},
            log=log,
        )


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.train_run = args.train_run.resolve()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.python_bin = args.python_bin.resolve()

    required = (
        args.python_bin,
        args.train_run / "config.yaml",
        args.repo / "scripts/run_bio_benchmark_all.sh",
        args.benchmark_root,
    )
    if not all(path.exists() for path in required):
        raise SystemExit(f"missing required path(s): {[str(path) for path in required if not path.exists()]}")

    state_root = args.output_root / "_state"
    claim_root = state_root / "claims"
    done_root = state_root / "done"
    failure_root = state_root / "failures"
    log_root = args.output_root / "logs"
    for path in (claim_root, done_root, failure_root, log_root, state_root / "workers"):
        path.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    worker = f"{host}-gpu{args.gpu}-pid{os.getpid()}"
    worker_status = state_root / "workers" / f"{host}_gpu{args.gpu}.json"

    while True:
        done_count = len(list(done_root.glob("*.json")))
        failed_ids = {
            path.name.split(".", 1)[0]
            for path in failure_root.glob("*.terminal.json")
        }
        if done_count + len(failed_ids) >= EXPECTED_SNAPSHOTS:
            atomic_json(worker_status, {"state": "complete", "updated_at_utc": utc_now(), "worker": worker})
            return 0

        snapshots = discover_snapshots(args.train_run, args.ready_age_seconds)
        progressed = False
        if gpu_is_idle(args.gpu):
            for checkpoint_id, source in snapshots:
                if (done_root / f"{checkpoint_id}.json").exists() or str(checkpoint_id) in failed_ids:
                    continue
                failures = sorted(failure_root.glob(f"{checkpoint_id}.attempt*.json"))
                if len(failures) >= args.max_attempts:
                    atomic_json(
                        failure_root / f"{checkpoint_id}.terminal.json",
                        {"checkpoint_id": checkpoint_id, "failed_at_utc": utc_now(), "attempts": len(failures)},
                    )
                    continue

                claim = claim_root / f"{checkpoint_id}.lock"
                try:
                    claim.mkdir()
                except FileExistsError:
                    continue

                attempt = len(failures) + 1
                owner = {
                    "attempt": attempt,
                    "batch_protocol": {"frozen": 32, "seg_feature": 32, "seg_probe": 32, "detection": 8},
                    "checkpoint_id": checkpoint_id,
                    "claimed_at_utc": utc_now(),
                    "core_jobs_per_gpu": args.core_jobs_per_gpu,
                    "gpu": args.gpu,
                    "host": host,
                    "pid": os.getpid(),
                    "seg_jobs_per_gpu": args.seg_jobs_per_gpu,
                    "segmentation_protocol": SEGMENTATION_PROTOCOL,
                    "source": str(source),
                    "worker": worker,
                }
                atomic_json(claim / "owner.json", owner)
                atomic_json(worker_status, {**owner, "state": "running"})
                prepare_adapter(args.input_root, checkpoint_id, source)
                point_log = log_root / f"checkpoint_{checkpoint_id}.{host}.attempt{attempt}.log"
                started = time.time()
                returncode = evaluate_point(args, checkpoint_id, point_log)
                elapsed = time.time() - started

                if returncode == 0:
                    atomic_json(
                        done_root / f"{checkpoint_id}.json",
                        {**owner, "completed_at_utc": utc_now(), "elapsed_seconds": elapsed},
                    )
                else:
                    atomic_json(
                        failure_root / f"{checkpoint_id}.attempt{attempt}.json",
                        {
                            **owner,
                            "elapsed_seconds": elapsed,
                            "failed_at_utc": utc_now(),
                            "log": str(point_log),
                            "returncode": returncode,
                        },
                    )
                    shutil.rmtree(claim, ignore_errors=True)
                    time.sleep(min(300, 60 * attempt))
                progressed = True
                break

        atomic_json(
            worker_status,
            {
                "discovered_snapshots": len(snapshots),
                "done_points": done_count,
                "gpu_idle": gpu_is_idle(args.gpu),
                "host": host,
                "pid": os.getpid(),
                "state": "scanning" if progressed else "idle",
                "updated_at_utc": utc_now(),
                "worker": worker,
            },
        )
        if not progressed:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
