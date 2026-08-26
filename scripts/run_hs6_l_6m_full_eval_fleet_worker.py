#!/usr/bin/env python3
"""Claim full-suite HS6-L 6M evaluation shards from a shared 3090 queue."""

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
from dataclasses import dataclass
from pathlib import Path


OFFICIAL_EPOCH_LENGTH = 5899
SNAPSHOT_PERIOD = 488
FULL_EVAL_PERIOD = 2 * SNAPSHOT_PERIOD
TOTAL_UPDATES = 15 * OFFICIAL_EPOCH_LENGTH
EXPECTED_CHECKPOINTS = TOTAL_UPDATES // FULL_EVAL_PERIOD


@dataclass(frozen=True)
class Lane:
    name: str
    tasks: str
    datasets_env: str
    datasets: str
    jobs: int = 2
    extra_env: tuple[tuple[str, str], ...] = ()


LANES = (
    Lane(
        "classification_a",
        "classification",
        "CLASSIFICATION_DATASETS",
        "bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist",
    ),
    Lane(
        "classification_b",
        "classification",
        "CLASSIFICATION_DATASETS",
        "dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle",
    ),
    Lane(
        "classification_c",
        "classification",
        "CLASSIFICATION_DATASETS",
        "cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1",
    ),
    Lane(
        "classification_d",
        "classification",
        "CLASSIFICATION_DATASETS",
        "chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 "
        "chammi-hpa-task1 chammi-hpa-task2",
    ),
    Lane(
        "regression",
        "regression",
        "REGRESSION_DATASETS",
        "bbbc013 bbbc005 conic-cell-count livecell-cell-count",
    ),
    Lane(
        "retrieval",
        "retrieval",
        "RETRIEVAL_DATASETS",
        "lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross",
    ),
    Lane(
        "detection",
        "detection",
        "DETECTION_DATASETS",
        "livecell bbbc038 conic",
        extra_env=(("DET_BATCH_SIZE", "4"),),
    ),
    Lane(
        "segmentation_a",
        "segmentation",
        "SEGMENTATION_DATASETS",
        "bbbc038 conic",
        extra_env=(("SEGMENTATION_DATASETS_PER_JOB", "1"),),
    ),
    Lane(
        "segmentation_b",
        "segmentation",
        "SEGMENTATION_DATASETS",
        "pannuke tissuenet",
        extra_env=(("SEGMENTATION_DATASETS_PER_JOB", "1"),),
    ),
    Lane(
        "segmentation_c",
        "segmentation",
        "SEGMENTATION_DATASETS",
        "livecell multimodal_cellseg",
        extra_env=(("SEGMENTATION_DATASETS_PER_JOB", "1"),),
    ),
    Lane(
        "segmentation_d",
        "segmentation",
        "SEGMENTATION_DATASETS",
        "monuseg cellpose",
        extra_env=(("SEGMENTATION_DATASETS_PER_JOB", "1"),),
    ),
    Lane("ood", "ood", "OOD_TASKS", "xray cryo", jobs=1),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    run_name = "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--train-run", type=Path, default=repo / "outputs/01_training_runs" / run_name)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo / "outputs/02_eval_inputs/hs6_l_6m_full_1m_20260826",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "outputs/02_eval_runs/hs6_l_6m_full_1m_3090fleet_20260826",
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker", default="")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--ready-age-seconds", type=float, default=30)
    parser.add_argument("--claim-stale-seconds", type=float, default=12 * 3600)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--once", action="store_true", help="Exit after one lane or one idle scan.")
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
        if (checkpoint_id + 1) % FULL_EVAL_PERIOD:
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


def lane_key(checkpoint_id: int, lane: Lane) -> str:
    return f"ckpt_{checkpoint_id}__{lane.name}"


def failure_count(failure_root: Path, key: str) -> int:
    return len(list(failure_root.glob(f"{key}.attempt*.json")))


def clear_stale_claim(claim: Path, stale_seconds: float) -> bool:
    owner_path = claim / "owner.json"
    try:
        owner = json.loads(owner_path.read_text())
        age = time.time() - float(owner["claimed_at_unix"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        age = time.time() - claim.stat().st_mtime
        owner = {}
    if age < stale_seconds:
        return False
    if owner.get("host") == socket.gethostname():
        try:
            os.kill(int(owner["pid"]), 0)
            return False
        except (KeyError, ProcessLookupError, ValueError):
            pass
        except PermissionError:
            return False
    shutil.rmtree(claim, ignore_errors=True)
    return not claim.exists()


def claim_lane(claim_root: Path, key: str, owner: dict[str, object], stale_seconds: float) -> Path | None:
    claim = claim_root / f"{key}.lock"
    try:
        claim.mkdir()
    except FileExistsError:
        if not clear_stale_claim(claim, stale_seconds):
            return None
        try:
            claim.mkdir()
        except FileExistsError:
            return None
    atomic_json(claim / "owner.json", owner)
    return claim


def base_env(args: argparse.Namespace, checkpoint_id: int, jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    # bio_benchmark assigns each child by writing CUDA_VISIBLE_DEVICES itself.
    # Passing a remapped zero here would send every remote worker to physical GPU 0.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "CHECKPOINT_ITERS": str(checkpoint_id),
            "GPUS": args.gpu,
            "JOBS_PER_GPU": str(jobs),
            "MAX_CONCURRENT_JOBS": str(jobs),
            "MAX_CPU_JOBS": str(jobs),
            "CONCURRENT_TASK_GROUPS": "0",
            "FROZEN_DATASETS_PER_JOB": "1",
            "FROZEN_BATCH_SIZE": "32",
            "FROZEN_CHANNEL_POLICY": "auto",
            "FROZEN_SPLIT_PROTOCOL": "current",
            "AUTOCAST_DTYPE": "bf16",
            "CLASSIFICATION_RESOLUTION_PROTOCOL": "best",
            "REGRESSION_RESOLUTION_PROTOCOL": "best",
            "SEGMENTATION_PROTOCOL": "best",
            "SEGMENTATION_CHANNEL_POLICY": "auto",
            "SEG_FEATURE_BATCH_SIZE": "4",
            "SEG_FEATURE_NUM_WORKERS": "2",
            "SEG_PROBE_EPOCHS": "50",
            "SEG_PROBE_BATCH_SIZE": "16",
            "SEG_PROBE_NUM_WORKERS": "2",
            "DET_EPOCHS": "5",
            "DET_BATCH_SIZE": "4",
            "DETECTION_CHANNEL_POLICY": "auto",
            "OOD_DEVICE": f"cuda:{args.gpu}",
            "OOD_BATCH_SIZE": "32",
            "OOD_NUM_WORKERS": "2",
            "NUM_WORKERS": "2",
            "EVAL_BLAS_THREADS": "1",
            "SEED": "0",
            "DRY_RUN": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def run_lane(args: argparse.Namespace, checkpoint_id: int, lane: Lane, log_path: Path) -> int:
    output = args.output_root / f"point_{checkpoint_id}" / lane.name
    env = base_env(args, checkpoint_id, lane.jobs)
    env.update({"TASKS": lane.tasks, lane.datasets_env: lane.datasets})
    env.update(dict(lane.extra_env))
    command = [
        "bash",
        str(args.repo / "scripts/run_bio_benchmark_all.sh"),
        str(args.input_root),
        str(args.train_run / "config.yaml"),
        str(output),
        str(args.benchmark_root),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        print(f"[{utc_now()}] lane={lane.name} command={' '.join(command)}", file=log)
        result = subprocess.run(
            command,
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(f"[{utc_now()}] lane={lane.name} rc={result.returncode}", file=log)
    return result.returncode


def update_checkpoint_status(output_root: Path, checkpoint_id: int) -> None:
    done_root = output_root / "_state/done"
    terminal_root = output_root / "_state/terminal"
    online_root = output_root / "_online_status"
    keys = [lane_key(checkpoint_id, lane) for lane in LANES]
    done = [key for key in keys if (done_root / f"{key}.json").exists()]
    terminal = [key for key in keys if (terminal_root / f"{key}.json").exists()]
    payload: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "done_lanes": done,
        "expected_lanes": len(keys),
        "terminal_lanes": terminal,
        "updated_at_utc": utc_now(),
    }
    atomic_json(online_root / f"ckpt_{checkpoint_id}.status.json", payload)
    if len(done) == len(keys):
        atomic_json(online_root / f"ckpt_{checkpoint_id}.done", payload)
    elif terminal:
        atomic_json(online_root / f"ckpt_{checkpoint_id}.failed", payload)


def validate_args(args: argparse.Namespace) -> None:
    required = (
        args.python_bin,
        args.train_run / "config.yaml",
        args.repo / "scripts/run_bio_benchmark_all.sh",
        args.benchmark_root,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required path(s): {missing}")


def main() -> int:
    args = parse_args()
    for name in ("repo", "train_run", "input_root", "output_root", "benchmark_root", "python_bin"):
        setattr(args, name, getattr(args, name).resolve())
    validate_args(args)

    state_root = args.output_root / "_state"
    claim_root = state_root / "claims"
    done_root = state_root / "done"
    failure_root = state_root / "failures"
    terminal_root = state_root / "terminal"
    log_root = args.output_root / "logs"
    worker_root = state_root / "workers"
    for path in (claim_root, done_root, failure_root, terminal_root, log_root, worker_root):
        path.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    worker = args.worker or f"{host}-gpu{args.gpu}-pid{os.getpid()}"
    worker_status = worker_root / f"{worker}.json"
    while True:
        snapshots = discover_snapshots(args.train_run, args.ready_age_seconds)
        progressed = False
        if gpu_is_idle(args.gpu):
            for checkpoint_id, source in snapshots:
                prepare_adapter(args.input_root, checkpoint_id, source)
                update_checkpoint_status(args.output_root, checkpoint_id)
                for lane in LANES:
                    key = lane_key(checkpoint_id, lane)
                    if (done_root / f"{key}.json").exists() or (terminal_root / f"{key}.json").exists():
                        continue
                    attempts = failure_count(failure_root, key)
                    if attempts >= args.max_attempts:
                        atomic_json(
                            terminal_root / f"{key}.json",
                            {"checkpoint_id": checkpoint_id, "lane": lane.name, "attempts": attempts},
                        )
                        update_checkpoint_status(args.output_root, checkpoint_id)
                        continue
                    owner: dict[str, object] = {
                        "attempt": attempts + 1,
                        "checkpoint_id": checkpoint_id,
                        "claimed_at_unix": time.time(),
                        "claimed_at_utc": utc_now(),
                        "gpu": args.gpu,
                        "host": host,
                        "lane": lane.name,
                        "pid": os.getpid(),
                        "source": str(source),
                        "worker": worker,
                    }
                    claim = claim_lane(claim_root, key, owner, args.claim_stale_seconds)
                    if claim is None:
                        continue
                    atomic_json(worker_status, {**owner, "state": "running"})
                    started = time.time()
                    log_path = log_root / f"{key}.{host}.attempt{attempts + 1}.log"
                    returncode = run_lane(args, checkpoint_id, lane, log_path)
                    result = {
                        **owner,
                        "elapsed_seconds": time.time() - started,
                        "log": str(log_path),
                        "returncode": returncode,
                    }
                    if returncode == 0:
                        atomic_json(done_root / f"{key}.json", {**result, "completed_at_utc": utc_now()})
                    else:
                        atomic_json(
                            failure_root / f"{key}.attempt{attempts + 1}.json",
                            {**result, "failed_at_utc": utc_now()},
                        )
                        shutil.rmtree(claim, ignore_errors=True)
                    update_checkpoint_status(args.output_root, checkpoint_id)
                    progressed = True
                    break
                if progressed:
                    break

        complete_points = len(list((args.output_root / "_online_status").glob("ckpt_*.done")))
        atomic_json(
            worker_status,
            {
                "complete_checkpoints": complete_points,
                "discovered_checkpoints": len(snapshots),
                "expected_checkpoints": EXPECTED_CHECKPOINTS,
                "gpu": args.gpu,
                "gpu_idle": gpu_is_idle(args.gpu),
                "host": host,
                "pid": os.getpid(),
                "state": "scanning" if progressed else "idle",
                "updated_at_utc": utc_now(),
                "worker": worker,
            },
        )
        if complete_points >= EXPECTED_CHECKPOINTS:
            return 0
        if args.once:
            return 0
        if not progressed:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
