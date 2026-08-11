#!/usr/bin/env python3
"""Run external-foundation-model benchmark jobs on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MODELS = [
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
]
FORMAL_STAGES = [
    "hpa", "classification", "nct-cross", "lc25000-diagnostic",
    "rxrx1-cross", "segmentation",
]
BATCH_SIZE = {
    "gigapath": 1,
    "hoptimus0": 1,
    "virchow2": 2,
    "phikon2": 4,
    "uni": 4,
    "siglip2": 4,
    "conch": 8,
    "pe": 8,
    "dinov2": 16,
    "mae": 16,
    "bioclip": 16,
    "jump_cp": 16,
    "cytoself": 32,
    # Keras/Torch retains allocator state across datasets; batch 1 is the
    # reliable setting on 24 GiB 3090 cards.
    "cytoimagenet": 1,
}
EXCLUSIVE_MODELS = {"cytoimagenet", "virchow2", "gigapath", "hoptimus0"}


@dataclass(frozen=True)
class Job:
    model: str
    stage: str
    command: list[str]
    log_path: str


@dataclass
class ActiveJob:
    index: int
    job: Job
    process: subprocess.Popen
    log_handle: object
    started: float
    slots: int


def gpu_state(gpu: str) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = output.split(",")
    return int(memory.strip()), int(utilization.strip())


def wait_for_idle(gpu: str, memory_limit: int, utilization_limit: int, poll_seconds: int) -> None:
    consecutive = 0
    while consecutive < 2:
        try:
            memory, utilization = gpu_state(gpu)
            idle = memory <= memory_limit and utilization <= utilization_limit
            consecutive = consecutive + 1 if idle else 0
            print(
                f"[gpu-wait] GPU{gpu} memory={memory}MiB utilization={utilization}% "
                f"idle_checks={consecutive}/2",
                flush=True,
            )
        except Exception as exc:
            consecutive = 0
            print(f"[gpu-wait] GPU{gpu} query failed: {exc}", flush=True)
        if consecutive < 2:
            time.sleep(poll_seconds)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    output_root = Path(args.output_root)
    feature_root = output_root / "feature_cache"
    log_root = Path(args.log_root)
    jobs: list[Job] = []
    stages = ["smoke"] if args.mode == "smoke" else args.stages
    for stage in stages:
        for model in args.models:
            batch_size = args.batch_size or BATCH_SIZE[model]
            if stage == "smoke":
                output = output_root / "smoke" / model
                command = [
                    args.python, args.classification_script,
                    "--model", model,
                    "--datasets", "bbbc005",
                    "--max-samples", "32",
                    "--output-dir", str(output),
                    "--feature-root", str(feature_root),
                    "--batch-size", str(batch_size),
                    "--num-workers", "0",
                    "--no-save-features",
                ]
            elif stage == "classification":
                output = output_root / "classification" / model
                command = [
                    args.python, args.classification_script,
                    "--model", model,
                    "--output-dir", str(output),
                    "--feature-root", str(feature_root),
                    "--batch-size", str(batch_size),
                    "--num-workers", str(args.num_workers),
                ]
            elif stage == "segmentation":
                output = output_root / "segmentation"
                command = [
                    args.python, args.dense_probe_script,
                    "--models", model,
                    "--datasets", "bbbc038", "conic", "livecell", "monuseg", "pannuke", "tissuenet",
                    "--out-root", str(output),
                    "--extract-batch-size", str(batch_size),
                    "--num-workers", str(args.num_workers),
                    "--probe-num-workers", "0",
                    "--epochs", "20",
                    "--eval-every", "5",
                    "--max-feature-side", "32",
                ]
            else:
                output = output_root / "retrieval_clustering" / model
                command = [
                    args.python, args.retrieval_script,
                    "--model", model,
                    "--protocols", stage,
                    "--output-dir", str(output),
                    "--feature-root", str(feature_root),
                    "--batch-size", str(batch_size),
                    "--num-workers", "0" if stage == "rxrx1-cross" else str(args.num_workers),
                    "--metric-chunk-size", str(args.metric_chunk_size),
                ]
            jobs.append(Job(model, stage, command, str(log_root / f"{stage}_{model}.log")))
    return jobs


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--stages", nargs="+", default=FORMAL_STAGES, choices=FORMAL_STAGES)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", default="/home/bbnc/venvs/external_fm/bin/python")
    parser.add_argument(
        "--classification-script",
        default="/mnt/huawei_deepcad/dinov3/scripts/run_external_fm_linear_probe.py",
    )
    parser.add_argument(
        "--retrieval-script",
        default="/mnt/huawei_deepcad/dinov3/scripts/run_external_fm_retrieval_clustering.py",
    )
    parser.add_argument(
        "--dense-probe-script",
        default="/mnt/huawei_deepcad/benchmark_model/run_dense_probe_benchmark.py",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/external_fm_fair_protocol_20260721",
    )
    parser.add_argument(
        "--log-root",
        default="/mnt/huawei_deepcad/dinov3/outputs/04_external_fm_logs_20260721",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metric-chunk-size", type=int, default=128)
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent lightweight jobs; large encoders still reserve all slots.",
    )
    parser.add_argument("--max-memory-used-mb", type=int, default=1500)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-wait-idle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = log_root / f"{args.mode}_status.json"
    results: list[dict] = []
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": (
            "/mnt/huawei_deepcad/benchmark_model/_vendor:"
            "/mnt/huawei_deepcad/benchmark_model:/mnt/huawei_deepcad/dinov3"
        ),
        "DENSE_PROBE_METRIC_PYTHON": "/home/bbnc/anaconda3/envs/dinov3/bin/python",
        "KERAS_BACKEND": "torch",
    })

    if args.jobs_per_gpu <= 0:
        raise ValueError("--jobs-per-gpu must be positive")
    if not args.no_wait_idle:
        wait_for_idle(
            args.gpu, args.max_memory_used_mb, args.max_utilization, args.poll_seconds
        )

    active: list[ActiveJob] = []
    next_index = 0
    while next_index < len(jobs) or active:
        used_slots = sum(item.slots for item in active)
        while next_index < len(jobs):
            job = jobs[next_index]
            slots = args.jobs_per_gpu if job.model in EXCLUSIVE_MODELS else 1
            if used_slots + slots > args.jobs_per_gpu:
                break
            index = next_index + 1
            print(
                f"[job {index}/{len(jobs)}] {job.stage}/{job.model} slots={slots}",
                flush=True,
            )
            print(f"[command] {shlex.join(job.command)}", flush=True)
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a")
            handle.write(f"\n$ {shlex.join(job.command)}\n")
            handle.flush()
            process = subprocess.Popen(
                job.command,
                env=environment,
                cwd="/mnt/huawei_deepcad/dinov3",
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append(ActiveJob(index, job, process, handle, time.time(), slots))
            used_slots += slots
            next_index += 1

        if not active:
            raise RuntimeError("Scheduler has pending jobs but cannot allocate any slot")
        time.sleep(5)
        finished = [item for item in active if item.process.poll() is not None]
        for item in finished:
            item.log_handle.close()
            result = {
                **asdict(item.job),
                "returncode": int(item.process.returncode),
                "elapsed_seconds": time.time() - item.started,
                "slots": item.slots,
            }
            results.append(result)
            active.remove(item)
            print(
                f"[done] {item.job.stage}/{item.job.model} "
                f"returncode={item.process.returncode} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        write_status(status_path, {
            "mode": args.mode,
            "gpu": args.gpu,
            "jobs_per_gpu": args.jobs_per_gpu,
            "completed_jobs": len(results),
            "total_jobs": len(jobs),
            "active": [
                {"index": item.index, **asdict(item.job), "slots": item.slots}
                for item in active
            ],
            "results": results,
        })
    failures = [result for result in results if result["returncode"] != 0]
    print(f"[queue-complete] jobs={len(results)} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
