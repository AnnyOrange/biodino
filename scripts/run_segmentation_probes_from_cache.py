#!/usr/bin/env python3
"""Run deterministic segmentation probes against existing feature caches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


DATASETS = (
    "bbbc038",
    "cellpose",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
    "multimodal_cellseg",
)


@dataclass(frozen=True)
class Job:
    dataset: str
    train_cache: Path
    val_cache: Path
    test_cache: Path
    output_dir: Path


def find_cache(cache_root: Path, dataset: str, checkpoint: str, split: str) -> Path:
    matches = sorted(
        path.resolve()
        for path in cache_root.glob(f"*/{dataset}/{checkpoint}/{dataset}_{split}_*.npz")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {split} cache for {dataset}/{checkpoint}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def is_complete(path: Path, batch_size: int, epochs: int, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
        meta = result["_meta"]
        return (
            "test" in result
            and meta.get("probe_rng_seeded") is True
            and int(meta["probe_batch_size"]) == batch_size
            and int(meta["probe_epochs"]) == epochs
            and int(meta["seed"]) == seed
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def probe_command(
    job: Job,
    batch_size: int,
    epochs: int,
    seed: int,
    num_workers: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dinov3.eval.bio_segmentation.linear_probe",
        "--dataset",
        job.dataset,
        "--use-cached-features",
        "--train-cache",
        str(job.train_cache),
        "--val-cache",
        str(job.val_cache),
        "--test-cache",
        str(job.test_cache),
        "--output-dir",
        str(job.output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--num-workers",
        str(num_workers),
        "--eval-every",
        str(epochs),
        "--seed",
        str(seed),
        "--semantic-only",
    ]
    if job.dataset == "conic":
        command.extend(["--class-weight-mode", "sqrt_inverse"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers_per_gpu <= 0:
        parser.error("--workers-per-gpu must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in args.datasets:
        jobs.append(
            Job(
                dataset=dataset,
                train_cache=find_cache(args.cache_root, dataset, args.checkpoint, "train"),
                val_cache=find_cache(args.cache_root, dataset, args.checkpoint, "val"),
                test_cache=find_cache(args.cache_root, dataset, args.checkpoint, "test"),
                output_dir=(args.output_root / dataset).resolve(),
            )
        )

    manifest = {
        "checkpoint": args.checkpoint,
        "cache_root": str(args.cache_root.resolve()),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "semantic_only": True,
        "jobs": [
            {
                "dataset": job.dataset,
                "train_cache": str(job.train_cache),
                "val_cache": str(job.val_cache),
                "test_cache": str(job.test_cache),
                "output_dir": str(job.output_dir),
            }
            for job in jobs
        ],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    pending = deque(
        job
        for job in jobs
        if not is_complete(
            job.output_dir / "results.json", args.batch_size, args.epochs, args.seed
        )
    )
    print(f"jobs={len(jobs)} complete={len(jobs) - len(pending)} pending={len(pending)}", flush=True)
    if args.dry_run:
        for job in pending:
            print(" ".join(probe_command(job, args.batch_size, args.epochs, args.seed, args.num_workers)))
        return 0

    running: list[tuple[subprocess.Popen, str, Job, object]] = []
    failures: list[Job] = []
    while pending or running:
        for gpu in args.gpus:
            active = sum(active_gpu == gpu for _, active_gpu, _, _ in running)
            while pending and active < args.workers_per_gpu:
                job = pending.popleft()
                job.output_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (job.output_dir / "probe.log").open("w")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    env[variable] = "2"
                process = subprocess.Popen(
                    probe_command(job, args.batch_size, args.epochs, args.seed, args.num_workers),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running.append((process, gpu, job, log_handle))
                active += 1
                print(f"START gpu={gpu} pid={process.pid} dataset={job.dataset}", flush=True)

        time.sleep(0.5)
        still_running = []
        for process, gpu, job, log_handle in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((process, gpu, job, log_handle))
                continue
            log_handle.close()
            status = "DONE" if returncode == 0 else "FAIL"
            print(
                f"{status} gpu={gpu} rc={returncode} dataset={job.dataset}",
                flush=True,
            )
            if returncode != 0:
                failures.append(job)
        running = still_running

    print(f"finished={len(jobs) - len(failures)} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
