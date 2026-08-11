#!/usr/bin/env python3
"""Evaluate each S+ data-scaling run as soon as its training completes."""

from __future__ import annotations

import os
import argparse
import subprocess
import time
from pathlib import Path


RUN_PREFIX = "DscaleFinal_splus_sigreg005_random"
LABELS = (10, 20, 50)
CHECKPOINTS = "1024 2049 4099 6149 8199 10249 12299 15374"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-suffix", default="qi4gbs64acc4")
    parser.add_argument("--eval-gpus", default="0 1 5")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    args = parser.parse_args()
    eval_gpus = args.eval_gpus.split()
    if not eval_gpus:
        parser.error("--eval-gpus must contain at least one GPU index")
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be positive")
    max_jobs = len(eval_gpus) * args.jobs_per_gpu
    roots = {
        label: Path(
            f"outputs/01_training_runs/{RUN_PREFIX}{label}_fixed15M_b1024_seed0_{args.run_suffix}"
        )
        for label in LABELS
    }
    base_env = {
        **os.environ,
        "PYTHON_BIN": "/home/bbnc/anaconda3/envs/dinov3/bin/python",
        "CHECKPOINT_ITERS": CHECKPOINTS,
        "GPUS": " ".join(eval_gpus),
        "MAX_CONCURRENT_JOBS": str(max_jobs),
        "MAX_CPU_JOBS": str(max_jobs),
        "JOBS_PER_GPU": str(args.jobs_per_gpu),
        "FROZEN_DATASETS_PER_JOB": "1",
        "CONCURRENT_TASK_GROUPS": "1",
        "FROZEN_BATCH_SIZE": "64",
        "NUM_WORKERS": "2",
        "FROZEN_CHANNEL_POLICY": "auto",
        "DRY_RUN": "0",
    }
    for label, root in roots.items():
        final_checkpoint = root / "ckpt/15374/checkpoint.pth"
        while not final_checkpoint.is_file():
            print(
                f"[data-scaling-watcher] waiting for random{label}: {final_checkpoint}",
                flush=True,
            )
            time.sleep(120)

        output = Path(
            f"outputs/02_eval_runs/{RUN_PREFIX}{label}_fixed15M_{args.run_suffix}__compute_proxy_curve"
        )
        success_marker = output / "._proxy_complete"
        if success_marker.is_file():
            print(f"[data-scaling-watcher] random{label} already complete", flush=True)
            continue
        print(
            f"[data-scaling-watcher] evaluating random{label} on GPUs {args.eval_gpus} "
            f"with {args.jobs_per_gpu} jobs/GPU",
            flush=True,
        )
        subprocess.run(
            [
                "bash",
                "docs/scaling_law/bio_sweet_spot/scripts/14_eval_splus_objective_proxy.sh",
                str(root),
                str(output),
            ],
            env=base_env,
            check=True,
        )
        success_marker.touch()
        print(f"[data-scaling-watcher] random{label} complete", flush=True)
    print("[data-scaling-watcher] all proxy curves complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
