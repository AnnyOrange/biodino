#!/usr/bin/env python3
"""Finalize and evaluate the locked S+ seed-1 replication after training."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def wait_for(path: Path, poll_seconds: int) -> None:
    while not path.is_file():
        print(f"[seed1-watcher] waiting for {path}", flush=True)
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    train_root = Path(
        "outputs/01_training_runs/"
        "S6sigreg005_seed1_repro_b1024_officialprec_e15_local5090_20260721"
    )
    interpolation_root = Path(
        "outputs/01_training_runs/"
        "S6interp_official_sigreg005_seed1_ck8199_a075_20260721"
    )
    eval_root = Path(
        "outputs/02_eval_runs/"
        "S6interp_official_sigreg005_seed1_ck8199_a075_20260721__full_dense_local"
    )

    # Wait for the final checkpoint so evaluation starts only after all 5090s
    # have been released by training. The selected parent remains ck8199.
    wait_for(train_root / "ckpt/15374/checkpoint.pth", args.poll_seconds)

    interpolated = interpolation_root / "ckpt/75/checkpoint.pth"
    if not interpolated.is_file():
        subprocess.run(
            [
                "/home/lxy/miniconda3/envs/dinov3/bin/python",
                "scripts/interpolate_teacher_checkpoints.py",
                "--official-checkpoint",
                "/mnt/huawei_deepcad/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                "--bio-checkpoint",
                str(train_root / "ckpt/8199/checkpoint.pth"),
                "--bio-config",
                str(train_root / "config.yaml"),
                "--output-root",
                str(interpolation_root),
                "--alphas",
                "0.75",
            ],
            check=True,
        )

    env = {
        "PYTHON_BIN": "/home/lxy/miniconda3/envs/dinov3/bin/python",
        "CHECKPOINT_ITERS": "75",
        "TASKS": "classification regression retrieval detection segmentation",
        "REGRESSION_DATASETS": "bbbc005",
        "GPUS": "0 1 2 3 4 5 6 7",
        "MAX_CONCURRENT_JOBS": "8",
        "MAX_CPU_JOBS": "8",
        "JOBS_PER_GPU": "1",
        "FROZEN_DATASETS_PER_JOB": "1",
        "FROZEN_BATCH_SIZE": "128",
        "NUM_WORKERS": "2",
        "FROZEN_CHANNEL_POLICY": "auto",
        "SEGMENTATION_CHANNEL_POLICY": "auto",
        "DRY_RUN": "0",
    }
    subprocess.run(
        [
            "env",
            *[f"{key}={value}" for key, value in env.items()],
            "bash",
            "docs/scaling_law/bio_sweet_spot/scripts/01_eval_splus_sweep.sh",
        ],
        env={**__import__("os").environ, "TRAIN_DIR": str(interpolation_root), "OUTPUT_DIR": str(eval_root), **env},
        check=True,
    )
    print(f"[seed1-watcher] complete: {eval_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
