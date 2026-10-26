#!/usr/bin/env python3
"""Resume the isolated five-batch 1PB replay after transient train failures."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
RUN = REPO / "outputs/01_training_runs/HS6_L_random1pb_replay_missing_4x3090qi_20261020"
QUEUE = Path("/mnt/huawei_blm/rolling_1pb_replay_missing_4x3090qi_20261020/queue")
BASE = REPO / "outputs/01_training_runs/HS6_L_random1pb1m_clean_robust_biosafe256_gb2048_lr1e4_e1_seed0_8x5090zxr_ddp_b32_20261012"
PYTHON = Path("/home/bbnc/anaconda3/envs/dinov3/bin/python")
EXPECTED = {f"batch-{number:06d}" for number in range(1, 6)}


def completed() -> set[str]:
    journal = RUN / "rolling_controller.jsonl"
    if not journal.is_file():
        return set()
    records = [json.loads(line) for line in journal.read_text().splitlines() if line]
    return {record["batch_id"] for record in records
            if record["batch_id"] in EXPECTED and Path(record["checkpoint"]).is_file()}


def main() -> None:
    failures_without_progress = 0
    while True:
        done = completed()
        if done == EXPECTED:
            print("REPLAY_COMPLETE all five checkpoints retained", flush=True)
            return
        remaining = len(EXPECTED - done)
        command = [str(PYTHON), "-u", "scripts/run_rolling_1pb_train.py",
                   "--queue-root", str(QUEUE), "--base-config", str(BASE / "config.yaml"),
                   "--run-dir", str(RUN),
                   "--baseline-checkpoint", str(BASE / "eval/training_487/teacher_checkpoint.pth"),
                   "--initial-full-checkpoint", str(BASE / "ckpt/487/checkpoint.pth"),
                   "--python", str(PYTHON), "--world-size", "4",
                   "--cuda-visible-devices", "2,5,6,7",
                   "--effective-global-batch-size", "1024", "--batch-size-per-gpu", "32",
                   "--batch-updates", "488", "--total-updates", "40016",
                   "--max-batches", str(remaining), "--master-port", "31972",
                   "--train-retries", "3", "--allow-segment-reconfigure"]
        print(f"REPLAY_RESUME completed={sorted(done)} remaining={remaining}", flush=True)
        result = subprocess.run(command, cwd=REPO, check=False)
        advanced = completed()
        failures_without_progress = 0 if len(advanced) > len(done) else failures_without_progress + 1
        if failures_without_progress >= 3:
            raise RuntimeError(f"Replay failed repeatedly without a new checkpoint; exit={result.returncode}")
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"REPLAY_SUPERVISOR_ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
