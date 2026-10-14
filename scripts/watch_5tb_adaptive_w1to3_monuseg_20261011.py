#!/usr/bin/env python3
"""Run the pinned MoNuSeg 30/7/14 campaign after the adaptive CLS pilot ends."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/data/xuzijing/monuseg_t30v7_remote_20260930")
RUN = Path("/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/") / (
    "hs6_l5_v2_recovery_fork29279_20260930/global_cls_adaptive_w1to3"
)
STEPS = tuple(range(29767, 35136, 488))
SITE = BASE / "site_lyxxr_adaptive_w1to3_20261011.json"
OUTPUT = BASE / "campaign_adaptive_w1to3_20261011"
STATUS = BASE / "watch_adaptive_w1to3_20261011.json"
PYTHON = Path("/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python")
LAUNCHER = BASE / "launch_monuseg_train30val7_remote_20260930.py"


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def readiness():
    snapshots = [RUN / "eval" / f"training_{step}" / "teacher_checkpoint.pth" for step in STEPS]
    missing = [str(path) for path in snapshots if not path.is_file() or path.stat().st_size < 10**9]
    full = RUN / "ckpt/35135/checkpoint.pth"
    if not full.is_file() or full.stat().st_size < 8 * 10**9:
        missing.append(str(full))
    processes = subprocess.check_output(["ps", "-eo", "args"], text=True)
    training_alive = any(
        "dinov3/train/train.py" in line and str(RUN) in line
        for line in processes.splitlines()
    )
    return snapshots, missing, training_alive


def prepare(snapshots):
    source = json.loads((BASE / "site_lyxxr_v3_round5_20261006.json").read_text())
    source.update(name="lyxxr-adaptive-w1to3", output=str(OUTPUT), gpus=[0, 1, 2, 3])
    source["assets"] = [
        dict(
            arm="v2_global_cls_adaptive_w1to3",
            checkpoint_id=str(step),
            path=str(path),
            config="/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/bin/config.yaml",
            kind="dinov3",
            model_id="",
            reserve_mib=4096,
        )
        for step, path in zip(STEPS, snapshots)
    ]
    save(SITE, source)
    subprocess.run([str(PYTHON), str(LAUNCHER), "prepare", "--site", str(SITE)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Check readiness without starting work")
    args = parser.parse_args()
    while True:
        snapshots, missing, training_alive = readiness()
        ready = not missing and not training_alive
        save(STATUS, dict(
            checked_utc=datetime.now(timezone.utc).isoformat(), ready=ready,
            missing=missing, training_alive=training_alive, checkpoint_count=len(STEPS),
            campaign=str(OUTPUT),
        ))
        if args.once:
            return
        if not ready:
            time.sleep(60)
            continue
        try:
            prepare(snapshots)
            subprocess.run([str(PYTHON), str(LAUNCHER), "worker", "--site", str(SITE)], check=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(300)


if __name__ == "__main__":
    main()
