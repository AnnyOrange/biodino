#!/usr/bin/env python3
"""Watch the 40k run, audit it, and run selected locked v4 ID checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from launch_hs6_long_100tb_deepcad4_40k_20261010 import DATA, ROOT, RUN_NAME, FULL_POOL_RUN_NAME


PYTHON = "/home/deepcad/anaconda3/envs/dinov3/bin/python"
RUN = ROOT / "outputs/01_training_runs" / RUN_NAME
STATUS = DATA / "TRAINING_100TB_40K_STATUS.json"
EVAL_BASE = ROOT / "plot/fig2/data_quality/long_100tb_40k_v4_id_20261010"
STEPS = (39999, 27999, 15999, 3999)
SESSION = "hs6_100tb_40k_20261010"
LABEL_TAG = "deepcad40k"
FULL_POOL = False


def record(stage: str, **detail) -> None:
    STATUS.write_text(json.dumps({"stage": stage, "time_unix": time.time(), **detail}, indent=2) + "\n")


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a", buffering=1) as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {log_path}")


def wait_for_training() -> None:
    retries = 0
    while True:
        exit_path = RUN / "exit.json"
        if exit_path.is_file():
            code = json.loads(exit_path.read_text())["returncode"]
            if code == 0:
                return
            checkpoints = list((RUN / "ckpt").glob("*/checkpoint.pth"))
            if retries >= 3 or not checkpoints:
                raise RuntimeError(f"Training exited {code}; resumable checkpoints: {len(checkpoints)}")
            retries += 1
            latest = max(checkpoints, key=lambda path: int(path.parent.name))
            record("RESUMING", attempt=retries, latest_checkpoint=str(latest))
            with (DATA / "logs" / "resume_100tb_40k.log").open("a", buffering=1) as output:
                subprocess.run([
                    PYTHON, "-u", str(ROOT / "scripts/launch_hs6_long_100tb_deepcad4_40k_20261010.py"),
                    "--resume", *(["--full-pool"] if FULL_POOL else []),
                ], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            continue
        if subprocess.run(["tmux", "has-session", "-t", SESSION],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Training tmux session ended without an exit record")
        record("TRAINING", run=str(RUN), metrics=str(RUN / "raw_loss_metrics.jsonl"))
        time.sleep(180)


def evaluate(step: int) -> None:
    campaign = EVAL_BASE / f"100tb_ck{step}"
    label = f"dq_long_{LABEL_TAG}_100tb_ck{step}"
    if not (campaign / "prepared.json").is_file():
        record("PREPARING_EVALUATION", step=step)
        run_logged([
            PYTHON, "-u", str(ROOT / "scripts/prepare_hs6_long_v4_id_20261009.py"),
            "--arm", "100tb", "--step", str(step), "--train-root", str(RUN),
            "--eval-base", str(EVAL_BASE), "--label-tag", LABEL_TAG,
        ], DATA / "logs" / f"prepare_eval_{label}.log")
    status_path = campaign / "driver_status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("state") == "COMPLETE":
        return
    record("EVALUATING", step=step, campaign=str(campaign))
    run_logged([
        PYTHON, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
        "--arm", label, "--eval-root", str(campaign), "--gpus", "1", "5", "6", "7",
    ], DATA / "logs" / f"eval_{label}.log")
    if not status_path.is_file() or json.loads(status_path.read_text()).get("state") != "COMPLETE":
        raise RuntimeError(f"V4 ID evaluation did not complete: {campaign}")
    record("EVALUATION_COMPLETE", step=step, campaign=str(campaign))


def main() -> None:
    global RUN, STATUS, EVAL_BASE, SESSION, LABEL_TAG, FULL_POOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pool", action="store_true")
    args = parser.parse_args()
    FULL_POOL = args.full_pool
    if FULL_POOL:
        RUN = ROOT / "outputs/01_training_runs" / FULL_POOL_RUN_NAME
        STATUS = DATA / "TRAINING_100TB_FULL_POOL_40K_STATUS.json"
        EVAL_BASE = ROOT / "plot/fig2/data_quality/long_100tb_full_pool_40k_v4_id_20261010"
        SESSION = "hs6_100tb_fullpool_40k_20261010"
        LABEL_TAG = "deepcad40kfullpool"
    try:
        wait_for_training()
        record("AUDITING", run=str(RUN))
        run_logged([PYTHON, "-u", str(ROOT / "scripts/audit_hs6_long_100tb_40k_20261010.py"),
                    "--run", str(RUN)], DATA / "logs" / "audit_100tb_40k.log")
        for step in STEPS:
            evaluate(step)
        record("COMPLETE", run=str(RUN), audit=str(RUN / "audit.json"), evaluations=str(EVAL_BASE))
    except Exception as exc:
        record("ERROR", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
