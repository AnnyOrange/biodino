#!/usr/bin/env python3
"""Wait for eight idle 3090s, then run and audit the matched finite arms."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
DATA = Path("/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009")
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
STATUS = DATA / "TRAINING_STATUS.json"
PREP_STATUS = DATA / "PREPARATION_STATUS.json"
EVAL_BASE = ROOT / "plot/fig2/data_quality/long_norepeat_eval_20261009"


def record(stage: str, **detail) -> None:
    STATUS.write_text(json.dumps({"stage": stage, "time_unix": time.time(), **detail}, indent=2) + "\n")


def gpus_idle() -> bool:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
    ).splitlines()
    if len(inventory) != 8:
        raise RuntimeError(f"Expected eight GPUs, found {len(inventory)}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    return not any(line.strip().isdigit() for line in active)


def wait_for_gpus() -> None:
    while True:
        if gpus_idle():
            time.sleep(30)
            if gpus_idle():
                return
        record("WAITING_FOR_8_IDLE_GPUS")
        time.sleep(120)


def wait_for_100tb() -> None:
    while True:
        if PREP_STATUS.is_file():
            prep = json.loads(PREP_STATUS.read_text())
            if prep["state"] == "READY":
                return
            if prep["state"] == "ERROR":
                raise RuntimeError(f"100TB preparation failed: {prep['error']}")
            record("WAITING_FOR_100TB_DATA", preparation=prep["state"])
        else:
            record("WAITING_FOR_100TB_DATA", preparation="NOT_STARTED")
        time.sleep(180)


def run_arm(arm: str, smoke: bool) -> None:
    suffix = "_smoke" if smoke else ""
    run = ROOT / f"outputs/01_training_runs/HS6_L_quality_long_noreplace_{arm}_ddp_gb1024_16k_20261009{suffix}"
    if (run / "exit.json").is_file():
        exit_record = json.loads((run / "exit.json").read_text())
        if exit_record["returncode"] != 0:
            raise RuntimeError(f"Earlier {arm}{suffix} run failed: {run}")
    else:
        if run.exists():
            raise RuntimeError(f"Incomplete {arm}{suffix} run needs inspection: {run}")
        wait_for_gpus()
        record("TRAINING", arm=arm, smoke=smoke, run=str(run))
        command = [PYTHON, "-u", str(ROOT / "scripts/launch_hs6_long_finite_quality_20261009.py"),
                   "--arm", arm, "--master-port", "31859"]
        if smoke:
            command.append("--smoke")
        log_path = DATA / "logs" / f"train_{arm}{suffix}.log"
        with log_path.open("a", buffering=1) as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{arm}{suffix} training failed: {log_path}")
    if not (run / "audit.json").is_file():
        record("AUDITING", arm=arm, smoke=smoke)
        command = [PYTHON, str(ROOT / "scripts/audit_hs6_long_finite_quality_20261009.py"), "--arm", arm]
        if smoke:
            command.append("--smoke")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log_path = DATA / "logs" / f"audit_{arm}{suffix}.log"
        log_path.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{arm}{suffix} audit failed: {log_path}")
    record("ARM_COMPLETE", arm=arm, smoke=smoke, audit=str(run / "audit.json"))


def run_evaluation(arm: str, step: int) -> None:
    campaign = EVAL_BASE / f"{arm}_ck{step}"
    label = f"long_{arm}_ck{step}"
    prepared = campaign / "prepared.json"
    if not prepared.is_file():
        record("PREPARING_EVALUATION", arm=arm, checkpoint=step)
        command = [PYTHON, "-u", str(ROOT / "scripts/prepare_hs6_long_v4_id_20261009.py"),
                   "--arm", arm, "--step", str(step)]
        with (DATA / "logs" / f"prepare_eval_{label}.log").open("a") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Preparing v4 ID evaluation failed for {label}")
    status_path = campaign / "driver_status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("state") == "COMPLETE":
        return
    wait_for_gpus()
    record("EVALUATING", arm=arm, checkpoint=step, campaign=str(campaign))
    command = [PYTHON, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
               "--arm", label, "--eval-root", str(campaign),
               "--gpus", *map(str, range(8))]
    with (DATA / "logs" / f"eval_{label}.log").open("a") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not status_path.is_file() or \
            json.loads(status_path.read_text()).get("state") != "COMPLETE":
        raise RuntimeError(f"V4 ID evaluation failed for {label}")
    record("EVALUATION_COMPLETE", arm=arm, checkpoint=step, campaign=str(campaign))


def main() -> None:
    try:
        run_arm("20tb", True)
        run_arm("20tb", False)
        wait_for_100tb()
        run_arm("100tb", True)
        run_arm("100tb", False)
        record("TRAINING_COMPLETE", arms=["20tb", "100tb"])
        for step in (3999, 7999, 11999, 15999):
            for arm in ("20tb", "100tb"):
                run_evaluation(arm, step)
        record("COMPLETE", arms=["20tb", "100tb"], evaluated_steps=[4000, 8000, 12000, 16000])
    except Exception as exc:
        record("ERROR", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
