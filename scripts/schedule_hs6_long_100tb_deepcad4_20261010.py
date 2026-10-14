#!/usr/bin/env python3
"""Smoke-test and then train the four-A100 100TB arm."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from launch_hs6_long_100tb_deepcad4_20261010 import DATA, ROOT, RUN_NAME


PYTHON = "/home/deepcad/anaconda3/envs/dinov3/bin/python"
STATUS = DATA / "TRAINING_100TB_DEEPCAD4_STATUS.json"
EVAL_BASE = ROOT / "plot/fig2/data_quality/long_norepeat_eval_deepcad4_20261010"
TWENTY_RUN = ROOT / "outputs/01_training_runs/HS6_L_quality_long_noreplace_20tb_ddp_gb1024_16k_20261009"


def record(stage: str, **details) -> None:
    STATUS.write_text(json.dumps({"stage": stage, "time_unix": time.time(), **details}, indent=2) + "\n")


def run_stage(smoke: bool) -> None:
    run = ROOT / "outputs/01_training_runs" / (RUN_NAME + ("_smoke" if smoke else ""))
    label = "SMOKE" if smoke else "FULL"
    if not (run / "exit.json").is_file():
        record("TRAINING_" + label, run=str(run))
        command = [PYTHON, "-u", str(ROOT / "scripts/launch_hs6_long_100tb_deepcad4_20261010.py")]
        if smoke:
            command.append("--smoke")
        log = DATA / "logs" / f"train_100tb_deepcad4_{label.lower()}.log"
        with log.open("a", buffering=1) as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"100TB {label} training failed: {log}")
    if json.loads((run / "exit.json").read_text())["returncode"]:
        raise RuntimeError(f"100TB {label} run did not exit successfully: {run}")
    record("AUDITING_" + label, run=str(run))
    audit = [PYTHON, str(ROOT / "scripts/audit_hs6_long_finite_quality_20261009.py"),
             "--arm", "100tb", "--run", str(run)]
    if smoke:
        audit.append("--smoke")
    log = DATA / "logs" / f"audit_100tb_deepcad4_{label.lower()}.log"
    with log.open("a", buffering=1) as output:
        result = subprocess.run(audit, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"100TB {label} audit failed: {log}")
    record("COMPLETE_" + label, run=str(run), audit=str(run / "audit.json"))


def evaluate(arm: str, train: Path, step: int) -> None:
    campaign = EVAL_BASE / f"{arm}_ck{step}"
    label = f"long_deepcad4_{arm}_ck{step}"
    if not (campaign / "prepared.json").is_file():
        record("PREPARING_EVALUATION", arm=arm, step=step)
        command = [PYTHON, "-u", str(ROOT / "scripts/prepare_hs6_long_v4_id_20261009.py"),
                   "--arm", arm, "--step", str(step), "--train-root", str(train),
                   "--eval-base", str(EVAL_BASE), "--label-tag", "deepcad4"]
        with (DATA / "logs" / f"prepare_eval_{label}.log").open("a", buffering=1) as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Preparing v4 ID evaluation failed for {label}")
    status_path = campaign / "driver_status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("state") == "COMPLETE":
        return
    record("EVALUATING", arm=arm, step=step, campaign=str(campaign))
    command = [PYTHON, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
               "--arm", label, "--eval-root", str(campaign), "--gpus", "1", "5", "6", "7"]
    with (DATA / "logs" / f"eval_{label}.log").open("a", buffering=1) as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode or not status_path.is_file() or \
            json.loads(status_path.read_text()).get("state") != "COMPLETE":
        raise RuntimeError(f"V4 ID evaluation failed for {label}")
    record("EVALUATION_COMPLETE", arm=arm, step=step, campaign=str(campaign))


def main() -> None:
    try:
        run_stage(True)
        run_stage(False)
        hundred = ROOT / "outputs/01_training_runs" / RUN_NAME
        for step in (3999, 7999, 11999, 15999):
            evaluate("100tb", hundred, step)
        while not (TWENTY_RUN / "audit.json").is_file():
            status = json.loads((DATA / "TRAINING_STATUS.json").read_text())
            if status["stage"] == "ERROR":
                raise RuntimeError(f"20TB training failed: {status}")
            record("WAITING_FOR_20TB_AUDIT", run=str(TWENTY_RUN))
            time.sleep(180)
        for step in (3999, 7999, 11999, 15999):
            evaluate("20tb", TWENTY_RUN, step)
        record("COMPLETE", runs=[str(hundred), str(TWENTY_RUN)], evaluations=str(EVAL_BASE))
    except Exception as exc:
        record("ERROR", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
