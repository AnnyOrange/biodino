#!/usr/bin/env python3
"""Finish 20TB training and all five matched-budget v4 ID evaluations."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from continue_data_quality_3090_20261002 import command as launch_command
from continue_data_quality_3090_20261002 import exit_code, verify_smoke


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
ARMS = ("1tb", "5tb", "20tb", "100tb", "1pb")
PAIRS = ((('100tb', '0,1,2,3'), ('5tb', '4,5,6,7')),
         (('1tb', '0,1,2,3'), ('1pb', '4,5,6,7')),
         (('20tb', '0,1,2,3,4,5,6,7'),))


def status(state: str, **fields) -> None:
    path = OUT / "finalizer_3090_status.json"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(state=state, time_unix=time.time(), **fields), indent=2) + "\n")
    os.replace(temp, path)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def wait_for_training(arms: tuple[str, ...]) -> None:
    while True:
        codes = {arm: exit_code(arm) for arm in arms}
        status("WAIT_TRAINING", arms=codes)
        if any(code is not None and code != 0 for code in codes.values()):
            raise RuntimeError(f"Training failed: {codes}")
        if all(code == 0 for code in codes.values()):
            return
        time.sleep(180)


def run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed rc={result.returncode}; inspect {path}")


def audit(arm: str) -> None:
    if (OUT / "training" / arm / "audit.json").is_file():
        result = json.loads((OUT / "training" / arm / "audit.json").read_text())
        if result["status"] == "PASS":
            return
    log(f"Auditing {arm} training")
    run_logged([PYTHON, "-u", str(ROOT / "scripts/audit_data_quality_1m_20261002.py"),
                "--arm", arm], OUT / "training" / arm / "audit_console.log")


def wait_for_20tb_sample() -> None:
    complete = Path("/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002/extraction_complete.json")
    while not complete.is_file():
        state_path = OUT / "20tb_extraction_status.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        status("WAIT_20TB_SAMPLE", extraction=state)
        if state.get("state") == "ERROR":
            raise RuntimeError(f"20TB extraction failed: {state}")
        time.sleep(180)
    result = json.loads(complete.read_text())
    if result["status"] != "PASS" or result["samples"] != 1_000_000 or result["shards"] != 500:
        raise ValueError("20TB sample completion record is invalid")


def train_20tb() -> None:
    if (OUT / "training/20tb").exists():
        log("20TB training already has an output directory; waiting for it")
        wait_for_training(("20tb",))
        audit("20tb")
        return
    smoke = OUT / "smoke/20tb"
    if not smoke.exists():
        log("Starting 20TB smoke test on GPUs 0-3")
        run_logged(launch_command("20tb", "0,1,2,3", 29615, smoke=True),
                   OUT / "20tb_smoke_launcher.log")
    verify_smoke("20tb")
    log("Starting 20TB 976-update training on GPUs 0-3")
    with (OUT / "20tb_launcher.log").open("w") as stream:
        process = subprocess.Popen(launch_command("20tb", "0,1,2,3", 29615),
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            status("TRAIN_20TB", launcher_pid=process.pid)
            time.sleep(180)
        if process.wait() != 0:
            raise RuntimeError("20TB training launcher failed; inspect 20tb_launcher.log")
    wait_for_training(("20tb",))
    audit("20tb")


def prepare_evaluations() -> None:
    for arm in ARMS:
        audit(arm)
        if (OUT / "eval" / arm / "prepared.json").is_file():
            continue
        log(f"Preparing v4 ID evaluation for {arm}")
        run_logged([PYTHON, "-u", str(ROOT / "scripts/prepare_data_quality_v4_id_20261002.py"),
                    "--arm", arm], OUT / "eval" / f"{arm}_prepare.log")


def evaluate() -> None:
    for group in PAIRS:
        processes = []
        for arm, gpu_group in group:
            external = OUT / "eval" / arm / "remote_eval_start.json"
            if external.is_file():
                completion = OUT / "eval" / arm / "remote_eval_exit.json"
                while not completion.is_file():
                    status("WAIT_EXTERNAL_EVAL", arm=arm, start=json.loads(external.read_text()))
                    time.sleep(180)
                result = json.loads(completion.read_text())
                driver = OUT / "eval" / arm / "driver_status.json"
                if result.get("returncode") != 0 or not driver.is_file() or \
                        json.loads(driver.read_text()).get("state") != "COMPLETE":
                    raise RuntimeError(f"External v4 ID evaluation failed for {arm}: {result}")
                log(f"{arm}: external evaluation completed on {result['host']}")
                continue
            record = OUT / "eval" / arm / "driver_status.json"
            if record.is_file() and json.loads(record.read_text()).get("state") == "COMPLETE":
                continue
            command = [PYTHON, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
                       "--arm", arm, "--gpus", *gpu_group.split(",")]
            path = OUT / "eval" / arm / "driver_console.log"
            stream = path.open("w")
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((arm, process, stream))
            log(f"Evaluating {arm} on GPUs {gpu_group}")
        while any(process.poll() is None for _, process, _ in processes):
            current = {arm: process.poll() for arm, process, _ in processes}
            status("EVALUATING", workers=current)
            time.sleep(180)
        failures = []
        for arm, process, stream in processes:
            code = process.wait()
            stream.close()
            if code != 0:
                failures.append((arm, code))
        if failures:
            raise RuntimeError(f"v4 ID evaluation workers failed: {failures}")


def main() -> None:
    wait_for_training(("1tb", "1pb", "5tb", "100tb"))
    early_start = OUT / "early_eval_3090_start.json"
    early_exit = OUT / "early_eval_3090_exit.json"
    if early_start.is_file():
        while not early_exit.is_file():
            status("WAIT_EARLY_EVAL_3090")
            time.sleep(180)
        if json.loads(early_exit.read_text()).get("returncode") != 0:
            raise RuntimeError("Early 3090 evaluation failed; inspect early_eval_3090.log")
    for arm in ("1tb", "1pb", "5tb", "100tb"):
        audit(arm)
    wait_for_20tb_sample()
    external_20tb = OUT / "20tb_eval_5090_controller_start.json"
    external_20tb_exit = OUT / "20tb_eval_5090_controller_exit.json"
    if external_20tb.is_file():
        while not external_20tb_exit.is_file():
            status("WAIT_20TB_EXTERNAL_EVAL")
            time.sleep(180)
        if json.loads(external_20tb_exit.read_text()).get("returncode") != 0:
            raise RuntimeError("20TB 5090 evaluation failed; inspect 20tb_eval_5090_controller.log")
    train_20tb()
    prepare_evaluations()
    evaluate()
    log("Collecting all validated ID results")
    run_logged([PYTHON, "-u", str(ROOT / "scripts/collect_data_quality_v4_id_20261002.py")],
               OUT / "collect_console.log")
    run_logged([PYTHON, "-u", str(OUT / "plot_data_quality_v4_id.py")],
               OUT / "plot_console.log")
    status("COMPLETE", arms=ARMS, result_csv=str(OUT / "v4_id_cell_scores.csv"),
           figure=str(OUT / "data_quality_v4_id_matched_1m.svg"))
    log("All five arms, v4 ID scores, and figure are complete")


if __name__ == "__main__":
    main()
