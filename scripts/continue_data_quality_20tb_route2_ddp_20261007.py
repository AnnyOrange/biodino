#!/usr/bin/env python3
"""Train and evaluate the second 20TB sample with matched DDP settings."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
ARM = "20tb_route2_ddp"
STATUS = OUT / "20tb_route2_ddp_controller_status.json"


def save(state: str, **fields: object) -> None:
    STATUS.write_text(json.dumps(dict(state=state, time_unix=time.time(), **fields), indent=2) + "\n")


def run(command: list[str], name: str) -> None:
    log_path = OUT / f"20tb_route2_ddp_{name}.log"
    with log_path.open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        save("ERROR", stage=name, returncode=code, log=str(log_path))
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")


def wait_for_gpus() -> None:
    while True:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        cards = [[int(field.strip()) for field in line.split(",")] for line in output.splitlines()]
        if all(any(row[0] == gpu and row[1] >= 20_000 and row[2] <= 20 for row in cards)
               for gpu in range(4, 8)):
            return
        save("WAIT_GPU", gpu_status=cards)
        time.sleep(180)


def main() -> None:
    sample = json.loads(Path("/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002/extraction_complete.json").read_text())
    if sample.get("status") != "PASS" or sample.get("samples") != 1_000_000:
        raise ValueError("Route-2 one-million sample is not complete")
    training = OUT / "training" / ARM
    if not training.exists():
        wait_for_gpus()
        save("TRAINING")
        run([PYTHON, "-u", str(ROOT / "scripts/launch_data_quality_1m_20261002.py"),
             "--arm", ARM, "--gpu-group", "4,5,6,7", "--master-port", "29742"], "train")
    if json.loads((training / "exit.json").read_text()).get("returncode") != 0:
        raise RuntimeError("Route-2 DDP training did not finish")
    save("AUDITING")
    run([PYTHON, "-u", str(ROOT / "scripts/audit_data_quality_1m_20261002.py"),
         "--arm", ARM], "audit")
    save("PREPARING_EVAL")
    if not (OUT / "eval" / ARM / "prepared.json").exists():
        run([PYTHON, "-u", str(ROOT / "scripts/prepare_data_quality_v4_id_20261002.py"),
             "--arm", ARM], "prepare")
    save("EVALUATING")
    run([PYTHON, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
         "--arm", ARM, "--gpus", "4", "5", "6", "7"], "eval")
    save("COMPLETE", training_audit=str(training / "audit.json"),
         evaluation_status=str(OUT / "eval" / ARM / "driver_status.json"))


if __name__ == "__main__":
    main()
