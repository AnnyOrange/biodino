#!/usr/bin/env python3
"""Audit and evaluate the 20TB arm after its 5090 training completes."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
ARM = "20tb"
GPUS = (0, 1, 2, 3)


def save(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def run_logged(command: list[str], path: Path) -> None:
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with {result.returncode}: {path}")


def main() -> None:
    start = OUT / "20tb_eval_5090_controller_start.json"
    finish = OUT / "20tb_eval_5090_controller_exit.json"
    if start.exists() or finish.exists():
        raise FileExistsError("20TB 5090 evaluation controller already registered")
    save(start, dict(host="5090-zxr", gpus=GPUS, started_unix=time.time()))
    code = 1
    try:
        run = OUT / "training" / ARM
        while not (run / "exit.json").is_file():
            save(OUT / "20tb_eval_5090_controller_status.json",
                 dict(state="WAIT_TRAINING", time_unix=time.time(),
                      host=platform.node(), gpus=GPUS))
            time.sleep(180)
        if json.loads((run / "exit.json").read_text())["returncode"] != 0:
            raise RuntimeError("20TB training failed")
        audit = run / "audit.json"
        if not audit.is_file():
            run_logged([sys.executable, "-u", str(ROOT / "scripts/audit_data_quality_1m_20261002.py"),
                        "--arm", ARM], run / "audit_console.log")
        if json.loads(audit.read_text())["status"] != "PASS":
            raise RuntimeError("20TB training audit failed")
        evaluation = OUT / "eval" / ARM
        if not (evaluation / "prepared.json").is_file():
            evaluation.parent.mkdir(parents=True, exist_ok=True)
            run_logged([sys.executable, "-u", str(ROOT / "scripts/prepare_data_quality_v4_id_20261002.py"),
                        "--arm", ARM], OUT / "eval" / "20tb_prepare.log")
        command = [sys.executable, "-u", str(ROOT / "scripts/run_data_quality_remote_eval_20261003.py"),
                   "--arm", ARM, "--gpus", *map(str, GPUS)]
        save(OUT / "20tb_eval_5090_controller_status.json",
             dict(state="EVALUATING", time_unix=time.time(), host=platform.node(), gpus=GPUS))
        run_logged(command, evaluation / "remote_driver_console.log")
        code = 0
    finally:
        save(finish, dict(host="5090-zxr", returncode=code, ended_unix=time.time()))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
