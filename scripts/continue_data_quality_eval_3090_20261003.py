#!/usr/bin/env python3
"""Evaluate the matched 1TB and 1PB arms as their 3090 training finishes."""

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
ARMS = {"1tb": (0, 1, 2, 3), "1pb": (4, 5, 6, 7)}


def save(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def run_logged(command: list[str], path: Path) -> None:
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with {result.returncode}: {path}")


def train_exit(arm: str) -> int | None:
    path = OUT / "training" / arm / "exit.json"
    return json.loads(path.read_text())["returncode"] if path.is_file() else None


def main() -> None:
    marker = OUT / "early_eval_3090_start.json"
    completion = OUT / "early_eval_3090_exit.json"
    if marker.exists() or completion.exists():
        raise FileExistsError("Early 3090 evaluation already registered")
    save(marker, dict(host=platform.node(), arms=list(ARMS), started_unix=time.time()))
    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    completed: set[str] = set()
    code = 1
    try:
        while len(completed) < len(ARMS):
            for arm, gpus in ARMS.items():
                if arm in completed or arm in processes:
                    continue
                exit_code = train_exit(arm)
                if exit_code is None:
                    continue
                if exit_code != 0:
                    raise RuntimeError(f"{arm} training failed: {exit_code}")
                audit = OUT / "training" / arm / "audit.json"
                if not audit.is_file():
                    run_logged([sys.executable, "-u", str(ROOT / "scripts/audit_data_quality_1m_20261002.py"),
                                "--arm", arm], OUT / "training" / arm / "audit_console.log")
                if json.loads(audit.read_text())["status"] != "PASS":
                    raise RuntimeError(f"{arm} training audit failed")
                directory = OUT / "eval" / arm
                if not (directory / "prepared.json").is_file():
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    run_logged([sys.executable, "-u", str(ROOT / "scripts/prepare_data_quality_v4_id_20261002.py"),
                                "--arm", arm], OUT / "eval" / f"{arm}_prepare.log")
                command = [sys.executable, "-u", str(ROOT / "scripts/run_data_quality_remote_eval_20261003.py"),
                           "--arm", arm, "--gpus", *map(str, gpus)]
                log = (directory / "remote_driver_console.log").open("w")
                processes[arm] = (subprocess.Popen(command, cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT), log)
                print(f"START {arm} v4 ID evaluation on GPUs {gpus}", flush=True)
            for arm, (process, log) in list(processes.items()):
                result = process.poll()
                if result is None:
                    continue
                log.close()
                del processes[arm]
                if result != 0:
                    raise RuntimeError(f"{arm} v4 ID evaluation failed: {result}")
                completed.add(arm)
                print(f"COMPLETE {arm} v4 ID evaluation", flush=True)
            save(OUT / "early_eval_3090_status.json", dict(
                host=platform.node(), time_unix=time.time(),
                training={arm: train_exit(arm) for arm in ARMS},
                evaluating=list(processes), completed=sorted(completed)))
            if len(completed) < len(ARMS):
                time.sleep(180)
        code = 0
    finally:
        save(completion, dict(host=platform.node(), returncode=code,
                              completed=sorted(completed), ended_unix=time.time()))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
