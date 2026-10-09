#!/usr/bin/env python3
"""Continue matched 1TB and 1PB training when the 3090 GPU groups are free."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
LAUNCHER = ROOT / "scripts/launch_data_quality_1m_20261002.py"
WAIT_FOR = ("100tb", "5tb")
NEXT = (("1tb", "0,1,2,3", 29613), ("1pb", "4,5,6,7", 29614))


def save_status(value: dict) -> None:
    path = OUT / "scheduler_3090_status.json"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def exit_code(arm: str) -> int | None:
    path = OUT / "training" / arm / "exit.json"
    return json.loads(path.read_text())["returncode"] if path.is_file() else None


def wait_for_current() -> None:
    while True:
        states = {arm: exit_code(arm) for arm in WAIT_FOR}
        save_status(dict(state="WAIT_CURRENT_TRAINING", arms=states, time_unix=time.time()))
        if any(code is not None and code != 0 for code in states.values()):
            raise RuntimeError(f"Current training failed: {states}")
        if all(code == 0 for code in states.values()):
            return
        time.sleep(180)


def verify_smoke(arm: str) -> None:
    run = OUT / "smoke" / arm
    record = json.loads((run / "exit.json").read_text())
    if record["returncode"] != 0:
        raise RuntimeError(f"Smoke test failed: {arm}")
    keys = set()
    for rank in range(4):
        path = run / f"consumed_sample_keys_rank{rank:02d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != 32 or any(len(row) != 8 for row in rows):
            raise ValueError(f"Smoke {arm}: rank {rank} did not consume 256 images")
        for row in rows:
            for key in row:
                if key in keys:
                    raise ValueError(f"Smoke {arm}: repeated image {key}")
                keys.add(key)
    if len(keys) != 1024:
        raise ValueError(f"Smoke {arm}: expected 1024 unique images, got {len(keys)}")


def command(arm: str, gpus: str, port: int, smoke: bool = False) -> list[str]:
    return [PYTHON, "-u", str(LAUNCHER), "--arm", arm,
            "--gpu-group", gpus, "--master-port", str(port)] + (["--smoke"] if smoke else [])


def main() -> None:
    wait_for_current()
    for arm in WAIT_FOR:
        log(f"{arm}: auditing all consumed IDs and final checkpoint")
        audit = [PYTHON, "-u", str(ROOT / "scripts/audit_data_quality_1m_20261002.py"),
                 "--arm", arm]
        with (OUT / "training" / arm / "audit_console.log").open("w") as stream:
            result = subprocess.run(audit, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Training audit failed for {arm}: {result.returncode}")
    log("100TB and 5TB passed audit; checking next groups")
    active = []
    for arm, gpus, port in NEXT:
        if (OUT / "training" / arm).exists():
            log(f"{arm}: training directory already exists; leaving it alone")
            continue
        smoke = OUT / "smoke" / arm
        if not smoke.exists():
            log(f"{arm}: starting smoke test")
            result = subprocess.run(command(arm, gpus, port, smoke=True), cwd=ROOT)
            if result.returncode != 0:
                raise RuntimeError(f"Smoke launcher failed for {arm}: {result.returncode}")
        verify_smoke(arm)
        log(f"{arm}: 1024 distinct smoke images verified")
        active.append((arm, gpus, port))
    processes = {}
    for arm, gpus, port in active:
        log(f"{arm}: starting 976-update training on GPUs {gpus}")
        processes[arm] = subprocess.Popen(command(arm, gpus, port), cwd=ROOT)
    while processes:
        states = {}
        for arm, process in list(processes.items()):
            code = process.poll()
            states[arm] = code
            if code is not None:
                log(f"{arm}: launcher exited with code {code}")
                del processes[arm]
                if code != 0:
                    raise RuntimeError(f"Training launcher failed for {arm}: {code}")
        save_status(dict(state="TRAIN_NEXT_GROUPS", processes=states, time_unix=time.time()))
        if processes:
            time.sleep(180)
    save_status(dict(state="COMPLETE", time_unix=time.time()))
    log("3090 continuation completed")


if __name__ == "__main__":
    main()
