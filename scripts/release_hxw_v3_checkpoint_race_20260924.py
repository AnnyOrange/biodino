#!/usr/bin/env python3
"""Release only queue failures caused by a checkpoint-transfer race."""
import json
from pathlib import Path
import shutil
import subprocess
import time

QUEUE = Path("/data/hs6_5tb_v3_parallel_queue_20260923")
LROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
ARCHIVE = QUEUE / "failures_recovered_checkpoint_race_20260924"


def sweep():
    ARCHIVE.mkdir(exist_ok=True)
    running = subprocess.check_output(["ps", "-eo", "args="], text=True)
    for receipt in (QUEUE / "failures").glob("l__*.json"):
        try:
            row = json.loads(receipt.read_text())
            point = int(receipt.name.split("__")[1])
            log = Path(row.get("log", ""))
            checkpoint = LROOT / "adapters" / str(point) / "checkpoint.pth"
            if (log.is_file() and "Full transferred checkpoint missing" in log.read_text()
                    and checkpoint.is_file() and checkpoint.stat().st_size == 1401909871):
                dataset = receipt.name.split("__")[2]
                active = any("run_hplus_l_5tb_v3_dense_hxw_20260922.py" in line
                             and f"--point {point} " in line
                             and f"--dataset {dataset} " in line
                             for line in running.splitlines())
                if active:
                    continue
                shutil.move(str(receipt), str(ARCHIVE / receipt.name))
                claim = QUEUE / "claims" / receipt.stem
                if claim.is_dir() and not any(claim.iterdir()):
                    claim.rmdir()
                print("released", receipt.name, flush=True)
        except (OSError, ValueError, KeyError):
            continue


if __name__ == "__main__":
    while True:
        sweep()
        time.sleep(30)
