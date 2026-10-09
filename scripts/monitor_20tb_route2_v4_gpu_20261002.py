#!/usr/bin/env python3
"""Record GPU memory and queue progress for the 20TB route2 v4 continuation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

OUTPUT = Path("/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/20tb_route2_remaining_v4_monu30_20261002")
HOSTS = ("cpu1", "cpu2", "cpu5", "cpu8", "cpu9", "cpu10", "cpu11", "cpu12", "cpu15", "cpu19")


def sample_host(host: str) -> dict:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=12)
    if result.returncode:
        return {"host": host, "error": result.stderr.strip()[-200:] or "nvidia-smi unavailable"}
    try:
        used, total = (int(value.strip()) for value in result.stdout.strip().splitlines()[0].split(","))
    except (IndexError, ValueError):
        return {"host": host, "error": "invalid nvidia-smi output"}
    return {"host": host, "used_mib": used, "total_mib": total,
            "fraction": round(used / total, 4), "above_half": used > total / 2}


def sample() -> dict:
    state = OUTPUT / "_state"
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        cards = list(pool.map(sample_host, HOSTS))
    return {
        "time_unix": time.time(), "cards": cards,
        "done": len(list((state / "done").glob("*.json"))),
        "running": len(list((state / "running").glob("*.json"))),
        "failed": len(list((state / "failed_resource").glob("*.json"))),
        "checkpoint_inputs_registered": len(list((state / "inputs").glob("*.json"))),
        "paused": (state / "PAUSED.json").exists(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    log = OUTPUT / "gpu_memory_samples.jsonl"
    while True:
        record = sample()
        with log.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(args.interval)
