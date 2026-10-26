#!/usr/bin/env python3
"""Retire old fleet producers after their in-flight WDS batches publish."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


QUEUE = Path("/mnt/huawei_blm/rolling_1pb_wds_v1/queue")
IN_FLIGHT = {
    "cpu1": "batch-000021",
    "cpu2": "batch-000028",
    "cpu4": "batch-000014",
    "cpu7": "batch-000018",
    "cpu8": "batch-000030",
    "cpu9": "batch-000015",
    "cpu10": "batch-000022",
    "cpu12": "batch-000017",
    "cpu18": "batch-000025",
    "cpu19": "batch-000024",
    "cpu20": "batch-000023",
}


def published(batch: str) -> bool:
    if (QUEUE / ".building" / batch).exists():
        return False
    for state in ("ready", "active", "completed"):
        manifest = QUEUE / state / batch / "READY.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("status") == "PASS":
            return True
    return False


def retire(host: str) -> bool:
    token = ("partitions_expansion_cpu2_20261019/sources.csv" if host == "cpu2"
             else f"partitions_fleet_20261019/{host}.csv")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         "ps -eo pid=,args="], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"{host}: cannot inspect old producer: {result.stderr.strip()}")
    pids = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid, args = fields
        if (args.startswith("/home/inspur/") and
                "start_rolling_1pb_producer.py" in args and
                "--daemon" in args and token in args):
            pids.append(pid)
    if not pids:
        print(f"{host}: old producer already exited", flush=True)
        return True
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         "kill", *pids], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"{host}: cannot stop old producer: {result.stderr.strip()}")
    print(f"{host}: published {IN_FLIGHT[host]}; stopped old producer {', '.join(pids)}",
          flush=True)
    return True


def main() -> None:
    pending = dict(IN_FLIGHT)
    print(f"waiting for {len(pending)} old batches to publish", flush=True)
    while pending:
        for host, batch in list(pending.items()):
            if not published(batch):
                continue
            try:
                if retire(host):
                    del pending[host]
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"{host}: retrying retirement: {exc}", flush=True)
        if pending:
            time.sleep(10)
    print("all old fleet producers retired", flush=True)


if __name__ == "__main__":
    main()
