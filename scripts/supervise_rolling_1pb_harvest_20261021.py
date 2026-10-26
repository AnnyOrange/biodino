#!/usr/bin/env python3
"""Keep the temporary 1PB harvest fleet running until 80k-step data is saved."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
QUEUE = Path("/mnt/huawei_blm/rolling_1pb_wds_v1/queue")
STATUS = QUEUE.parent / "harvest_20261021_status.json"
SESSION = "rolling1pb_harvest_20261021"
HOSTS = ("cpu1", "cpu2", "cpu4", "cpu5", "cpu7", "cpu8", "cpu10",
         "cpu11", "cpu12", "cpu15", "cpu18", "cpu19", "cpu20",
         "deepcad", "3090-qi", "5090-zxr")
AUX = {
    "deepcad": ("/home/deepcad/anaconda3/envs/Dinov3/bin/python", 4, 20261021001, ()),
    "3090-qi": ("/home/bbnc/anaconda3/envs/dinov3/bin/python", 2, 20261021002,
                ("/slfm", "/home/inspur/as13000_sam2p",
                 "/home/inspur/huaweimount_sam2p",
                 "/mnt/mount_100t_zzf_2", "/mnt/mount_100t_zzf_3")),
    "5090-zxr": ("/home/lxy/miniconda3/envs/dinov3/bin/python", 2, 20261021003, ()),
}
TARGET_BATCHES = 163
MIN_FREE_GIB = 2048


def verified_count() -> dict[str, int]:
    counts = {}
    for state in ("completed", "active", "ready"):
        counts[state] = sum(
            json.loads(path.read_text()).get("status") == "PASS"
            for path in (QUEUE / state).glob("batch-*/READY.json")
        )
    counts["total"] = sum(counts.values())
    return counts


def check_host(host: str, relaunch: bool) -> dict[str, str]:
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host]
    session = "rolling1pb_harvest_aux_20261021" if host in AUX else SESSION
    try:
        check = subprocess.run(base + [f"tmux has-session -t {session} 2>/dev/null"],
                               capture_output=True, text=True, timeout=12)
        if check.returncode == 0:
            return {"host": host, "state": "running"}
        if check.returncode != 1:
            return {"host": host, "state": "unreachable", "error": check.stderr.strip()}
        if not relaunch:
            return {"host": host, "state": "stopped"}
        if host in AUX:
            python, workers, seed, excluded = AUX[host]
            arguments = [python, "-u", "scripts/start_rolling_1pb_producer.py",
                         "--metadata-csv", str(QUEUE.parent / "inventory/partitions_global_20261019/cpu9.csv"),
                         "--partition-summary", str(QUEUE.parent / "inventory/partitions_global_20261019/READY.json"),
                         "--queue-root", str(QUEUE), "--workers", str(workers),
                         "--verify-workers", str(workers), "--max-live-batches", "130",
                         "--min-free-gib", str(MIN_FREE_GIB), "--seed", str(seed),
                         "--poll-seconds", "30", "--daemon"]
            for root in excluded:
                arguments += ["--exclude-root", root]
            command = (f"cd {shlex.quote(str(REPO))} && {shlex.join(arguments)} "
                       "> /tmp/rolling1pb_harvest_aux_20261021.log 2>&1")
        else:
            partial = " --allow-partial-roots" if host == "cpu8" else ""
            command = (
                f"cd {REPO} && /home/inspur/anaconda3/envs/dinov3/bin/python -u "
                f"scripts/launch_rolling_1pb_global.py --host {host} "
                f"--max-live-batches 130 --min-free-gib {MIN_FREE_GIB}{partial} "
                f"> /tmp/rolling1pb_harvest_20261021.log 2>&1"
            )
        launch = subprocess.run(base + [f"tmux new-session -d -s {session} "
                                       f"{shlex.quote(command)}"],
                                capture_output=True, text=True, timeout=12)
        return {"host": host, "state": "restarted" if launch.returncode == 0 else "failed",
                "error": launch.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"host": host, "state": "timeout"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    while True:
        counts = verified_count()
        stats = os.statvfs(QUEUE)
        free_gib = stats.f_bavail * stats.f_frsize // 1024**3
        relaunch = counts["total"] < TARGET_BATCHES and free_gib > MIN_FREE_GIB + 256
        with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
            hosts = list(pool.map(lambda host: check_host(host, relaunch), HOSTS))
        report = {"updated_unix": time.time(), "target_batches": TARGET_BATCHES,
                  "batches": counts, "free_gib": free_gib, "hosts": hosts}
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, STATUS)
        print(json.dumps({"batches": counts, "free_gib": free_gib,
                          "hosts": {item["host"]: item["state"] for item in hosts}}),
              flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
