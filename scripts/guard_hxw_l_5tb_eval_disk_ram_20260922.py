#!/usr/bin/env python3
"""Pause only added L worker launch shells when hxw cache space/RAM is low."""
import datetime as dt
import os
from pathlib import Path
import signal
import time

ROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
SLOTS = (1436271, 1436272, 1436273, 1436274, 1436275, 1436276,
         1439477, 1439478, 1439479, 1439480, 1470971, 1470972)


def gib(value):
    return value / (1024 ** 3)


def resources():
    fs = os.statvfs(ROOT)
    disk = gib(fs.f_bavail * fs.f_frsize)
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            values["available"] = gib(int(line.split()[1]) * 1024)
            break
    return disk, values["available"]


def targets():
    for pid in SLOTS:
        proc = Path("/proc") / str(pid)
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if not command.startswith("bash /data/hs6_l_5tb_nogram_eval_20260921/bin/keep_l_5tb_eval_slot_busy_20260922.sh "):
                continue
            state = next(line.split()[1] for line in (proc / "status").read_text().splitlines() if line.startswith("State:"))
            yield pid, state in ("T", "t")
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue


def main():
    while True:
        disk, ram = resources()
        for pid, stopped in targets():
            try:
                if (disk < 230 or ram < 60) and not stopped:
                    os.kill(pid, signal.SIGSTOP)
                    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} PAUSE slot={pid} disk_gib={disk:.1f} mem_available_gib={ram:.1f}", flush=True)
                elif disk > 300 and ram > 110 and stopped:
                    os.kill(pid, signal.SIGCONT)
                    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} RESUME slot={pid} disk_gib={disk:.1f} mem_available_gib={ram:.1f}", flush=True)
            except ProcessLookupError:
                pass
        time.sleep(15)


if __name__ == "__main__":
    main()
