#!/usr/bin/env python3
"""Stop only speculative LIVECell probes if host memory becomes unsafe."""

import datetime as dt
import os
import signal
import time
from pathlib import Path


PROBE_GROUPS = (631035, 631036)
MIN_AVAILABLE_GIB = 45


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing")


while True:
    alive = [pid for pid in PROBE_GROUPS if Path(f"/proc/{pid}").exists()]
    if not alive:
        break
    available = available_gib()
    if available < MIN_AVAILABLE_GIB:
        for pid in alive:
            try:
                os.killpg(pid, signal.SIGTERM)
                print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} "
                      f"stopped_speculative_probe_group={pid} mem_available_gib={available:.1f}",
                      flush=True)
            except ProcessLookupError:
                pass
        break
    time.sleep(10)
