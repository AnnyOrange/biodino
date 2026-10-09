#!/usr/bin/env python3
"""Resume hxw v3 evaluation slots when GPUs 0-2 are free of other users' jobs."""

import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

QUEUE = Path("/data/hs6_5tb_v3_parallel_queue_20260923")
WORKER = Path("/data/hs6_l_5tb_nogram_eval_20260921/bin/run_hplus_l_5tb_v3_queue_hxw_20260923.py")
PYTHON = Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python")
GPUS = (0, 1, 2)
TARGET = 6


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def memory_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def gpu_owners():
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    indices = {row.split(", ")[1]: int(row.split(", ")[0]) for row in gpu_rows}
    owners = {gpu: set() for gpu in GPUS}
    app_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    ).splitlines()
    for row in app_rows:
        uuid, pid = row.split(", ")
        gpu = indices.get(uuid)
        if gpu not in owners:
            continue
        try:
            owners[gpu].add(Path(f"/proc/{int(pid)}").stat().st_uid)
        except FileNotFoundError:
            pass
    return owners


def active_slots(gpu):
    result = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(WORKER) not in command or f"--gpu {gpu} " not in command:
            continue
        for slot in range(1, TARGET + 1):
            if f"--slot auto_wait_g{gpu}_s{slot}" in command:
                result.add(slot)
    return result


def run():
    QUEUE.joinpath("supervisors").mkdir(parents=True, exist_ok=True)
    with (QUEUE / "admin" / "resume_after_foreign_gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        next_gpu = 0
        while True:
            try:
                owners = gpu_owners()
                order = GPUS[next_gpu:] + GPUS[:next_gpu]
                for gpu in order:
                    if any(uid != os.getuid() for uid in owners[gpu]):
                        print(f"{utc()} gpu={gpu} waiting_for_foreign_job", flush=True)
                        continue
                    slots = active_slots(gpu)
                    if len(slots) >= TARGET:
                        continue
                    mem = memory_available_gib()
                    data = shutil.disk_usage("/data").free / 2**30
                    system = shutil.disk_usage("/").free / 2**30
                    if mem < 160 or data < 120 or system < 200:
                        print(f"{utc()} admission_wait gpu={gpu} mem={mem:.1f} "
                              f"data={data:.1f} system={system:.1f}", flush=True)
                        continue
                    slot = next(i for i in range(1, TARGET + 1) if i not in slots)
                    name = f"auto_wait_g{gpu}_s{slot}"
                    log = QUEUE / "supervisors" / f"{name}.log"
                    with log.open("a") as output:
                        proc = subprocess.Popen(
                            [str(PYTHON), "-u", str(WORKER), "--gpu", str(gpu), "--slot", name],
                            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    print(f"{utc()} launched gpu={gpu} slot={name} pid={proc.pid}", flush=True)
                    next_gpu = (GPUS.index(gpu) + 1) % len(GPUS)
                    break  # Admit one worker per poll to avoid a memory surge.
            except Exception as exc:
                print(f"{utc()} monitor_error {type(exc).__name__}: {exc}", flush=True)
            time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
