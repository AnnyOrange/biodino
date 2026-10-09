#!/usr/bin/env python3
"""Restart a leaking training job at a checkpoint boundary instead of losing progress.

DataLoader worker RSS grows ~3 GB/h per rank on the packwds_robust pipeline until the
host OOM-kills a worker (20TB route2 run, 2026-09-25).  Two thresholds on host
MemAvailable:
  soft: arm; as soon as the next complete optimizer checkpoint lands, stop the job
        and exit 75 so the caller resumes from it (no lost updates).
  hard: stop immediately (exit 137); the caller resumes from the newest complete one.
"""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

PLANNED_RESTART = 75


def log(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] [memory-guard] {message}", flush=True)


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def complete_checkpoints(ckpt_root, min_bytes, settle_seconds):
    done = set()
    if not ckpt_root.is_dir():
        return done
    now = time.time()
    for entry in ckpt_root.iterdir():
        if not (entry.is_dir() and entry.name.isdigit()):
            continue
        ckpt = entry / "checkpoint.pth"
        try:
            st = ckpt.stat()
        except OSError:
            continue
        if st.st_size >= min_bytes and now - st.st_mtime >= settle_seconds:
            done.add(int(entry.name))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--soft-available-gb", type=float, default=350)
    p.add_argument("--hard-available-gb", type=float, default=100)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--min-ckpt-bytes", type=int, default=6_000_000_000)
    p.add_argument("--settle-seconds", type=float, default=30)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command or a.hard_available_gb >= a.soft_available_gb:
        p.error("a command is required and hard threshold must be below soft threshold")
    ckpt_root = a.run_dir / "ckpt"

    child = subprocess.Popen(command, start_new_session=True)

    def kill_group():
        # torchrun workers create their own process groups; walk descendants too.
        parents = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
        descendants = [child.pid]
        for parent in descendants:
            descendants.extend(pid for pid, ppid in parents.items() if ppid == parent)
        for pid in reversed(descendants):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def interrupted(signum, frame):
        kill_group()
        child.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    log(f"START pgid={child.pid} soft<{a.soft_available_gb}GB hard<{a.hard_available_gb}GB run={a.run_dir}")

    armed_baseline = None
    last_log = 0.0
    try:
        while child.poll() is None:
            avail = mem_available_gb()
            if avail < a.hard_available_gb:
                log(f"HARD_LIMIT MemAvailable={avail:.1f}GB; SIGKILL training now")
                kill_group()
                child.wait()
                return 137
            if armed_baseline is None and avail < a.soft_available_gb:
                armed_baseline = complete_checkpoints(ckpt_root, a.min_ckpt_bytes, 0)
                log(f"SOFT_ARMED MemAvailable={avail:.1f}GB; restart after next checkpoint "
                    f"(have up to ck{max(armed_baseline) if armed_baseline else None})")
            if armed_baseline is not None:
                new = complete_checkpoints(ckpt_root, a.min_ckpt_bytes, a.settle_seconds) - armed_baseline
                if new:
                    log(f"PLANNED_RESTART checkpoint ck{max(new)} complete; MemAvailable={avail:.1f}GB")
                    kill_group()
                    child.wait()
                    return PLANNED_RESTART
            if time.monotonic() - last_log >= 600:
                log(f"MemAvailable={avail:.1f}GB armed={armed_baseline is not None}")
                last_log = time.monotonic()
            time.sleep(a.poll_seconds)
    except BaseException:
        kill_group()
        child.wait()
        raise
    log(f"EXIT training returncode={child.returncode}")
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
