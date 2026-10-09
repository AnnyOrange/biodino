#!/usr/bin/env python3
"""Stop the launched training group before deepcad's user memory limit is hit."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time


def log(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] [memory-guard] {message}", flush=True)


def resident_bytes(stat_path):
    stats = dict(line.split() for line in stat_path.read_text().splitlines())
    return int(stats["total_rss"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-gb", type=float, default=350)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--memory-stat", type=Path, default=Path(
        "/sys/fs/cgroup/memory/user.slice/user-1008.slice/memory.stat"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.limit_gb <= 0 or args.poll_seconds <= 0:
        parser.error("a command and positive memory limit/poll interval are required")
    limit = int(args.limit_gb * 1_000_000_000)
    usage = resident_bytes(args.memory_stat)
    if usage >= limit:
        log(f"REFUSED user RSS={usage} bytes >= limit={limit} bytes")
        return 137
    child = subprocess.Popen(command, start_new_session=True)

    def kill_group():
        # Torchrun workers create separate process groups; include descendants.
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
    log(f"START pgid={child.pid} limit={limit} bytes poll={args.poll_seconds}s scope=user total_rss (excludes reclaimable file cache)")
    last_log = 0
    try:
        while child.poll() is None:
            usage = resident_bytes(args.memory_stat)
            if usage >= limit:
                log(f"LIMIT_EXCEEDED user RSS={usage} bytes; SIGKILL training pgid={child.pid}")
                kill_group()
                child.wait()
                return 137
            if time.monotonic() - last_log >= 60:
                log(f"user RSS={usage / 1_000_000_000:.3f} GB")
                last_log = time.monotonic()
            time.sleep(args.poll_seconds)
    except BaseException:
        kill_group()
        child.wait()
        raise
    log(f"EXIT training returncode={child.returncode}")
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
