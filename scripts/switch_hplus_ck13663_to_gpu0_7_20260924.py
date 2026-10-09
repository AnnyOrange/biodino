#!/usr/bin/env python3
"""Move the running H+ DDP job to GPUs 0-7 after full ck13663 is saved."""

import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path("/data/xuzijing/biodino/outputs/01_training_runs")
OLD = BASE / "HS6_Hplus_5tb_no_fsdp_fromck13175_bs64_4x5090lyxxr_20260924"
NEW = BASE / "HS6_Hplus_5tb_no_fsdp_fromck13663_bs64_8x5090lyxxr_20260924"
SOURCE = OLD / "ckpt/13663/checkpoint.pth"
STAGED = NEW / "ckpt/13663/checkpoint.pth"
OLD_LOG = OLD / "logs/log.txt"
LOG_DIR = Path("/data/xuzijing/biodino/outputs/auto_train_logs")
SWITCH_LOG = LOG_DIR / "hplus_ck13663_to_gpu0_7_20260924.log"
TRAIN_LOG = LOG_DIR / "hplus_ck13663_8x5090lyxxr_20260924.log"
LAUNCHER = LOG_DIR / "resume_hs6_hplus_5tb_ddp_8x5090lyxxr_20260924.sh"
MARKER = f"Saved consolidated checkpoint: {SOURCE}".encode()


def log(message):
    line = f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with SWITCH_LOG.open("a") as f:
        print(line, file=f, flush=True)


def trainer_pids(output_dir):
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "-m torch.distributed.run" in cmd and f"--output-dir {output_dir}" in cmd:
            matches.append(int(proc.name))
    return matches


def gpu_apps(indices):
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    uuids = {line.split(",", 1)[1].strip() for line in gpu_lines
             if int(line.split(",", 1)[0].strip()) in indices}
    app_lines = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    ).splitlines()
    return [line.strip() for line in app_lines if line.split(",", 1)[0].strip() in uuids]


def checkpoint_saved():
    if not SOURCE.is_file() or SOURCE.stat().st_size < 8_000_000_000:
        return False
    with OLD_LOG.open("rb") as f:
        f.seek(max(0, OLD_LOG.stat().st_size - 100_000))
        return MARKER in f.read()


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / ".hplus_ck13663_to_gpu0_7.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("ERROR another switch watcher holds the lock")
            return 2
        if not LAUNCHER.is_file():
            log("ERROR missing eight GPU launcher")
            return 2
        log("WAIT for complete H+ checkpoint ck13663; GPUs 0-7")
        last_heartbeat = 0.0
        while True:
            old_pids = trainer_pids(OLD)
            if len(old_pids) != 1:
                log(f"ERROR expected one original H+ torchrun, found {old_pids}")
                return 2
            if checkpoint_saved():
                size = SOURCE.stat().st_size
                time.sleep(10)
                if checkpoint_saved() and SOURCE.stat().st_size == size:
                    break
            if time.monotonic() - last_heartbeat > 300:
                log(f"WAIT ck13663; old torchrun pid={old_pids[0]}")
                last_heartbeat = time.monotonic()
            time.sleep(15)

        log(f"checkpoint ck13663 confirmed complete ({size} bytes)")
        while gpu_apps({4, 5, 6, 7}):
            log("WAIT GPUs 4-7 occupied; keeping existing H+ trainer running")
            time.sleep(30)
        if NEW.exists():
            log(f"ERROR target output already exists: {NEW}")
            return 2
        old_pids = trainer_pids(OLD)
        if len(old_pids) != 1:
            log(f"ERROR original H+ trainer changed: {old_pids}")
            return 2
        STAGED.parent.mkdir(parents=True)
        os.link(SOURCE, STAGED)
        log(f"staged full checkpoint at {STAGED}; stopping old H+ torchrun pid={old_pids[0]}")
        os.kill(old_pids[0], signal.SIGTERM)
        for _ in range(90):
            if not trainer_pids(OLD) and not gpu_apps({0, 1, 2, 3}):
                break
            time.sleep(2)
        else:
            log("ERROR old trainer did not release GPUs 0-3 within 180s")
            return 2
        if gpu_apps({0, 1, 2, 3, 4, 5, 6, 7}):
            log("ERROR GPUs 0-7 became occupied before launch")
            return 2
        env = dict(os.environ, OUTPUT_DIR=str(NEW), SOURCE_ITER="13663")
        with TRAIN_LOG.open("a") as train_log:
            child = subprocess.Popen(["bash", str(LAUNCHER)],
                                     cwd="/data/xuzijing/biodino_hplus_ddp_20260922",
                                     env=env, stdin=subprocess.DEVNULL,
                                     stdout=train_log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log(f"START eight GPU H+ torchrun launcher pid={child.pid}, output={NEW}")
        for _ in range(180):
            if child.poll() is not None:
                log(f"ERROR eight GPU H+ launcher exited rc={child.returncode}; inspect {TRAIN_LOG}")
                return 3
            try:
                with (NEW / "logs/log.txt").open("rb") as f:
                    content = f.read()[-100_000:]
                if b"Starting training from iteration 13664" in content:
                    log("SUCCESS resumed from ck13663 on GPUs 0-7 at iteration 13664")
                    return 0
            except OSError:
                pass
            time.sleep(5)
        log("WARNING launcher still alive; training start not confirmed within 15 min")
        return 4


if __name__ == "__main__":
    sys.exit(main())
