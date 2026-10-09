#!/usr/bin/env python3
"""On 70% host RAM, wait for a new full H+ checkpoint, then switch GPUs 4/5 to DDP."""

import datetime
import os
from pathlib import Path
import signal
import subprocess
import time


BASE = Path("/data/xuzijing/biodino/outputs/01_training_runs")
OLD_RUN = BASE / "HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_5tb_resume6831_2x5090lyxxr_20260921"
DDP_CODE = Path("/data/xuzijing/biodino_hplus_ddp_20260922")
DDP_LAUNCHER = DDP_CODE / "scripts/resume_hs6_hplus_5tb_ddp_2x5090lyxxr_20260922.sh"
LOG = Path("/data/xuzijing/biodino/outputs/auto_train_logs/hplus_5tb_ddp_memory70_switch_20260922.log")
MIN_CHECKPOINT_BYTES = 16_000_000_000
POLL_SECONDS = 30


def log(message):
    with LOG.open("a") as out:
        line = f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        print(line, file=out, flush=True)


def used_memory_fraction():
    lines = subprocess.check_output(["free", "-b"], text=True).splitlines()
    fields = next(line.split() for line in lines if line.startswith("Mem:"))
    return int(fields[2]) / int(fields[1]), int(fields[2]), int(fields[1])


def latest_checkpoint():
    candidates = []
    for ckpt in (OLD_RUN / "ckpt").glob("[0-9]*/checkpoint.pth"):
        try:
            if ckpt.stat().st_size >= MIN_CHECKPOINT_BYTES:
                candidates.append((int(ckpt.parent.name), ckpt))
        except (OSError, ValueError):
            continue
    return max(candidates, default=(0, None))


def current_trainer_pids():
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            if "-m torch.distributed.run" in cmdline and f"--output-dir {OLD_RUN}" in cmdline:
                matches.append(int(proc.name))
        except OSError:
            continue
    return matches


def gpu_4_5_pids():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    ids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    uuids = {line.split(",", 1)[1].strip() for line in ids.splitlines()
             if line.split(",", 1)[0].strip() in {"4", "5"}}
    return [line.strip() for line in result.splitlines()
            if line.split(",", 1)[0].strip() in uuids]


def main():
    if not DDP_LAUNCHER.is_file():
        raise SystemExit(f"missing DDP launcher: {DDP_LAUNCHER}")
    initial_id, _ = latest_checkpoint()
    log(f"WATCH 70% of host RAM (free -b Mem.used/Mem.total); latest H+ full ck{initial_id}; GPUs 4,5 only")
    last_heartbeat = 0.0
    while True:
        fraction, used, total = used_memory_fraction()
        if len(current_trainer_pids()) != 1:
            log("STOP: expected exactly one original H+ torchrun; no processes killed")
            return 2
        if fraction > 0.70:
            trigger_ckpt, _ = latest_checkpoint()
            log(f"TRIGGER used={used}/{total} ({fraction:.2%}) >70%; wait for a NEW full checkpoint after ck{trigger_ckpt}")
            break
        if time.monotonic() - last_heartbeat >= 300:
            log(f"WAIT used={used}/{total} ({fraction:.2%}); latest ck{latest_checkpoint()[0]}")
            last_heartbeat = time.monotonic()
        time.sleep(POLL_SECONDS)

    while True:
        if len(current_trainer_pids()) != 1:
            log("STOP: old H+ trainer ended before next checkpoint; no processes killed")
            return 2
        next_id, source = latest_checkpoint()
        if next_id > trigger_ckpt and source is not None:
            size = source.stat().st_size
            time.sleep(15)
            if source.is_file() and source.stat().st_size == size:
                break
        time.sleep(POLL_SECONDS)

    run = BASE / f"HS6_Hplus_5tb_no_fsdp_fromck{next_id}_bs64_2x5090lyxxr_20260922"
    staged = run / "ckpt" / str(next_id) / "checkpoint.pth"
    if run.exists():
        log(f"STOP: output directory already exists: {run}; no processes killed")
        return 2
    staged.parent.mkdir(parents=True)
    os.link(source, staged)
    log(f"STAGED full checkpoint ck{next_id} ({size} bytes) by hardlink; stopping original H+ torchrun")

    pids = current_trainer_pids()
    if len(pids) != 1:
        log("STOP: original trainer PID changed before signal; no processes killed")
        return 2
    os.kill(pids[0], signal.SIGTERM)
    for _ in range(60):
        if not current_trainer_pids() and not gpu_4_5_pids():
            break
        time.sleep(2)
    else:
        log("STOP: original H+ did not release GPUs 4,5 within 120s; DDP not started")
        return 2

    if gpu_4_5_pids():
        log("STOP: GPUs 4,5 acquired by another task; DDP not started")
        return 2
    log("START DDP no-FSDP, GPUs 4,5, batch 64 per GPU, accum 8, full activation checkpointing")
    env = dict(os.environ, OUTPUT_DIR=str(run), SOURCE_ITER=str(next_id))
    with LOG.open("a") as out:
        result = subprocess.run(["bash", str(DDP_LAUNCHER)], cwd=DDP_CODE, env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    log(f"DDP_EXIT rc={result.returncode} output={run}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
