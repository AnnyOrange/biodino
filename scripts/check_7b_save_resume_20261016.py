#!/usr/bin/env python3
"""Bounded full-size 7B save/reload check; produces no performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--original-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if occupied.strip():
        raise RuntimeError("7B smoke requires all eight GPUs to be free")
    args.output.mkdir(parents=True)
    record = dict(purpose="Actual 7B FSDP + CPU AdamW moments save/reload check, not an efficacy run",
                  init="Original official pretrained 7B; fresh optimizer; no claim of resuming failed step 249",
                  config=str(args.original_config), repo=str(args.repo), started=time.time(),
                  config_sha256=hashlib.sha256(args.original_config.read_bytes()).hexdigest(),
                  train_sha256=hashlib.sha256((args.repo / "dinov3/train/train.py").read_bytes()).hexdigest(),
                  global_batch=288, world_size=8, batch_per_gpu=6, accumulation=6, phases=[])
    manifest = args.output / "save_resume_check.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONPATH=str(args.repo),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MALLOC_ARENA_MAX="2",
               PYTORCH_ALLOC_CONF="expandable_segments:True", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
               NCCL_DEBUG="WARN", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    for updates in (1, 2):
        cmd = [args.python, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=8",
               "--master_addr=127.0.0.1", "--master_port=29719", "dinov3/train/train.py",
               "--config-file", str(args.original_config), "--output-dir", str(args.output),
               "--expected-start-iteration", str(updates - 1), "--seed", "0",
               f"train.max_updates={updates}", "train.batch_size_per_gpu=6", "train.num_workers=1",
               "train.wds_shuffle_buffer=128", "train.prefetch_factor=1", "optim.gradient_accumulation_steps=6",
               "evaluation.eval_period_iterations=1", "checkpointing.period=1", "checkpointing.max_to_keep=2"]
        phase = dict(updates=updates, command=cmd, started=time.time())
        record["phases"].append(phase)
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        with (args.output / f"phase{updates}.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            min_ram = float("inf")
            while proc.poll() is None:
                ram = next(int(s.split()[1]) / 1024**2 for s in Path("/proc/meminfo").read_text().splitlines()
                           if s.startswith("MemAvailable:"))
                min_ram = min(min_ram, ram)
                if ram < 24 or time.time() - phase["started"] > 2700:
                    os.killpg(proc.pid, signal.SIGTERM)
                    phase["abort_reason"] = "RAM below 24 GiB or phase exceeded 45 minutes"
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(10)
            phase.update(returncode=proc.wait(), finished=time.time(), minimum_available_ram_gib=min_ram)
        metadata = args.output / "ckpt" / str(updates - 1) / ".metadata"
        phase["completed_checkpoint"] = metadata.is_file()
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        if phase["returncode"] or not metadata.is_file():
            raise RuntimeError(f"7B save/resume phase {updates} failed; see {args.output}")
    record.update(status="PASS", finished=time.time())
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
