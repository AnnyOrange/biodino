#!/usr/bin/env python3
"""Consume verified rolling 1PB WDS groups on an 8-GPU DDP host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from rolling_1pb_wds import claim, complete, release, write_json_atomic


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    candidates = []
    for path in (run_dir / "ckpt").glob("*/checkpoint.pth"):
        try:
            candidates.append((int(path.parent.name), path))
        except ValueError:
            continue
    return max(candidates, default=None, key=lambda row: row[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_campaign_manifest(args, run_dir: Path) -> None:
    path = run_dir / "campaign_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("protocol_id") != "rolling-1pb-wds-v1" or
                manifest.get("data", {}).get("queue_root") != str(args.queue_root.resolve())):
            raise ValueError(f"Existing campaign manifest does not match this run: {path}")
        resume = {
            "initial_full_checkpoint": str(args.initial_full_checkpoint.resolve()),
            "initial_iteration": int(args.initial_full_checkpoint.parent.name),
            "state": "model_and_optimizer",
        }
        if manifest.get("resume") not in (None, resume):
            raise ValueError(f"Existing campaign resume source differs: {path}")
        if manifest.get("resume") is None:
            manifest["resume"] = resume
            write_json_atomic(path, manifest)
        return
    write_json_atomic(path, {
        "protocol_id": "rolling-1pb-wds-v1",
        "status": "RUNNING",
        "created_unix": time.time(),
        "host": os.uname().nodename,
        "topology": {"world_size": args.world_size,
                     "cuda_visible_devices": args.cuda_visible_devices},
        "training": {
            "base_config": str(args.base_config.resolve()),
            "base_config_sha256": sha256(args.base_config),
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
            "initial_full_checkpoint": str(args.initial_full_checkpoint.resolve()),
            "initial_checkpoint_iteration": int(args.initial_full_checkpoint.parent.name),
            "initial_checkpoint_bytes": args.initial_full_checkpoint.stat().st_size,
            "batch_updates": args.batch_updates,
            "total_updates": args.total_updates,
            "global_batch_size": 2048,
            "checkpoint_interval_updates": args.batch_updates,
            "max_checkpoints": 2,
        },
        "data": {
            "source": "raw_1pb_inventory",
            "queue_root": str(args.queue_root.resolve()),
            "format": "finite_webdataset_tar",
            "crop_size": 512,
            "target_samples_per_batch": 499712,
            "shard_samples": 1952,
            "max_live_batches": 2,
            "batch_reclamation": "after finite DDP process exits and checkpoint validates",
        },
        "controller": {
            "script": str(Path(__file__).resolve()),
            "repo": str(args.repo.resolve()),
            "python": args.python,
        },
    })


def seed_initial_checkpoint(run_dir: Path, checkpoint: Path) -> None:
    if latest_checkpoint(run_dir) is not None:
        return
    iteration = int(checkpoint.parent.name)
    destination = run_dir / "ckpt" / str(iteration) / "checkpoint.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(checkpoint, destination)
    except OSError:
        shutil.copy2(checkpoint, destination)


def build_assignment(batch_dir: Path, world_size: int) -> Path:
    ready = json.loads((batch_dir / "READY.json").read_text())
    if ready.get("status") != "PASS":
        raise ValueError(f"Batch is not verified: {batch_dir}")
    shards = sorted((batch_dir / "shards").glob("*.tar"))
    if len(shards) != int(ready["shard_count"]):
        raise ValueError("READY shard count differs from on-disk inventory")
    batch_samples = int(ready["samples"])
    samples_per_shard = int(ready["samples_per_shard"])
    if batch_samples % world_size or (batch_samples // world_size) % samples_per_shard:
        raise ValueError("Batch cannot be split into exact per-rank finite streams")
    shards_per_rank = batch_samples // world_size // samples_per_shard
    if len(shards) != shards_per_rank * world_size:
        raise ValueError("Shard count does not balance evenly across DDP ranks")
    rank_shards = [[] for _ in range(world_size)]
    for index, shard in enumerate(shards):
        rank_shards[index % world_size].append(str(shard.resolve()))
    counts = [len(group) * samples_per_shard for group in rank_shards]
    target = batch_samples // world_size
    if counts != [target] * world_size:
        raise ValueError(f"Uneven per-rank sample counts: {counts}, target={target}")
    path = batch_dir / "finite_assignment.json"
    write_json_atomic(path, {
        "status": "PASS", "format": "rolling_1pb_finite_wds_v1",
        "world_size": world_size, "target_per_rank": target,
        "rank_sample_counts": counts, "rank_shards": rank_shards,
        "samples_per_shard": samples_per_shard,
    })
    return path


def segment_config(base_config: Path, run_dir: Path, batch_dir: Path,
                   baseline_checkpoint: Path,
                   total_updates: int, batch_updates: int) -> dict:
    with base_config.open() as source:
        cfg = yaml.safe_load(source)
    train = cfg["train"]
    optim = cfg["optim"]
    train["dataset_path"] = f"packwds_once_robust:{batch_dir}/shards/*.tar::pct=1,99"
    train["output_dir"] = str(run_dir)
    train["OFFICIAL_EPOCH_LENGTH"] = batch_updates
    train["max_updates"] = min(total_updates, batch_updates)
    train["num_workers"] = 1
    train["wds_shuffle_buffer"] = 50
    train["wds_deterministic_resampling"] = True
    train["wds_finite_skip_samples"] = 0
    train["finite_stream_fresh_segment"] = True
    train["finite_stream_id"] = batch_dir.name
    optim["epochs"] = (total_updates + batch_updates - 1) // batch_updates
    cfg["checkpointing"]["period"] = batch_updates
    cfg["checkpointing"]["max_to_keep"] = 2
    cfg["evaluation"]["eval_period_iterations"] = 0
    cfg["student"]["resume_from_teacher_chkpt"] = ""
    return cfg


def run(args) -> int:
    queue_root = args.queue_root.resolve()
    run_dir = args.run_dir.resolve()
    if (not args.base_config.is_file() or not args.baseline_checkpoint.is_file() or
            not args.initial_full_checkpoint.is_file()):
        raise FileNotFoundError("Base config and baseline teacher/full checkpoints must exist")
    try:
        int(args.initial_full_checkpoint.parent.name)
    except ValueError as exc:
        raise ValueError("Initial full checkpoint must be stored under an integer iteration directory") from exc
    run_dir.mkdir(parents=True, exist_ok=True)
    write_campaign_manifest(args, run_dir)
    seed_initial_checkpoint(run_dir, args.initial_full_checkpoint)
    segment_root = run_dir / "segment_configs"
    segment_root.mkdir(exist_ok=True)
    controller_log = run_dir / "rolling_controller.jsonl"
    completed_batches = 0

    while args.max_batches == 0 or completed_batches < args.max_batches:
        batch_dir = claim(queue_root)
        if batch_dir is None:
            print("no READY batch; waiting", flush=True)
            time.sleep(args.poll_seconds)
            continue
        batch_name = batch_dir.name
        training_started = False
        try:
            start = latest_checkpoint(run_dir)
            start_update = 0 if start is None else start[0] + 1
            if start_update >= args.total_updates:
                release(queue_root, batch_name)
                print(f"training schedule reached {args.total_updates} updates", flush=True)
                return 0
            end_update = min(args.total_updates, start_update + args.batch_updates)
            if end_update - start_update != args.batch_updates:
                raise ValueError("total_updates must be a multiple of batch_updates")
            assignment = build_assignment(batch_dir, args.world_size)
            cfg = segment_config(args.base_config, run_dir, batch_dir,
                                 args.baseline_checkpoint,
                                 args.total_updates, args.batch_updates)
            cfg["train"]["max_updates"] = end_update
            config_path = segment_root / f"{batch_name}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "DQ_FINITE_SHARD_ASSIGNMENT": str(assignment),
                "PYTHONUNBUFFERED": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "NCCL_DEBUG": "WARN",
                "NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "OMP_NUM_THREADS": str(args.omp_threads),
            })
            command = [args.python, "-m", "torch.distributed.run",
                       "--nnodes=1", "--node_rank=0",
                       f"--nproc_per_node={args.world_size}",
                       "--master_addr=127.0.0.1", f"--master_port={args.master_port}",
                       "dinov3/train/train.py", "--config-file", str(config_path)]
            log_path = run_dir / "logs" / f"{batch_name}.log"
            log_path.parent.mkdir(exist_ok=True)
            print(f"training {batch_name}: updates {start_update}..{end_update - 1}", flush=True)
            training_started = True
            with log_path.open("w", buffering=1) as log:
                process = subprocess.run(command, cwd=args.repo, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                print(f"training failed for {batch_name}; retained ACTIVE data for inspection; log={log_path}",
                      file=sys.stderr, flush=True)
                return process.returncode
            checkpoint_iteration = end_update - 1
            checkpoint = run_dir / "ckpt" / str(checkpoint_iteration) / "checkpoint.pth"
            done_dir = complete(queue_root, batch_name, checkpoint,
                                checkpoint_iteration, args.min_checkpoint_bytes)
            with controller_log.open("a", buffering=1) as output:
                output.write(json.dumps({
                    "batch_id": batch_name, "start_update": start_update,
                    "end_update": end_update, "checkpoint": str(checkpoint),
                    "completed_data_dir": str(done_dir), "log": str(log_path),
                    "time_unix": time.time(),
                }) + "\n")
            completed_batches += 1
        except Exception:
            if not training_started and (queue_root / "active" / batch_name).exists():
                release(queue_root, batch_name)
            raise
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True,
                        help="Teacher snapshot from the completed 1M baseline run")
    parser.add_argument("--initial-full-checkpoint", type=Path, required=True,
                        help="Full checkpoint from the 1M run, including optimizer state")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-updates", type=int, default=244)
    parser.add_argument("--total-updates", type=int, default=40_016)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 waits and consumes until total_updates is reached")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=31971)
    parser.add_argument("--min-checkpoint-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    if args.total_updates % args.batch_updates:
        parser.error("total-updates must be a multiple of batch-updates")
    if args.world_size != 8:
        parser.error("This campaign is fixed to the validated 8-GPU DDP topology")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
