#!/usr/bin/env python3
"""Audit 40k 100TB training, including each pass's no-replacement property."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import yaml

from launch_hs6_long_100tb_deepcad4_40k_20261010 import (
    ACCUM, FULL_POOL_ACCUM, FULL_POOL_MICROBATCH, INDEX, MICROBATCH, ROOT,
    RUN_NAME, UPDATES, WORLD_SIZE,
)


def audit(run: Path) -> dict:
    launch = json.loads((run / "launch_manifest.json").read_text())
    completion = json.loads((run / "exit.json").read_text())
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    index = Path(launch["stream_index"])
    assignment = json.loads((index / "assignment.json").read_text())
    full_pool = assignment["format"] == "100tb_full_pool_stream_v1"
    microbatch = FULL_POOL_MICROBATCH if full_pool else MICROBATCH
    accum = FULL_POOL_ACCUM if full_pool else ACCUM
    if completion["returncode"] != 0 or launch["target_updates"] != UPDATES:
        raise ValueError("Training did not finish all 40,000 updates")
    if (not full_pool and (index != INDEX or launch["candidate_sha256"] != assignment["candidate_sha256"])) or \
            (full_pool and assignment["total_items"] != 26_614_979):
        raise ValueError("100TB source index changed")
    if cfg["compute_precision"]["distributed_mode"] != "ddp" or cfg["gram"]["use_loss"]:
        raise ValueError("Expected no-GRAM DDP")
    if cfg["train"]["max_updates"] != UPDATES or cfg["train"]["batch_size_per_gpu"] != microbatch or \
            cfg["optim"]["gradient_accumulation_steps"] != accum or \
            WORLD_SIZE * microbatch * accum != 1024:
        raise ValueError("Optimizer budget or batch differs from the launch")
    if cfg["optim"]["lr"] != 0.0001 or cfg["optim"]["warmup_epochs"] != 3 or \
            cfg["train"]["OFFICIAL_EPOCH_LENGTH"] != 4098:
        raise ValueError("Schedule differs from the 5TB/20TB long-training reference")
    expected_prefix = "raw100tb_full_pool_robust:" if full_pool else "raw100tb_repeat_robust:"
    if cfg["train"]["dataset_path"] != expected_prefix + str(index):
        raise ValueError("Expected the audited 100TB repeated stream")

    first_loss = last_loss = None
    count = 0
    with (run / "raw_loss_metrics.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["optimizer_update"] != count - 1 or row["samples_seen"] != count * 1024 or \
                    row["effective_global_batch_size"] != 1024 or not math.isfinite(row["total_loss"]):
                raise ValueError(f"Bad metrics at optimizer update {count - 1}")
            first_loss = row["total_loss"] if count == 1 else first_loss
            last_loss = row["total_loss"]
    if count != UPDATES:
        raise ValueError(f"Expected {UPDATES} optimizer updates, found {count}")

    pass_counts = []
    if full_pool:
        if sum(assignment["rank_counts"]) != assignment["total_items"] or \
                sum(map(len, assignment["rank_shards"])) != len(assignment["shards"]):
            raise ValueError("Full-pool rank assignments do not cover the source pool")
    else:
        per_pass = assignment["target_per_rank"]
        per_rank = UPDATES * accum * microbatch
        if per_rank != 10_240_000 or per_pass != 4_096_000:
            raise ValueError("Unexpected per-rank image visit or pass size")
        with contextlib.ExitStack() as stack:
            streams = [stack.enter_context((run / f"consumed_sample_keys_rank{rank:02d}.jsonl").open())
                       for rank in range(WORLD_SIZE)]
            remaining = per_rank
            while remaining:
                per_rank_this_pass = min(remaining, per_pass)
                seen = bytearray(26_852_135)
                count_this_pass = 0
                for rank, stream in enumerate(streams):
                    for _ in range(per_rank_this_pass // microbatch):
                        line = stream.readline()
                        if not line:
                            raise ValueError(f"Rank {rank} audit ended early")
                        keys = json.loads(line)
                        if len(keys) != microbatch:
                            raise ValueError(f"Rank {rank} has an incomplete microbatch")
                        for key in keys:
                            item_id = int(key.rsplit("::", 1)[-1])
                            if not 0 < item_id < len(seen) or seen[item_id]:
                                raise ValueError(f"Repeated or invalid 100TB item within pass: {key}")
                            seen[item_id] = 1
                            count_this_pass += 1
                pass_counts.append(count_this_pass)
                remaining -= per_rank_this_pass
            if any(stream.readline() for stream in streams):
                raise ValueError("Consumed-key log extends beyond 40,000 updates")
        if pass_counts != [16_384_000, 16_384_000, 8_192_000]:
            raise ValueError(f"Unexpected per-pass unique counts: {pass_counts}")

    checkpoints = list(range(3999, 40000, 4000))
    for step in checkpoints:
        teacher = run / f"eval/training_{step}/teacher_checkpoint.pth"
        optimizer = run / f"ckpt/{step}/checkpoint.pth"
        if not teacher.is_file() or teacher.stat().st_size < 600_000_000 or \
                not optimizer.is_file() or optimizer.stat().st_size < 3_000_000_000:
            raise ValueError(f"Checkpoint {step} is missing or incomplete")
    failure_pattern = "raw100tb_full_pool_failures_rank*.jsonl" if full_pool else "raw100tb_failures_rank*.jsonl"
    failure_counts = {path.name: sum(1 for _ in path.open())
                      for path in sorted(run.glob(failure_pattern))}
    result = {
        "status": "PASS", "protocol": "100tb_40k_full_pool_v1" if full_pool else "100tb_40k_repeated_v1",
        "arm": "100tb", "optimizer_updates": UPDATES, "effective_batch": 1024,
        "image_visits": UPDATES * 1024,
        "source_subset_size": assignment["total_items"] if full_pool else assignment["target_total"],
        "unique_visits_per_pass": pass_counts, "world_size": WORLD_SIZE,
        "first_loss": first_loss, "last_loss": last_loss,
        "read_failure_log_lines": failure_counts, "checkpoints": checkpoints,
    }
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/01_training_runs" / RUN_NAME)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))


if __name__ == "__main__":
    main()
