#!/usr/bin/env python3
"""Audit each new local route2 optimizer checkpoint without changing training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

ROOT = Path("/mnt/huawei_deepcad/dinov3")
RUN = ROOT / ("outputs/01_training_runs/"
              "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_"
              "20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924")
LOG = ROOT / "outputs/auto_train_logs/route2_optimizer_retention_audit_20261004.jsonl"
BASELINE = (32695, 33183, 33671, 34159, 34647, 35135, 35623, 36111)


def record(event: str, **details):
    entry = {"time_unix": time.time(), "event": event, **details}
    with LOG.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def scan(verified: set[int]):
    config = yaml.safe_load((RUN / "config.yaml").read_text())
    if config["checkpointing"]["max_to_keep"] is not None:
        record("ERROR_RETENTION_LIMIT", value=config["checkpointing"]["max_to_keep"])
    checkpoints = {int(path.name): path for path in (RUN / "ckpt").iterdir()
                   if path.is_dir() and path.name.isdigit()}
    missing = sorted(set(BASELINE) - checkpoints.keys())
    if missing:
        record("ERROR_HISTORICAL_CHECKPOINT_MISSING", steps=missing)
    for step, directory in sorted(checkpoints.items()):
        if step <= max(BASELINE) or step in verified:
            continue
        checkpoint = directory / "checkpoint.pth"
        if not checkpoint.is_file():
            continue
        stat = checkpoint.stat()
        if stat.st_size < 6_000_000_000 or time.time() - stat.st_mtime < 180:
            continue
        try:
            payload = torch.load(checkpoint, map_location="cpu", mmap=True,
                                 weights_only=False)
            if not isinstance(payload.get("optimizer"), dict) or not payload["optimizer"]:
                record("ERROR_OPTIMIZER_MISSING", step=step, bytes=stat.st_size)
                continue
            record("VERIFIED_OPTIMIZER", step=step, bytes=stat.st_size,
                   iteration=payload.get("iteration"),
                   retained_numeric_checkpoints=len(checkpoints))
            verified.add(step)
        except Exception as error:
            record("ERROR_CHECKPOINT_READ", step=step, error=str(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    verified = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("event") == "VERIFIED_OPTIMIZER":
                verified.add(int(entry["step"]))
    while True:
        scan(verified)
        if args.once:
            break
        time.sleep(120)
