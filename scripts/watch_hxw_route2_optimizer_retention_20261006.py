#!/usr/bin/env python3
"""Audit every new r0r9 optimizer checkpoint on the hxw output mount."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

ROOT = Path("/home/xzj/route2_20tb_20261006")
RUN = ROOT / "output"
DATA_MERGED = Path("/home/xzj/storage/merged")
LOG = ROOT / "optimizer_audit.jsonl"
BASELINE = 7319


def record(event: str, **details):
    entry = {"time_unix": time.time(), "event": event, **details}
    with LOG.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def scan(verified: set[int]):
    if not os.path.ismount(RUN) or not os.path.ismount(DATA_MERGED):
        record("ERROR_MOUNT_MISSING", output=os.path.ismount(RUN),
               merged_data=os.path.ismount(DATA_MERGED))
        return
    config_path = RUN / "config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text())
        if config["checkpointing"]["max_to_keep"] is not None:
            record("ERROR_RETENTION_LIMIT", value=config["checkpointing"]["max_to_keep"])
    checkpoints = {int(path.name): path for path in (RUN / "ckpt").iterdir()
                   if path.is_dir() and path.name.isdigit()}
    for step in sorted(verified - checkpoints.keys()):
        record("ERROR_PREVIOUS_CHECKPOINT_MISSING", step=step)
    if BASELINE not in checkpoints:
        record("ERROR_BOOTSTRAP_CHECKPOINT_MISSING", step=BASELINE)
    for step, directory in sorted(checkpoints.items()):
        if step <= BASELINE or step in verified:
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
                   iteration=payload.get("iteration"), retained_numeric_checkpoints=len(checkpoints))
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
    record("START", baseline=BASELINE, output=str(RUN))
    while True:
        scan(verified)
        if args.once:
            break
        time.sleep(120)
