#!/usr/bin/env python3
"""Preserve full rolling 1PB checkpoints while an older controller is running."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def retain(ckpt_dir: Path, min_bytes: int) -> None:
    for source in sorted(ckpt_dir.glob("*/checkpoint.pth")):
        if not source.parent.name.isdecimal():
            continue
        if source.stat().st_size < min_bytes:
            raise RuntimeError(f"Checkpoint is unexpectedly small: {source}")
        target = ckpt_dir / "retained" / source.parent.name / "checkpoint.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
            print(f"RETAINED step={source.parent.name} path={target}", flush=True)
        except FileExistsError:
            if not source.samefile(target):
                raise RuntimeError(f"Retained checkpoint differs from source: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--min-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error("interval-seconds must be positive")
    while True:
        try:
            retain(args.ckpt_dir, args.min_bytes)
        except OSError as error:
            if args.once:
                raise
            print(f"RETAIN_RETRY {type(error).__name__}: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
