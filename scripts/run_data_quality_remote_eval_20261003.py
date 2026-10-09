#!/usr/bin/env python3
"""Record ownership of a data-quality evaluation running on another host."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
ARMS = ("1tb", "5tb", "20tb", "100tb", "1pb")


def save(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    args = parser.parse_args()
    directory = OUT / "eval" / args.arm
    if not (directory / "prepared.json").is_file():
        raise FileNotFoundError(directory / "prepared.json")
    start = directory / "remote_eval_start.json"
    exit_record = directory / "remote_eval_exit.json"
    if start.exists() or exit_record.exists():
        raise FileExistsError(f"Remote evaluation already registered for {args.arm}")
    save(start, dict(arm=args.arm, host=platform.node(), gpus=args.gpus,
                     pid=os.getpid(), started_unix=time.time()))
    command = [sys.executable, "-u", str(ROOT / "scripts/run_data_quality_v4_id_20261002.py"),
               "--arm", args.arm, "--gpus", *map(str, args.gpus)]
    code = 1
    try:
        code = subprocess.call(command, cwd=ROOT)
    finally:
        save(exit_record, dict(arm=args.arm, host=platform.node(), gpus=args.gpus,
                               returncode=code, ended_unix=time.time()))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
