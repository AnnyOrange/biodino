#!/usr/bin/env python3
"""Run the locked shared v4 worker with an eight-slot per-GPU ceiling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/run_retest_fleet_20260918.py"
PREPARER = Path("/mnt/huawei_deepcad/dinov3/scripts/prepare_hs6_long_v4_id_20261009.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--target-per-gpu", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--max-host-jobs", type=int, default=64)
    parser.add_argument("--max-global-jobs", type=int, default=160)
    args = parser.parse_args()
    manifest = json.loads((args.output / "campaign_manifest.json").read_text())
    mismatches = []
    for name, expected in manifest.get("external_source_hashes", {}).items():
        path = Path(name)
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append({"path": name, "expected_sha256": expected, "actual_sha256": actual})
    preparer_path = str(PREPARER)
    if mismatches:
        raise RuntimeError(f"Unexpected external source mismatch: {mismatches}")

    sys.path.insert(0, str(WORKER.parent))
    spec = importlib.util.spec_from_file_location("locked_v4_fleet", WORKER)
    fleet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fleet)
    fleet.queue.command = fleet.command
    fleet.queue.validate_cell = fleet.validate
    fleet.queue.worker(SimpleNamespace(
        output=args.output, host=args.host, gpus=args.gpus,
        target_per_gpu=args.target_per_gpu, max_host_jobs=args.max_host_jobs,
        max_global_jobs=args.max_global_jobs, task_family="mixed",
    ))


if __name__ == "__main__":
    main()
