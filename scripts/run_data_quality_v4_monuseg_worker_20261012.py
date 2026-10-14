#!/usr/bin/env python3
"""Run the audited MoNuSeg queue with an explicit preparer-hash exception."""

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


REPO = Path("/mnt/huawei_deepcad/dinov3")
OUTPUT = REPO / "plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999/monuseg"
WORKER = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929/scripts/run_retest_fleet_20260918.py")
PREPARER = REPO / "scripts/prepare_hs6_long_v4_id_20261009.py"
EXPECTED_PREPARER_SHA256 = "5b0350f39986af8c0c29834e9dd8a4ce34b69e8d05be04d8590892ea27c7cf97"
ALLOWED_PREPARER_SHA256 = "248ad2d4b03392a48c73961bd309feb2df9adaaeef571ecae0fe170c0a9a169c"
SPLIT_ID = "monuseg2018-train30-extra7val-test14-v1"
COUNTS = {"train": 30, "val": 7, "test": 14}


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
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()

    root = OUTPUT
    campaign = json.loads((root / "campaign_manifest.json").read_text())
    prepared = json.loads((root.parent / "prepared.json").read_text())
    external = campaign.get("external_source_hashes", {})
    mismatches = []
    for name, expected in external.items():
        path = Path(name)
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append({"path": name, "expected_sha256": expected, "actual_sha256": actual})

    if mismatches != [{"path": str(PREPARER),
                       "expected_sha256": EXPECTED_PREPARER_SHA256,
                       "actual_sha256": ALLOWED_PREPARER_SHA256}]:
        raise RuntimeError(f"Unexpected external source mismatch: {mismatches}")
    if sha256(PREPARER) != ALLOWED_PREPARER_SHA256:
        raise RuntimeError("The explicitly audited preparer revision changed")

    monuseg = json.loads((root / "campaign_manifest.json").read_text())
    if prepared.get("monuseg_counts") != COUNTS:
        raise RuntimeError("Prepared campaign does not record the 30/7/14 split")
    if len(monuseg.get("datasets", [])) != 1 or \
            monuseg["datasets"][0].get("counts") != COUNTS or \
            monuseg["datasets"][0].get("split_protocol_id") != SPLIT_ID or \
            len(monuseg.get("tasks", [])) != 1:
        raise RuntimeError("MoNuSeg campaign inventory or split protocol mismatch")

    split_path = Path("/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted/monuseg2018_train30_extra7val_test14_manifest.json")
    split_sha = sha256(split_path)
    if external.get(str(split_path)) != split_sha:
        raise RuntimeError("The 30/7/14 split manifest hash does not match the campaign")

    save(root / "source_integrity_exception.json", {
        "reason": "The preparer gained partial-checkpoint audit and exact-hash split-path fallback after this campaign was prepared; evaluator and data-source hashes remain checked by the locked worker.",
        "preparer_mismatch": mismatches[0],
        "split_protocol_id": SPLIT_ID,
        "split_counts": COUNTS,
        "split_manifest_sha256": split_sha,
        "recorded_unix": time.time(),
    })

    sys.path.insert(0, str(WORKER.parent))
    spec = importlib.util.spec_from_file_location("locked_v4_fleet", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load locked worker {WORKER}")
    fleet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fleet)
    fleet.queue.command = fleet.command
    fleet.queue.validate_cell = fleet.validate
    fleet.queue.worker(SimpleNamespace(
        output=root,
        host=args.host,
        gpus=args.gpus,
        target_per_gpu=1,
        max_host_jobs=1,
        max_global_jobs=160,
        task_family="segmentation",
    ))


if __name__ == "__main__":
    main()
