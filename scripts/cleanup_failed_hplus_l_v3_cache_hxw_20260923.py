#!/usr/bin/env python3
"""Delete only failed-cell feature caches and release their retry claims."""
from __future__ import annotations

import datetime as dt
import argparse
import json
from pathlib import Path
import shutil


QUEUE = Path("/data/hs6_5tb_v3_parallel_queue_20260923")
ROOTS = {
    "hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
    "l": Path("/data/hs6_l_5tb_nogram_eval_20260921"),
}


def key_for_cell(campaign: str, point: str, cell: Path) -> str:
    _, dataset, split = cell.name.split("__", 2)
    return f"{campaign}__{point}__{dataset}__{split.replace('-', '_')}"


def cell_for_key(key: str) -> Path:
    campaign, point, *_ = key.split("__", 3)
    matches = [cell for cell in ROOTS[campaign].joinpath("v3/cells").glob(f"point_{point}__*")
               if key_for_cell(campaign, point, cell) == key]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact failed cell for {key}, got {matches}")
    return matches[0]


def active_keys() -> set[str]:
    active = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError):
            continue
        if not any(arg.endswith("run_hplus_l_5tb_v3_dense_hxw_20260922.py") for arg in args):
            continue
        try:
            campaign = args[args.index("--campaign") + 1]
            point = args[args.index("--point") + 1]
            dataset = args[args.index("--dataset") + 1]
            split = args[args.index("--fold") + 1] if "--fold" in args else "formal-static-v1"
        except (ValueError, IndexError):
            continue
        active.add(f"{campaign}__{point}__{dataset}__{split.replace('-', '_')}")
    return active


def remove_caches(key: str, cell: Path) -> list[dict]:
    campaign = key.split("__", 1)[0]
    removed = []
    for cache in (cell / "cache", Path("/home/xzj/hs6_5tb_v3_scratch") / campaign / cell.name):
        if cache.exists():
            size = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            shutil.rmtree(cache)
            removed.append({"path": str(cache), "bytes": size})
    return removed


def audit_row(audit: Path, key: str, cell: Path, removed: list[dict], action: str) -> None:
    row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "key": key,
           "cell": str(cell), "removed": removed, "action": action,
           "preserved": [str(cell / "results"), str(cell / "pipeline.log")]}
    with audit.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def quarantine_stale() -> None:
    failures = QUEUE / "failures"
    audit = QUEUE / "stale_cache_cleanup.jsonl"
    active = active_keys()
    for claim in sorted(QUEUE.joinpath("claims").iterdir()):
        if not claim.is_dir() or claim.name in active:
            continue
        key = claim.name
        cell = cell_for_key(key)
        removed = remove_caches(key, cell)
        report = cell / "validation_report.json"
        valid = report.exists() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE"
        if valid:
            claim.rmdir()
            action = "release_valid_complete"
        else:
            receipt = failures / f"{key}.json"
            if not receipt.exists():
                temporary = receipt.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "key": key,
                    "status": "ADMIN_ABORT_RESOURCE_PRESSURE", "retryable": True,
                    "reason": "Extra slot terminated before host OOM; temporary cache deleted directly",
                }, indent=2) + "\n")
                temporary.replace(receipt)
            action = "quarantine_incomplete"
        audit_row(audit, key, cell, removed, action)


def release_failures() -> None:
    failures = QUEUE / "failures"
    audit = QUEUE / "failed_cache_cleanup.jsonl"
    for receipt in sorted(failures.glob("*.json")):
        key = receipt.stem
        cell = cell_for_key(key)
        report = cell / "validation_report.json"
        if report.exists() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE":
            raise RuntimeError(f"Refusing failed-cache cleanup of validated cell: {cell}")
        removed = remove_caches(key, cell)
        claim = QUEUE / "claims" / key
        if claim.exists():
            claim.rmdir()
        receipt.unlink()
        audit_row(audit, key, cell, removed, "release_for_retry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine-stale", action="store_true",
                        help="Delete caches for inactive claims but keep incomplete claims quarantined")
    args = parser.parse_args()
    if args.quarantine_stale:
        quarantine_stale()
    else:
        release_failures()


if __name__ == "__main__":
    main()
