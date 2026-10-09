#!/usr/bin/env python3
"""Immediately discard reproducible features of VALID completed H+ cells only."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


ROOT = Path("/data/hs6_hplus_5tb_eval_20260921")
OLD = ROOT / "old"
AUDIT = ROOT / "logs" / "cache_cleanup.jsonl"


def active_commands() -> list[str]:
    commands = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            commands.append((proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (PermissionError, ProcessLookupError, FileNotFoundError):
            continue
    return commands


def finite_result(path: Path, expected_id: str) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        raw = path.read_bytes()
        row = json.loads(raw)
        if not isinstance(row, dict) or row.get("error"):
            return None
        checkpoint = row.get("checkpoint")
        if checkpoint and Path(checkpoint).parent.name != expected_id:
            return None
        metrics = ("balanced_accuracy", "macro_auc", "r2", "recall_at_1", "nmi")
        rows = row.get("rows") if isinstance(row.get("rows"), list) else [row]
        if not rows or not any(
            math.isfinite(float(item[key]))
            for item in rows if isinstance(item, dict)
            for key in metrics if item.get(key) is not None
        ):
            return None
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def finite_segmentation_result(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        raw = path.read_bytes()
        row = json.loads(raw)
        if not isinstance(row, dict) or row.get("error"):
            return None
        test = row.get("test")
        if not isinstance(test, dict) or not any(
            math.isfinite(float(test[key])) for key in ("mIoU", "mDice", "Dice", "AP50")
            if test.get(key) is not None
        ):
            return None
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def verified_cache_targets(commands: list[str]):
    if ROOT.is_symlink() or OLD.is_symlink() or not OLD.is_dir():
        raise RuntimeError("Campaign root is missing or a symlink")
    for point in sorted(OLD.glob("point_*")):
        if point.is_symlink() or not point.is_dir():
            continue
        expected_id = point.name.removeprefix("point_")
        if not expected_id.isdigit():
            continue
        for family in ("bio_classification", "bio_regression", "bio_retrieval"):
            for job in point.glob(f"*/{family}/*/{expected_id}"):
                if job.is_symlink() or not job.is_dir():
                    continue
                features = job / "features"
                if not features.is_dir() or features.is_symlink():
                    continue
                digest = finite_result(job / "last_result.json", expected_id)
                if digest is None or any(str(job) in command for command in commands):
                    continue
                yield features, digest
        # Segmentation folds are independent; release a completed fold's
        # extracted features without disturbing the next fold still training.
        for lane in point.glob("segmentation_*"):
            if lane.is_symlink() or not lane.is_dir():
                continue
            for features in lane.glob(f"cache/bio_segmentation/*/*/{expected_id}"):
                if features.is_symlink() or not features.is_dir():
                    continue
                dataset = features.parent.name
                run_name = features.parent.parent.name
                result = lane / "bio_segmentation" / run_name / dataset / expected_id / "results.json"
                digest = finite_segmentation_result(result)
                if digest is None or any(str(features) in command for command in commands):
                    continue
                yield features, digest
    v3 = ROOT / "v3" / "cells"
    if v3.is_dir() and not v3.is_symlink():
        for cell in v3.iterdir():
            if cell.is_symlink() or not cell.is_dir():
                continue
            cache = cell / "cache"
            report = cell / "validation_report.json"
            if not cache.is_dir() or cache.is_symlink() or not report.is_file():
                continue
            try:
                raw = report.read_bytes()
                if json.loads(raw).get("status") != "VALID_COMPLETE":
                    continue
            except (OSError, ValueError):
                continue
            if any(str(cell) in command for command in commands):
                continue
            yield cache, hashlib.sha256(raw).hexdigest()


def sweep(dry_run: bool = False) -> tuple[int, int]:
    count = freed = 0
    for target, digest in verified_cache_targets(active_commands()):
        # Defense in depth: never follow a symlink out of this campaign.
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"Unsafe cache path: {target}")
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file() and not p.is_symlink())
        if dry_run:
            print(json.dumps(dict(would_delete=str(target), logical_bytes=size,
                                  validated_result_sha256=digest), sort_keys=True), flush=True)
            count += 1
            freed += size
            continue
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            # Another sweep/process may have removed a completed fold between
            # discovery and deletion. This is already the desired state.
            continue
        entry = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), path=str(target),
                     logical_bytes=size, validated_result_sha256=digest,
                     recovery="Reextract features from the retained source checkpoint")
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("a") as out:
            out.write(json.dumps(entry, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        print(json.dumps(entry, sort_keys=True), flush=True)
        count += 1
        freed += size
    return count, freed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List validated targets without deleting")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--campaign", choices=("hplus", "l"), default="hplus")
    args = parser.parse_args()
    if args.campaign == "l":
        ROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
        OLD = ROOT / "results"
        AUDIT = ROOT / "logs" / "cache_cleanup.jsonl"
    if args.interval < 10:
        parser.error("interval must be >=10 seconds")
    while True:
        sweep(dry_run=args.dry_run)
        if args.once:
            break
        time.sleep(args.interval)
