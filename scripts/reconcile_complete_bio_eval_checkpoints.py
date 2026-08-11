#!/usr/bin/env python3
"""Mark split bio-evaluation checkpoints complete after a strict result audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


EXPECTED_RESULTS = {
    "bio_classification": ("last_result.json", 25),
    "bio_regression": ("last_result.json", 2),
    "bio_retrieval": ("last_result.json", 4),
    "bio_detection": ("results_bio_detection.json", 1),
    "bio_segmentation": ("results.json", 8),
}
DEFAULT_CHECKPOINTS = (1024, 2049, 3074, 4099, 5124, 6149, 7174, 8199, 9224,
                       10249, 11274, 12299, 13324, 14349, 15374)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--watch-seconds", type=float, default=0)
    parser.add_argument("--min-result-age-seconds", type=float, default=10)
    return parser.parse_args()


def check_finite(value: Any, source: Path, key_path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            check_finite(nested, source, f"{key_path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            check_finite(nested, source, f"{key_path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{source}: non-finite value at {key_path}: {value}")
    elif isinstance(value, str) and value.strip().lower() in {
        "nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    }:
        raise ValueError(f"{source}: non-finite string at {key_path}: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_inventory(eval_root: Path, checkpoints: set[int]) -> dict[int, dict[str, list[Path]]]:
    inventory = {
        checkpoint: {family: [] for family in EXPECTED_RESULTS}
        for checkpoint in checkpoints
    }
    checkpoint_names = {str(checkpoint): checkpoint for checkpoint in checkpoints}
    for family, (filename, _) in EXPECTED_RESULTS.items():
        for path in (eval_root / family).rglob(filename):
            checkpoint = checkpoint_names.get(path.parent.name)
            if checkpoint is not None:
                inventory[checkpoint][family].append(path)
    return inventory


def collect_results(
    checkpoint: int,
    min_age: float,
    inventory: dict[int, dict[str, list[Path]]],
) -> tuple[list[Path], dict[str, int]]:
    now = time.time()
    result_paths: list[Path] = []
    counts: dict[str, int] = {}
    for family, (filename, expected_count) in EXPECTED_RESULTS.items():
        matches = sorted(inventory[checkpoint][family])
        counts[family] = len(matches)
        if len(matches) != expected_count:
            raise RuntimeError(f"{family}={len(matches)}/{expected_count}")
        for path in matches:
            if now - path.stat().st_mtime < min_age:
                raise RuntimeError(f"result is still fresh: {path}")
            with path.open() as handle:
                check_finite(json.load(handle), path)
        result_paths.extend(matches)
    return result_paths, counts


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def mark_complete(
    eval_root: Path,
    checkpoint: int,
    min_age: float,
    inventory: dict[int, dict[str, list[Path]]],
) -> bool:
    status_root = eval_root / "_online_status"
    done_marker = status_root / f"ckpt_{checkpoint}.done"
    failed_marker = status_root / f"ckpt_{checkpoint}.failed"
    if done_marker.is_file() and not failed_marker.exists():
        return True
    try:
        results, counts = collect_results(checkpoint, min_age, inventory)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
        return False

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "checkpoint": checkpoint,
        "created_at_utc": created_at,
        "result_count": len(results),
        "result_counts": counts,
        "results": [
            {"path": str(path.relative_to(eval_root)), "sha256": sha256(path)}
            for path in results
        ],
    }
    write_json_atomic(eval_root / "_completion_audit" / f"ckpt_{checkpoint}.json", manifest)
    status_root.mkdir(parents=True, exist_ok=True)
    temporary = status_root / f".ckpt_{checkpoint}.done.tmp"
    temporary.write_text(created_at + "\n")
    os.replace(temporary, done_marker)
    failed_marker.unlink(missing_ok=True)
    print(f"marked complete: {eval_root} checkpoint={checkpoint} results={len(results)}", flush=True)
    return True


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    pending = set(args.checkpoints)
    while pending:
        completed = {
            checkpoint for checkpoint in pending
            if (eval_root / "_online_status" / f"ckpt_{checkpoint}.done").is_file()
            and not (eval_root / "_online_status" / f"ckpt_{checkpoint}.failed").exists()
        }
        pending.difference_update(completed)
        if not pending:
            break
        inventory = collect_inventory(eval_root, pending)
        for checkpoint in sorted(pending):
            if mark_complete(eval_root, checkpoint, args.min_result_age_seconds, inventory):
                pending.remove(checkpoint)
        print(f"progress: {len(args.checkpoints) - len(pending)}/{len(args.checkpoints)} complete", flush=True)
        if not pending or args.watch_seconds <= 0:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
