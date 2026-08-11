#!/usr/bin/env python3
"""Safely remove reconstructable feature caches after a complete bio evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_RESULTS = {
    "bio_classification": ("last_result.json", 25),
    "bio_regression": ("last_result.json", 2),
    "bio_retrieval": ("last_result.json", 4),
    "bio_detection": ("results_bio_detection.json", 1),
    "bio_segmentation": ("results.json", 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["classification", "regression", "retrieval", "detection", "segmentation", "ood"],
        help="Task families to audit. By default, infer them from result directories present in eval-root.",
    )
    parser.add_argument("--delete", action="store_true", help="Delete caches after all safety checks pass.")
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
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        raise ValueError(f"{source}: non-finite string at {key_path}: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_result_families(eval_root: Path, tasks: list[str] | None) -> list[str]:
    if tasks:
        families = [f"bio_{task}" for task in tasks if task != "ood"]
    else:
        families = [family for family in EXPECTED_RESULTS if (eval_root / family).is_dir()]
    if not families:
        raise RuntimeError(f"No auditable result families found under {eval_root}")
    return list(dict.fromkeys(families))


def collect_results(
    eval_root: Path, checkpoint: int, families: list[str]
) -> tuple[list[Path], dict[str, int]]:
    result_paths: list[Path] = []
    counts: dict[str, int] = {}
    for family in families:
        filename, expected_count = EXPECTED_RESULTS[family]
        family_root = eval_root / family
        matches = sorted(
            path
            for path in family_root.rglob(filename)
            if path.parent.name == str(checkpoint)
        )
        counts[family] = len(matches)
        if len(matches) != expected_count:
            raise RuntimeError(
                f"checkpoint {checkpoint}: {family} has {len(matches)} {filename} files; "
                f"expected {expected_count}"
            )
        result_paths.extend(matches)
    for path in result_paths:
        with path.open() as handle:
            payload = json.load(handle)
        check_finite(payload, path)
    return result_paths, counts


def collect_cache_paths(eval_root: Path, checkpoint: int) -> list[Path]:
    cache_root = (eval_root / "cache").resolve()
    if not cache_root.is_dir():
        return []
    paths: list[Path] = []
    for path in cache_root.rglob(str(checkpoint)):
        if not path.is_dir() or path.name != str(checkpoint):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(cache_root):
            raise RuntimeError(f"cache path escapes cache root: {path} -> {resolved}")
        paths.append(resolved)
    return sorted(set(paths))


def disk_usage_kib(paths: list[Path]) -> int:
    if not paths:
        return 0
    result = subprocess.run(
        ["du", "-sk", *map(str, paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(line.split(maxsplit=1)[0]) for line in result.stdout.splitlines() if line.strip())


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    checkpoint = args.checkpoint
    done_marker = eval_root / "_online_status" / f"ckpt_{checkpoint}.done"
    failed_marker = eval_root / "_online_status" / f"ckpt_{checkpoint}.failed"
    if not done_marker.is_file():
        raise SystemExit(f"Refusing cache cleanup without completion marker: {done_marker}")
    if failed_marker.exists():
        raise SystemExit(f"Refusing cache cleanup with failure marker present: {failed_marker}")

    audited_families = select_result_families(eval_root, args.tasks)
    result_paths, result_counts = collect_results(eval_root, checkpoint, audited_families)
    cache_paths = collect_cache_paths(eval_root, checkpoint)
    cache_kib = disk_usage_kib(cache_paths)
    manifest_path = eval_root / "_cache_cleanup" / f"ckpt_{checkpoint}.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open() as handle:
            previous_manifest = json.load(handle)
    previously_deleted = bool(previous_manifest.get("deleted")) and not cache_paths
    payload: dict[str, Any] = {
        "checkpoint": checkpoint,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "delete_requested": bool(args.delete),
        "deleted": previously_deleted,
        "completion_marker": str(done_marker),
        "audited_families": audited_families,
        "result_counts": result_counts,
        "result_count": len(result_paths),
        "results": [
            {"path": str(path.relative_to(eval_root)), "sha256": sha256(path)}
            for path in result_paths
        ],
        "cache_paths": [str(path.relative_to(eval_root)) for path in cache_paths],
        "cache_kib": cache_kib,
    }
    if previously_deleted:
        payload["deleted_at_utc"] = previous_manifest.get("deleted_at_utc")
        payload["reverified_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(manifest_path, payload)

    if args.delete:
        for path in cache_paths:
            shutil.rmtree(path)
        remaining = collect_cache_paths(eval_root, checkpoint)
        if remaining:
            raise RuntimeError(f"cache cleanup incomplete: {remaining}")
        payload["deleted"] = True
        payload["deleted_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_manifest(manifest_path, payload)

    print(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "result_count": len(result_paths),
                "cache_paths": len(cache_paths),
                "cache_gib": cache_kib / 1024 / 1024,
                "deleted": payload["deleted"],
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
