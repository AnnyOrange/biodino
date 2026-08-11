#!/usr/bin/env python3
"""Remove a tuning run's checkpoint after its scalar evaluation is complete."""

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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=15374)
    parser.add_argument("--delete", action="store_true")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_usage_kib(path: Path) -> int:
    if not path.exists():
        return 0
    result = subprocess.run(
        ["du", "-sk", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split(maxsplit=1)[0])


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def collect_results(eval_root: Path, checkpoint: int) -> tuple[list[Path], dict[str, int]]:
    paths: list[Path] = []
    counts: dict[str, int] = {}
    for family, (filename, expected_count) in EXPECTED_RESULTS.items():
        matches = sorted(
            path
            for path in (eval_root / family).rglob(filename)
            if path.parent.name == str(checkpoint)
        )
        counts[family] = len(matches)
        if len(matches) != expected_count:
            raise RuntimeError(
                f"{family} has {len(matches)} results for checkpoint {checkpoint}; "
                f"expected {expected_count}"
            )
        paths.extend(matches)
    for path in paths:
        with path.open() as handle:
            payload = json.load(handle)
        check_finite(payload, path)
    return paths, counts


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint
    eval_root = run_dir / "eval_scalar"
    complete_marker = eval_root / "_complete"
    checkpoint_root = run_dir / "ckpt"
    final_checkpoint = checkpoint_root / str(checkpoint)
    manifest_path = eval_root / "checkpoint_prune.json"

    if not complete_marker.is_file():
        raise SystemExit(f"Refusing prune without scalar completion marker: {complete_marker}")
    for evidence in (run_dir / "config.yaml", run_dir / "raw_loss_metrics.jsonl"):
        if not evidence.is_file():
            raise SystemExit(f"Refusing prune without training evidence: {evidence}")

    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open() as handle:
            previous = json.load(handle)
    previously_deleted = bool(previous.get("deleted")) and not checkpoint_root.exists()
    if not previously_deleted and not (
        (final_checkpoint / "checkpoint.pth").is_file()
        or (final_checkpoint / ".metadata").is_file()
    ):
        raise SystemExit(f"Refusing prune without final checkpoint: {final_checkpoint}")

    result_paths, result_counts = collect_results(eval_root, checkpoint)
    checkpoint_kib = disk_usage_kib(checkpoint_root)
    checkpoint_entries = sorted(path.name for path in checkpoint_root.iterdir()) if checkpoint_root.is_dir() else []
    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "delete_requested": bool(args.delete),
        "deleted": previously_deleted,
        "checkpoint_entries": checkpoint_entries,
        "checkpoint_kib": checkpoint_kib,
        "result_counts": result_counts,
        "result_count": len(result_paths),
        "results": [
            {"path": str(path.relative_to(run_dir)), "sha256": sha256(path)}
            for path in result_paths
        ],
        "training_evidence": {
            name: sha256(run_dir / name)
            for name in ("config.yaml", "raw_loss_metrics.jsonl")
        },
    }
    if previously_deleted:
        payload["deleted_at_utc"] = previous.get("deleted_at_utc")
        payload["reverified_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(manifest_path, payload)

    if args.delete and checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
        if checkpoint_root.exists():
            raise RuntimeError(f"checkpoint prune incomplete: {checkpoint_root}")
        payload["deleted"] = True
        payload["deleted_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_manifest(manifest_path, payload)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint": checkpoint,
                "result_count": len(result_paths),
                "checkpoint_gib": checkpoint_kib / 1024 / 1024,
                "deleted": payload["deleted"],
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
