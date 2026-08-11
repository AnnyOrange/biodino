#!/usr/bin/env python3
"""Audit a selected classification panel and write an atomic done marker."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--panel-name", default="partial_classification")
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


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    results = []
    for dataset in args.datasets:
        path = eval_root / "bio_classification" / dataset / str(args.checkpoint) / "last_result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as handle:
            payload = json.load(handle)
        if payload.get("error"):
            raise RuntimeError(f"{path}: {payload['error']}")
        if payload.get("dataset") != dataset:
            raise RuntimeError(f"{path}: dataset={payload.get('dataset')!r}, expected {dataset!r}")
        check_finite(payload, path)
        results.append({"dataset": dataset, "path": str(path.relative_to(eval_root)), "sha256": sha256(path)})

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    audit = {
        "checkpoint": args.checkpoint,
        "created_at_utc": created_at,
        "dataset_count": len(results),
        "datasets": list(args.datasets),
        "panel_name": args.panel_name,
        "results": results,
    }
    audit_path = eval_root / "_completion_audit" / f"{args.panel_name}_ckpt_{args.checkpoint}.json"
    write_atomic(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    marker = eval_root / "_online_status" / f"ckpt_{args.checkpoint}_{args.panel_name}.done"
    write_atomic(marker, created_at + "\n")
    print(json.dumps({"audit": str(audit_path), "marker": str(marker), "results": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
