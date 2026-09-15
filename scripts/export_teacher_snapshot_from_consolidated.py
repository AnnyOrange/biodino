#!/usr/bin/env python3
"""Export the EMA teacher used by ``do_test`` from a consolidated train checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _trusted_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_teacher(path: Path, expected_iteration: int | None) -> dict[str, torch.Tensor]:
    payload = _trusted_load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Not a consolidated DINOv3 training checkpoint: {path}")
    iteration = int(payload.get("iteration", -1))
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(
            f"Checkpoint iteration is {iteration}, expected {expected_iteration}: {path}"
        )
    prefix = "teacher."
    teacher = {
        key.removeprefix(prefix): value
        for key, value in payload["model"].items()
        if key.startswith(prefix)
    }
    if not teacher:
        raise ValueError(f"Checkpoint has no {prefix!r} model entries: {path}")
    invalid = [key for key, value in teacher.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f"Teacher state contains non-tensors: {invalid[:8]}")
    return teacher


def verify_exact(actual: dict[str, torch.Tensor], reference_path: Path) -> None:
    reference = _trusted_load(reference_path)
    if not isinstance(reference, dict) or not isinstance(reference.get("teacher"), dict):
        raise ValueError(f"Reference is not a teacher snapshot: {reference_path}")
    expected = reference["teacher"]
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise ValueError(
            f"Teacher key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for key, value in actual.items():
        other = expected[key]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            raise ValueError(f"Teacher tensor differs at {key}")


def atomic_save(path: Path, teacher: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save({"teacher": teacher}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    if args.output is None and args.verify_against is None:
        parser.error("at least one of --output or --verify-against is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {args.output}")

    teacher = extract_teacher(args.checkpoint, args.expected_iteration)
    if args.verify_against is not None:
        verify_exact(teacher, args.verify_against)
    if args.output is not None:
        atomic_save(args.output, teacher)
    print(
        json.dumps(
            {
                "status": "VALID_EXACT_TEACHER_EXPORT",
                "checkpoint": str(args.checkpoint.resolve()),
                "output": str(args.output.resolve()) if args.output is not None else None,
                "verified_against": (
                    str(args.verify_against.resolve())
                    if args.verify_against is not None
                    else None
                ),
                "tensor_count": len(teacher),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
