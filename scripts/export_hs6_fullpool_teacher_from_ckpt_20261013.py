#!/usr/bin/env python3
"""Export and verify the EMA teacher from one consolidated full-pool checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    source = args.run / "ckpt" / str(args.step) / "checkpoint.pth"
    target = args.run / "eval" / f"training_{args.step}" / "teacher_checkpoint.pth"
    if not source.is_file() or source.stat().st_size < 3_000_000_000:
        raise FileNotFoundError(f"Incomplete optimizer checkpoint: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        provenance = target.parent / "EXPORT_PROVENANCE.json"
        if provenance.is_file() and json.loads(provenance.read_text()).get("source_sha256") == sha(source):
            print(f"REUSED {target}", flush=True)
            return
        raise FileExistsError(f"Teacher exists without matching provenance: {target}")

    torch.set_num_threads(2)
    state = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if int(state.get("iteration", -1)) != args.step:
        raise ValueError(f"Checkpoint iteration mismatch: {state.get('iteration')} != {args.step}")
    model = state.get("model", {})
    teacher = {key.removeprefix("teacher."): value for key, value in model.items()
               if key.startswith("teacher.")}
    if not teacher or not any(key.startswith("backbone.") for key in teacher):
        raise ValueError("Checkpoint does not contain EMA teacher backbone tensors")
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save({"teacher": teacher}, temp)
    check = torch.load(temp, map_location="cpu", mmap=True, weights_only=False)["teacher"]
    if check.keys() != teacher.keys() or any(not torch.equal(value, check[key]) for key, value in teacher.items()):
        temp.unlink(missing_ok=True)
        raise ValueError("Exported teacher failed exact round-trip verification")
    temp.replace(target)
    provenance = dict(source=str(source), source_sha256=sha(source), source_bytes=source.stat().st_size,
                      target=str(target), target_sha256=sha(target), target_bytes=target.stat().st_size,
                      tensor_count=len(teacher), exact_roundtrip=True)
    (target.parent / "EXPORT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
