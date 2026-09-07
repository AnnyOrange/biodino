#!/usr/bin/env python3
"""Strip optimizer/student state from a Scout training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="teacher.backbone.")
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.input, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.input, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    selected = {key: value for key, value in state.items() if key.startswith(args.prefix)}
    if not selected:
        raise ValueError(f"No keys begin with {args.prefix!r} in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": checkpoint.get("iteration"), "model": selected}, args.output)
    print(f"saved {len(selected)} tensors to {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
