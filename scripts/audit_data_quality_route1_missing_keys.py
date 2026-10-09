#!/usr/bin/env python3
"""Find tar metadata keys that the failed route-1 loader never emitted."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def keys(path: str) -> tuple[str, set[str]]:
    found: set[str] = set()
    with tarfile.open(path, "r") as archive:
        while member := archive.next():
            if member.name.endswith(".meta.json"):
                found.add(f"{Path(path).name}::{member.name[:-10]}")
    return path, found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--consumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.assignment.read_text())
    # The loader expands the primary pattern before the backfill pattern.
    paths = sorted(source["shard_sample_counts"], key=lambda path: ("/backfill_final/" in path, path))
    random.Random(2).shuffle(paths)
    old_rank0 = paths[::4]
    with ThreadPoolExecutor(max_workers=12) as pool:
        inventory = dict(pool.map(keys, old_rank0))
    consumed = {key for line in args.consumed.open() for key in json.loads(line)}
    all_keys = set().union(*inventory.values())
    missing = sorted(all_keys - consumed)
    by_shard = Counter(key.split("::", 1)[0] for key in missing)
    result = dict(metadata=len(all_keys), consumed=len(consumed), missing=len(missing),
                  missing_keys=missing, missing_by_shard=dict(by_shard.most_common()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "missing_keys"}, indent=2))


if __name__ == "__main__":
    main()
