#!/usr/bin/env python3
"""Count route-1 WDS samples and assign whole shards evenly to four ranks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PATTERNS = (
    "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/packed/filtered_projection_20TB_nested-r*.tar",
    "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/backfill_final/filtered_projection_20TB_nested-r*.tar",
)


def count(path: str) -> tuple[str, int]:
    count_meta = 0
    with tarfile.open(path, "r") as archive:
        while member := archive.next():
            count_meta += member.name.endswith(".meta.json")
    return path, count_meta


def assign(items: list[tuple[str, int]], seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    groups: list[list[str]] = [[] for _ in range(4)]
    totals = [0] * 4
    for path, size in items:
        rank = min(range(4), key=lambda index: totals[index])
        groups[rank].append(path)
        totals[rank] += size
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    paths = sorted(set(path for pattern in PATTERNS for path in glob.glob(pattern)))
    if len(paths) != 275:
        raise ValueError(f"Expected 275 shards, found {len(paths)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        items = list(pool.map(count, paths))
    if sum(size for _, size in items) != 1_000_000:
        raise ValueError(f"Metadata sample count is {sum(size for _, size in items)}, expected 1M")
    candidates = [assign(items.copy(), seed) for seed in range(1000)]
    groups = max(candidates, key=lambda candidate: min(sum(dict(items)[path] for path in group) for group in candidate))
    counts = dict(items)
    totals = [sum(counts[path] for path in group) for group in groups]
    if min(totals) < 249_856:
        raise ValueError(f"No assignment covers the 976-update budget: {totals}")
    result = {
        "status": "PASS", "sample_count": 1_000_000, "target_per_rank": 249_856,
        "rank_sample_counts": totals, "rank_shards": groups,
        "shard_sample_counts": counts,
        "inventory_sha256": hashlib.sha256("\n".join(f"{path}:{Path(path).stat().st_size}" for path in paths).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rank_sample_counts": totals}, indent=2))


if __name__ == "__main__":
    main()
