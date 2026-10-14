#!/usr/bin/env python3
"""Split the existing global 100TB candidate list into small rank-local indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


DEFAULT_CANDIDATES = Path(
    "/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009/"
    "manifests/100tb_candidates_16500000.csv"
)
DEFAULT_OUTPUT = DEFAULT_CANDIDATES.parent.parent / "stream_index_exact"
TARGET_TOTAL = 16_000 * 1024


def build(candidates: Path, output: Path, world_size: int = 8, target_total: int = TARGET_TOTAL) -> dict:
    if world_size < 1 or target_total < 1 or target_total % world_size:
        raise ValueError("Target total must be positive and divisible by world size")
    if output.exists():
        raise FileExistsError(f"Stream index already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"Incomplete prior index build: {building}")
    primary_counts = Counter()
    backup_counts = Counter()
    digest = hashlib.sha256()
    with candidates.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with candidates.open(newline="") as source:
        rows = csv.DictReader(source)
        for number, row in enumerate(rows):
            if int(row["priority"]) != number:
                raise ValueError(f"Candidate priority changed at row {number}")
            if number < target_total:
                primary_counts[row["shard_path"]] += 1
            else:
                backup_counts[row["shard_path"]] += 1
    counts = primary_counts + backup_counts
    total = sum(counts.values())
    if total < target_total:
        raise ValueError("Candidate list is smaller than the requested training budget")
    if world_size == 8 and total != 16_500_000:
        raise ValueError(f"Expected 16.5M candidates, found {total}")
    shuffled = list(primary_counts)
    random.Random(20261009).shuffle(shuffled)
    target_per_rank = target_total // world_size
    primary_routes = {}
    primary_groups = [set() for _ in range(world_size)]
    rank_primary_counts = [0] * world_size
    rank = 0
    for path in shuffled:
        remaining = primary_counts[path]
        primary_routes[path] = []
        while remaining:
            amount = min(remaining, target_per_rank - rank_primary_counts[rank])
            primary_routes[path].append((rank, amount))
            primary_groups[rank].add(path)
            rank_primary_counts[rank] += amount
            remaining -= amount
            if rank_primary_counts[rank] == target_per_rank:
                rank += 1
    if rank_primary_counts != [target_per_rank] * world_size:
        raise ValueError(f"Primary rank quotas are not exact: {rank_primary_counts}")
    shards = sorted(counts)
    shard_ids = {path: index for index, path in enumerate(shards)}
    rank_backup_counts = [0] * world_size
    backup_groups = [set() for _ in range(world_size)]
    route_positions = Counter()
    route_used = Counter()
    building.mkdir(parents=True)
    handles = [(building / f"rank{rank:02d}.tsv").open("w", newline="") for rank in range(world_size)]
    writers = [csv.writer(handle, delimiter="\t", lineterminator="\n") for handle in handles]
    try:
        with candidates.open(newline="") as source:
            for row in csv.DictReader(source):
                path = row["shard_path"]
                priority = int(row["priority"])
                if priority < target_total:
                    position = route_positions[path]
                    route_rank, route_count = primary_routes[path][position]
                    rank_for_row = route_rank
                    route_used[path] += 1
                    if route_used[path] == route_count:
                        route_positions[path] += 1
                        route_used[path] = 0
                else:
                    rank_for_row = (priority - target_total) % world_size
                    rank_backup_counts[rank_for_row] += 1
                    backup_groups[rank_for_row].add(path)
                writers[rank_for_row].writerow((
                    shard_ids[path], row["storage_item_id"], row["channel_count"], row["priority"]
                ))
    finally:
        for handle in handles:
            handle.close()
    report = {
        "status": "PASS", "format": "100tb_raw_tar_stream_v2",
        "candidate_manifest": str(candidates), "candidate_sha256": digest.hexdigest(),
        "candidate_count": total, "sampling_unit": "deduplicated exported stored item",
        "sampling_method": "global random permutation without replacement; no dataset quotas",
        "shards": shards,
        "rank_shards": [[shard_ids[path] for path in sorted(paths)] for paths in primary_groups],
        "rank_backup_shards": [[shard_ids[path] for path in sorted(paths)] for paths in backup_groups],
        "rank_primary_counts": rank_primary_counts,
        "rank_backup_counts": rank_backup_counts,
        "rank_sample_counts": [p + b for p, b in zip(rank_primary_counts, rank_backup_counts)],
        "target_total": target_total, "target_per_rank": target_per_rank,
        "world_size": world_size, "seed": 20261009,
    }
    (building / "assignment.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    os.replace(building, output)
    return {key: report[key] for key in ("status", "candidate_count", "rank_primary_counts",
                                         "rank_backup_counts", "world_size")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    args = parser.parse_args()
    print(json.dumps(build(args.candidates, args.output, args.world_size, args.target_total), indent=2))
