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
DEFAULT_OUTPUT = DEFAULT_CANDIDATES.parent.parent / "stream_index"
TARGET_PER_RANK = 16_000 * 16 * 8


def build(candidates: Path, output: Path, world_size: int = 8) -> dict:
    if output.exists():
        raise FileExistsError(f"Stream index already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"Incomplete prior index build: {building}")
    counts = Counter()
    digest = hashlib.sha256()
    with candidates.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with candidates.open(newline="") as source:
        rows = csv.DictReader(source)
        for number, row in enumerate(rows):
            if int(row["priority"]) != number:
                raise ValueError(f"Candidate priority changed at row {number}")
            counts[row["shard_path"]] += 1
    total = sum(counts.values())
    if world_size == 8 and total != 16_500_000:
        raise ValueError(f"Expected 16.5M candidates, found {total}")
    shuffled = list(counts)
    random.Random(20261009).shuffle(shuffled)
    shuffled.sort(key=lambda path: counts[path], reverse=True)
    groups = [[] for _ in range(world_size)]
    rank_counts = [0] * world_size
    for path in shuffled:
        rank = min(range(world_size), key=lambda index: rank_counts[index])
        groups[rank].append(path)
        rank_counts[rank] += counts[path]
    if world_size == 8 and min(rank_counts) < TARGET_PER_RANK:
        raise ValueError(f"A rank cannot cover the 16k-update budget: {rank_counts}")
    shards = sorted(counts)
    shard_ids = {path: index for index, path in enumerate(shards)}
    rank_for_path = {path: rank for rank, paths in enumerate(groups) for path in paths}
    building.mkdir(parents=True)
    handles = [(building / f"rank{rank:02d}.tsv").open("w", newline="") for rank in range(world_size)]
    writers = [csv.writer(handle, delimiter="\t", lineterminator="\n") for handle in handles]
    try:
        with candidates.open(newline="") as source:
            for row in csv.DictReader(source):
                path = row["shard_path"]
                writers[rank_for_path[path]].writerow((
                    shard_ids[path], row["storage_item_id"], row["channel_count"], row["priority"]
                ))
    finally:
        for handle in handles:
            handle.close()
    report = {
        "status": "PASS", "format": "100tb_raw_tar_stream_v1",
        "candidate_manifest": str(candidates), "candidate_sha256": digest.hexdigest(),
        "candidate_count": total, "sampling_unit": "deduplicated exported stored item",
        "sampling_method": "global random permutation without replacement; no dataset quotas",
        "shards": shards,
        "rank_shards": [[shard_ids[path] for path in paths] for paths in groups],
        "rank_sample_counts": rank_counts, "target_per_rank": TARGET_PER_RANK if world_size == 8 else None,
        "world_size": world_size, "seed": 20261009,
    }
    (building / "assignment.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    os.replace(building, output)
    return {key: report[key] for key in ("status", "candidate_count", "rank_sample_counts", "world_size")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args.candidates, args.output, args.world_size), indent=2))
