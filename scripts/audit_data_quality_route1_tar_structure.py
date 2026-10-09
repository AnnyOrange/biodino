#!/usr/bin/env python3
"""Check that every route-1 metadata record has a recognized image member."""

from __future__ import annotations

import argparse
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def inspect(path: str) -> tuple[str, dict]:
    meta: set[str] = set()
    channels: set[str] = set()
    with tarfile.open(path, "r") as archive:
        while member := archive.next():
            name = member.name
            if name.endswith(".meta.json"):
                meta.add(name[:-10])
            elif name.endswith(".tif") and ".ch" in name:
                prefix, channel = name.rsplit(".ch", 1)
                if channel[:-4].isdigit() and 1 <= int(channel[:-4]) <= 8:
                    channels.add(prefix)
    return path, dict(metadata=len(meta), with_channel=len(meta & channels),
                      without_channel=len(meta - channels), channel_without_metadata=len(channels - meta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignment = json.loads(args.assignment.read_text())
    paths = list(assignment["shard_sample_counts"])
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = dict(pool.map(inspect, paths))
    totals = [sum(rows[path]["with_channel"] for path in group) for group in assignment["rank_shards"]]
    result = dict(rank_with_channel_counts=totals, rank_target=assignment["target_per_rank"],
                  total_with_channel=sum(row["with_channel"] for row in rows.values()),
                  total_without_channel=sum(row["without_channel"] for row in rows.values()),
                  shard_counts=rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "shard_counts"}, indent=2))


if __name__ == "__main__":
    main()
