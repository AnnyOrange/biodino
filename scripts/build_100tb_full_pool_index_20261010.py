#!/usr/bin/env python3
"""Index every exported 100TB stored item without copying image bytes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from sample_full_storage_global_candidates import DEFAULT_DB_CONFIG, STORAGE_ROOT, psycopg2


EXPECTED_ITEMS = 26_614_979


def build(output: Path, world_size: int = 4) -> dict:
    if output.exists() or output.with_name(output.name + ".building").exists():
        raise FileExistsError(f"Index path already exists: {output}")
    building = output.with_name(output.name + ".building")
    building.mkdir(parents=True)
    conn = psycopg2.connect(**DEFAULT_DB_CONFIG)
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.shard_id, s.shard_path
                FROM curated_100t_shards s
                JOIN curated_100t_runs r ON r.run_id=s.run_id
                WHERE s.run_id IN (1,5) AND r.storage_root=%s
            """, (STORAGE_ROOT,))
            shards = {int(shard_id): path for shard_id, path in cur.fetchall()}
        if not shards:
            raise ValueError("No 100TB source shards found")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT i.shard_id, count(*)
                FROM curated_100t_items i
                JOIN curated_100t_runs r ON r.run_id=i.run_id
                WHERE i.run_id IN (1,5) AND i.status='exported'
                  AND r.storage_root=%s
                GROUP BY i.shard_id
            """, (STORAGE_ROOT,))
            shard_counts = {int(shard_id): int(count) for shard_id, count in cur.fetchall()}
        if set(shard_counts) - set(shards) or sum(shard_counts.values()) != EXPECTED_ITEMS:
            raise ValueError("100TB shard inventory differs from the exported item population")
        rank_targets = [0] * world_size
        routes = {}
        for shard_id, count in sorted(shard_counts.items(), key=lambda row: (-row[1], row[0])):
            rank = min(range(world_size), key=lambda candidate: (rank_targets[candidate], candidate))
            routes[shard_id] = rank
            rank_targets[rank] += count
        handles = [(building / f"rank{rank:02d}.tsv").open("w", newline="")
                   for rank in range(world_size)]
        writers = [csv.writer(handle, delimiter="\t", lineterminator="\n") for handle in handles]
        counts = [0] * world_size
        rank_shards = [set() for _ in range(world_size)]
        try:
            cur = conn.cursor(name="full_100tb_stream_index")
            cur.itersize = 100_000
            cur.execute("""
                SELECT i.shard_id, i.item_id, COALESCE(o.channel_count, 0)
                FROM curated_100t_items i
                JOIN curated_100t_runs r ON r.run_id=i.run_id
                LEFT JOIN original_images_all o
                  ON i.source_table_code=1 AND o.id=i.source_id
                WHERE i.run_id IN (1,5) AND i.status='exported'
                  AND r.storage_root=%s
            """, (STORAGE_ROOT,))
            try:
                for shard_id, item_id, hint in cur:
                    shard_id = int(shard_id)
                    rank = routes[shard_id]
                    writers[rank].writerow((shard_id, item_id, hint or 0))
                    counts[rank] += 1
                    rank_shards[rank].add(shard_id)
                    if sum(counts) % 1_000_000 == 0:
                        print(f"indexed={sum(counts):,}/{EXPECTED_ITEMS:,}", flush=True)
            finally:
                cur.close()
        finally:
            for handle in handles:
                handle.close()
        if sum(counts) != EXPECTED_ITEMS:
            raise ValueError(f"100TB exported population changed: {sum(counts)} != {EXPECTED_ITEMS}")
        if counts != rank_targets:
            raise ValueError(f"Full-pool rank counts changed during index build: {counts} != {rank_targets}")
        used_shards = {str(shard_id): shards[shard_id] for group in rank_shards for shard_id in group}
        missing = [path for path in used_shards.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} source tars unavailable, first: {missing[0]}")
        report = {
            "status": "PASS", "format": "100tb_full_pool_stream_v1",
            "population": "exported stored items, runs 1/5, all source shards; no dataset quotas",
            "sampling": "random shard permutation plus bounded in-memory item shuffle per rank",
            "world_size": world_size, "total_items": sum(counts), "rank_counts": counts,
            "shards": used_shards,
            "rank_shards": [sorted(group) for group in rank_shards],
            "created_unix": time.time(),
        }
        (building / "assignment.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
        os.replace(building, output)
        return {key: report[key] for key in ("status", "total_items", "rank_counts", "world_size")}
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.world_size), indent=2))
