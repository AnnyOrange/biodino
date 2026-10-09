#!/usr/bin/env python3
"""Draw an ordered, dataset-agnostic SRS from the complete storage inventory.

100TB units are exported storage items in curated_100t_items (runs 1/5).
1PB units are original_images_all rows plus individual frames enumerated by
original_image_all_2p_parsed.count.  This writes source candidates only;
decoding/technical rejection and the final first-1M acceptance happen later.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterator

import psycopg2

PIPE = Path("/home/inspur/xzj/pre_data/scalinglaw_sampling")
sys.path.insert(0, str(PIPE))
from slice_from_dataset_plan import DEFAULT_DB_CONFIG  # noqa: E402

FIELDS = ["priority", "pool", "storage_item_id", "source_table_code", "source_id",
          "frame_idx", "file_path", "image_shape", "channel_count", "h", "w", "shard_path"]
STORAGE_ROOT = "/mnt/huawei_blm/deepcad_100t"


def permutation_prefix(low: int, high: int, seed: int) -> Iterator[int]:
    """Lazy Fisher-Yates: every existing ID has equal inclusion probability."""
    rng = random.Random(seed)
    n = high - low + 1
    swaps: dict[int, int] = {}
    for i in range(n):
        j = rng.randrange(i, n)
        result = swaps.get(j, j)
        swaps[j] = swaps.get(i, i)
        yield low + result


def source_counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), min(id), max(id) FROM original_images_all")
        ori_count, ori_min, ori_max = (int(x) for x in cur.fetchone())
        cur.execute('SELECT coalesce(sum("count"),0) FROM original_image_all_2p_parsed WHERE "count" > 0')
        frame_count = int(cur.fetchone()[0])
        cur.execute("""
            SELECT count(*), min(i.item_id), max(i.item_id)
            FROM curated_100t_items i
            JOIN curated_100t_runs r ON r.run_id=i.run_id
            WHERE i.run_id IN (1,5) AND i.status='exported' AND r.storage_root=%s
        """, (STORAGE_ROOT,))
        storage_count, storage_min, storage_max = (int(x) for x in cur.fetchone())
    return {"ori_count": ori_count, "ori_min": ori_min, "ori_max": ori_max,
            "frame_count": frame_count, "storage_count": storage_count,
            "storage_min": storage_min, "storage_max": storage_max}


def batched(it: Iterator[int], size: int) -> Iterator[list[int]]:
    while True:
        batch = []
        for _ in range(size):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        if not batch:
            break
        yield batch


def sample_exported_items(conn, n: int, counts: dict, seed: int) -> tuple[list[dict], int]:
    if n > counts["storage_count"]:
        raise ValueError("Requested more than exported storage items")
    selected: list[dict] = []
    proposed = 0
    perm = permutation_prefix(counts["storage_min"], counts["storage_max"], seed)
    query = """
        SELECT i.item_id, i.source_table_code, i.source_id, i.frame_idx,
               o.file_path, o.image_shape, o.channel_count,
               NULL, NULL, s.shard_path
        FROM curated_100t_items i
        JOIN curated_100t_runs r ON r.run_id=i.run_id
        LEFT JOIN original_images_all o ON i.source_table_code=1 AND o.id=i.source_id
        LEFT JOIN curated_100t_shards s ON s.shard_id=i.shard_id
        WHERE i.item_id=ANY(%s) AND i.run_id IN (1,5)
          AND i.status='exported' AND r.storage_root=%s
    """
    for ids in batched(perm, 10000):
        proposed += len(ids)
        with conn.cursor() as cur:
            cur.execute(query, (ids, STORAGE_ROOT))
            found = {int(r[0]): r for r in cur.fetchall()}
        for item_id in ids:
            row = found.get(item_id)
            if row is None:
                continue
            selected.append(dict(zip(FIELDS[2:], row)))
            if len(selected) == n:
                fill_2p_metadata(conn, selected)
                return selected, proposed
        if len(selected) % 100000 < 10000:
            print(f"100tb selected={len(selected)}/{n} proposed={proposed}", flush=True)
    raise RuntimeError("Exhausted storage item IDs before reaching requested candidates")


def fill_2p_metadata(conn, selected: list[dict]) -> None:
    # The 2P source table has no id index; scan it once after selecting items.
    ids = {int(row["source_id"]) for row in selected if int(row["source_table_code"]) == 2}
    if not ids:
        return
    metadata = fetch_2p_metadata(conn, ids)
    for row in selected:
        if int(row["source_table_code"]) != 2:
            continue
        _id, path, shape, channels, h, w, _count = metadata[int(row["source_id"])]
        row.update(file_path=path, image_shape=shape, channel_count=channels, h=h, w=w)


def sample_ori_rows(conn, n: int, counts: dict, seed: int) -> tuple[list[tuple], int]:
    if n > counts["ori_count"]:
        raise ValueError("Requested more than ORI source rows")
    selected: list[tuple] = []
    proposed = 0
    perm = permutation_prefix(counts["ori_min"], counts["ori_max"], seed)
    for ids in batched(perm, 10000):
        proposed += len(ids)
        with conn.cursor() as cur:
            cur.execute("SELECT id,file_path,image_shape,channel_count FROM original_images_all WHERE id=ANY(%s)", (ids,))
            found = {int(r[0]): r for r in cur.fetchall()}
        for source_id in ids:
            row = found.get(source_id)
            if row is None:
                continue
            selected.append(row)
            if len(selected) == n:
                return selected, proposed
        if len(selected) % 100000 < 10000:
            print(f"1pb ORI selected={len(selected)}/{n} proposed={proposed}", flush=True)
    raise RuntimeError("Exhausted ORI IDs before reaching requested candidates")


def frame_index(conn, expected_frames: int) -> tuple[array, array]:
    ids = array("q")
    cumulative = array("q")
    total = 0
    cur = conn.cursor(name="full_storage_frame_index")
    cur.itersize = 100000
    cur.execute('SELECT id,"count" FROM original_image_all_2p_parsed WHERE "count" > 0 ORDER BY id')
    try:
        for source_id, count in cur:
            total += int(count)
            ids.append(int(source_id))
            cumulative.append(total)
            if len(ids) % 500000 == 0:
                print(f"2p indexed_files={len(ids)} frames={total}", flush=True)
    finally:
        cur.close()
    if total != expected_frames:
        raise ValueError(f"2P frame count changed during sampling: {total} != {expected_frames}")
    return ids, cumulative


def fetch_2p_metadata(conn, ids: set[int]) -> dict[int, tuple]:
    metadata: dict[int, tuple] = {}
    # This source table has no index on id.  Repeated id=ANY batches would
    # rescan all 4.3M files for every batch; stream it once instead.
    cur = conn.cursor(name="full_storage_2p_metadata")
    cur.itersize = 100000
    cur.execute('SELECT id,file_path,image_shape,channel_count,h,w,"count" '
                'FROM original_image_all_2p_parsed')
    try:
        for scanned, row in enumerate(cur, 1):
            source_id = int(row[0])
            if source_id in ids:
                metadata[source_id] = row
            if scanned % 500000 == 0:
                print(f"2p metadata scanned_files={scanned} matched={len(metadata)}/{len(ids)}", flush=True)
    finally:
        cur.close()
    if len(metadata) != len(ids):
        raise ValueError(f"Missing 2P metadata: {len(metadata)} of {len(ids)}")
    return metadata


def sample_1pb(conn, n: int, counts: dict, seed: int) -> tuple[list[dict], dict]:
    n_ori = counts["ori_count"]
    n_frames = counts["frame_count"]
    total = n_ori + n_frames
    if n > total:
        raise ValueError("Requested more than full 1PB storage units")
    # This single draw fixes a random ORI/2P split; there are no fixed source
    # type or dataset quotas.  Every file image and every 2P frame has weight 1.
    ranks = random.Random(seed).sample(range(total), n)
    ori_needed = sum(rank < n_ori for rank in ranks)
    ori_rows, proposed_ids = sample_ori_rows(conn, ori_needed, counts, seed + 1)
    frame_ranks = [rank - n_ori for rank in ranks if rank >= n_ori]
    frame_ids, cumulative = frame_index(conn, n_frames)
    frame_refs = []
    for rank in frame_ranks:
        index = bisect.bisect_right(cumulative, rank)
        previous = cumulative[index - 1] if index else 0
        frame_refs.append((int(frame_ids[index]), int(rank - previous)))
    metadata = fetch_2p_metadata(conn, {source_id for source_id, _ in frame_refs})
    rows = []
    ori_iter = iter(ori_rows)
    frame_iter = iter(frame_refs)
    for priority, rank in enumerate(ranks):
        if rank < n_ori:
            source_id, path, shape, channels = next(ori_iter)
            rows.append({"priority": priority, "pool": "1pb", "storage_item_id": "",
                         "source_table_code": 1, "source_id": source_id, "frame_idx": "",
                         "file_path": path, "image_shape": shape,
                         "channel_count": channels, "h": "", "w": "", "shard_path": ""})
        else:
            source_id, frame = next(frame_iter)
            _id, path, shape, channels, h, w, count = metadata[source_id]
            if not 0 <= frame < count:
                raise ValueError(f"Frame outside indexed file: {source_id}:{frame}/{count}")
            rows.append({"priority": priority, "pool": "1pb", "storage_item_id": "",
                         "source_table_code": 2, "source_id": source_id, "frame_idx": frame,
                         "file_path": path, "image_shape": shape,
                         "channel_count": channels, "h": h, "w": w, "shard_path": ""})
    return rows, {"ori_id_proposals": proposed_ids, "selected_ori": ori_needed,
                  "selected_2p_frames": n - ori_needed, "indexed_2p_files": len(frame_ids)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", choices=("100tb", "1pb"), required=True)
    p.add_argument("--candidates", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--out-csv", type=Path, required=True)
    a = p.parse_args()
    if a.candidates <= 0:
        raise ValueError("--candidates must be positive")
    conn = psycopg2.connect(**DEFAULT_DB_CONFIG)
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    try:
        counts = source_counts(conn)
        print(json.dumps(counts), flush=True)
        if a.pool == "100tb":
            sampled, proposed = sample_exported_items(conn, a.candidates, counts, a.seed)
            rows = []
            for priority, row in enumerate(sampled):
                rows.append({"priority": priority, "pool": "100tb", **row})
            details = {"storage_id_proposals": proposed}
        else:
            rows, details = sample_1pb(conn, a.candidates, counts, a.seed)
    finally:
        conn.close()
    if len(rows) != a.candidates or len({r["priority"] for r in rows}) != len(rows):
        raise ValueError("Incorrect candidate count or repeated priority")
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if a.out_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {a.out_csv}")
    with a.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    source_counts_by_code = Counter(int(r["source_table_code"]) for r in rows)
    report = {"pool": a.pool, "seed": a.seed, "sampling_unit": "stored item" if a.pool == "100tb" else "ORI file image or 2P frame",
              "method": "global random permutation without replacement; no dataset quotas",
              "candidate_count": len(rows), "candidate_source_table_counts": source_counts_by_code,
              "universe": counts, "details": details, "manifest_csv": str(a.out_csv)}
    report_path = a.out_csv.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
