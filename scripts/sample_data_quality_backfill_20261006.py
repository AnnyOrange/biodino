#!/usr/bin/env python3
"""Draw an ordered, disjoint reserve for unreadable 20TB route-1 candidates."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path("/mnt/deepcad_nfs/deepcad_100t/projection_match_20_25tb/selected_20tb.parquet")
OUT = Path("/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006")
INITIAL = OUT / "selected_1m.parquet"
BACKFILL = OUT / "backfill_selected.parquet"
SEED = 20261006
SIZE = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if BACKFILL.exists() or (OUT / "backfill_selection.json").exists():
        raise FileExistsError("Backfill selection already exists")
    source = pq.ParquetFile(SOURCE)
    initial = pq.read_table(INITIAL, columns=["pool_row_index"])["pool_row_index"].to_pylist()
    if len(initial) != 1_000_000 or len(set(initial)) != len(initial):
        raise ValueError("Initial random sample is not one million unique pool rows")
    initial_set = set(initial)
    rng = random.Random(SEED)
    reserve = []
    reserve_set = set()
    while len(reserve) < SIZE:
        index = rng.randrange(source.metadata.num_rows)
        if index not in initial_set and index not in reserve_set:
            reserve.append(index)
            reserve_set.add(index)
    priorities = {index: priority for priority, index in enumerate(reserve)}
    ordered = sorted(reserve)
    schema = source.schema_arrow.append(pa.field("pool_row_index", pa.int64()))
    schema = schema.append(pa.field("backfill_priority", pa.int64()))
    temporary = BACKFILL.with_suffix(".parquet.part")
    cursor = 0
    count = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=32768):
            end = cursor + batch.num_rows
            stop = bisect.bisect_left(ordered, end, count)
            if stop > count:
                selected = ordered[count:stop]
                subset = batch.take(pa.array([index - cursor for index in selected], type=pa.int64()))
                subset = subset.append_column("pool_row_index", pa.array(selected, type=pa.int64()))
                subset = subset.append_column("backfill_priority", pa.array(
                    [priorities[index] for index in selected], type=pa.int64()))
                writer.write_batch(subset)
                count = stop
            cursor = end
    if count != SIZE or cursor != source.metadata.num_rows:
        raise ValueError("Incomplete backfill selection")
    os.replace(temporary, BACKFILL)
    digest = hashlib.sha256()
    for index in reserve:
        digest.update(struct.pack("<Q", index))
    record = {
        "status": "PASS",
        "algorithm": "ordered rejection sampling without replacement from initial complement",
        "seed": SEED,
        "reserve_size": SIZE,
        "pool_size": source.metadata.num_rows,
        "initial_manifest_sha256": sha256(INITIAL),
        "source_manifest_sha256": sha256(SOURCE),
        "backfill_manifest_sha256": sha256(BACKFILL),
        "reserve_order_sha256_le64": digest.hexdigest(),
    }
    path = OUT / "backfill_selection.json"
    temporary_record = path.with_suffix(".json.part")
    temporary_record.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary_record, path)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
