#!/usr/bin/env python3
"""Select one million patch occurrences uniformly from a complete Parquet pool."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import struct
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--expected-pool-size", type=int, required=True)
    args = parser.parse_args()

    source = pq.ParquetFile(args.source)
    pool_size = source.metadata.num_rows
    if pool_size != args.expected_pool_size or not 0 < args.size <= pool_size:
        raise ValueError(f"Unexpected pool/sample size: {pool_size}/{args.size}")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = args.output / "selected_1m.parquet"
    manifest = args.output / "selection.json"
    if selected.exists() or manifest.exists():
        raise FileExistsError("Selection already exists; refusing to replace it")

    indices = sorted(random.Random(args.seed).sample(range(pool_size), args.size))
    index_hash = hashlib.sha256()
    for index in indices:
        index_hash.update(struct.pack("<Q", index))
    schema = source.schema_arrow.append(pa.field("pool_row_index", pa.int64()))
    temporary = selected.with_suffix(".parquet.part")
    count = 0
    estimated_bytes = 0
    cursor = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=32768):
            end = cursor + batch.num_rows
            stop = bisect.bisect_left(indices, end, count)
            if stop > count:
                chosen = indices[count:stop]
                subset = batch.take(pa.array([i - cursor for i in chosen], type=pa.int64()))
                subset = subset.append_column("pool_row_index", pa.array(chosen, type=pa.int64()))
                writer.write_batch(subset)
                estimated_bytes += sum(int(x or 0) for x in subset.column(
                    subset.schema.get_field_index("estimated_bytes")
                ).to_pylist())
                count = stop
            cursor = end
    if count != args.size or cursor != pool_size:
        raise ValueError(f"Incomplete selection: {count}/{args.size}, pool {cursor}/{pool_size}")
    os.replace(temporary, selected)
    record = {
        "status": "PASS",
        "sampling_unit": "one selected image patch occurrence per Parquet row",
        "algorithm": "random.Random(seed).sample(range(pool_size), size)",
        "seed": args.seed,
        "pool_size": pool_size,
        "sample_size": count,
        "estimated_sample_bytes": estimated_bytes,
        "source_manifest": str(args.source),
        "source_manifest_sha256": sha256(args.source),
        "selected_manifest": str(selected),
        "selected_manifest_sha256": sha256(selected),
        "sampled_pool_row_indices_sha256_le64": index_hash.hexdigest(),
        "created_unix": time.time(),
    }
    temporary_manifest = manifest.with_suffix(".json.part")
    temporary_manifest.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary_manifest, manifest)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
