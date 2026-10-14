#!/usr/bin/env python3
"""Check exact primary selection, backup use, and deterministic resume locally."""

import csv
import io
import itertools
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from dinov3.data.raw_100tb_stream import Raw100TBRepeatedStream, Raw100TBStream
from scripts.build_100tb_stream_index_20261009 import build


def make_tar(path: Path, ids: list[int]):
    with tarfile.open(path, "w") as output:
        for item_id in ids:
            image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + item_id
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, image)
            payload = buffer.getvalue()
            member = tarfile.TarInfo(f"{item_id:018d}.tiff")
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))


def collect(index: Path, rank: int, skip: int = 0):
    os.environ["RANK"] = str(rank)
    dataset = Raw100TBStream(str(index), None, target_channels=3, seed=41,
                             shuffle_buffer=2, skip_samples=skip)
    return [int(sample["__key__"]) for sample in dataset]


def main():
    with tempfile.TemporaryDirectory(prefix="raw100tb_exact_") as directory:
        root = Path(directory)
        shards = [root / "a.tar", root / "b.tar", root / "c.tar"]
        make_tar(shards[0], [1, 2, 3])
        make_tar(shards[1], [4])
        make_tar(shards[2], [5, 6])
        candidates = root / "candidates.csv"
        with candidates.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["priority", "storage_item_id", "shard_path", "channel_count"])
            writer.writeheader()
            for priority, item_id in enumerate(range(1, 7)):
                writer.writerow({"priority": priority, "storage_item_id": item_id,
                                 "shard_path": str(shards[0 if item_id <= 3 else 1 if item_id == 4 else 2]),
                                 "channel_count": 1})
        index = root / "index"
        build(candidates, index, world_size=2, target_total=4)
        first = collect(index, 0)
        second = collect(index, 1)
        assert len(first) == len(second) == 2
        assert set(first + second) == {1, 2, 3, 4}
        assert collect(index, 0, skip=1) == first[1:]
        assert collect(index, 1, skip=1) == second[1:]
        os.environ["RANK"] = "0"
        repeated = Raw100TBRepeatedStream(str(index), None, target_channels=3, seed=41,
                                          shuffle_buffer=2)
        two_passes = [int(sample["__key__"]) for sample in itertools.islice(repeated, 4)]
        assert set(two_passes[:2]) == set(two_passes[2:]) == set(first)
        resumed = Raw100TBRepeatedStream(str(index), None, target_channels=3, seed=41,
                                         shuffle_buffer=2, skip_samples=3)
        assert [int(sample["__key__"]) for sample in itertools.islice(resumed, 1)] == two_passes[3:4]
        # Removing one selected member must consume a later priority as backup.
        affected = next(rank for rank, keys in enumerate((first, second)) if 4 in keys)
        make_tar(shards[1], [])
        repaired = collect(index, affected)
        assert len(repaired) == 2 and 4 not in repaired
        assert any(item_id in (5, 6) for item_id in repaired)
        make_tar(shards[1], [4])
        with shards[1].open("r+b") as broken:
            broken.truncate(512 + 64)
        os.environ["DQ_STREAM_FAILURE_LOG_DIR"] = str(root)
        try:
            repaired = collect(index, affected)
        finally:
            os.environ.pop("DQ_STREAM_FAILURE_LOG_DIR", None)
        assert len(repaired) == 2 and 4 not in repaired
        assert any(item_id in (5, 6) for item_id in repaired)
        failure_log = (root / f"raw100tb_failures_rank{affected:02d}.jsonl").read_text()
        assert "shard_read_error:ReadError" in failure_log
    print("PASS: exact primary set, deterministic resume, missing and truncated tar backup replacement")


if __name__ == "__main__":
    main()
