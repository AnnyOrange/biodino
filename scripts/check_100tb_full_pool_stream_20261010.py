#!/usr/bin/env python3
"""Check full-pool rank coverage, repeated passes, and a broken source tar."""

import csv
import io
import itertools
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch

from dinov3.data.raw_100tb_stream import Raw100TBFullPoolStream


def make_tar(path: Path, ids: list[int]):
    with tarfile.open(path, "w") as output:
        for item_id in ids:
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, np.full((32, 32), item_id, dtype=np.uint16))
            payload = buffer.getvalue()
            member = tarfile.TarInfo(f"{item_id:018d}.tiff")
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))


def main():
    with tempfile.TemporaryDirectory(prefix="raw100tb_fullpool_") as directory:
        root = Path(directory)
        paths = [root / f"{name}.tar" for name in "abc"]
        for path, ids in zip(paths, ([1, 2], [3, 4], [5, 6])):
            make_tar(path, ids)
        index = root / "index"
        index.mkdir()
        assignment = {
            "status": "PASS", "format": "100tb_full_pool_stream_v1", "world_size": 2,
            "total_items": 6, "rank_counts": [4, 2],
            "shards": {str(i): str(path) for i, path in enumerate(paths)},
            "rank_shards": [[0, 1], [2]],
        }
        (index / "assignment.json").write_text(json.dumps(assignment))
        for rank, rows in enumerate(([(0, 1), (0, 2), (1, 3), (1, 4)], [(2, 5), (2, 6)])):
            with (index / f"rank{rank:02d}.tsv").open("w", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerows((shard, item, 1) for shard, item in rows)
        for rank, expected in ((0, {1, 2, 3, 4}), (1, {5, 6})):
            os.environ["RANK"] = str(rank)
            stream = Raw100TBFullPoolStream(str(index), None, target_channels=3,
                                            seed=41, shuffle_buffer=2, active_shards=2)
            samples = [int(sample["__key__"]) for sample in itertools.islice(stream, 2 * len(expected))]
            assert set(samples[:len(expected)]) == set(samples[len(expected):]) == expected
        os.environ["RANK"] = "0"
        stream = Raw100TBFullPoolStream(str(index), None, target_channels=3,
                                        seed=41, shuffle_buffer=2, active_shards=2)
        loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=2)
        samples = [int(sample["__key__"]) for sample in itertools.islice(loader, 4)]
        assert set(samples) == {1, 2, 3, 4}
        make_tar(paths[0], [])
        os.environ["RANK"] = "0"
        os.environ["DQ_STREAM_FAILURE_LOG_DIR"] = str(root)
        try:
            stream = Raw100TBFullPoolStream(str(index), None, target_channels=3,
                                            seed=41, shuffle_buffer=2, active_shards=2)
            samples = [int(sample["__key__"]) for sample in itertools.islice(stream, 4)]
        finally:
            os.environ.pop("DQ_STREAM_FAILURE_LOG_DIR", None)
        assert set(samples) == {3, 4}
        assert "member_missing" in (root / "raw100tb_full_pool_failures_rank00_worker00.jsonl").read_text()
        make_tar(paths[0], [1, 2])
        with paths[0].open("r+b") as broken:
            broken.truncate(512 + 64)
        stream = Raw100TBFullPoolStream(str(index), None, target_channels=3,
                                        seed=41, shuffle_buffer=2, active_shards=2)
        samples = [int(sample["__key__"]) for sample in itertools.islice(stream, 4)]
        assert set(samples) == {3, 4}
    print("PASS: full-pool rank coverage, repeated passes, and broken-tar continuation")


if __name__ == "__main__":
    main()
