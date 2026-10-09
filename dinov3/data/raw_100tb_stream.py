"""Finite, rank-disjoint training stream from the original 100TB export tars."""

from __future__ import annotations

import csv
import itertools
import json
import logging
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

from .wds_decoder import _robust_per_channel, _to_float_tensor

logger = logging.getLogger("dinov3")


def decode_raw_item(data: bytes, item_id: int, channel_hint: int, target_channels: int) -> torch.Tensor:
    # Apply the exact crop and floating-point conversion used by the earlier
    # materializer; only the intermediate tar write/read is omitted.
    from scripts.materialize_global_100tb_wds import image_planes

    planes, _ = image_planes(data, str(item_id), channel_hint, 20260924)
    decoded = [_robust_per_channel(_to_float_tensor(plane[None])[0], 1.0, 99.0)
               for plane in planes[:target_channels]]
    h, w = decoded[0].shape
    result = torch.zeros(target_channels, h, w, dtype=torch.float32)
    if len(decoded) == 1:
        result[:] = decoded[0]
    else:
        for channel, plane in enumerate(decoded):
            result[channel] = plane
    return result


class Raw100TBStream(torch.utils.data.IterableDataset):
    def __init__(self, index_root: str, transform: Callable | None, *, target_channels: int,
                 seed: int, shuffle_buffer: int, skip_samples: int = 0):
        super().__init__()
        self.index_root = Path(index_root)
        self.transform = transform
        self.target_channels = target_channels
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.skip_samples = skip_samples
        if shuffle_buffer < 1 or skip_samples < 0:
            raise ValueError("Invalid shuffle buffer or resume skip")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers != 1:
            raise ValueError("Raw100TBStream requires one worker per DDP rank")
        assignment = json.loads((self.index_root / "assignment.json").read_text())
        rank = int(os.environ.get("RANK", "0"))
        if assignment["format"] != "100tb_raw_tar_stream_v1" or rank >= assignment["world_size"]:
            raise ValueError("Invalid 100TB streaming assignment or rank")
        assigned = assignment["rank_shards"][rank]
        wanted = defaultdict(dict)
        with (self.index_root / f"rank{rank:02d}.tsv").open(newline="") as source:
            for shard_id, item_id, hint, _priority in csv.reader(source, delimiter="\t"):
                shard_id, item_id = int(shard_id), int(item_id)
                if item_id in wanted[shard_id]:
                    raise ValueError(f"Repeated candidate item {item_id}")
                wanted[shard_id][item_id] = int(hint or 0)
        if sum(map(len, wanted.values())) != assignment["rank_sample_counts"][rank] or \
                set(wanted) != set(assigned):
            raise ValueError(f"Rank {rank} candidate index differs from assignment")
        rng = random.Random(self.seed + rank * 1_000_003)
        assigned = list(assigned)
        rng.shuffle(assigned)
        buffer = []
        failures = 0
        failure_root = os.environ.get("DQ_STREAM_FAILURE_LOG_DIR")
        failure_log = (Path(failure_root) / f"raw100tb_failures_rank{rank:02d}.jsonl").open("a", buffering=1) \
            if failure_root else None

        def record_failure(item_id: int, path: str, reason: str):
            nonlocal failures
            failures += 1
            if failure_log is not None:
                failure_log.write(json.dumps({"item_id": item_id, "shard": path, "reason": reason}) + "\n")

        def decoded_samples():
            for shard_id in assigned:
                path = assignment["shards"][shard_id]
                remaining = wanted.pop(shard_id)
                with tarfile.open(path, "r:") as source:
                    for member in source:
                        if not member.isfile():
                            continue
                        name = member.name.rsplit("/", 1)[-1].split(".", 1)[0]
                        if not name.isdigit():
                            continue
                        item_id = int(name)
                        if item_id not in remaining:
                            continue
                        hint = remaining.pop(item_id)
                        payload = source.extractfile(member)
                        if payload is None:
                            raise OSError(f"Cannot extract {item_id} from {path}")
                        buffer.append((item_id, hint, payload.read(), path))
                        if len(buffer) >= self.shuffle_buffer:
                            yield from drain_one()
                if remaining:
                    for item_id in remaining:
                        record_failure(item_id, path, "member_missing")
                    logger.warning("100TB shard %s missing %d selected items", path, len(remaining))
            while buffer:
                yield from drain_one()

        def drain_one():
            index = rng.randrange(len(buffer))
            item_id, hint, data, path = buffer[index]
            buffer[index] = buffer[-1]
            buffer.pop()
            try:
                image = decode_raw_item(data, item_id, hint, self.target_channels)
            except Exception as exc:
                record_failure(item_id, path, f"{type(exc).__name__}:{str(exc)[:200]}")
                logger.warning("100TB item %d failed to decode: %s", item_id, exc)
                return
            yield {"image": image, "__key__": str(item_id), "__url__": path}

        stream = decoded_samples()
        if self.skip_samples:
            stream = itertools.islice(stream, self.skip_samples, None)
        try:
            for sample in stream:
                if self.transform is None:
                    yield sample
                else:
                    transformed = self.transform(sample["image"])
                    transformed["__key__"] = sample["__key__"]
                    transformed["__url__"] = sample["__url__"]
                    yield transformed, ()
        finally:
            if failure_log is not None:
                failure_log.close()
            logger.info("100TB finite raw stream rank=%d finished; failures=%d", rank, failures)
