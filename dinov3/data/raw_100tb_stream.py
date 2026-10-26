"""Finite, rank-disjoint training stream from the original 100TB export tars."""

from __future__ import annotations

import csv
import io
import itertools
import json
import logging
import math
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import tifffile

from .wds_decoder import _robust_per_channel, _to_float_tensor

logger = logging.getLogger("dinov3")
MAX_FULL_POOL_DECODED_BYTES = 512 * 1024 * 1024


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
        if assignment["format"] != "100tb_raw_tar_stream_v2" or rank >= assignment["world_size"]:
            raise ValueError("Invalid 100TB streaming assignment or rank")
        primary_shards = assignment["rank_shards"][rank]
        backup_shards = assignment["rank_backup_shards"][rank]
        primary = defaultdict(dict)
        backup = defaultdict(dict)
        with (self.index_root / f"rank{rank:02d}.tsv").open(newline="") as source:
            for shard_id, item_id, hint, priority in csv.reader(source, delimiter="\t"):
                shard_id, item_id = int(shard_id), int(item_id)
                wanted = primary if int(priority) < assignment["target_total"] else backup
                if item_id in wanted[shard_id]:
                    raise ValueError(f"Repeated candidate item {item_id}")
                wanted[shard_id][item_id] = int(hint or 0)
        if sum(map(len, primary.values())) != assignment["rank_primary_counts"][rank] or \
                sum(map(len, backup.values())) != assignment["rank_backup_counts"][rank] or \
                set(primary) != set(primary_shards) or set(backup) != set(backup_shards):
            raise ValueError(f"Rank {rank} candidate index differs from assignment")
        rng = random.Random(self.seed + rank * 1_000_003)
        primary_shards = list(primary_shards)
        backup_shards = list(backup_shards)
        rng.shuffle(primary_shards)
        rng.shuffle(backup_shards)
        buffer = []
        failures = 0
        successes = 0
        failure_root = os.environ.get("DQ_STREAM_FAILURE_LOG_DIR")
        failure_log = (Path(failure_root) / f"raw100tb_failures_rank{rank:02d}.jsonl").open("a", buffering=1) \
            if failure_root else None

        def record_failure(item_id: int, path: str, reason: str):
            nonlocal failures
            failures += 1
            if failure_log is not None:
                failure_log.write(json.dumps({"item_id": item_id, "shard": path, "reason": reason}) + "\n")

        def read_phase(shards, selected, buffer_limit):
            for shard_id in shards:
                path = assignment["shards"][shard_id]
                remaining = selected.pop(shard_id)
                try:
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
                            try:
                                payload = source.extractfile(member)
                                if payload is None:
                                    raise OSError(f"Cannot extract {item_id} from {path}")
                                data = payload.read()
                            except (OSError, tarfile.TarError, EOFError):
                                remaining[item_id] = hint
                                raise
                            buffer.append((item_id, hint, data, path))
                            if len(buffer) >= buffer_limit:
                                yield from drain_one()
                except (OSError, tarfile.TarError, EOFError) as exc:
                    logger.warning("100TB shard %s stopped after read error: %s", path, exc)
                    for item_id in remaining:
                        record_failure(item_id, path, f"shard_read_error:{type(exc).__name__}:{str(exc)[:160]}")
                    continue
                if remaining:
                    for item_id in remaining:
                        record_failure(item_id, path, "member_missing")
                    logger.warning("100TB shard %s missing %d selected items", path, len(remaining))
            while buffer:
                yield from drain_one()

        def drain_one():
            nonlocal successes
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
            successes += 1
            yield {"image": image, "__key__": str(item_id), "__url__": path}

        def decoded_samples():
            yield from read_phase(primary_shards, primary, self.shuffle_buffer)
            if successes < assignment["target_per_rank"]:
                needed = assignment["target_per_rank"] - successes
                logger.warning("100TB rank %d needs %d backup samples", rank, needed)
                yield from itertools.islice(read_phase(backup_shards, backup, 1), needed)
            if successes != assignment["target_per_rank"]:
                raise RuntimeError(f"100TB rank {rank} yielded {successes} samples, "
                                   f"expected {assignment['target_per_rank']}")

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


class Raw100TBRepeatedStream(torch.utils.data.IterableDataset):
    """Repeat the audited random subset, reshuffling independently on each pass."""

    def __init__(self, index_root: str, transform: Callable | None, *, target_channels: int,
                 seed: int, shuffle_buffer: int, skip_samples: int = 0):
        super().__init__()
        self.index_root = index_root
        self.transform = transform
        self.target_channels = target_channels
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.skip_samples = skip_samples
        if skip_samples < 0:
            raise ValueError("Invalid resume skip")

    def __iter__(self):
        assignment = json.loads((Path(self.index_root) / "assignment.json").read_text())
        rank = int(os.environ.get("RANK", "0"))
        per_pass = assignment["target_per_rank"]
        pass_index, within_pass = divmod(self.skip_samples, per_pass)
        while True:
            stream = Raw100TBStream(
                self.index_root, self.transform,
                target_channels=self.target_channels,
                seed=self.seed + pass_index * 1_000_000_007,
                shuffle_buffer=self.shuffle_buffer,
                skip_samples=within_pass,
            )
            logger.info("100TB rank=%d random pass=%d resume_skip=%d", rank, pass_index, within_pass)
            yield from stream
            pass_index += 1
            within_pass = 0


class Raw100TBFullPoolStream(torch.utils.data.IterableDataset):
    """Continuously mix all exported items across random, rank-disjoint tar shards."""

    def __init__(self, index_root: str, transform: Callable | None, *, target_channels: int,
                 seed: int, shuffle_buffer: int, active_shards: int = 4):
        super().__init__()
        self.index_root = Path(index_root)
        self.transform = transform
        self.target_channels = target_channels
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.active_shards = active_shards
        if shuffle_buffer < 1 or active_shards < 1:
            raise ValueError("Invalid online shuffle size")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        assignment = json.loads((self.index_root / "assignment.json").read_text())
        rank = int(os.environ.get("RANK", "0"))
        if assignment["status"] != "PASS" or assignment["format"] != "100tb_full_pool_stream_v1" or \
                rank >= assignment["world_size"]:
            raise ValueError("Invalid 100TB full-pool assignment or rank")
        all_selected = defaultdict(dict)
        with (self.index_root / f"rank{rank:02d}.tsv").open(newline="") as source:
            for shard_id, item_id, hint in csv.reader(source, delimiter="\t"):
                shard_id, item_id = int(shard_id), int(item_id)
                if item_id in all_selected[shard_id]:
                    raise ValueError(f"Repeated full-pool item {item_id}")
                all_selected[shard_id][item_id] = int(hint)
        all_shards = assignment["rank_shards"][rank]
        if sum(map(len, all_selected.values())) != assignment["rank_counts"][rank] or \
                set(all_selected) != set(all_shards):
            raise ValueError(f"Rank {rank} full-pool index differs from assignment")
        worker_loads = [0] * worker_count
        owners = {}
        for shard_id in sorted(all_shards, key=lambda key: (-len(all_selected[key]), key)):
            owner = min(range(worker_count), key=lambda candidate: (worker_loads[candidate], candidate))
            owners[shard_id] = owner
            worker_loads[owner] += len(all_selected[shard_id])
        shard_ids = [shard_id for shard_id in all_shards if owners[shard_id] == worker_id]
        selected = {shard_id: all_selected[shard_id] for shard_id in shard_ids}
        del all_selected
        rng = random.Random(self.seed + rank * 1_000_003 + worker_id * 1_000_000_007)
        failure_root = os.environ.get("DQ_STREAM_FAILURE_LOG_DIR")
        failure_log = (Path(failure_root) / f"raw100tb_full_pool_failures_rank{rank:02d}_worker{worker_id:02d}.jsonl").open(
            "a", buffering=1) if failure_root else None
        buffer = []

        def record_failure(item_id: int | None, path: str, reason: str, count: int = 1):
            if failure_log is not None:
                failure_log.write(json.dumps({"item_id": item_id, "shard": path,
                                              "reason": reason, "count": count}) + "\n")

        def read_shard(shard_id: int):
            path = assignment["shards"][str(shard_id)]
            remaining = selected[shard_id].copy()
            try:
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
                        if member.size > MAX_FULL_POOL_DECODED_BYTES:
                            record_failure(item_id, path, f"raw_item_too_large:{member.size}")
                            continue
                        try:
                            payload = source.extractfile(member)
                            if payload is None:
                                raise OSError(f"Cannot extract {item_id} from {path}")
                            data = payload.read()
                        except (OSError, tarfile.TarError, EOFError):
                            remaining[item_id] = hint
                            raise
                        try:
                            with tifffile.TiffFile(io.BytesIO(data)) as tiff:
                                series = tiff.series[0]
                                decoded_bytes = math.prod(series.shape) * series.dtype.itemsize
                            if decoded_bytes > MAX_FULL_POOL_DECODED_BYTES:
                                raise ValueError(f"decoded_image_too_large:{decoded_bytes}")
                            image = decode_raw_item(data, item_id, hint, self.target_channels)
                        except Exception as exc:
                            record_failure(item_id, path, f"{type(exc).__name__}:{str(exc)[:200]}")
                            logger.warning("100TB full-pool item %d failed to decode: %s", item_id, exc)
                            continue
                        del data, tiff, series
                        yield item_id, image, path
            except (OSError, tarfile.TarError, EOFError) as exc:
                logger.warning("100TB full-pool shard %s stopped after read error: %s", path, exc)
                record_failure(None, path, f"shard_read_error:{type(exc).__name__}:{str(exc)[:160]}",
                               len(remaining))
                return
            if remaining:
                record_failure(None, path, "member_missing", len(remaining))
                logger.warning("100TB full-pool shard %s missing %d items", path, len(remaining))

        def drain_one():
            index = rng.randrange(len(buffer))
            item_id, image, path = buffer[index]
            buffer[index] = buffer[-1]
            buffer.pop()
            sample = {"image": image, "__key__": str(item_id), "__url__": path}
            if self.transform is None:
                return sample
            transformed = self.transform(image)
            transformed["__key__"] = sample["__key__"]
            transformed["__url__"] = sample["__url__"]
            return transformed, ()

        try:
            pass_index = 0
            while True:
                order = list(shard_ids)
                rng.shuffle(order)
                pending = iter(order)
                active = []
                successful = 0

                def refill():
                    while len(active) < self.active_shards:
                        try:
                            active.append(read_shard(next(pending)))
                        except StopIteration:
                            break

                refill()
                while active:
                    index = rng.randrange(len(active))
                    try:
                        buffer.append(next(active[index]))
                    except StopIteration:
                        active.pop(index)
                        refill()
                        continue
                    if len(buffer) >= self.shuffle_buffer:
                        sample = drain_one()
                        if sample is not None:
                            successful += 1
                            yield sample
                while buffer:
                    sample = drain_one()
                    if sample is not None:
                        successful += 1
                        yield sample
                logger.info("100TB full-pool rank=%d worker=%d pass=%d successful=%d indexed=%d",
                            rank, worker_id, pass_index, successful, worker_loads[worker_id])
                if not successful:
                    raise RuntimeError(f"100TB full-pool rank {rank} read no images in pass {pass_index}")
                pass_index += 1
        finally:
            if failure_log is not None:
                failure_log.close()
