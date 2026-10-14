"""Online reader for filesystem-inventoried raw 1PB TIFF images and frames."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .wds_decoder import _robust_per_channel, _to_float_tensor

logger = logging.getLogger("dinov3")
MAX_DECODED_SHUFFLE_BUFFER = 32


def _decode_frame(frame: np.ndarray, identity: str, channel_hint: int,
                  target_channels: int, seed: int) -> torch.Tensor:
    from scripts.materialize_global_100tb_wds import array_planes

    planes, _ = array_planes(frame, identity, channel_hint, seed)
    decoded = [_robust_per_channel(_to_float_tensor(plane[None])[0], 1.0, 99.0)
               for plane in planes[:target_channels]]
    if not decoded:
        raise ValueError("no_decodable_channels")
    height, width = decoded[0].shape
    result = torch.zeros(target_channels, height, width, dtype=torch.float32)
    if len(decoded) == 1:
        result[:] = decoded[0]
    else:
        for channel, image in enumerate(decoded):
            result[channel] = image
    return result


class Raw1PBStream(torch.utils.data.IterableDataset):
    """Shuffle files and frames online; never materialize image bytes to disk."""

    def __init__(self, index_path: str, transform: Callable | None, *, target_channels: int,
                 seed: int, shuffle_buffer: int = 32):
        super().__init__()
        self.index_path = Path(index_path)
        self.transform = transform
        self.target_channels = int(target_channels)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        if self.target_channels < 1 or self.shuffle_buffer < 1:
            raise ValueError("target_channels and shuffle_buffer must be positive")
        if self.shuffle_buffer > MAX_DECODED_SHUFFLE_BUFFER:
            logger.warning("raw1pb shuffle buffer capped at %d decoded samples (requested %d)",
                           MAX_DECODED_SHUFFLE_BUFFER, self.shuffle_buffer)
            self.shuffle_buffer = MAX_DECODED_SHUFFLE_BUFFER

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rank = int(os.environ.get("RANK", "0"))
        connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM inventory_meta"))
            if metadata.get("status") != "PASS":
                raise ValueError("Raw 1PB inventory is incomplete; run the inventory builder first")
            world_size = int(metadata["world_size"])
            if not 0 <= rank < world_size:
                raise ValueError(f"DDP rank {rank} exceeds inventory world_size={world_size}")
            files = list(connection.execute(
                "SELECT path,frame_count,channel_hint FROM files "
                "WHERE readable=1 AND rank=? ORDER BY path", (rank,)))
        finally:
            connection.close()
        if not files:
            raise RuntimeError(f"No readable raw 1PB TIFF files assigned to rank {rank}")
        files = [row for row in files if zlib.crc32(row[0].encode()) % worker_count == worker_id]
        if not files:
            raise RuntimeError(f"No raw 1PB TIFF files assigned to rank {rank} worker {worker_id}/{worker_count}")

        from scripts.probe_global_1pb_frames import read_frame_from_tif
        pass_index = 0
        failure_root = os.environ.get("RAW1PB_FAILURE_LOG_DIR")
        failure_path = (Path(failure_root) /
                        f"raw1pb_failures_rank{rank:02d}_worker{worker_id:02d}.jsonl") if failure_root else None
        if failure_path:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            rng = random.Random(self.seed + rank * 1_000_003 + worker_id * 1_000_000_007
                                + pass_index * 1_000_000_009)
            order = list(files)
            rng.shuffle(order)
            buffer: list[tuple[torch.Tensor, str, str]] = []
            successful = 0

            def report_failure(path: str, frame_index: int | None, reason: str) -> None:
                logger.warning("raw1pb read failed path=%s frame=%s: %s", path, frame_index, reason)
                if failure_path:
                    with failure_path.open("a", buffering=1) as output:
                        output.write(json.dumps({"path": path, "frame_idx": frame_index,
                                                 "reason": reason, "pass": pass_index,
                                                 "time_unix": time.time()}) + "\n")

            def emit_one():
                nonlocal successful
                index = rng.randrange(len(buffer))
                image, key, path = buffer[index]
                buffer[index] = buffer[-1]
                buffer.pop()
                successful += 1
                if self.transform is None:
                    return {"image": image, "__key__": key, "__url__": path}
                sample = self.transform(image)
                sample["__key__"] = key
                sample["__url__"] = path
                return sample, ()

            for path, frame_count, channel_hint in order:
                try:
                    import tifffile
                    with tifffile.TiffFile(path) as tif:
                        for frame_index in rng.sample(range(frame_count), frame_count):
                            try:
                                frame, _ = read_frame_from_tif(tif, frame_index, channel_hint)
                                key = hashlib.blake2b(
                                    f"{path}|{frame_index}".encode(), digest_size=12).hexdigest()
                                image = _decode_frame(frame, key, channel_hint,
                                                      self.target_channels,
                                                      self.seed + pass_index * 1_000_000_007)
                                buffer.append((image, key, path))
                                if len(buffer) >= self.shuffle_buffer:
                                    yield emit_one()
                            except Exception as exc:
                                report_failure(path, frame_index,
                                               f"{type(exc).__name__}:{str(exc)[:180]}")
                except Exception as exc:
                    report_failure(path, None, f"{type(exc).__name__}:{str(exc)[:180]}")
            while buffer:
                yield emit_one()
            logger.info("raw1pb rank=%d worker=%d/%d pass=%d files=%d samples=%d",
                        rank, worker_id, worker_count, pass_index, len(order), successful)
            if successful == 0:
                raise RuntimeError(f"Raw 1PB rank {rank} produced no samples in pass {pass_index}")
            pass_index += 1
