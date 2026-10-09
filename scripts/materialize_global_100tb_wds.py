#!/usr/bin/env python3
"""Materialize a global-priority 100TB sample from exported storage tar files.

This stage preserves every candidate's priority.  A separate finalization step
must select the first 1M successful priorities; worker output is not training
data until that step has completed and passed its audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import tarfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile


def worker_number(path: str, count: int) -> int:
    return zlib.crc32(path.encode()) % count


def array_planes(arr: np.ndarray, source_id: str, channel_hint: int, seed: int):
    arr = np.squeeze(np.asarray(arr))
    source_dtype = str(arr.dtype)
    conversion = "none"
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("nonfinite_float")
        # The packed decoder clips floating TIFFs to [0,1].  Original 2P
        # exports often contain float intensity counts far above 1, so that
        # decoder would destroy contrast.  Quantize on the documented uint16
        # count scale, recording which rule was applied for audit.
        maximum = float(np.max(arr))
        if maximum <= 1.0 and float(np.min(arr)) >= 0.0:
            arr = np.rint(arr * 65535.0).astype(np.uint16)
            conversion = "normalized_float_to_uint16"
        else:
            arr = np.rint(np.clip(arr, 0, 65535)).astype(np.uint16)
            conversion = "float_counts_clipped_to_uint16"
    if arr.ndim == 2:
        planes = [arr]
    elif arr.ndim == 3:
        axes = [axis for axis, size in enumerate(arr.shape) if size == channel_hint and 1 <= size <= 8]
        if not axes:
            axes = [axis for axis, size in enumerate(arr.shape) if 1 <= size <= 8]
        if not axes:
            raise ValueError(f"unsupported_3d_shape:{arr.shape}")
        hwc = np.moveaxis(arr, axes[0], -1)
        planes = [hwc[..., ch] for ch in range(min(3, hwc.shape[-1]))]
    else:
        raise ValueError(f"unsupported_shape:{arr.shape}")
    if any(x.ndim != 2 for x in planes):
        raise ValueError(f"non_2d_channel:{arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"non_integer_dtype:{arr.dtype}")
    h, w = planes[0].shape
    if h < 16 or w < 16:
        raise ValueError(f"tiny_image:{h}x{w}")
    digest = hashlib.blake2b(f"{seed}|{source_id}".encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "little")
    y0 = draw % (max(0, h - 512) + 1)
    x0 = (draw >> 32) % (max(0, w - 512) + 1)
    y1, x1 = min(h, y0 + 512), min(w, x0 + 512)
    cropped = [np.ascontiguousarray(x[y0:y1, x0:x1]) for x in planes]
    return cropped, {"original_shape": list(arr.shape), "source_dtype": source_dtype,
                     "wds_dtype": str(arr.dtype), "conversion": conversion,
                     "crop_yx": [y0, x0, y1, x1], "channels_written": len(cropped)}


def image_planes(data: bytes, source_id: str, channel_hint: int, seed: int):
    return array_planes(tifffile.imread(io.BytesIO(data)), source_id, channel_hint, seed)


class TarWriter:
    def __init__(self, root: Path, worker_id: int, samples_per_shard: int):
        self.root = root
        self.worker_id = worker_id
        self.samples_per_shard = samples_per_shard
        self.count = 0
        self.shard = -1
        self.tar = None

    def add(self, key: str, planes: list[np.ndarray], meta: dict):
        if self.count % self.samples_per_shard == 0:
            self.close()
            self.shard += 1
            path = self.root / f"staged_w{self.worker_id:02d}_{self.shard:05d}.tar"
            self.tar = tarfile.open(path, "w")
        assert self.tar is not None
        for ch, plane in enumerate(planes, 1):
            buf = io.BytesIO()
            tifffile.imwrite(buf, plane, photometric="minisblack")
            payload = buf.getvalue()
            info = tarfile.TarInfo(f"{key}.ch{ch}.tif")
            info.size = len(payload)
            self.tar.addfile(info, io.BytesIO(payload))
        payload = json.dumps(meta, separators=(",", ":")).encode()
        info = tarfile.TarInfo(f"{key}.meta.json")
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))
        self.count += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def run_worker(args):
    wid, workers, csv_path, max_candidates, out_root, samples_per_shard, seed = args
    grouped = defaultdict(dict)
    direct = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            priority = int(row["priority"])
            if priority >= max_candidates:
                break
            path = row["shard_path"]
            if worker_number(path, workers) == wid:
                if row["source_table_code"] == "1":
                    direct.append(row)
                else:
                    grouped[path][int(row["storage_item_id"])] = row
    writer = TarWriter(out_root, wid, samples_per_shard)
    status_path = out_root / f"status_w{wid:02d}.csv"
    seen = 0
    success = 0
    with status_path.open("w", newline="") as handle:
        status = csv.writer(handle)
        status.writerow(["priority", "storage_item_id", "status", "reason"])

        def record(row, data, source_location, member_name=None):
            nonlocal success
            item_id = int(row["storage_item_id"])
            priority = int(row["priority"])
            try:
                planes, image_info = image_planes(
                    data, str(item_id), int(row["channel_count"] or 0), seed)
                key = f"p{priority:08d}"
                meta = {"pool": "100tb", "priority": priority,
                        "storage_item_id": item_id,
                        "source_table_code": int(row["source_table_code"]),
                        "source_id": int(row["source_id"]),
                        "frame_idx": int(row["frame_idx"]) if row["frame_idx"] else None,
                        "source_shard": row["shard_path"],
                        "source_path": row["file_path"],
                        "read_from": source_location,
                        "source_member": member_name, **image_info}
                writer.add(key, planes, meta)
                status.writerow([priority, item_id, "success", ""])
                success += 1
            except Exception as exc:
                status.writerow([priority, item_id, "failed", f"{type(exc).__name__}:{str(exc)[:150]}"])

        # The 100TB ORI exporter copies source bytes verbatim into its tar.
        # Direct source reads avoid indexing thousands of multi-GB tar files.
        # A missing source falls back to the exported tar copy below.
        for row in direct:
            seen += 1
            try:
                with open(row["file_path"], "rb") as original:
                    payload = original.read()
                record(row, payload, "original_source")
            except Exception:
                grouped[row["shard_path"]][int(row["storage_item_id"])] = row
            if seen % 10000 == 0:
                print(f"worker={wid} candidates={seen} success={success} phase=direct", flush=True)

        for shard_path, wanted in grouped.items():
            seen += sum(row["source_table_code"] != "1" for row in wanted.values())
            try:
                with tarfile.open(shard_path, "r:") as source:
                    members = {}
                    for member in source.getmembers():
                        try:
                            item_id = int(Path(member.name).stem)
                        except ValueError:
                            continue
                        if item_id in wanted and member.isfile():
                            members[item_id] = member
                    for item_id, row in wanted.items():
                        member = members.get(item_id)
                        if member is None:
                            status.writerow([row["priority"], item_id, "failed", "member_missing"])
                            continue
                        try:
                            extracted = source.extractfile(member)
                            if extracted is None:
                                raise ValueError("member_unreadable")
                            record(row, extracted.read(), "exported_tar", member.name)
                        except Exception as exc:
                            status.writerow([row["priority"], item_id, "failed", f"{type(exc).__name__}:{str(exc)[:150]}"])
            except Exception as exc:
                for item_id, row in wanted.items():
                    status.writerow([row["priority"], item_id, "failed", f"shard_{type(exc).__name__}:{str(exc)[:150]}"])
            if seen % 10000 < len(wanted):
                print(f"worker={wid} candidates={seen} success={success} source_shards_done={len(grouped)}", flush=True)
    writer.close()
    return {"worker": wid, "candidates": seen, "success": success, "source_shards": len(grouped),
            "staged_shards": writer.shard + 1, "status_csv": str(status_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--samples-per-shard", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260924)
    a = p.parse_args()
    if a.out_root.exists() and any(a.out_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {a.out_root}")
    a.out_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    jobs = [(i, a.workers, a.candidates, a.max_candidates, a.out_root,
             a.samples_per_shard, a.seed) for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        results = pool.map(run_worker, jobs)
    report = {"pool": "100tb", "sampling": "full exported storage pool priority order",
              "max_candidates": a.max_candidates, "seed": a.seed,
              "results": results, "elapsed_seconds": time.monotonic() - started}
    (a.out_root / "materialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
