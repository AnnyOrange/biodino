#!/usr/bin/env python3
"""Scan every WDS image for exact pre-robust decoder RGB mean/std.

Unsigned TIFFs use dtype-max scaling; signed TIFFs use the per-plane min/max
rule in wds_decoder.py; floating TIFFs clip to [0,1].  Source dtypes and any
materialization conversions are reported separately so these numbers are not
mistaken for a common physical-intensity scale.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from pathlib import Path
import re
import tarfile

import numpy as np
import tifffile

CHANNEL = re.compile(r"^(?P<key>.+)\.ch(?P<channel>\d+)\.tiff?$", re.IGNORECASE)


def resolve_shards(root: Path) -> list[Path]:
    return sorted(root.glob("filtered_mixed_train*.tar"))


def iter_tar_samples(path: Path):
    key = None
    members = {}
    meta = None
    with tarfile.open(path, "r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            matched = CHANNEL.match(member.name)
            if matched:
                next_key = matched.group("key")
                channel = int(matched.group("channel"))
            elif member.name.endswith(".meta.json"):
                next_key = member.name[:-len(".meta.json")]
                channel = None
            else:
                continue
            if key is not None and next_key != key:
                yield key, members, meta
                members, meta = {}, None
            key = next_key
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable tar member: {path}:{member.name}")
            payload = stream.read()
            if channel is None:
                meta = payload
            else:
                members[channel] = payload
        if key is not None:
            yield key, members, meta


def pre_robust_plane(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        return np.asarray(arr, dtype=np.float64) / np.iinfo(arr.dtype).max
    if np.issubdtype(arr.dtype, np.signedinteger):
        out = np.asarray(arr, dtype=np.float64)
        lo, hi = float(out.min()), float(out.max())
        return (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite floating WDS plane")
        return np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    raise ValueError(f"Unsupported stored dtype: {arr.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, action="append", required=True,
                   help="WDS directory; repeat for a mixture stored in separate directories")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--expected-samples", type=int, default=1_000_000)
    a = p.parse_args()
    shards = [shard for root in a.root for shard in resolve_shards(root)]
    if not shards:
        raise ValueError(f"No WDS shards in {a.root}")
    sums = np.zeros(3, dtype=np.float64)
    sumsqs = np.zeros(3, dtype=np.float64)
    pixels = np.zeros(3, dtype=np.int64)
    samples = 0
    stored_dtypes = Counter()
    source_dtypes = Counter()
    conversions = Counter()
    by_source_dtype = {}
    started = time.monotonic()
    for shard_idx, shard in enumerate(shards, 1):
        for key, members, meta_bytes in iter_tar_samples(shard):
            if not members:
                continue
            meta = json.loads(meta_bytes) if meta_bytes else {}
            source_dtype = str(meta.get("source_dtype", "unspecified"))
            source_dtypes[source_dtype] += 1
            conversions[str(meta.get("conversion", "unspecified"))] += 1
            bucket = by_source_dtype.setdefault(source_dtype, {
                "images": 0, "pixels": np.zeros(3, dtype=np.int64),
                "sum": np.zeros(3, dtype=np.float64),
                "sumsq": np.zeros(3, dtype=np.float64)})
            planes = []
            for ch in sorted(members)[:3]:
                arr = tifffile.imread(io.BytesIO(members[ch]))
                if arr.ndim != 2:
                    raise ValueError(f"Unsupported TIFF plane {shard} {key} ch{ch}: {arr.shape}, {arr.dtype}")
                stored_dtypes[str(arr.dtype)] += 1
                planes.append(pre_robust_plane(arr))
            while len(planes) < 3:
                planes.append(planes[-1])
            for ch in range(3):
                pixels[ch] += planes[ch].size
                plane_sum = planes[ch].sum(dtype=np.float64)
                plane_sumsq = np.square(planes[ch], dtype=np.float64).sum(dtype=np.float64)
                sums[ch] += plane_sum
                sumsqs[ch] += plane_sumsq
                bucket["pixels"][ch] += planes[ch].size
                bucket["sum"][ch] += plane_sum
                bucket["sumsq"][ch] += plane_sumsq
            samples += 1
            bucket["images"] += 1
        if shard_idx % 10 == 0 or shard_idx == len(shards):
            print(f"shards={shard_idx}/{len(shards)} samples={samples} elapsed_sec={time.monotonic()-started:.0f}", flush=True)
    if samples != a.expected_samples:
        raise ValueError(f"Expected {a.expected_samples} WDS samples, read {samples}")
    mean = sums / pixels
    std = np.sqrt(np.maximum(0, sumsqs / pixels - mean * mean))
    per_dtype = {}
    for dtype, bucket in by_source_dtype.items():
        dtype_mean = bucket["sum"] / bucket["pixels"]
        dtype_std = np.sqrt(np.maximum(0, bucket["sumsq"] / bucket["pixels"] - dtype_mean * dtype_mean))
        per_dtype[dtype] = {"images": bucket["images"],
                            "pixels_per_channel": bucket["pixels"].tolist(),
                            "rgb_mean": dtype_mean.tolist(),
                            "rgb_std_population": dtype_std.tolist()}
    out = {"definition": "Every WDS image; first three TIFF channels as RGB, missing channels repeat last; pre-robust packwds decoder scale by stored dtype; population std",
           "root": str(a.root[0]) if len(a.root) == 1 else None,
           "roots": [str(root) for root in a.root],
           "shards_read": len(shards), "images_read": samples,
           "pixels_per_channel": pixels.tolist(), "rgb_mean": mean.tolist(),
           "rgb_std_population": std.tolist(),
           "stored_plane_dtype_counts": stored_dtypes,
           "source_image_dtype_counts": source_dtypes,
           "materialization_conversion_counts": conversions,
           "pre_robust_stats_by_source_dtype": per_dtype}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
