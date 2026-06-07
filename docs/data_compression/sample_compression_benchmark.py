#!/usr/bin/env python
"""Sample-shard TIFF compression benchmark for the 1TB WebDataset.

Goal: validate the ~24.6% (deflate) / 27.2% (LZW) extrapolation from the
single test pack on a larger, randomly sampled set of real shards, before
committing to a full repack of the dataset.

For each sampled shard we walk the tar, decode every (capped) `.tif` member,
re-encode it with several TIFF-internal lossless codecs, and accumulate the
encoded payload sizes. We also optionally check that the re-encode is truly
lossless (decode-back == original array) and measure an 8-bit quantization
variant (LOSSY, per-image min/max) for reference.

Run with the dinov3 env which has tifffile + imagecodecs:

    /home/lxy/miniconda3/envs/dinov3/bin/python \
        docs/data_compression/sample_compression_benchmark.py \
        --data-dir /mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle \
        --num-shards 15 --max-tifs-per-shard 400 --verify \
        --report docs/data_compression/sample_compression_results.md

Notes:
- `--max-tifs-per-shard` caps decode work for a fast estimate. Set to 0 to
  decode every TIFF in the sampled shards (slower, more accurate).
- Compression ratios are reported against the *TIFF payload* (the dominant
  cost), not the whole tar; JSON metadata is negligible.
- The uint8 quant row is LOSSY and only a size reference, not a recommendation.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile


# TIFF-internal lossless codecs to test. Names are passed straight to
# tifffile.imwrite(compression=...). imagecodecs must be installed.
LOSSLESS_METHODS = ["deflate", "lzw", "zstd"]


@dataclass
class Acc:
    """Accumulates encoded payload bytes and per-image counts."""

    orig_bytes: int = 0
    n_tifs: int = 0
    method_bytes: Dict[str, int] = field(default_factory=dict)
    quant8_bytes: int = 0
    verify_checked: int = 0
    verify_failures: int = 0


def encode(array: np.ndarray, compression) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, array, compression=compression)
    return buf.getvalue()


def quantize_uint8(array: np.ndarray) -> np.ndarray:
    """Per-image min/max linear map uint16 -> uint8 (LOSSY reference only)."""
    a = array.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0, 255).astype(np.uint8)


def process_shard(tar_path: Path, acc: Acc, max_tifs: int, verify: bool,
                  do_quant8: bool) -> None:
    seen = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".tif"):
                continue
            if max_tifs and seen >= max_tifs:
                break
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            acc.orig_bytes += len(raw)
            acc.n_tifs += 1
            seen += 1

            array = tifffile.imread(io.BytesIO(raw))

            for m in LOSSLESS_METHODS:
                enc = encode(array, m)
                acc.method_bytes[m] = acc.method_bytes.get(m, 0) + len(enc)
                if verify and acc.verify_checked < 200:
                    back = tifffile.imread(io.BytesIO(enc))
                    acc.verify_checked += 1
                    if not np.array_equal(back, array):
                        acc.verify_failures += 1

            if do_quant8:
                q = quantize_uint8(array)
                acc.quant8_bytes += len(encode(q, "deflate"))


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def build_report(acc: Acc, shards: List[Path], elapsed: float,
                 max_tifs: int) -> str:
    lines: List[str] = []
    lines.append("# Sample shard compression results\n")
    lines.append(f"- shards sampled: {len(shards)}")
    lines.append(f"- TIFFs decoded: {acc.n_tifs}")
    lines.append(f"- max tifs/shard cap: {max_tifs or 'none (full)'}")
    lines.append(f"- original TIFF payload: {fmt_gb(acc.orig_bytes)}")
    if acc.verify_checked:
        status = "PASS" if acc.verify_failures == 0 else f"FAIL ({acc.verify_failures})"
        lines.append(f"- lossless verify: {status} on {acc.verify_checked} samples")
    lines.append(f"- elapsed: {elapsed:.1f}s\n")

    lines.append("| method (lossless) | encoded payload | ratio vs original |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| original (none) | {fmt_gb(acc.orig_bytes)} | 100.0% |")
    for m in LOSSLESS_METHODS:
        b = acc.method_bytes.get(m, 0)
        pct = 100.0 * b / acc.orig_bytes if acc.orig_bytes else 0.0
        lines.append(f"| {m} | {fmt_gb(b)} | {pct:.1f}% |")
    if acc.quant8_bytes:
        pct = 100.0 * acc.quant8_bytes / acc.orig_bytes if acc.orig_bytes else 0.0
        lines.append(f"| uint8+deflate (LOSSY ref) | {fmt_gb(acc.quant8_bytes)} | {pct:.1f}% |")

    lines.append("\n## Extrapolation to full dataset\n")
    lines.append("Assuming the sampled ratio holds, a 966G TIFF payload maps to:\n")
    for m in LOSSLESS_METHODS:
        b = acc.method_bytes.get(m, 0)
        if acc.orig_bytes:
            est = 966 * b / acc.orig_bytes
            lines.append(f"- {m}: ~{est:.0f} GB")
    lines.append("\n_Sampled estimate only; rerun with --max-tifs-per-shard 0 "
                 "for a tighter number before committing to a full repack._")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--num-shards", type=int, default=15)
    ap.add_argument("--max-tifs-per-shard", type=int, default=400,
                    help="0 = decode every TIFF (slow, accurate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="check decode(encode(x)) == x on up to 200 samples")
    ap.add_argument("--quant8", action="store_true",
                    help="also measure LOSSY uint8 (per-image min/max) + deflate")
    ap.add_argument("--glob", default="*.tar")
    ap.add_argument("--report", type=Path, default=None,
                    help="optional path to write a markdown report")
    args = ap.parse_args()

    all_shards = sorted(p for p in args.data_dir.glob(args.glob) if p.is_file())
    if not all_shards:
        print(f"no shards matching {args.glob} in {args.data_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n = min(args.num_shards, len(all_shards))
    shards = sorted(rng.sample(all_shards, n))

    print(f"sampling {n}/{len(all_shards)} shards (seed={args.seed})", file=sys.stderr)
    acc = Acc()
    t0 = time.time()
    for i, sp in enumerate(shards, 1):
        print(f"[{i}/{n}] {sp.name}", file=sys.stderr)
        process_shard(sp, acc, args.max_tifs_per_shard, args.verify, args.quant8)
    elapsed = time.time() - t0

    report = build_report(acc, shards, elapsed, args.max_tifs_per_shard)
    print("\n" + report)
    if args.report:
        args.report.write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    if acc.verify_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
