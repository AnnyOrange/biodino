#!/usr/bin/env python3
"""Compute per-channel mean / std from packed WebDataset shards.

Outputs values ready to paste into the YAML ``crops.rgb_mean`` /
``crops.rgb_std`` fields.

Only samples that **actually contain** a given channel contribute to
that channel's statistics — zero-filled channels from missing data are
excluded.

Usage::

    # Sample 5000 patches from all shards (fast, ~2 min)
    python data/repackage/compute_channel_stats.py \
        --shard-pattern "/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar" \
        --max-channels 8 \
        --max-samples 5000

    # Full pass over all data (slow but exact)
    python data/repackage/compute_channel_stats.py \
        --shard-pattern "/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar" \
        --max-channels 8 \
        --max-samples 0

Algorithm:
    Welford online accumulation per channel.  Each uint16 pixel is
    normalised to [0, 1] by dividing by 65535 (consistent with
    ``wds_decoder._to_float_tensor``).
"""

from __future__ import annotations

import argparse
import glob as _glob
import io
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("channel_stats")

# WebDataset tar member names look like "<key>.ch3.tif"
# We match the channel suffix after the last dot-group.
MEMBER_CH_RE = re.compile(r"\.ch(\d+)\.tiff?$", re.IGNORECASE)


class ChannelAccumulator:
    """Numerically stable online mean/variance per channel (Welford)."""

    def __init__(self, max_channels: int) -> None:
        self.n = max_channels
        self.count = np.zeros(max_channels, dtype=np.int64)
        self.mean = np.zeros(max_channels, dtype=np.float64)
        self.m2 = np.zeros(max_channels, dtype=np.float64)

    def update(self, ch_idx: int, pixels: np.ndarray) -> None:
        """Incorporate a flat float64 pixel array for channel *ch_idx*."""
        for val in _chunk_iter(pixels, chunk_size=100_000):
            n = len(val)
            batch_mean = float(val.mean())
            batch_var = float(val.var())

            old_count = self.count[ch_idx]
            new_count = old_count + n
            delta = batch_mean - self.mean[ch_idx]

            new_mean = self.mean[ch_idx] + delta * n / new_count
            # Parallel variance merge (Chan et al.)
            self.m2[ch_idx] += batch_var * n + delta ** 2 * old_count * n / new_count

            self.count[ch_idx] = new_count
            self.mean[ch_idx] = new_mean

    def channel_mean(self) -> np.ndarray:
        return self.mean.copy()

    def channel_std(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            var = self.m2 / np.maximum(self.count, 1)
        return np.sqrt(var)


def _chunk_iter(arr, chunk_size=100_000):
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


def expand_pattern(pattern: str) -> List[str]:
    from braceexpand import braceexpand

    expanded = list(braceexpand(pattern))
    resolved: List[str] = []
    for p in expanded:
        if any(c in p for c in ("*", "?", "[")):
            resolved.extend(sorted(_glob.glob(p)))
        else:
            resolved.append(p)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std from packed shards."
    )
    parser.add_argument(
        "--shard-pattern",
        required=True,
        help="Brace+glob pattern for packed tar shards.",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="Total channel count (defines output length).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Stop after this many samples (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shard order.",
    )
    args = parser.parse_args()

    # ---- resolve shards ------------------------------------------------
    shards = expand_pattern(args.shard_pattern)
    if not shards:
        logger.error("No shards found matching: %s", args.shard_pattern)
        sys.exit(1)

    import random
    rng = random.Random(args.seed)
    rng.shuffle(shards)
    logger.info("Found %d shards", len(shards))

    # ---- accumulate stats ----------------------------------------------
    import tarfile
    import tifffile

    acc = ChannelAccumulator(args.max_channels)
    sample_count = 0
    t0 = time.monotonic()

    for si, shard_path in enumerate(shards):
        try:
            tf = tarfile.open(shard_path, "r")
        except Exception as exc:
            logger.warning("skip %s: %s", shard_path, exc)
            continue

        seen_keys: set = set()
        try:
            for member in tf:
                if not member.isfile():
                    continue
                m = MEMBER_CH_RE.search(member.name)
                if not m:
                    continue

                ch_num = int(m.group(1))
                if ch_num < 1 or ch_num > args.max_channels:
                    continue

                raw = tf.extractfile(member)
                if raw is None:
                    continue
                try:
                    arr = tifffile.imread(io.BytesIO(raw.read()))
                except Exception:
                    continue

                pixels = arr.astype(np.float64).ravel() / 65535.0
                acc.update(ch_num - 1, pixels)

                # Count unique samples by their key prefix
                sample_key = member.name[: m.start()]
                if sample_key not in seen_keys:
                    seen_keys.add(sample_key)
                    sample_count += 1
                    if args.max_samples > 0 and sample_count >= args.max_samples:
                        break
        finally:
            tf.close()

        if args.max_samples > 0 and sample_count >= args.max_samples:
            logger.info("Reached %d samples, stopping early.", sample_count)
            break

        if (si + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            logger.info(
                "  %d/%d shards | %d samples | %.0fs",
                si + 1, len(shards), sample_count, elapsed,
            )

    elapsed = time.monotonic() - t0

    # ---- report --------------------------------------------------------
    means = acc.channel_mean()
    stds = acc.channel_std()

    print()
    print("=" * 60)
    print(f"Per-channel statistics  ({sample_count} samples, {elapsed:.1f}s)")
    print("=" * 60)
    print()
    print(f"{'ch':<5} {'pixels':>14} {'mean':>10} {'std':>10}")
    print("-" * 42)
    for i in range(args.max_channels):
        if acc.count[i] > 0:
            print(f"ch{i+1:<3d} {acc.count[i]:>14,d} {means[i]:>10.6f} {stds[i]:>10.6f}")
        else:
            print(f"ch{i+1:<3d} {'(no data)':>14} {'—':>10} {'—':>10}")
    print()

    # ---- YAML output ---------------------------------------------------
    print("# ---- paste into YAML (crops section) ----")
    print("  rgb_mean:")
    for i in range(args.max_channels):
        print(f"  - {means[i]:.6f}")
    print("  rgb_std:")
    for i in range(args.max_channels):
        v = stds[i] if stds[i] > 0 else 0.1
        print(f"  - {v:.6f}")
    print()


if __name__ == "__main__":
    main()
