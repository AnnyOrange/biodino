#!/usr/bin/env python3
"""Compute RGB mean/std for packed WebDataset tensors.

This reports two statistics:
- present-only: only channels physically present in a sample contribute.
- train-tensor: missing channels are counted as zero, matching packwds RGB
  training where decode_packed_sample zero-fills absent channels.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import io
import random
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import tifffile

try:
    from braceexpand import braceexpand
except Exception:  # pragma: no cover - optional dependency guard
    braceexpand = None


CH_RE = re.compile(r"^(?P<key>.+)\.ch(?P<ch>\d+)\.tiff?$", re.IGNORECASE)


def parse_float_list(value: str | None, channels: int) -> np.ndarray | None:
    if value is None or value == "":
        return None
    cleaned = value.strip().strip("[]")
    parts = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]
    vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    if vals.size != channels:
        raise ValueError(f"Expected {channels} values, got {vals.size}: {value}")
    return vals


class TopK:
    def __init__(self, k: int) -> None:
        self.k = max(0, int(k))
        self.heap: list[tuple[float, str]] = []

    def add(self, score: float, text: str) -> None:
        if self.k <= 0 or not np.isfinite(score):
            return
        item = (float(score), text)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def lines(self) -> list[str]:
        return [text for _, text in sorted(self.heap, reverse=True)]


def expand_pattern(pattern: str) -> list[str]:
    patterns = list(braceexpand(pattern)) if braceexpand else [pattern]
    out: list[str] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            out.extend(sorted(glob.glob(pat)))
        else:
            out.append(pat)
    return out


def to_decoder_float(arr: np.ndarray) -> np.ndarray | None:
    """Match dinov3.data.wds_decoder._to_float_tensor for stats."""
    if arr is None or arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]

    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr, 0.0, 1.0).astype(np.float64, copy=False)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        return arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)

    arr64 = arr.astype(np.float64)
    mn = float(np.nanmin(arr64))
    mx = float(np.nanmax(arr64))
    if mx > mn:
        return (arr64 - mn) / (mx - mn)
    return np.zeros_like(arr64, dtype=np.float64)


def add_pixels(acc: dict[str, np.ndarray], idx: int, pixels: np.ndarray) -> int:
    flat = pixels.ravel()
    finite = np.isfinite(flat)
    bad = int((~finite).sum())
    if bad:
        flat = flat[finite]
    acc["count"][idx] += flat.size
    s = float(flat.sum())
    acc["sum"][idx] += s
    acc["sumsq"][idx] += float(np.dot(flat, flat))
    return bad


def report(name: str, acc: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = acc["count"].astype(np.float64)
    mean = acc["sum"] / np.maximum(count, 1)
    var = acc["sumsq"] / np.maximum(count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0))

    print(f"\n{name}")
    print(f"{'ch':<4} {'pixels':>16} {'mean':>10} {'std':>10}")
    for i in range(len(mean)):
        print(f"ch{i + 1:<2} {int(acc['count'][i]):>16,d} {mean[i]:>10.6f} {std[i]:>10.6f}")

    print("rgb_mean:")
    for value in mean:
        print(f"  - {value:.6f}")
    print("rgb_std:")
    for value in std:
        print(f"  - {value:.6f}")
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-pattern", required=True)
    parser.add_argument("--max-channels", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=10000, help="0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track-topk", type=int, default=20)
    parser.add_argument("--normalize-mean", default=None, help="Optional comma/list mean to track normalized extremes.")
    parser.add_argument("--normalize-std", default=None, help="Optional comma/list std to track normalized extremes.")
    parser.add_argument(
        "--normalize-channels",
        type=int,
        default=None,
        help="Only use the first N channels when ranking normalized extremes; defaults to --max-channels.",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Compute stats on decode_packed_sample_robust output (the packwds_robust: path): "
        "per-channel percentile clip+rescale, single-channel samples replicated.",
    )
    parser.add_argument("--pct-low", type=float, default=1.0, help="Lower clip percentile for --robust (default 1).")
    parser.add_argument("--pct-high", type=float, default=99.0, help="Upper clip percentile for --robust (default 99).")
    args = parser.parse_args()

    shards = expand_pattern(args.shard_pattern)
    if not shards:
        print(f"No shards matched: {args.shard_pattern}", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(shards)

    channels = args.max_channels
    norm_mean = parse_float_list(args.normalize_mean, channels)
    norm_std = parse_float_list(args.normalize_std, channels)
    if (norm_mean is None) != (norm_std is None):
        print("--normalize-mean and --normalize-std must be provided together", file=sys.stderr)
        return 1
    norm_channels = args.normalize_channels or channels
    if not 1 <= norm_channels <= channels:
        print(f"--normalize-channels must be in [1, {channels}], got {norm_channels}", file=sys.stderr)
        return 1

    present = {
        "count": np.zeros(channels, dtype=np.int64),
        "sum": np.zeros(channels, dtype=np.float64),
        "sumsq": np.zeros(channels, dtype=np.float64),
    }
    train_tensor = {
        "count": np.zeros(channels, dtype=np.int64),
        "sum": np.zeros(channels, dtype=np.float64),
        "sumsq": np.zeros(channels, dtype=np.float64),
    }

    robust_decode = None
    robust_acc = None
    if args.robust:
        if not 0.0 <= args.pct_low < args.pct_high <= 100.0:
            print(
                f"--pct-low/--pct-high must satisfy 0<=low<high<=100, got {args.pct_low},{args.pct_high}",
                file=sys.stderr,
            )
            return 1
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from dinov3.data.wds_decoder import decode_packed_sample_robust as robust_decode  # noqa: F401
        robust_acc = {
            "count": np.zeros(channels, dtype=np.int64),
            "sum": np.zeros(channels, dtype=np.float64),
            "sumsq": np.zeros(channels, dtype=np.float64),
        }

    sample_count = 0
    bad: list[tuple[str, str, str]] = []
    raw_nonfinite_details: list[tuple[str, str, int, str]] = []
    decoded_nonfinite_details: list[tuple[str, str, int]] = []
    raw_nonfinite_pixels = 0
    decoded_nonfinite_pixels = 0
    top_raw_abs = TopK(args.track_topk)
    top_norm_abs_present = TopK(args.track_topk)
    top_norm_abs_train_tensor = TopK(args.track_topk)
    t0 = time.time()

    def finish_sample(shard_name: str, key: str, items: list[tuple[int, bytes, str]]) -> bool:
        nonlocal sample_count, raw_nonfinite_pixels, decoded_nonfinite_pixels
        full_key = f"{shard_name}::{key}"
        if robust_decode is not None:
            # packwds_robust: stats — run the actual training decode and
            # accumulate per-channel stats on its (channels, H, W) output.
            sample_dict = {
                f"ch{ch_num}.tif": data
                for ch_num, data, _ in items
                if 1 <= ch_num <= channels
            }
            if not sample_dict:
                return False
            try:
                out = robust_decode(
                    sample_dict, target_channels=channels, p_low=args.pct_low, p_high=args.pct_high
                )
            except Exception as exc:  # pragma: no cover - defensive
                bad.append((full_key, "<robust>", f"{type(exc).__name__}: {exc}"))
                return False
            if out is None:
                return False
            out_np = out.detach().cpu().numpy()
            for i in range(channels):
                add_pixels(robust_acc, i, out_np[i])
            sample_count += 1
            return True
        sample_sum = np.zeros(channels, dtype=np.float64)
        sample_sumsq = np.zeros(channels, dtype=np.float64)
        sample_present = np.zeros(channels, dtype=bool)
        sample_min = np.full(channels, np.inf, dtype=np.float64)
        sample_max = np.full(channels, -np.inf, dtype=np.float64)
        sample_absmax = np.zeros(channels, dtype=np.float64)
        hw: tuple[int, int] | None = None
        ok = False

        for ch_num, data, member_name in items:
            if not (1 <= ch_num <= channels):
                continue
            try:
                raw_arr = tifffile.imread(io.BytesIO(data))
            except Exception as exc:
                bad.append((full_key, member_name, f"{type(exc).__name__}: {exc}"))
                continue
            if np.issubdtype(raw_arr.dtype, np.floating):
                raw_nonfinite = int((~np.isfinite(raw_arr)).sum())
                raw_nonfinite_pixels += raw_nonfinite
                if raw_nonfinite:
                    raw_nonfinite_details.append((full_key, member_name, raw_nonfinite, str(raw_arr.dtype)))
            arr = to_decoder_float(raw_arr)
            if arr is None:
                bad.append((full_key, member_name, "empty"))
                continue
            if hw is None:
                hw = arr.shape[-2:]
            elif hw != arr.shape[-2:]:
                bad.append((full_key, member_name, f"shape mismatch {hw} vs {arr.shape[-2:]}"))
                return False

            idx = ch_num - 1
            flat = arr.ravel()
            finite = np.isfinite(flat)
            n_decoded_nonfinite = int((~finite).sum())
            decoded_nonfinite_pixels += n_decoded_nonfinite
            if n_decoded_nonfinite:
                decoded_nonfinite_details.append((full_key, member_name, n_decoded_nonfinite))
            if not finite.all():
                flat = flat[finite]
            if flat.size == 0:
                bad.append((full_key, member_name, "all pixels non-finite"))
                continue
            present["count"][idx] += flat.size
            s = float(flat.sum())
            ss = float(np.dot(flat, flat))
            present["sum"][idx] += s
            present["sumsq"][idx] += ss
            sample_sum[idx] += s
            sample_sumsq[idx] += ss
            sample_present[idx] = True
            sample_min[idx] = min(sample_min[idx], float(flat.min()))
            sample_max[idx] = max(sample_max[idx], float(flat.max()))
            sample_absmax[idx] = max(sample_absmax[idx], float(np.abs(flat).max()))
            ok = True

        if not ok or hw is None:
            return False

        pixels_per_channel = int(hw[0]) * int(hw[1])
        # Missing channels contribute zero-valued pixels to the actual RGB tensor.
        train_tensor["count"] += pixels_per_channel
        train_tensor["sum"] += sample_sum
        train_tensor["sumsq"] += sample_sumsq

        present_channels = [i + 1 for i, is_present in enumerate(sample_present) if is_present]
        minmax = ", ".join(
            f"ch{i + 1}=[{sample_min[i]:.6g},{sample_max[i]:.6g}]"
            for i in range(channels)
            if sample_present[i]
        )
        raw_score = float(sample_absmax.max()) if sample_present.any() else 0.0
        top_raw_abs.add(raw_score, f"raw_absmax={raw_score:.6g} key={full_key} channels={present_channels} {minmax}")

        if norm_mean is not None and norm_std is not None:
            present_scores = np.zeros(channels, dtype=np.float64)
            train_scores = np.zeros(channels, dtype=np.float64)
            for i in range(norm_channels):
                if sample_present[i]:
                    vals = np.asarray(
                        [
                            (sample_min[i] - norm_mean[i]) / norm_std[i],
                            (sample_max[i] - norm_mean[i]) / norm_std[i],
                        ],
                        dtype=np.float64,
                    )
                    present_scores[i] = float(np.abs(vals).max())
                    train_scores[i] = present_scores[i]
                else:
                    train_scores[i] = abs((0.0 - norm_mean[i]) / norm_std[i])
            present_score = float(present_scores[:norm_channels].max())
            train_score = float(train_scores[:norm_channels].max())
            top_norm_abs_present.add(
                present_score,
                f"norm_absmax_present={present_score:.6g} key={full_key} channels={present_channels} {minmax}",
            )
            top_norm_abs_train_tensor.add(
                train_score,
                f"norm_absmax_train_tensor={train_score:.6g} key={full_key} channels={present_channels} {minmax}",
            )
        sample_count += 1
        return True

    print(f"found_shards={len(shards)} seed={args.seed}", flush=True)
    for shard_index, shard_path in enumerate(shards):
        shard_name = Path(shard_path).name
        current_key: str | None = None
        current_items: list[tuple[int, bytes, str]] = []
        try:
            with tarfile.open(shard_path, "r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    match = CH_RE.match(member.name)
                    if not match:
                        continue
                    key = match.group("key")
                    if current_key is not None and key != current_key:
                        finish_sample(shard_name, current_key, current_items)
                        if args.max_samples and sample_count >= args.max_samples:
                            break
                        current_items = []
                    current_key = key
                    raw = tf.extractfile(member)
                    if raw is not None:
                        current_items.append((int(match.group("ch")), raw.read(), member.name))
                if (not args.max_samples or sample_count < args.max_samples) and current_key is not None:
                    finish_sample(shard_name, current_key, current_items)
        except Exception as exc:
            bad.append((str(Path(shard_path).name), "<tar>", f"{type(exc).__name__}: {exc}"))

        if args.max_samples and sample_count >= args.max_samples:
            print(f"reached max_samples={sample_count}", flush=True)
            break
        if (shard_index + 1) % 10 == 0:
            print(
                f"{shard_index + 1}/{len(shards)} shards samples={sample_count} "
                f"elapsed={time.time() - t0:.1f}s bad={len(bad)} "
                f"raw_nonfinite={raw_nonfinite_pixels} decoded_nonfinite={decoded_nonfinite_pixels}",
                flush=True,
            )

    print(
        f"\nsamples={sample_count} elapsed={time.time() - t0:.1f}s bad={len(bad)} "
        f"raw_nonfinite_pixels={raw_nonfinite_pixels} "
        f"decoded_nonfinite_pixels={decoded_nonfinite_pixels}"
    )
    for item in bad[:20]:
        print("BAD", item)
    for item in raw_nonfinite_details[:50]:
        print("RAW_NONFINITE", item)
    if len(raw_nonfinite_details) > 50:
        print(f"RAW_NONFINITE ... {len(raw_nonfinite_details) - 50} more entries omitted")
    for item in decoded_nonfinite_details[:50]:
        print("DECODED_NONFINITE", item)
    if len(decoded_nonfinite_details) > 50:
        print(f"DECODED_NONFINITE ... {len(decoded_nonfinite_details) - 50} more entries omitted")
    print("\nTOP_RAW_ABS_SAMPLES")
    for line in top_raw_abs.lines():
        print(line)
    if norm_mean is not None and norm_std is not None:
        print(f"\nTOP_NORMALIZED_ABS_PRESENT_ONLY_SAMPLES_first{norm_channels}")
        for line in top_norm_abs_present.lines():
            print(line)
        print(f"\nTOP_NORMALIZED_ABS_TRAIN_TENSOR_SAMPLES_first{norm_channels}")
        for line in top_norm_abs_train_tensor.lines():
            print(line)
    if args.robust:
        report("ROBUST_TRAIN_TENSOR (packwds_robust: per-channel pct clip + 1ch-replicate)", robust_acc)
    else:
        report("PRESENT_ONLY_excludes_missing_channels", present)
        report("TRAIN_TENSOR_includes_zero_filled_missing_channels", train_tensor)
    return 1 if bad or raw_nonfinite_pixels or decoded_nonfinite_pixels else 0


if __name__ == "__main__":
    raise SystemExit(main())
