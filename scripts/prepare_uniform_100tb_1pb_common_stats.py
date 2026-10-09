#!/usr/bin/env python3
"""Pool the two audited RGB samples into shared training normalization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pooled(stats: list[dict], key: str) -> dict:
    entries = [s[key] for s in stats]
    counts = [e["pixels_per_channel"] for e in entries]
    means = [e["rgb_mean"] for e in entries]
    stds = [e["rgb_std_population"] for e in entries]
    pixels = [sum(c[ch] for c in counts) for ch in range(3)]
    mean = [sum(counts[i][ch] * means[i][ch] for i in range(2)) / pixels[ch] for ch in range(3)]
    variance = [sum(counts[i][ch] * (stds[i][ch] ** 2 + means[i][ch] ** 2) for i in range(2)) / pixels[ch] - mean[ch] ** 2 for ch in range(3)]
    return {"pixels_per_channel": pixels, "rgb_mean": mean,
            "rgb_std_population": [math.sqrt(max(0, v)) for v in variance]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    stats = []
    for pool in ("100tb", "1pb"):
        p = args.root / pool / "mean_std" / f"{pool}_1m_rgb_sample100k.json"
        s = json.loads(p.read_text())
        if s["samples_read"] != 100000 or s["samples_failed"] != 0:
            raise ValueError(f"Incomplete stats: {p}")
        for key in ("sample_rgb_stack_tifffile_dtype_max", "sample_rgb_stack_robust_pct"):
            if s[key]["images_read"] != 100000:
                raise ValueError(f"Incomplete {key}: {p}")
        stats.append(s)
    out = {
        "source_stat_files": [str(args.root / p / "mean_std" / f"{p}_1m_rgb_sample100k.json") for p in ("100tb", "1pb")],
        "raw_uint16_over_65535": pooled(stats, "sample_rgb_stack_tifffile_dtype_max"),
        "training_robust_pct_1_99": pooled(stats, "sample_rgb_stack_robust_pct"),
    }
    path = args.root / "manifests" / "common_rgb_mean_std.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
