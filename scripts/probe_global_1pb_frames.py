#!/usr/bin/env python3
"""Check exact 2P frame and channel extraction for global 1PB candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile


def read_frame_from_tif(tif: tifffile.TiffFile, frame_idx: int, expected_channels: int) -> tuple[np.ndarray, dict]:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        page_count = len(series.pages)
        if "T" in axes:
            frame_count = int(shape[axes.index("T")])
        elif len(shape) > 2 and axes[0] in "IQZ":
            frame_count = int(shape[0])
        else:
            frame_count = 1
        if not 0 <= frame_idx < frame_count:
            raise ValueError(f"Frame {frame_idx} outside {frame_count} in {axes}:{shape}")
        if page_count % frame_count != 0:
            raise ValueError(f"Cannot map pages to frames: {page_count}/{frame_count}")
        pages_per_frame = page_count // frame_count
        start = frame_idx * pages_per_frame
        frame = series.asarray(key=slice(start, start + pages_per_frame))
        frame = np.squeeze(frame)
        if frame.ndim == 2:
            actual_channels = 1
        elif frame.ndim == 3:
            actual_channels = next((int(v) for v in frame.shape if int(v) == expected_channels),
                                   next((int(v) for v in frame.shape if 1 <= int(v) <= 8), -1))
        else:
            raise ValueError(f"Unsupported extracted frame shape {frame.shape}")
        if actual_channels < 1:
            raise ValueError(f"Cannot locate channels: {axes}:{shape}->{frame.shape}")
        info = {"series_axes": axes, "series_shape": list(shape), "page_count": page_count,
                "frame_count": frame_count, "pages_per_frame": pages_per_frame,
                "metadata_channel_count": expected_channels,
                "extracted_channel_count": actual_channels,
                "metadata_channel_mismatch": actual_channels != expected_channels,
                "frame_shape": list(frame.shape), "dtype": str(frame.dtype),
                "minimum": float(np.min(frame)), "maximum": float(np.max(frame)),
                "mean": float(np.mean(frame)), "std": float(np.std(frame))}
        return frame, info


def read_frame(path: str, frame_idx: int, expected_channels: int) -> tuple[np.ndarray, dict]:
    with tifffile.TiffFile(path) as tif:
        return read_frame_from_tif(tif, frame_idx, expected_channels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--priorities", required=True, help="Comma-separated global priorities")
    a = p.parse_args()
    wanted = {int(x) for x in a.priorities.split(",")}
    with a.candidate_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            priority = int(row["priority"])
            if priority not in wanted:
                continue
            if row["source_table_code"] != "2":
                raise ValueError(f"Priority {priority} is not 2P")
            _frame, info = read_frame(row["file_path"], int(row["frame_idx"]), int(row["channel_count"]))
            print(json.dumps({"priority": priority, "source_id": int(row["source_id"]),
                              "frame_idx": int(row["frame_idx"]), **info}))
            wanted.remove(priority)
            if not wanted:
                break
    if wanted:
        raise ValueError(f"Priorities not found: {sorted(wanted)}")


if __name__ == "__main__":
    main()
