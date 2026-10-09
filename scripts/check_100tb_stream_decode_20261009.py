#!/usr/bin/env python3
"""Compare the raw 100TB decoder with the former in-memory repacking path."""

import csv
import io
import json
from pathlib import Path

import tifffile
import torch

from dinov3.data.raw_100tb_stream import decode_raw_item
from dinov3.data.wds_decoder import decode_packed_sample_robust
from scripts.materialize_global_100tb_wds import image_planes


CANDIDATES = Path(
    "/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009/manifests/prefix_smoke_10.csv"
)


def main():
    checked = {}
    with CANDIDATES.open(newline="") as source:
        for row in csv.DictReader(source):
            channel_hint = int(row["channel_count"])
            if channel_hint in checked or channel_hint not in (1, 2, 3):
                continue
            data = Path(row["file_path"]).read_bytes()
            item_id = int(row["storage_item_id"])
            actual = decode_raw_item(data, item_id, channel_hint, 3)
            planes, _ = image_planes(data, str(item_id), channel_hint, 20260924)
            packed = {}
            for index, plane in enumerate(planes, 1):
                buffer = io.BytesIO()
                tifffile.imwrite(buffer, plane, photometric="minisblack")
                packed[f"ch{index}.tif"] = buffer.getvalue()
            reference = decode_packed_sample_robust(packed, target_channels=3, p_low=1, p_high=99)
            if reference is None or not torch.equal(actual, reference):
                raise AssertionError(f"Online/packed decoder mismatch for item {item_id}")
            checked[channel_hint] = {"item_id": item_id, "shape": list(actual.shape)}
            if len(checked) == 3:
                break
    if set(checked) != {1, 2, 3}:
        raise AssertionError(f"Missing representative channel counts: {checked}")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
