#!/usr/bin/env python3
"""Audit source identity and global ORI/2P proportions in a candidate list."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    summary = json.loads(a.manifest.with_suffix(".json").read_text())
    pool = summary["pool"]
    universe = summary["universe"]
    expected_n = summary["candidate_count"]
    seen = set()
    types = Counter()
    missing_metadata = 0
    with a.manifest.open(newline="") as handle:
        for priority, row in enumerate(csv.DictReader(handle)):
            if int(row["priority"]) != priority or row["pool"] != pool:
                raise ValueError(f"Priority or pool mismatch at row {priority}")
            key = ((int(row["storage_item_id"]),) if pool == "100tb" else
                   (int(row["source_table_code"]), int(row["source_id"]), row["frame_idx"]))
            if key in seen:
                raise ValueError(f"Repeated storage image/frame at row {priority}: {key}")
            seen.add(key)
            types[int(row["source_table_code"])] += 1
            if not row["file_path"] or not row["image_shape"] or not row["channel_count"]:
                missing_metadata += 1
            if pool == "100tb" and not row["shard_path"]:
                missing_metadata += 1
    n = len(seen)
    if n != expected_n:
        raise ValueError(f"Expected {expected_n} candidates; saw {n}")
    if pool == "1pb":
        expected_fraction_2p = universe["frame_count"] / (universe["ori_count"] + universe["frame_count"])
    else:
        # The complete exported-item type counts are queried separately in the
        # storage audit; no fixed type quota is imposed by this candidate list.
        expected_fraction_2p = None
    observed_fraction_2p = types[2] / n
    z_score = ((observed_fraction_2p - expected_fraction_2p) /
               math.sqrt(expected_fraction_2p * (1 - expected_fraction_2p) / n)
               if expected_fraction_2p is not None else None)
    report = {"manifest": str(a.manifest), "pool": pool, "candidate_count": n,
              "unique_storage_image_or_frame_keys": n, "source_table_counts": types,
              "missing_index_metadata": missing_metadata,
              "observed_fraction_2p": observed_fraction_2p,
              "expected_fraction_2p": expected_fraction_2p, "two_p_fraction_z_score": z_score}
    path = a.manifest.with_name(a.manifest.stem + "_audit.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing_metadata:
        raise ValueError(f"{missing_metadata} candidates have missing index metadata")
    if z_score is not None and abs(z_score) > 6:
        raise ValueError(f"2P share differs materially from global SRS expectation: z={z_score:.2f}")


if __name__ == "__main__":
    main()
