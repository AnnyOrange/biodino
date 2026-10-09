#!/usr/bin/env python3
"""Recover 2P candidates lost when tar.getmembers hit a damaged tar tail.

The original whole-tar index pass marked every requested member failed if it
encountered a truncated tail.  This streams intact earlier members and keeps
the original candidate priorities.  Nonfinite float pixels are replaced with
finite endpoints and counted, preserving the selected storage item's identity.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile

from materialize_global_100tb_wds import TarWriter, array_planes


def failed_priorities(stage_root: Path) -> set[int]:
    failed = set()
    for status_path in glob.glob(str(stage_root / "status_w*.csv")):
        with open(status_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "success":
                    failed.add(int(row["priority"]))
    return failed


def decode(data: bytes, row: dict, seed: int):
    arr = tifffile.imread(io.BytesIO(data))
    nonfinite = 0
    if np.issubdtype(arr.dtype, np.floating):
        nonfinite = int((~np.isfinite(arr)).sum())
        if nonfinite:
            arr = np.nan_to_num(arr, nan=0.0, posinf=65535.0, neginf=0.0)
    planes, info = array_planes(arr, row["storage_item_id"], int(row["channel_count"] or 0), seed)
    info["nonfinite_pixels_replaced"] = nonfinite
    return planes, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, default=1_100_000)
    p.add_argument("--seed", type=int, default=20260924)
    a = p.parse_args()
    if a.out_root.exists() and any(a.out_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {a.out_root}")
    failures = failed_priorities(a.stage_root)
    grouped = defaultdict(dict)
    with a.candidates.open(newline="") as handle:
        for row in csv.DictReader(handle):
            priority = int(row["priority"])
            if priority >= a.max_candidates:
                break
            if priority in failures:
                grouped[row["shard_path"]][int(row["storage_item_id"])] = row
    if sum(map(len, grouped.values())) != len(failures):
        raise ValueError("Failed priority lookup coverage mismatch")
    a.out_root.mkdir(parents=True, exist_ok=True)
    writer = TarWriter(a.out_root, 0, 2000)
    outcomes = {}
    errors = Counter()
    for shard_path, wanted in grouped.items():
        try:
            with tarfile.open(shard_path, "r:") as archive:
                try:
                    for member in archive:
                        if not member.isfile():
                            continue
                        try:
                            item_id = int(Path(member.name).stem)
                        except ValueError:
                            continue
                        row = wanted.get(item_id)
                        if row is None:
                            continue
                        priority = int(row["priority"])
                        try:
                            stream = archive.extractfile(member)
                            if stream is None:
                                raise ValueError("member_unreadable")
                            planes, info = decode(stream.read(), row, a.seed)
                            meta = {"pool": "100tb", "priority": priority,
                                    "storage_item_id": item_id,
                                    "source_table_code": int(row["source_table_code"]),
                                    "source_id": int(row["source_id"]),
                                    "frame_idx": int(row["frame_idx"]) if row["frame_idx"] else None,
                                    "source_shard": shard_path, "source_member": member.name,
                                    "source_path": row["file_path"], "read_from": "exported_tar_repair",
                                    **info}
                            writer.add(f"p{priority:08d}", planes, meta)
                            outcomes[priority] = ("success", "")
                        except Exception as exc:
                            outcomes[priority] = ("failed", f"{type(exc).__name__}:{str(exc)[:150]}")
                except tarfile.ReadError as exc:
                    errors[f"ReadError:{str(exc)}"] += 1
        except Exception as exc:
            errors[f"open_{type(exc).__name__}:{str(exc)[:100]}"] += 1
    writer.close()
    with (a.out_root / "status_w00.csv").open("w", newline="") as handle:
        status = csv.writer(handle)
        status.writerow(["priority", "storage_item_id", "status", "reason"])
        for wanted in grouped.values():
            for item_id, row in wanted.items():
                priority = int(row["priority"])
                result, reason = outcomes.get(priority, ("failed", "member_not_recovered"))
                status.writerow([priority, item_id, result, reason])
    result_counts = Counter(result for result, _reason in outcomes.values())
    result_counts["missing_after_scan"] = len(failures) - len(outcomes)
    report = {"attempted_failed_candidates": len(failures), "outcomes": result_counts,
              "damaged_shard_errors": errors, "staged_shards": writer.shard + 1,
              "stage_root": str(a.stage_root), "out_root": str(a.out_root)}
    (a.out_root / "repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
