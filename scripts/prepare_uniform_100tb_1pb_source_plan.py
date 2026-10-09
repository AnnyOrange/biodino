#!/usr/bin/env python3
"""Allocate exact 1.2M source-image quotas without capacity or quality weights."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OLD = Path("/mnt/huawei_blm/random_1pb_100tb_each_1m")


def allocate(rows: list[dict], target: int) -> list[dict]:
    counts = [int(round(float(r["source_file_count"]))) for r in rows]
    total = sum(counts)
    if total <= 0:
        raise ValueError("No positive source image counts")
    floors = [(target * n) // total for n in counts]
    remainders = [(target * n) % total for n in counts]
    order = sorted(range(len(rows)), key=lambda i: (-remainders[i], rows[i]["dataset_id"]))
    for i in order[: target - sum(floors)]:
        floors[i] += 1
    out = []
    for row, count, quota in zip(rows, counts, floors):
        if quota == 0:
            continue
        if quota > count:
            raise ValueError(f"Quota exceeds sources: {row['dataset_id']}")
        item = dict(row)
        item["target_patches"] = str(quota)
        item["final_target_pool"] = "1000000"
        item["max_patches_per_image"] = "1"
        item["effective_capacity"] = str(count)
        item["sampling_strategy"] = f"uniform_fov_{row['source_pool']}"
        out.append(item)
    assert sum(int(r["target_patches"]) for r in out) == target
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--raw-target", type=int, default=1_200_000)
    parser.add_argument("--actual-counts-root", type=Path, required=True,
                        help="Directory containing complete actual_source_counts_{pool}.csv files")
    args = parser.parse_args()
    out_root = args.out_root
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    summary = {"raw_target_per_pool": args.raw_target, "final_target_per_pool": 1_000_000,
               "selection_unit": "one source image, one patch", "quota_weight": "source_file_count",
               "content_filter": "none", "source_order": "seeded hash",
               "100tb_pool": "curated_100t_items exported run 1/5",
               "1pb_pool": "full original_images_all, including curated 100TB records"}
    for pool in ("100tb", "1pb"):
        source = OLD / pool / "manifests" / f"raw_plan_{pool}.csv"
        with source.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or {r["source_pool"] for r in rows} != {pool}:
            raise ValueError(f"Unexpected source pool in {source}")
        counts_path = args.actual_counts_root / f"actual_source_counts_{pool}.csv"
        with counts_path.open(newline="") as f:
            count_rows = list(csv.DictReader(f))
        if len(count_rows) != len(rows):
            raise ValueError(f"Incomplete or duplicate counts in {counts_path}: {len(count_rows)} vs {len(rows)}")
        count_by_id = {r["dataset_id"]: r for r in count_rows}
        if len(count_by_id) != len(rows) or set(count_by_id) != {r["dataset_id"] for r in rows}:
            raise ValueError(f"Dataset ID mismatch in {counts_path}")
        count_column = "actual_curated_sources" if pool == "100tb" else "actual_sources"
        for row in rows:
            actual = count_by_id[row["dataset_id"]]
            if actual["dataset_path"] != row["dataset_path"]:
                raise ValueError(f"Path mismatch for {row['dataset_id']}")
            row["source_file_count"] = str(int(actual[count_column]))
        unique_rows = []
        by_path = {}
        dropped_duplicate_ids = []
        for row in rows:
            path = row["dataset_path"]
            if path in by_path:
                if by_path[path]["source_file_count"] != row["source_file_count"]:
                    raise ValueError(f"Duplicate path has different actual counts: {path}")
                dropped_duplicate_ids.append(row["dataset_id"])
            else:
                by_path[path] = row
                unique_rows.append(row)
        paths = sorted(by_path)
        for left, right in zip(paths, paths[1:]):
            if right.startswith(left + "/"):
                raise ValueError(f"Nested dataset paths could double count images: {left}, {right}")
        plan = allocate(unique_rows, args.raw_target)
        path = out_root / "manifests" / f"uniform_source_plan_{pool}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=plan[0]); writer.writeheader(); writer.writerows(plan)
        top = sorted(plan, key=lambda r: int(r["target_patches"]), reverse=True)
        summary[pool] = {"source_plan": str(source), "output_plan": str(path),
                         "source_images_actual": sum(int(r["source_file_count"]) for r in unique_rows),
                         "actual_counts": str(counts_path),
                         "dropped_duplicate_path_dataset_ids": dropped_duplicate_ids,
                         "unique_dataset_paths": len(unique_rows),
                         "dataset_count": len(plan),
                         "top10_quota_fraction": sum(int(r["target_patches"]) for r in top[:10]) / args.raw_target}
    (out_root / "manifests" / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
