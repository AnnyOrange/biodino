#!/usr/bin/env python3
"""Count eligible original-image records per 1PB dataset for exact quotas."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PIPE = Path("/home/inspur/xzj/pre_data/scalinglaw_sampling")
sys.path.insert(0, str(PIPE))
from slice_from_dataset_plan import DEFAULT_DB_CONFIG, prefix_upper_bound  # noqa: E402
import psycopg2  # noqa: E402

ROOT = Path("/mnt/huawei_blm/random_1pb_100tb_each_1m_uniform_v1/manifests")
PLAN = ROOT / "uniform_source_plan_1pb.csv"
OUT = ROOT / "actual_source_counts_1pb.csv"

QUERY = """
SELECT count(*) FROM original_images_all o
WHERE o.file_path >= %s AND o.file_path < %s
  AND o.channel_count BETWEEN 1 AND 10
"""


def main() -> None:
    with PLAN.open(newline="") as f:
        rows = list(csv.DictReader(f))
    completed = {}
    if OUT.exists():
        with OUT.open(newline="") as f:
            completed = {r["dataset_id"]: r for r in csv.DictReader(f)}
    conn = psycopg2.connect(**DEFAULT_DB_CONFIG)
    try:
        with OUT.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_id", "dataset_path", "actual_sources", "elapsed_sec"])
            if not completed:
                writer.writeheader()
            for i, row in enumerate(rows, 1):
                if row["dataset_id"] in completed:
                    continue
                start = time.monotonic()
                with conn.cursor() as cur:
                    cur.execute(QUERY, (row["dataset_path"], prefix_upper_bound(row["dataset_path"])))
                    count = int(cur.fetchone()[0])
                elapsed = time.monotonic() - start
                writer.writerow({"dataset_id": row["dataset_id"], "dataset_path": row["dataset_path"],
                                 "actual_sources": count, "elapsed_sec": round(elapsed, 3)})
                f.flush()
                print(f"{i}/{len(rows)} {row['dataset_id']} actual={count} plan_source_count={row['source_file_count']} sec={elapsed:.1f}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
