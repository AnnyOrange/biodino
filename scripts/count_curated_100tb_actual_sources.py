#!/usr/bin/env python3
"""Count actual exported curated source images per dataset prefix for uniform quotas."""
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
PLAN = ROOT / "uniform_source_plan_100tb.csv"
OUT = ROOT / "actual_source_counts_100tb.csv"

QUERY = """
SELECT count(*) FROM original_images_all o
WHERE o.file_path >= %s AND o.file_path < %s
  AND o.channel_count BETWEEN 1 AND 10
  AND EXISTS (
    SELECT 1 FROM curated_100t_items i
    WHERE i.source_table_code = 1 AND i.source_id = o.id
      AND i.run_id IN (1,5) AND i.status = 'exported'
  )
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
            writer = csv.DictWriter(f, fieldnames=["dataset_id", "dataset_path", "actual_curated_sources", "elapsed_sec"])
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
                                 "actual_curated_sources": count, "elapsed_sec": round(elapsed, 3)})
                f.flush()
                print(f"{i}/{len(rows)} {row['dataset_id']} actual={count} plan_source_count={row['source_file_count']} sec={elapsed:.1f}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
