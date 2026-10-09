#!/usr/bin/env python3
"""Check the realized 1M source-image sample against its eligible pool."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PIPE = Path("/home/inspur/xzj/pre_data/scalinglaw_sampling")
sys.path.insert(0, str(PIPE))
from slice_from_dataset_plan import DEFAULT_DB_CONFIG  # noqa: E402
import psycopg2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", choices=("100tb", "1pb"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--table", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", a.table):
        raise ValueError("Unsafe table identifier")
    plan_path = a.root / "manifests" / f"uniform_source_plan_{a.pool}.csv"
    with plan_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["dataset_id"]: int(r["source_file_count"]) for r in rows}
    if len(by_id) != len(rows) or len({r["dataset_path"] for r in rows}) != len(rows):
        raise ValueError("Plan has duplicate IDs or paths")
    conn = psycopg2.connect(**DEFAULT_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*), count(DISTINCT original_image_id) FROM {a.table}")
            total, unique = map(int, cur.fetchone())
            cur.execute(f"SELECT dataset_id, count(*) FROM {a.table} GROUP BY dataset_id")
            observed = {str(k): int(v) for k, v in cur.fetchall()}
    finally:
        conn.close()
    if total != 1_000_000 or unique != total or set(observed) - set(by_id):
        raise ValueError(f"Invalid sample: rows={total}, unique_source_ids={unique}, unknown_datasets={set(observed)-set(by_id)}")
    n_sources = sum(by_id.values())
    expected = {k: total * v / n_sources for k, v in by_id.items()}
    tv = sum(abs(observed.get(k, 0) / total - v / n_sources) for k, v in by_id.items()) / 2
    worst = sorted(by_id, key=lambda k: abs(observed.get(k, 0) - expected[k]), reverse=True)[:15]
    report = {
        "pool": a.pool, "table": a.table, "samples": total,
        "unique_original_image_ids": unique, "eligible_source_images": n_sources,
        "unique_dataset_paths": len(rows), "total_variation_from_eligible_source_mix": tv,
        "largest_absolute_dataset_deviations": [
            {"dataset_id": k, "eligible_sources": by_id[k],
             "observed_samples": observed.get(k, 0), "expected_samples": expected[k]}
            for k in worst
        ],
    }
    path = a.root / "manifests" / f"{a.pool}_uniformity_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if tv > 0.02:
        raise ValueError(f"Realized sample differs from eligible source mix: TV={tv:.4%}")


if __name__ == "__main__":
    main()
