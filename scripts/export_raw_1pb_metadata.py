#!/usr/bin/env python3
"""Export raw 1PB image/frame metadata for filesystem reconciliation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PIPE = Path("/home/inspur/xzj/pre_data/scalinglaw_sampling")
sys.path.insert(0, str(PIPE))
from slice_from_dataset_plan import DEFAULT_DB_CONFIG  # noqa: E402
import psycopg2  # noqa: E402


QUERIES = (
    ("1", "SELECT id,file_path,1 FROM original_images_all WHERE file_path IS NOT NULL"),
    ('2', 'SELECT id,file_path,"count" FROM original_image_all_2p_parsed WHERE file_path IS NOT NULL'),
)


def export(output: Path, fetch_size: int = 50000) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = psycopg2.connect(**DEFAULT_DB_CONFIG)
    total = {"1": 0, "2": 0}
    try:
        with output.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
            for source_code, query in QUERIES:
                cursor = connection.cursor(name=f"raw1pb_metadata_{source_code}")
                cursor.itersize = fetch_size
                cursor.execute(query)
                try:
                    for source_id, file_path, frame_count in cursor:
                        writer.writerow((file_path, source_id, source_code, frame_count))
                        total[source_code] += 1
                finally:
                    cursor.close()
    finally:
        connection.close()
    report = {"metadata_csv": str(output.resolve()), "ori_rows": total["1"],
              "two_photon_file_rows": total["2"], "total_file_rows": sum(total.values())}
    print(report, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-size", type=int, default=50000)
    args = parser.parse_args()
    if args.fetch_size < 1:
        parser.error("--fetch-size must be positive")
    export(args.output, args.fetch_size)


if __name__ == "__main__":
    main()
