#!/usr/bin/env python3
"""Wait for the raw inventory, then run the bounded 1PB WDS producer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rolling_1pb_wds import produce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=Path("/mnt/huawei_blm/rolling_1pb_wds_v1/inventory/raw_1pb_inventory.sqlite"))
    parser.add_argument("--metadata-csv", type=Path,
                        default=Path("/mnt/huawei_blm/rolling_1pb_wds_v1/inventory/raw_1pb_source_metadata.csv"))
    parser.add_argument("--preparation-report", type=Path,
                        default=Path("/mnt/huawei_blm/rolling_1pb_wds_v1/inventory/preparation.json"))
    parser.add_argument("--queue-root", type=Path,
                        default=Path("/mnt/huawei_blm/rolling_1pb_wds_v1/queue"))
    parser.add_argument("--batch-samples", type=int, default=499_712)
    parser.add_argument("--max-live-batches", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples-per-shard", type=int, default=1952)
    parser.add_argument("--file-block", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--daemon", action="store_true",
                        help="Keep producing batches as the trainer reclaims queue capacity")
    args = parser.parse_args()
    while True:
        if args.preparation_report.is_file():
            report = json.loads(args.preparation_report.read_text())
            if report.get("state") in ("INCOMPLETE", "ERROR"):
                raise SystemExit(f"Raw inventory preparation failed: {args.preparation_report}")
            metadata_ready = (report.get("metadata_export") == "PASS" and
                              args.metadata_csv.is_file())
            inventory_ready = report.get("state") == "PASS" and args.index.is_file()
            if metadata_ready or inventory_ready:
                break
        print("waiting for metadata export or raw 1PB inventory", flush=True)
        time.sleep(args.poll_seconds)
    produce(args)


if __name__ == "__main__":
    main()
