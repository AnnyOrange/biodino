#!/usr/bin/env python3
"""Publish a root-limited metadata slice for a newly mounted producer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


def filter_partition(source: Path, output_dir: Path, root: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "sources.csv"
    temporary = output_dir / ".sources.csv.tmp"
    total = selected = 0
    with source.open(newline="", encoding="utf-8") as input_file, temporary.open(
            "w", newline="", encoding="utf-8", buffering=1024 * 1024) as output_file:
        reader = csv.reader(input_file)
        header = next(reader)
        if "file_path" not in header:
            raise ValueError("Metadata has no file_path column")
        path_index = header.index("file_path")
        writer = csv.writer(output_file)
        writer.writerow(header)
        for row in reader:
            total += 1
            if len(row) != len(header):
                raise ValueError(f"Malformed metadata row {total}")
            if row[path_index].startswith(root.rstrip("/") + "/"):
                writer.writerow(row)
                selected += 1
        output_file.flush()
        os.fsync(output_file.fileno())
    if selected == 0:
        raise ValueError(f"No sources under {root}")
    os.replace(temporary, destination)
    report = {
        "status": "PASS", "source": str(source), "source_rows": total,
        "selected_rows": selected, "root": root,
        "overlap_policy": "source paths may overlap the existing producer partition",
        "created_unix": time.time(),
    }
    manifest = output_dir / ".READY.json.tmp"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(manifest, output_dir / "READY.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", default="/mnt/deepcad_nfs")
    args = parser.parse_args()
    print(json.dumps(filter_partition(args.source, args.output_dir, args.root), indent=2))


if __name__ == "__main__":
    main()
