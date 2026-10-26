#!/usr/bin/env python3
"""Partition the raw source catalog by nodes that can read each source mount."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zlib
from contextlib import ExitStack
from pathlib import Path


REQUIRED_COLUMNS = {"file_path", "source_id", "source_table_code", "frame_count"}


def partition(metadata_csv: Path, fleet_json: Path, output_dir: Path) -> dict:
    plan = json.loads(fleet_json.read_text())
    hosts = plan["hosts"]
    root_hosts = {
        root: (list(hosts) if candidates == "all" else candidates)
        for root, candidates in plan["root_hosts"].items()
    }
    if not hosts or not root_hosts:
        raise ValueError("Fleet needs hosts and source roots")
    for root, candidates in root_hosts.items():
        if not root.startswith("/") or not candidates or any(host not in hosts for host in candidates):
            raise ValueError(f"Invalid source assignment: {root}: {candidates}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "READY.json").exists():
        raise FileExistsError(f"Partition already published: {output_dir}")
    roots = sorted(root_hosts, key=len, reverse=True)
    counts = {host: 0 for host in hosts}
    by_root = {root: 0 for root in roots}
    frames_by_root = {root: 0 for root in roots}
    host_roots = {host: {root: 0 for root in roots} for host in hosts}
    unmatched = 0
    started = time.time()
    with ExitStack() as stack:
        handles = {}
        writers = {}
        for host in hosts:
            handle = stack.enter_context((output_dir / f".{host}.csv.tmp").open(
                "w", newline="", encoding="utf-8", buffering=1024 * 1024))
            handles[host] = handle
            writers[host] = csv.writer(handle)
        with metadata_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            header = next(reader)
            if not REQUIRED_COLUMNS.issubset(header):
                raise ValueError(f"Missing metadata columns: {REQUIRED_COLUMNS - set(header)}")
            path_index = header.index("file_path")
            frame_index = header.index("frame_count")
            for writer in writers.values():
                writer.writerow(header)
            for row_number, row in enumerate(reader, 1):
                if len(row) != len(header):
                    raise ValueError(f"Malformed metadata row {row_number}")
                path = row[path_index]
                root = next((root for root in roots if path.startswith(root + "/")), None)
                if root is None:
                    unmatched += 1
                    continue
                candidates = root_hosts[root]
                host = candidates[zlib.crc32(path.encode("utf-8")) % len(candidates)]
                writers[host].writerow(row)
                counts[host] += 1
                by_root[root] += 1
                host_roots[host][root] += 1
                frames_by_root[root] += max(1, int(row[frame_index] or 0))
                if row_number % 1_000_000 == 0:
                    print(f"partitioned {row_number:,} rows in {time.time() - started:.0f}s", flush=True)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())

    total = sum(counts.values()) + unmatched
    if not total or unmatched / total > 0.05 or any(count == 0 for count in counts.values()):
        raise ValueError(f"Source coverage or host assignment failed: {unmatched=}, {counts=}")
    for host in hosts:
        os.replace(output_dir / f".{host}.csv.tmp", output_dir / f"{host}.csv")
    summary = {"status": "PASS", "source": str(metadata_csv),
               "source_bytes": metadata_csv.stat().st_size,
               "fleet": str(fleet_json), "rows": total, "unmatched": unmatched,
               "host_rows": counts, "root_rows": by_root,
               "root_frame_units": frames_by_root, "host_root_rows": host_roots,
               "elapsed_seconds": time.time() - started}
    temporary = output_dir / ".READY.json.tmp"
    with temporary.open("w") as target:
        json.dump(summary, target, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output_dir / "READY.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--fleet-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(partition(args.metadata_csv, args.fleet_json, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
