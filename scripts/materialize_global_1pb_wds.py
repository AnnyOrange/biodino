#!/usr/bin/env python3
"""Materialize global 1PB file/frame candidates without dataset quotas.

Unavailable source prefixes can be excluded before any filesystem access.  The
candidate priority is unchanged, so finalization can take the first successful
million from the currently readable pool and audit every excluded candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import tifffile

from materialize_global_100tb_wds import TarWriter, array_planes, image_planes, worker_number
from probe_global_1pb_frames import read_frame_from_tif


ROOTS = ("/home/inspur/huaweimount_sam2p/", "/home/inspur/as13000_sam2p/",
         "/mnt/deepcad_nfs/", "/mnt/mount_100t_zzf_2/", "/mnt/mount_100t_zzf_3/", "/slfm/")


def source_root(path: str) -> str:
    return next((root for root in ROOTS if path.startswith(root)), "other")


def resolve_path(path: str, mappings: dict[str, str]) -> str:
    for old in sorted(mappings, key=len, reverse=True):
        prefix = old.rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            return mappings[old].rstrip("/") + path[len(prefix):]
    return path


def excluded_prefix(path: str, prefixes: tuple[str, ...]) -> str | None:
    return next((prefix for prefix in prefixes
                 if path == prefix or path.startswith(prefix + "/")), None)


def run_worker(args):
    wid, workers, csv_path, max_candidates, out_root, per_shard, seed, mappings, excluded = args
    grouped = defaultdict(list)
    writer = TarWriter(out_root, wid, per_shard)
    status_path = out_root / f"status_w{wid:02d}.csv"
    outcomes = Counter()
    roots = defaultdict(Counter)
    with status_path.open("w", newline="", buffering=65536) as handle:
        status = csv.writer(handle)
        status.writerow(["priority", "source_table_code", "source_root", "status", "reason"])

        with csv_path.open(newline="") as candidates:
            for row in csv.DictReader(candidates):
                if int(row["priority"]) >= max_candidates:
                    break
                path = row["file_path"]
                if worker_number(path, workers) != wid:
                    continue
                resolved = resolve_path(path, mappings)
                prefix = excluded_prefix(path, excluded) or excluded_prefix(resolved, excluded)
                if prefix is not None:
                    origin = source_root(path)
                    status.writerow([row["priority"], row["source_table_code"], origin,
                                     "excluded", f"excluded_prefix:{prefix}"])
                    outcomes["excluded"] += 1
                    roots[origin]["excluded"] += 1
                else:
                    grouped[path].append(row)
        handle.flush()
        last_flush = sum(outcomes.values())
        last_report = last_flush // 1000
        print(f"worker={wid} indexed_files={len(grouped)} "
              f"preexcluded={outcomes['excluded']}", flush=True)

        def record(row, array_or_bytes, extra, read_from):
            priority = int(row["priority"])
            origin = source_root(row["file_path"])
            try:
                identity = f"{row['source_table_code']}:{row['source_id']}:{row['frame_idx']}"
                hint = int(row["channel_count"] or 0)
                if isinstance(array_or_bytes, bytes):
                    planes, image_info = image_planes(array_or_bytes, identity, hint, seed)
                else:
                    planes, image_info = array_planes(array_or_bytes, identity, hint, seed)
                meta = {"pool": "1pb", "priority": priority,
                        "source_table_code": int(row["source_table_code"]),
                        "source_id": int(row["source_id"]),
                        "frame_idx": int(row["frame_idx"]) if row["frame_idx"] else None,
                        "source_path": row["file_path"], "read_from": read_from,
                        **extra, **image_info}
                writer.add(f"p{priority:08d}", planes, meta)
                status.writerow([priority, row["source_table_code"], origin, "success", ""])
                outcomes["success"] += 1
                roots[origin]["success"] += 1
            except Exception as exc:
                status.writerow([priority, row["source_table_code"], origin, "failed",
                                 f"{type(exc).__name__}:{str(exc)[:150]}"])
                outcomes["failed"] += 1
                roots[origin]["failed"] += 1

        for indexed_path, rows in grouped.items():
            resolved = resolve_path(indexed_path, mappings)
            code = rows[0]["source_table_code"]
            try:
                if code == "1":
                    with open(resolved, "rb") as source:
                        data = source.read()
                    for row in rows:
                        record(row, data, {"resolved_source_path": resolved}, "original_source")
                elif code == "2":
                    with tifffile.TiffFile(resolved) as tif:
                        for row in rows:
                            try:
                                frame, info = read_frame_from_tif(
                                    tif, int(row["frame_idx"]), int(row["channel_count"]))
                                record(row, frame, {"resolved_source_path": resolved, **info}, "2p_frame")
                            except Exception as exc:
                                origin = source_root(indexed_path)
                                status.writerow([row["priority"], code, origin, "failed",
                                                 f"frame_{type(exc).__name__}:{str(exc)[:150]}"])
                                outcomes["failed"] += 1
                                roots[origin]["failed"] += 1
                else:
                    raise ValueError(f"Unknown source_table_code {code}")
            except Exception as exc:
                origin = source_root(indexed_path)
                for row in rows:
                    status.writerow([row["priority"], code, origin, "failed",
                                     f"file_{type(exc).__name__}:{str(exc)[:150]}"])
                    outcomes["failed"] += 1
                    roots[origin]["failed"] += 1
            done = sum(outcomes.values())
            if done - last_flush >= 1000:
                handle.flush()
                last_flush = done
            if done // 1000 > last_report:
                print(f"worker={wid} candidates={done} success={outcomes['success']} "
                      f"failed={outcomes['failed']} excluded={outcomes['excluded']}", flush=True)
                last_report = done // 1000
    writer.close()
    return {"worker": wid, "candidates": sum(outcomes.values()), "outcomes": outcomes,
            "source_roots": roots, "source_files": len(grouped),
            "staged_shards": writer.shard + 1, "status_csv": str(status_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--samples-per-shard", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--path-map-json", type=Path, help="Only rewrites source mount prefixes; never changes candidates")
    p.add_argument("--exclude-prefix", action="append", default=[],
                   help="Skip this unavailable source prefix before opening any file; may be repeated")
    a = p.parse_args()
    if a.out_root.exists() and any(a.out_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {a.out_root}")
    mappings = json.loads(a.path_map_json.read_text()) if a.path_map_json else {}
    if not isinstance(mappings, dict) or any(not str(k).startswith("/") or not str(v).startswith("/")
                                             for k, v in mappings.items()):
        raise ValueError("Path mappings must be an object of absolute source and destination prefixes")
    excluded = tuple(prefix.rstrip("/") for prefix in a.exclude_prefix)
    if any(not prefix.startswith("/") or prefix == "" for prefix in excluded):
        raise ValueError("Excluded prefixes must be absolute paths")
    a.out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(i, a.workers, a.candidates, a.max_candidates, a.out_root,
             a.samples_per_shard, a.seed, mappings, excluded) for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        results = pool.map(run_worker, jobs)
    report = {"pool": "1pb", "candidate_count": a.max_candidates,
              "method": "global random priority, no dataset quotas",
              "sampling_population": "currently readable indexed images and frames",
              "path_mappings": mappings, "excluded_prefixes": excluded, "results": results}
    (a.out_root / "materialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
