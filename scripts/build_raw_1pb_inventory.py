#!/usr/bin/env python3
"""Build a resumable, metadata-only inventory of raw TIFF image files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import tifffile


TIFF_SUFFIXES = (".tif", ".tiff")


def normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def iter_tiff_paths(roots: list[Path]) -> Iterator[tuple[str, str | None]]:
    for root in roots:
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(TIFF_SUFFIXES):
                                yield entry.path, None
                        except OSError as exc:
                            yield entry.path, f"stat:{type(exc).__name__}:{str(exc)[:180]}"
            except OSError as exc:
                yield str(current), f"scandir:{type(exc).__name__}:{str(exc)[:180]}"


def inspect_tiff(path: str) -> dict:
    stat = os.stat(path)
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError("no_series")
        series = tif.series[0]
        shape = tuple(int(x) for x in series.shape)
        axes = str(series.axes)
        if not shape:
            raise ValueError("empty_shape")
        if "T" in axes:
            frame_axis = axes.index("T")
        elif len(shape) > 2 and axes[0] in "IQZ":
            frame_axis = 0
        else:
            frame_axis = None
        frames = shape[frame_axis] if frame_axis is not None else 1
        channels = shape[axes.index("C")] if "C" in axes else 0
        page_count = len(series.pages)
        if frames < 1 or page_count % frames:
            raise ValueError(f"invalid_frame_layout:{page_count}/{frames}")
        remaining_axes = [axis for i, axis in enumerate(axes) if i != frame_axis]
        remaining_shape = [size for i, size in enumerate(shape) if i != frame_axis]
        spatial = [size for axis, size in zip(remaining_axes, remaining_shape) if axis not in "SC"]
        if len(spatial) < 2:
            spatial = remaining_shape[-2:]
        height, width = spatial[-2:]
        tiled = bool(series.pages and series.pages[0].is_tiled)
    return {
        "path": normalize_path(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "frame_count": int(frames), "channel_hint": int(channels),
        "height": int(height), "width": int(width), "axes": axes,
        "shape": json.dumps(shape), "tiled": int(tiled), "readable": 1, "error": "",
    }


def create_database(path: Path, world_size: int) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-32768")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS inventory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, source_root TEXT NOT NULL, size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, frame_count INTEGER NOT NULL DEFAULT 0,
            channel_hint INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL DEFAULT 0, axes TEXT NOT NULL DEFAULT '',
            shape TEXT NOT NULL DEFAULT '', tiled INTEGER NOT NULL DEFAULT 0,
            readable INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
            rank INTEGER NOT NULL DEFAULT -1, scanned_unix REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS files_rank_path ON files(rank, readable, path);
        CREATE TABLE IF NOT EXISTS metadata_paths (
            path TEXT PRIMARY KEY, source_id TEXT, source_table_code TEXT,
            frame_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS scan_errors (
            path TEXT NOT NULL, error TEXT NOT NULL, scanned_unix REAL NOT NULL
        );
    """)
    saved_world = db.execute("SELECT value FROM inventory_meta WHERE key='world_size'").fetchone()
    if saved_world and int(saved_world[0]) != world_size:
        raise ValueError(f"Inventory world_size={saved_world[0]}, requested {world_size}; use a new index")
    db.execute("INSERT OR REPLACE INTO inventory_meta VALUES ('world_size', ?)", (str(world_size),))
    db.commit()
    return db


def import_metadata_csv(db: sqlite3.Connection, csv_path: Path) -> int:
    count = 0
    db.execute("DELETE FROM metadata_paths")
    with csv_path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "file_path" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain a file_path column")
        for row in reader:
            raw_path = row.get("file_path", "")
            if not raw_path:
                continue
            db.execute(
                "INSERT OR REPLACE INTO metadata_paths VALUES (?, ?, ?, ?)",
                (normalize_path(raw_path), row.get("source_id", ""),
                 row.get("source_table_code", ""), int(row.get("frame_count") or row.get("count") or 0)),
            )
            count += 1
            if count % 10000 == 0:
                db.commit()
    db.commit()
    return count


def build(roots: list[Path], output: Path, world_size: int, workers: int,
          metadata_csv: Path | None = None, batch_size: int = 512) -> dict:
    roots = [Path(normalize_path(str(root))) for root in roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise FileNotFoundError("Every --root must be an accessible directory")
    roots = list(dict.fromkeys(roots))
    db = create_database(output, world_size)
    try:
        normalized_roots = [normalize_path(str(root)) for root in roots]
        saved_roots = db.execute("SELECT value FROM inventory_meta WHERE key='roots'").fetchone()
        if saved_roots and json.loads(saved_roots[0]) != normalized_roots:
            raise ValueError("Inventory roots changed; use a new index to avoid mixing stale roots")
        db.execute("INSERT OR REPLACE INTO inventory_meta VALUES ('roots', ?)",
                   (json.dumps(normalized_roots),))
        started_unix = time.time()
        db.execute("DELETE FROM scan_errors")
        db.execute("INSERT OR REPLACE INTO inventory_meta VALUES ('status','SCANNING')")
        db.commit()
        if metadata_csv:
            import_metadata_csv(db, metadata_csv)
        loads = [0] * world_size
        for rank, units in db.execute(
                "SELECT rank, COALESCE(SUM(frame_count),0) FROM files "
                "WHERE readable=1 AND rank>=0 GROUP BY rank"):
            loads[int(rank)] = int(units)

        pending: list[tuple[str, str | None]] = []
        seen = 0
        added = refreshed = unchanged = failed = 0
        started = time.monotonic()

        def flush(rows: list[tuple[str, str | None]]) -> None:
            nonlocal seen, added, refreshed, unchanged, failed
            todo = []
            for raw_path, walk_error in rows:
                seen += 1
                path = normalize_path(raw_path)
                if walk_error:
                    db.execute("INSERT INTO scan_errors VALUES (?, ?, ?)",
                               (path, walk_error, time.time()))
                    failed += 1
                    continue
                try:
                    stat = os.stat(path)
                except OSError as exc:
                    db.execute("INSERT INTO scan_errors VALUES (?, ?, ?)",
                               (path, f"stat:{type(exc).__name__}:{str(exc)[:180]}", time.time()))
                    failed += 1
                    continue
                previous = db.execute("SELECT size,mtime_ns FROM files WHERE path=?", (path,)).fetchone()
                if previous == (stat.st_size, stat.st_mtime_ns):
                    unchanged += 1
                    db.execute("UPDATE files SET scanned_unix=? WHERE path=?", (time.time(), path))
                    continue
                todo.append((path, str(next((str(root) for root in roots
                                             if path == normalize_path(str(root)) or
                                             path.startswith(normalize_path(str(root)) + os.sep)), roots[0]))))

            def one(item: tuple[str, str]):
                path, source_root = item
                try:
                    result = inspect_tiff(path)
                    result["source_root"] = source_root
                    return result
                except Exception as exc:
                    return {"path": path, "source_root": source_root, "size": 0,
                            "mtime_ns": 0, "frame_count": 0, "channel_hint": 0,
                            "height": 0, "width": 0, "axes": "", "shape": "",
                            "tiled": 0, "readable": 0,
                            "error": f"{type(exc).__name__}:{str(exc)[:180]}"}

            for result in executor.map(one, todo):
                if not result["readable"]:
                    failed += 1
                previous = db.execute(
                    "SELECT rank,frame_count,readable FROM files WHERE path=?", (result["path"],)
                ).fetchone()
                if previous and int(previous[2]):
                    loads[int(previous[0])] -= int(previous[1])
                if result["readable"]:
                    if previous and int(previous[0]) >= 0:
                        rank = int(previous[0])
                        refreshed += 1
                    else:
                        rank = min(range(world_size), key=lambda candidate: loads[candidate])
                        added += 1
                    result["rank"] = rank
                    loads[rank] += int(result["frame_count"])
                else:
                    result["rank"] = -1
                db.execute("""
                    INSERT OR REPLACE INTO files
                    (path,source_root,size,mtime_ns,frame_count,channel_hint,height,width,
                     axes,shape,tiled,readable,error,rank,scanned_unix)
                    VALUES (:path,:source_root,:size,:mtime_ns,:frame_count,:channel_hint,
                            :height,:width,:axes,:shape,:tiled,:readable,:error,:rank,:scanned_unix)
                """, {**result, "scanned_unix": time.time()})
            db.commit()
            if seen and seen % (batch_size * 10) < batch_size:
                print(f"seen={seen:,} new={added:,} refreshed={refreshed:,} "
                      f"unchanged={unchanged:,} unreadable={failed:,}", flush=True)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for item in iter_tiff_paths(roots):
                pending.append(item)
                if len(pending) >= batch_size:
                    flush(pending)
                    pending = []
            if pending:
                flush(pending)

        readable, units = db.execute(
            "SELECT count(*), COALESCE(sum(frame_count),0) FROM files WHERE readable=1").fetchone()
        filesystem_scan_errors = int(db.execute(
            "SELECT count(*) FROM scan_errors"
        ).fetchone()[0])
        if not filesystem_scan_errors:
            for root in roots:
                normalized_root = normalize_path(str(root))
                db.execute("""
                    UPDATE files SET readable=0,rank=-1,error='not_seen_in_latest_scan'
                    WHERE source_root=? AND scanned_unix<? AND readable=1
                """, (normalized_root, started_unix))
            db.commit()
            readable, units = db.execute(
                "SELECT count(*), COALESCE(sum(frame_count),0) FROM files WHERE readable=1").fetchone()
            loads = [0] * world_size
            for rank, rank_units in db.execute(
                    "SELECT rank, COALESCE(SUM(frame_count),0) FROM files "
                    "WHERE readable=1 AND rank>=0 GROUP BY rank"):
                loads[int(rank)] = int(rank_units)
        matched, metadata_only, inventory_only, frame_mismatch = db.execute("""
            SELECT
                SUM(CASE WHEN f.path IS NOT NULL AND f.readable=1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN f.path IS NULL OR f.readable=0 THEN 1 ELSE 0 END),
                (SELECT count(*) FROM files f2 LEFT JOIN metadata_paths m2 ON m2.path=f2.path
                 WHERE f2.readable=1 AND m2.path IS NULL),
                SUM(CASE WHEN f.path IS NOT NULL AND f.readable=1 AND m.frame_count>0
                          AND m.frame_count!=f.frame_count THEN 1 ELSE 0 END)
            FROM metadata_paths m LEFT JOIN files f ON f.path=m.path
        """).fetchone() if db.execute("SELECT 1 FROM metadata_paths LIMIT 1").fetchone() else (0, 0, 0, 0)
        report = {
            "status": "PASS" if int(readable) and not filesystem_scan_errors else "INCOMPLETE",
            "format": "raw_1pb_tiff_inventory_v1",
            "roots": [normalize_path(str(root)) for root in roots],
            "world_size": world_size, "readable_files": int(readable),
            "readable_image_or_frame_units": int(units),
            "metadata_rows_matched_to_readable_files": int(matched or 0),
            "metadata_paths_missing_or_unreadable": int(metadata_only or 0),
            "filesystem_files_absent_from_metadata": int(inventory_only or 0),
            "metadata_frame_count_mismatches": int(frame_mismatch or 0),
            "filesystem_scan_errors": filesystem_scan_errors,
            "rank_frame_units": loads, "scan_entries": seen,
            "unchanged_files_reused": unchanged, "new_files": added,
            "refreshed_files": refreshed, "unreadable_or_walk_errors": failed,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "index_db": str(output.resolve()),
        }
        report_path = output.with_suffix(output.suffix + ".report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        db.execute("INSERT OR REPLACE INTO inventory_meta VALUES ('status',?)", (report["status"],))
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True,
                        help="A data-only root to inventory; repeat for each mounted source")
    parser.add_argument("--output", type=Path, required=True, help="SQLite index path")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--metadata-csv", type=Path,
                        help="Optional export with file_path and optional count/source_id fields")
    args = parser.parse_args()
    if args.world_size < 1 or args.workers < 1 or args.batch_size < 1:
        parser.error("world-size, workers and batch-size must be positive")
    build(args.root, args.output, args.world_size, args.workers, args.metadata_csv, args.batch_size)


if __name__ == "__main__":
    main()
