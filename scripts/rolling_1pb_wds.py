#!/usr/bin/env python3
"""Bounded producer/consumer queue for cropped 1PB WebDataset batches."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import random
import sqlite3
import tarfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

from materialize_global_100tb_wds import TarWriter, array_planes


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _queue_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".queue.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _directories(root: Path) -> None:
    for name in (".building", "ready", "active", ".reaping", "completed"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _live_count(root: Path) -> int:
    return sum(len(list((root / name).glob("batch-*")))
               for name in (".building", "ready", "active", ".reaping"))


def _inventory_rows(index_path: Path) -> list[tuple]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM inventory_meta"))
        if metadata.get("status") != "PASS":
            raise ValueError("1PB inventory status is not PASS")
        rows = list(connection.execute("""
            SELECT f.path, f.frame_count, f.channel_hint,
                   m.source_id, m.source_table_code
            FROM files f LEFT JOIN metadata_paths m ON m.path=f.path
            WHERE f.readable=1 AND f.frame_count>0
        """))
        if not rows:
            raise ValueError("1PB inventory has no readable TIFF files")
        return rows
    finally:
        connection.close()


def _metadata_rows(metadata_csv: Path) -> list[tuple]:
    rows = []
    with metadata_csv.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"file_path", "source_id", "source_table_code", "frame_count"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata CSV is missing columns: {required}")
        for row in reader:
            raw_path = row.get("file_path", "")
            if not raw_path:
                continue
            rows.append((os.path.abspath(raw_path),
                         max(1, int(row.get("frame_count") or 0)), 0,
                         row.get("source_id", ""), row.get("source_table_code", "")))
    if not rows:
        raise ValueError(f"Metadata CSV has no source paths: {metadata_csv}")
    return rows


def _inventory_is_ready(index_path: Path) -> bool:
    if not index_path.is_file():
        return False
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=5)
    try:
        row = connection.execute(
            "SELECT value FROM inventory_meta WHERE key='status'").fetchone()
        return bool(row and row[0] == "PASS")
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def _assign_files(rows: list[tuple], workers: int, seed: int) -> list[list[tuple]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    assignments: list[list[tuple]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for row in sorted(rows, key=lambda x: int(x[1]), reverse=True):
        owner = min(range(workers), key=lambda i: loads[i])
        assignments[owner].append(row)
        loads[owner] += max(1, int(row[1]))
    return assignments


def _source_root(path: str, roots: list[str]) -> str:
    return next((root for root in roots if path.startswith(root + "/")), "")


def _root_counts_for_shard(weights: dict[str, int], shard_index: int,
                           samples_per_shard: int) -> dict[str, int]:
    total_weight = sum(weights.values())
    if not weights or total_weight <= 0:
        raise ValueError("Source weights must be positive")
    dominant = max(weights, key=weights.get)
    counts = {
        root: (((shard_index + 1) * samples_per_shard * weight) // total_weight -
               (shard_index * samples_per_shard * weight) // total_weight)
        for root, weight in weights.items() if root != dominant
    }
    counts[dominant] = samples_per_shard - sum(counts.values())
    return counts


def _read_frame(tif: tifffile.TiffFile, frame_idx: int) -> tuple[np.ndarray, dict]:
    series = tif.series[0]
    axes, shape = series.axes, series.shape
    if "T" in axes:
        frame_count = int(shape[axes.index("T")])
    elif len(shape) > 2 and axes[0] in "IQZ":
        frame_count = int(shape[0])
    else:
        frame_count = 1
    if frame_idx < 0 or frame_idx >= frame_count:
        raise ValueError(f"frame {frame_idx} outside 0..{frame_count - 1}")
    if len(series.pages) % frame_count:
        raise ValueError(f"invalid frame layout: {len(series.pages)}/{frame_count}")
    pages_per_frame = len(series.pages) // frame_count
    start = frame_idx * pages_per_frame
    frame = np.squeeze(series.asarray(key=slice(start, start + pages_per_frame)))
    if frame.ndim not in (2, 3):
        raise ValueError(f"unsupported extracted frame shape {frame.shape}")
    return frame, {"series_axes": axes, "series_shape": list(shape),
                   "frame_shape": list(frame.shape), "dtype": str(frame.dtype)}


def _worker(args: tuple) -> dict:
    if len(args) > 8 and args[8]:
        return _weighted_worker(args)
    worker_id, batch_name, rows, quota, out_dir, samples_per_shard, file_block, seed = args[:8]
    rng = random.Random(seed + worker_id * 1_000_003)
    rows = list(rows)
    writer = TarWriter(out_dir, worker_id, samples_per_shard)
    failures_path = out_dir / f"failures_w{worker_id:02d}.jsonl"
    provenance_path = out_dir / "provenance.jsonl"
    provenance = provenance_path.open("w", buffering=65536)
    failures = files_seen = passes = 0
    count = 0

    def fail(path: str, frame_idx: int | None, error: Exception) -> None:
        nonlocal failures
        failures += 1
        with failures_path.open("a", buffering=1) as output:
            output.write(json.dumps({
                "source_path": path,
                "frame_idx": frame_idx,
                "reason": f"{type(error).__name__}:{str(error)[:240]}",
                "time_unix": time.time(),
            }, ensure_ascii=False) + "\n")

    try:
        while count < quota:
            passes += 1
            rng.shuffle(rows)
            pass_success = 0
            for block_start in range(0, len(rows), file_block):
                block = rows[block_start:block_start + file_block]
                rng.shuffle(block)
                for path, frame_count, channel_hint, source_id, source_code in block:
                    if count >= quota:
                        break
                    files_seen += 1
                    try:
                        with tifffile.TiffFile(path) as tif:
                            indices = list(range(int(frame_count)))
                            rng.shuffle(indices)
                            for frame_idx in indices:
                                if count >= quota:
                                    break
                                try:
                                    frame, frame_info = _read_frame(tif, frame_idx)
                                    identity = (f"{source_code or 'fs'}:{source_id or path}:"
                                                f"{frame_idx}")
                                    crop_seed = seed + passes * 1_000_000_007
                                    planes, crop_info = array_planes(
                                        frame, identity, int(channel_hint or 0), crop_seed)
                                    key = hashlib.blake2b(
                                        f"{batch_name}|{identity}|{worker_id}|{count}".encode(),
                                        digest_size=12).hexdigest()
                                    meta = {
                                        "pool": "1pb", "source_path": path,
                                        "source_id": source_id,
                                        "source_table_code": source_code,
                                        "frame_idx": frame_idx,
                                        "source_identity": identity,
                                        "batch_id": batch_name,
                                        "worker_id": worker_id,
                                        "frame_info": frame_info,
                                        **crop_info,
                                    }
                                    writer.add(key, planes, meta)
                                    provenance.write(json.dumps({
                                        "key": key, "source_path": path,
                                        "source_id": source_id,
                                        "source_table_code": source_code,
                                        "frame_idx": frame_idx,
                                        "source_identity": identity,
                                        "crop_yx": crop_info["crop_yx"],
                                        "original_shape": crop_info["original_shape"],
                                    }, separators=(",", ":"), ensure_ascii=False) + "\n")
                                    count += 1
                                    pass_success += 1
                                except Exception as exc:
                                    fail(path, frame_idx, exc)
                    except Exception as exc:
                        fail(path, None, exc)
            if count < quota and pass_success == 0:
                raise RuntimeError(f"worker {worker_id} found no readable samples in pass {passes}")
    finally:
        writer.close()
        provenance.flush()
        os.fsync(provenance.fileno())
        provenance.close()
    if count != quota:
        raise RuntimeError(f"worker {worker_id} produced {count}, expected {quota}")
    return {"worker": worker_id, "samples": count, "files_seen": files_seen,
            "passes": passes, "failures": failures,
            "tar_shards": writer.shard + 1,
            "failure_log": str(failures_path) if failures_path.exists() else None}


def _weighted_worker(args: tuple) -> dict:
    (worker_id, batch_name, rows, quota, out_dir, samples_per_shard, _file_block,
     seed, weights, aliases) = args
    rng = random.Random(seed + worker_id * 1_000_003)
    roots = sorted(weights, key=len, reverse=True)
    grouped = {root: [] for root in roots}
    for row in rows:
        root = _source_root(row[0], roots)
        if root:
            grouped[root].append(row)
    if any(not grouped[root] for root in roots):
        raise ValueError(f"Worker {worker_id} lacks files for a weighted source root")
    states = {}
    for root, candidates in grouped.items():
        rng.shuffle(candidates)
        states[root] = {"rows": candidates, "position": 0, "passes": 1,
                        "tif": None, "indices": [], "row": None}
    writer = TarWriter(out_dir, worker_id, samples_per_shard)
    failures_path = out_dir / f"failures_w{worker_id:02d}.jsonl"
    provenance_path = out_dir / "provenance.jsonl"
    counts = {root: 0 for root in roots}
    failures = files_seen = 0
    available_roots = set(roots)
    unavailable_roots = []

    def fail(path: str, read_path: str, frame_idx: int | None, error: Exception) -> None:
        nonlocal failures
        failures += 1
        with failures_path.open("a", buffering=1) as output:
            output.write(json.dumps({
                "source_path": path, "read_path": read_path,
                "frame_idx": frame_idx,
                "reason": f"{type(error).__name__}:{str(error)[:240]}",
                "time_unix": time.time(),
            }, ensure_ascii=False) + "\n")

    def next_sample(root: str):
        nonlocal files_seen
        state = states[root]
        alias = aliases.get(root, root)
        failed_attempts = 0
        while failed_attempts < max(32, min(2048, len(state["rows"]) * 2)):
            if not state["indices"]:
                if state["tif"] is not None:
                    state["tif"].close()
                    state["tif"] = None
                if state["position"] == len(state["rows"]):
                    rng.shuffle(state["rows"])
                    state["position"] = 0
                    state["passes"] += 1
                row = state["rows"][state["position"]]
                state["position"] += 1
                state["row"] = row
                path, frame_count, *_ = row
                read_path = alias + path[len(root):]
                files_seen += 1
                try:
                    state["tif"] = tifffile.TiffFile(read_path)
                    state["indices"] = list(range(int(frame_count)))
                    rng.shuffle(state["indices"])
                except Exception as exc:
                    fail(path, read_path, None, exc)
                    failed_attempts += 1
                    continue
            path, _frame_count, channel_hint, source_id, source_code = state["row"]
            read_path = alias + path[len(root):]
            frame_idx = state["indices"].pop()
            try:
                frame, frame_info = _read_frame(state["tif"], frame_idx)
                identity = f"{source_code or 'fs'}:{source_id or path}:{frame_idx}"
                planes, crop_info = array_planes(
                    frame, identity, int(channel_hint or 0),
                    seed + state["passes"] * 1_000_000_007)
                return (path, read_path, source_id, source_code, frame_idx,
                        identity, planes, frame_info, crop_info)
            except Exception as exc:
                fail(path, read_path, frame_idx, exc)
                failed_attempts += 1
        raise RuntimeError(f"Worker {worker_id} found no readable samples for {root}")

    def sample_for(requested_root: str):
        root = requested_root
        while True:
            if root not in available_roots:
                choices = sorted(available_roots)
                if not choices:
                    raise RuntimeError(f"Worker {worker_id} has no readable source roots")
                root = rng.choices(choices, weights=[weights[item] for item in choices])[0]
            try:
                return root, next_sample(root)
            except RuntimeError as exc:
                available_roots.remove(root)
                unavailable_roots.append(root)
                print(f"{batch_name} worker {worker_id}: skipping unreadable root {root}: {exc}",
                      flush=True)

    try:
        with provenance_path.open("w", buffering=65536) as provenance:
            for shard_index in range(quota // samples_per_shard):
                schedule = [root for root, target in
                            _root_counts_for_shard(weights, shard_index,
                                                   samples_per_shard).items()
                            for _ in range(target)]
                rng.shuffle(schedule)
                for requested_root in schedule:
                    root, sample = sample_for(requested_root)
                    (path, read_path, source_id, source_code, frame_idx, identity,
                     planes, frame_info, crop_info) = sample
                    count = writer.count
                    key = hashlib.blake2b(
                        f"{batch_name}|{identity}|{worker_id}|{count}".encode(),
                        digest_size=12).hexdigest()
                    meta = {
                        "pool": "1pb", "source_path": path, "read_path": read_path,
                        "source_root": root, "source_id": source_id,
                        "source_table_code": source_code, "frame_idx": frame_idx,
                        "source_identity": identity, "batch_id": batch_name,
                        "worker_id": worker_id, "frame_info": frame_info,
                        **crop_info,
                    }
                    writer.add(key, planes, meta)
                    provenance.write(json.dumps({
                        "key": key, "source_path": path, "read_path": read_path,
                        "source_root": root, "source_id": source_id,
                        "source_table_code": source_code, "frame_idx": frame_idx,
                        "source_identity": identity, "crop_yx": crop_info["crop_yx"],
                        "original_shape": crop_info["original_shape"],
                    }, separators=(",", ":"), ensure_ascii=False) + "\n")
                    counts[root] += 1
            provenance.flush()
            os.fsync(provenance.fileno())
    finally:
        writer.close()
        for state in states.values():
            if state["tif"] is not None:
                state["tif"].close()
    if writer.count != quota:
        raise RuntimeError(f"Worker {worker_id} produced {writer.count}, expected {quota}")
    return {"worker": worker_id, "samples": writer.count, "files_seen": files_seen,
            "passes": max(state["passes"] for state in states.values()),
            "failures": failures, "tar_shards": writer.shard + 1,
            "root_sample_counts": counts,
            "unavailable_roots": unavailable_roots,
            "failure_log": str(failures_path) if failures_path.exists() else None}


def _verify_shard(tar_path: Path, samples_per_shard: int,
                  root_weights: dict[str, int] | None = None) -> tuple[int, dict, set[str], dict[str, int]]:
    worker_id = int(tar_path.name.split("-")[1][1:])
    groups: dict[str, dict] = defaultdict(dict)
    root_counts: dict[str, int] = defaultdict(int)
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.size <= 0:
                raise ValueError(f"Invalid tar member {tar_path}:{member.name}")
            name = Path(member.name).name
            if name.endswith(".meta.json"):
                suffix = "meta"
                key = name[:-len(".meta.json")]
            elif ".ch" in name and name.endswith(".tif"):
                key, channel = name.rsplit(".ch", 1)
                suffix = f"ch{channel[:-4]}"
            else:
                raise ValueError(f"Unexpected WDS member {tar_path}:{member.name}")
            if suffix in groups[key]:
                raise ValueError(f"Duplicate WDS member {tar_path}:{member.name}")
            groups[key][suffix] = member
        for key, members in groups.items():
            meta_member = members.get("meta")
            if meta_member is None:
                raise ValueError(f"Sample has no metadata: {tar_path}:{key}")
            source = archive.extractfile(meta_member)
            if source is None:
                raise ValueError(f"Unreadable metadata: {tar_path}:{key}")
            meta = json.load(source)
            required = ("source_path", "source_identity", "frame_idx", "crop_yx",
                        "original_shape", "channels_written")
            if meta.get("pool") != "1pb" or any(field not in meta for field in required):
                raise ValueError(f"Missing provenance fields: {tar_path}:{key}")
            if root_weights is not None:
                root = meta.get("source_root")
                if root not in root_weights or not meta["source_path"].startswith(root + "/"):
                    raise ValueError(f"Invalid source root: {tar_path}:{key}")
                root_counts[root] += 1
            expected_members = {"meta"} | {
                f"ch{i}" for i in range(1, int(meta["channels_written"]) + 1)}
            if set(members) != expected_members:
                raise ValueError(f"Incomplete sample {tar_path}:{key}: {set(members)}")
            y0, x0, y1, x1 = map(int, meta["crop_yx"])
            if not (0 <= y0 < y1 and 0 <= x0 < x1 and
                    y1 - y0 <= 512 and x1 - x0 <= 512):
                raise ValueError(f"Invalid 512 crop provenance: {tar_path}:{key}")
    if len(groups) != samples_per_shard:
        raise ValueError(f"{tar_path} has {len(groups)} samples, expected {samples_per_shard}")
    return worker_id, {"path": tar_path.name, "samples": len(groups),
                       "bytes": tar_path.stat().st_size}, set(groups), dict(root_counts)


def _verify_batch(root: Path, batch_name: str, expected_samples: int,
                  samples_per_shard: int, worker_reports: list[dict],
                  verify_workers: int = 4,
                  root_weights: dict[str, int] | None = None) -> dict:
    tar_paths = sorted((root / "shards").glob("*.tar"))
    if not tar_paths:
        raise RuntimeError("No WDS tar shards were produced")
    shards = []
    worker_sample_keys: dict[int, set[str]] = defaultdict(set)
    verified_root_counts: dict[str, int] = defaultdict(int)
    verify_started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, verify_workers)) as executor:
        results = executor.map(
            lambda path: _verify_shard(path, samples_per_shard, root_weights), tar_paths)
        for worker_id, shard, keys, root_counts in results:
            if worker_sample_keys[worker_id].intersection(keys):
                raise ValueError(f"Duplicate sample key across shards for worker {worker_id}")
            worker_sample_keys[worker_id].update(keys)
            shards.append(shard)
            for source_root, count in root_counts.items():
                verified_root_counts[source_root] += count
            if len(shards) % 32 == 0 or len(shards) == len(tar_paths):
                print(f"verifying {batch_name}: {len(shards)}/{len(tar_paths)} shards "
                      f"in {time.time() - verify_started:.0f}s", flush=True)
    samples = sum(shard["samples"] for shard in shards)
    if samples != expected_samples:
        raise ValueError(f"WDS batch has {samples} samples, expected {expected_samples}")
    if root_weights is not None:
        reported = {root: sum(report["root_sample_counts"][root]
                              for report in worker_reports) for root in root_weights}
        if any(verified_root_counts[root] != count for root, count in reported.items()):
            raise ValueError("Verified source counts differ from worker reports")
    provenance_files = []
    for report in worker_reports:
        provenance_path = root / f"provenance_w{int(report['worker']):02d}.jsonl"
        provenance_count = 0
        provenance_keys = set()
        with provenance_path.open() as source:
            for line in source:
                record = json.loads(line)
                if any(field not in record for field in (
                        "key", "source_path", "frame_idx", "source_identity", "crop_yx")):
                    raise ValueError(f"Incomplete provenance row in {provenance_path}")
                if root_weights is not None:
                    source_root = record.get("source_root")
                    if (source_root not in root_weights or
                            not record["source_path"].startswith(source_root + "/")):
                        raise ValueError(f"Invalid provenance source in {provenance_path}")
                key = str(record["key"])
                if key in provenance_keys or key not in worker_sample_keys[int(report["worker"])]:
                    raise ValueError(f"Provenance key mismatch in {provenance_path}: {key}")
                provenance_keys.add(key)
                provenance_count += 1
        if provenance_count != int(report["samples"]) or \
                provenance_keys != worker_sample_keys[int(report["worker"])]:
            raise ValueError(f"{provenance_path} has {provenance_count} rows, expected {report['samples']}")
        provenance_files.append({"path": provenance_path.name,
                                 "samples": provenance_count,
                                 "bytes": provenance_path.stat().st_size})
    return {"status": "PASS", "batch_id": batch_name, "samples": samples,
            "samples_per_shard": samples_per_shard, "shard_count": len(shards),
            "shards": shards, "workers": worker_reports,
            "root_sample_counts": dict(verified_root_counts),
            "root_frame_weights": root_weights,
            "source_mix_policy": ("weighted_with_unreadable_root_fallback"
                                  if root_weights is not None else None),
            "provenance_files": provenance_files,
            "verified_unix": time.time()}


def produce_batch(index_path: Path, queue_root: Path, batch_samples: int,
                  max_live_batches: int, workers: int, samples_per_shard: int,
                  file_block: int, seed: int,
                  inventory_rows: list[tuple] | None = None,
                  file_assignments: list[list[tuple]] | None = None,
                  source_pool: str = "raw_inventory",
                  verify_workers: int = 4,
                  root_weights: dict[str, int] | None = None,
                  root_aliases: dict[str, str] | None = None,
                  min_free_gib: int = 0) -> str:
    if min(batch_samples, max_live_batches, workers, samples_per_shard,
           file_block, verify_workers) < 1:
        raise ValueError("Batch, cache, worker, shard, and block sizes must be positive")
    if batch_samples % workers or (batch_samples // workers) % samples_per_shard:
        raise ValueError("batch_samples must divide evenly into worker quotas and full shards")
    _directories(queue_root)
    lock = _queue_lock(queue_root)
    try:
        if _live_count(queue_root) >= max_live_batches:
            return "FULL"
        if min_free_gib:
            stats = os.statvfs(queue_root)
            free_bytes = stats.f_bavail * stats.f_frsize
            building = len(list((queue_root / ".building").glob("batch-*")))
            reserved_bytes = (building + 1) * 256 * 1024**3
            if free_bytes < min_free_gib * 1024**3 + reserved_bytes:
                return "LOW_SPACE"
        state_path = queue_root / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"next_batch": 1}
        existing = [int(path.name.removeprefix("batch-"))
                    for name in (".building", "ready", "active", ".reaping", "completed")
                    for path in (queue_root / name).glob("batch-*")
                    if path.name.removeprefix("batch-").isdigit()]
        batch_number = max(int(state["next_batch"]), max(existing, default=0) + 1)
        while True:
            batch_name = f"batch-{batch_number:06d}"
            build_root = queue_root / ".building" / batch_name
            try:
                build_root.mkdir()
                break
            except FileExistsError:
                batch_number += 1
        state["next_batch"] = batch_number + 1
        write_json_atomic(state_path, state)
        if _live_count(queue_root) > max_live_batches:
            build_root.rmdir()
            return "FULL"
        write_json_atomic(build_root / "BUILDING.json", {
            "status": "BUILDING", "batch_id": batch_name,
            "target_samples": batch_samples, "seed": seed,
            "index_path": str(index_path), "source_pool": source_pool,
            "root_frame_weights": root_weights,
            "root_aliases": root_aliases,
            "started_unix": time.time(),
        })
    finally:
        lock.close()

    if file_assignments is None:
        rows = inventory_rows if inventory_rows is not None else _inventory_rows(index_path)
        assignments = _assign_files(rows, workers, seed)
    else:
        if len(file_assignments) != workers:
            raise ValueError("file_assignments length must match workers")
        assignments = file_assignments

    worker_root = build_root / "workers"
    shard_root = build_root / "shards"
    worker_root.mkdir()
    shard_root.mkdir()
    quota = batch_samples // workers
    jobs = [(worker_id, batch_name, assignments[worker_id], quota,
             worker_root / f"w{worker_id:02d}", samples_per_shard,
             file_block, seed, root_weights, root_aliases or {})
            for worker_id in range(workers)]
    for job in jobs:
        Path(job[4]).mkdir()
    context = mp.get_context("fork")
    with context.Pool(workers) as pool:
        reports = pool.map(_worker, jobs)
    for worker_id in range(workers):
        source_dir = worker_root / f"w{worker_id:02d}"
        for source in sorted(source_dir.glob("staged_w*.tar")):
            source.replace(shard_root / f"shard-w{worker_id:02d}-{source.stem.rsplit('_', 1)[-1]}.tar")
        (source_dir / "provenance.jsonl").replace(
            build_root / f"provenance_w{worker_id:02d}.jsonl")
        failure_log = source_dir / f"failures_w{worker_id:02d}.jsonl"
        if failure_log.exists():
            failure_log.replace(build_root / failure_log.name)
            reports[worker_id]["failure_log"] = str(
                queue_root / "ready" / batch_name / failure_log.name)
        source_dir.rmdir()
    worker_root.rmdir()

    summary = _verify_batch(build_root, batch_name, batch_samples,
                            samples_per_shard, reports, verify_workers, root_weights)
    summary.update({"seed": seed, "inventory_path": str(index_path),
                    "source_pool": source_pool,
                    "created_unix": time.time(), "failure_count": sum(x["failures"] for x in reports)})
    write_json_atomic(build_root / "READY.json", summary)
    (build_root / "BUILDING.json").unlink()
    ready_path = queue_root / "ready" / batch_name
    os.replace(build_root, ready_path)
    return batch_name


def produce(args) -> None:
    pool_kind = None
    rows = assignments = None
    root_weights = None
    if args.partition_summary is not None:
        partition = json.loads(args.partition_summary.read_text())
        if partition.get("status") != "PASS" or not partition.get("root_frame_units"):
            raise ValueError(f"Invalid global partition summary: {args.partition_summary}")
        root_weights = {root: int(weight) for root, weight in
                        partition["root_frame_units"].items()}
        for root in args.exclude_root:
            if root not in root_weights:
                raise ValueError(f"Unknown source root to exclude: {root}")
            del root_weights[root]
        if not root_weights:
            raise ValueError("No mounted source roots remain")
    root_aliases = dict(pair.split("=", 1) for pair in args.root_alias)
    while True:
        if _live_count(args.queue_root) >= args.max_live_batches:
            print(f"queue full ({args.max_live_batches} live groups); waiting", flush=True)
            if not args.daemon:
                return
            time.sleep(args.poll_seconds)
            continue
        if args.min_free_gib:
            stats = os.statvfs(args.queue_root)
            free_gib = stats.f_bavail * stats.f_frsize // 1024**3
            if free_gib < args.min_free_gib + 256:
                print(f"low space ({free_gib} GiB free); waiting", flush=True)
                if not args.daemon:
                    return
                time.sleep(args.poll_seconds)
                continue
        ready = _inventory_is_ready(args.index) if root_weights is None else False
        next_pool = ("global_partition" if root_weights is not None else
                     "raw_inventory" if ready else "metadata_candidates")
        if pool_kind != next_pool:
            rows = (_inventory_rows(args.index) if ready
                    else _metadata_rows(args.metadata_csv))
            assignments = _assign_files(rows, args.workers, args.seed)
            pool_kind = next_pool
            print(f"loaded {len(rows):,} {pool_kind} source files", flush=True)
        batch_name = produce_batch(args.index, args.queue_root, args.batch_samples,
                                   args.max_live_batches, args.workers,
                                   args.samples_per_shard, args.file_block, args.seed,
                                   inventory_rows=rows, file_assignments=assignments,
                                   source_pool=pool_kind,
                                   verify_workers=args.verify_workers,
                                   root_weights=root_weights,
                                   root_aliases=root_aliases,
                                   min_free_gib=args.min_free_gib)
        if batch_name in ("FULL", "LOW_SPACE"):
            print(f"queue admission paused ({batch_name}); waiting", flush=True)
            if not args.daemon:
                return
            time.sleep(args.poll_seconds)
            continue
        print(f"published {batch_name}", flush=True)
        args.seed += 1_000_000_007
        if not args.daemon:
            return


def claim(queue_root: Path) -> Path | None:
    _directories(queue_root)
    lock = _queue_lock(queue_root)
    try:
        active = list((queue_root / "active").glob("batch-*"))
        if active:
            raise RuntimeError(f"A batch is already active: {active[0]}")
        ready = sorted((queue_root / "ready").glob("batch-*"))
        if not ready:
            return None
        source = ready[0]
        if not (source / "READY.json").is_file():
            raise ValueError(f"Ready directory lacks READY.json: {source}")
        target = queue_root / "active" / source.name
        os.replace(source, target)
        write_json_atomic(target / "CLAIMED.json", {"status": "ACTIVE", "claimed_unix": time.time()})
        return target
    finally:
        lock.close()


def release(queue_root: Path, batch_name: str) -> None:
    lock = _queue_lock(queue_root)
    try:
        source = queue_root / "active" / batch_name
        if not (source / "READY.json").is_file() or (source / "CONSUMED.json").exists():
            raise ValueError(f"Cannot release non-active or consumed batch: {source}")
        (source / "CLAIMED.json").unlink(missing_ok=True)
        os.replace(source, queue_root / "ready" / batch_name)
    finally:
        lock.close()


def complete(queue_root: Path, batch_name: str, checkpoint: Path,
             checkpoint_iteration: int, min_checkpoint_bytes: int = 1_000_000_000) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size < min_checkpoint_bytes:
        raise ValueError(f"Checkpoint is absent or too small: {checkpoint}")
    lock = _queue_lock(queue_root)
    try:
        active = queue_root / "active" / batch_name
        if not (active / "CLAIMED.json").is_file() or (active / "CONSUMED.json").exists():
            raise ValueError(f"Batch is not active: {active}")
        receipt = {
            "status": "CONSUMED", "batch_id": batch_name,
            "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_iteration": checkpoint_iteration,
            "retained_shards": True,
            "completed_unix": time.time(),
        }
        destination = queue_root / "completed" / batch_name
        assignment_path = active / "finite_assignment.json"
        if assignment_path.is_file():
            assignment = json.loads(assignment_path.read_text())
            assignment["rank_shards"] = [
                [str(destination / Path(path).relative_to(active)) for path in rank_paths]
                for rank_paths in assignment["rank_shards"]
            ]
            write_json_atomic(assignment_path, assignment)
        write_json_atomic(active / "CONSUMED.json", receipt)
        os.replace(active, destination)
        return destination
    finally:
        lock.close()


def status(queue_root: Path) -> dict:
    _directories(queue_root)
    groups = {}
    for state in (".building", "ready", "active", "completed"):
        groups[state] = []
        for path in sorted((queue_root / state).glob("batch-*")):
            manifest = path / "READY.json"
            report = json.loads(manifest.read_text()) if manifest.exists() else {}
            groups[state].append({"batch_id": path.name, "samples": report.get("samples"),
                                  "shards": report.get("shard_count")})
    return {"live_limit_count": sum(len(groups[s]) for s in (".building", "ready", "active")),
            "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    producer = sub.add_parser("produce")
    producer.add_argument("--index", type=Path, required=True)
    producer.add_argument("--queue-root", type=Path, required=True)
    producer.add_argument("--batch-samples", type=int, default=499_712)
    producer.add_argument("--max-live-batches", type=int, default=2)
    producer.add_argument("--workers", type=int, default=8)
    producer.add_argument("--samples-per-shard", type=int, default=1952)
    producer.add_argument("--file-block", type=int, default=64)
    producer.add_argument("--verify-workers", type=int, default=4)
    producer.add_argument("--partition-summary", type=Path)
    producer.add_argument("--root-alias", action="append", default=[])
    producer.add_argument("--seed", type=int, default=20261012)
    producer.add_argument("--daemon", action="store_true")
    producer.add_argument("--poll-seconds", type=int, default=30)
    producer.set_defaults(func=produce)
    for command in ("claim", "status"):
        item = sub.add_parser(command)
        item.add_argument("--queue-root", type=Path, required=True)
        item.set_defaults(func=(lambda args: print(json.dumps(status(args.queue_root), indent=2))
                                if args.command == "status" else
                                print(str(claim(args.queue_root) or "EMPTY"))))
    ack = sub.add_parser("complete")
    ack.add_argument("--queue-root", type=Path, required=True)
    ack.add_argument("--batch-id", required=True)
    ack.add_argument("--checkpoint", type=Path, required=True)
    ack.add_argument("--checkpoint-iteration", type=int, required=True)
    ack.add_argument("--min-checkpoint-bytes", type=int, default=1_000_000_000)
    ack.set_defaults(func=lambda args: print(complete(args.queue_root, args.batch_id,
                                                       args.checkpoint, args.checkpoint_iteration,
                                                       args.min_checkpoint_bytes)))
    rel = sub.add_parser("release")
    rel.add_argument("--queue-root", type=Path, required=True)
    rel.add_argument("--batch-id", required=True)
    rel.set_defaults(func=lambda args: release(args.queue_root, args.batch_id))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
