#!/usr/bin/env python3
"""Draw one million images uniformly from the accepted 20TB route-2 WDS pool."""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923")
OUT = Path("/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002")
SOURCE_DIRS = {
    "phase15": ROOT / "route2_15tb_no_old5_micro_tars",
    "phase5": ROOT / "route2_5tb_boundary_tars",
}
EXPECTED_SAMPLES = 11_033_883 + 1_748_877 + 3_842_850
SAMPLE_SIZE = 1_000_000
SHARD_SIZE = 2_000
SEED = 20261002
CHANNEL = re.compile(r"^(.*)\.ch\d+\.tiff?$", re.IGNORECASE)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def source_paths() -> list[tuple[str, str, int]]:
    return [(phase, str(path), path.stat().st_size)
            for phase, directory in SOURCE_DIRS.items()
            for path in sorted(directory.glob("*.tar"))]


def count_tar(item: tuple[str, str, int]) -> dict:
    phase, path, size = item
    count = 0
    with tarfile.open(path, "r") as archive:
        for member in archive:
            count += member.name.endswith(".meta.json")
    return {"phase": phase, "path": path, "bytes": size, "samples": count}


def inventory(workers: int) -> None:
    if (OUT / "inventory.json").exists():
        raise FileExistsError(OUT / "inventory.json")
    acceptance = json.loads((ROOT / "phase15_no_old5_reconciliation.json").read_text())
    boundary = json.loads((ROOT / "phase5_no_old5_final_audit/FINAL_TAR_AUDIT.json").read_text())
    if acceptance["status"] != "PASS" or boundary["status"] != "PASS":
        raise RuntimeError("20TB source acceptance is not PASS")
    paths = source_paths()
    print(f"Inventorying {len(paths)} accepted tar files", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(count_tar, item): item for item in paths}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"inventory {len(rows)}/{len(paths)}", flush=True)
    rows.sort(key=lambda row: (row["phase"], row["path"]))
    total = sum(row["samples"] for row in rows)
    if total != EXPECTED_SAMPLES:
        raise RuntimeError(f"20TB samples: expected {EXPECTED_SAMPLES}, found {total}")
    atomic_json(OUT / "inventory.json", {
        "source_pool": "accepted complete route-2 15TB plus 5TB boundary WDS",
        "total_samples": total, "expected_samples": EXPECTED_SAMPLES,
        "source_tar_count": len(rows), "rows": rows,
    })
    print(f"Inventory PASS: {total} images in {len(rows)} tar files", flush=True)


def member_key(name: str) -> str:
    if name.endswith(".meta.json"):
        return name[:-10]
    match = CHANNEL.match(name)
    if not match:
        raise ValueError(f"Unexpected tar member: {name}")
    return match.group(1)


def extract_shard(task: tuple[int, list[tuple[str, str, list[int]]]]) -> dict:
    shard, sources = task
    output = OUT / f"filtered_mixed_train_{shard:05d}.tar"
    record_path = OUT / "sample_keys" / f"{shard:05d}.jsonl"
    if output.exists() or record_path.exists():
        raise FileExistsError(output)
    temp = output.with_name(f".{output.name}.{os.getpid()}.part")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_temp = record_path.with_name(f".{record_path.name}.{os.getpid()}.part")
    samples = 0
    with tarfile.open(temp, "w") as destination, record_temp.open("w") as records:
        for phase, path, selected in sources:
            wanted = set(selected)
            found = set()
            ordinal = -1
            prior_key = None
            with tarfile.open(path, "r") as archive:
                for member in archive:
                    key = member_key(member.name)
                    if key != prior_key:
                        ordinal += 1
                        prior_key = key
                    if ordinal > selected[-1]:
                        break
                    if ordinal not in wanted:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Non-file member: {path}:{member.name}")
                    renamed = copy.copy(member)
                    prefix = f"{phase}__{key}"
                    renamed.name = prefix + member.name[len(key):]
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable member: {path}:{member.name}")
                    if member.name.endswith(".meta.json"):
                        metadata = json.load(stream)
                        if metadata.get("sample_id") != key:
                            raise ValueError(f"Sample ID mismatch: {path}:{key}")
                        metadata["source_sample_id"] = key
                        metadata["sample_id"] = prefix
                        encoded = json.dumps(metadata, separators=(",", ":")).encode()
                        renamed.size = len(encoded)
                        destination.addfile(renamed, io.BytesIO(encoded))
                        records.write(json.dumps({"sample_id": prefix, "source_tar": path}) + "\n")
                        found.add(ordinal)
                        samples += 1
                    else:
                        destination.addfile(renamed, stream)
            if found != wanted:
                raise RuntimeError(f"Missing selected samples in {path}: {len(wanted - found)}")
    if samples != SHARD_SIZE:
        raise RuntimeError(f"Shard {shard}: expected {SHARD_SIZE}, wrote {samples}")
    os.replace(temp, output)
    os.replace(record_temp, record_path)
    return {"shard": shard, "samples": samples, "bytes": output.stat().st_size}


def extract(workers: int) -> None:
    inventory_path = OUT / "inventory.json"
    inventory_data = json.loads(inventory_path.read_text())
    rows = inventory_data["rows"]
    paths = source_paths()
    if [(row["phase"], row["path"], row["bytes"]) for row in rows] != paths:
        raise RuntimeError("20TB source inventory changed after counting")
    if inventory_data["total_samples"] != EXPECTED_SAMPLES:
        raise RuntimeError("20TB inventory total is invalid")
    if list(OUT.glob("filtered_mixed_train_*.tar")):
        raise FileExistsError("Existing extracted tar files; refusing overwrite")
    selected = sorted(random.Random(SEED).sample(range(EXPECTED_SAMPLES), SAMPLE_SIZE))
    ends = []
    total = 0
    for row in rows:
        total += row["samples"]
        ends.append(total)
    tasks = []
    for shard in range(SAMPLE_SIZE // SHARD_SIZE):
        per_source = {}
        for global_index in selected[shard * SHARD_SIZE:(shard + 1) * SHARD_SIZE]:
            row_index = bisect.bisect_right(ends, global_index)
            start = 0 if row_index == 0 else ends[row_index - 1]
            per_source.setdefault(row_index, []).append(global_index - start)
        sources = [(rows[index]["phase"], rows[index]["path"], ordinals)
                   for index, ordinals in sorted(per_source.items())]
        tasks.append((shard, sources))
    atomic_json(OUT / "selection.json", {
        "seed": SEED, "algorithm": "Python random.sample(range(total_samples), 1000000)",
        "sample_size": SAMPLE_SIZE, "total_samples": EXPECTED_SAMPLES,
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "source_phases": sorted(SOURCE_DIRS), "shard_size": SHARD_SIZE,
        "created_unix": time.time(),
    })
    print(f"Extracting {SAMPLE_SIZE} selected images into {len(tasks)} tar files", flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_shard, task): task[0] for task in tasks}
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 10 == 0:
                print(f"extracted {len(completed)}/{len(tasks)} shards", flush=True)
    atomic_json(OUT / "extraction_complete.json", {
        "shards": len(completed), "samples": sum(row["samples"] for row in completed),
        "bytes": sum(row["bytes"] for row in completed),
        "source_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "status": "PASS",
    })
    print("Extraction PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "extract"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    mp.set_start_method("spawn")
    OUT.mkdir(parents=True, exist_ok=True)
    {"inventory": inventory, "extract": extract}[args.stage](args.workers)


if __name__ == "__main__":
    main()
