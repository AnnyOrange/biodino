#!/usr/bin/env python3
"""Audit exact uniqueness and provenance of matched data-quality slices."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


DEFAULT_ROOT = Path(
    "/mnt/huawei_deepcad/deduplication/data_quality_ablation_1m_20260901/features"
)
DEFAULT_1TB_INDEX = Path("/mnt/huawei_deepcad/final-data/indexes/1TB.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--one-tb-index", type=Path, default=DEFAULT_1TB_INDEX)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ratio(unique: int, total: int) -> float:
    return unique / total if total else 0.0


def read_manifest(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if int(row["output_index"]) != line_number - 1:
                raise ValueError(f"nonsequential output_index at {path}:{line_number}")
            yield row


def audit_raw(path: Path) -> dict[str, object]:
    logical_keys: set[str] = set()
    source_paths: set[str] = set()
    records = 0
    for row in read_manifest(path):
        logical_keys.add(row["logical_key"])
        source_paths.add(row["source_path"])
        records += 1
    return {
        "records": records,
        "unique_logical_keys": len(logical_keys),
        "unique_source_paths": len(source_paths),
        "exact_source_path_unique_ratio": ratio(len(source_paths), records),
    }


def audit_hundred_tb(path: Path) -> dict[str, object]:
    logical_keys: set[str] = set()
    canonical_tiles: set[str] = set()
    source_paths: set[str] = set()
    source_items: set[int] = set()
    records = 0
    for row in read_manifest(path):
        logical_keys.add(row["logical_key"])
        canonical_tiles.add(row["canonical_id"])
        source_paths.add(row["source_path"])
        source_items.add(int(row["source_item_id"]))
        records += 1
    return {
        "records": records,
        "unique_logical_keys": len(logical_keys),
        "unique_canonical_tiles": len(canonical_tiles),
        "unique_source_items": len(source_items),
        "unique_source_paths": len(source_paths),
        "exact_tile_unique_ratio": ratio(len(canonical_tiles), records),
        "parent_source_path_ratio": ratio(len(source_paths), records),
    }


def audit_one_tb(path: Path, index_path: Path) -> dict[str, object]:
    selected_keys: set[str] = set()
    records = 0
    for row in read_manifest(path):
        selected_keys.add(row["logical_key"])
        records += 1
    if len(selected_keys) != records:
        raise RuntimeError(
            f"1TB manifest has {records:,} rows but {len(selected_keys):,} unique sample keys"
        )

    canonical_keys: set[str] = set()
    source_paths: set[str] = set()
    normalized_oids: set[str] = set()
    matched = 0
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA temp_store=MEMORY")
        cursor = connection.execute(
            "SELECT sample_key, canonical_key, "
            "COALESCE(NULLIF(src_path, ''), NULLIF(source_file_path, ''), original_path), "
            "norm_oid FROM samples"
        )
        for sample_key, canonical_key, source_path, normalized_oid in cursor:
            if sample_key not in selected_keys:
                continue
            matched += 1
            canonical_keys.add(canonical_key)
            if source_path:
                source_paths.add(source_path)
            if normalized_oid:
                normalized_oids.add(normalized_oid)
    finally:
        connection.close()
    if matched != records:
        raise RuntimeError(f"matched {matched:,} of {records:,} selected 1TB keys in {index_path}")
    return {
        "records": records,
        "unique_logical_keys": len(selected_keys),
        "unique_canonical_crops": len(canonical_keys),
        "unique_source_paths": len(source_paths),
        "unique_normalized_oids": len(normalized_oids),
        "exact_canonical_crop_unique_ratio": ratio(len(canonical_keys), records),
        "parent_source_path_ratio": ratio(len(source_paths), records),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output or args.root.parent / "audit" / "exact_uniqueness.json"
    result = {
        "definition": (
            "Exact identifier/path equality only; this is not perceptual or embedding-based "
            "near-duplicate detection."
        ),
        "raw_1pb": audit_raw(args.root / "raw_1pb" / "selected_records.jsonl"),
        "dedup_100tb": audit_hundred_tb(
            args.root / "dedup_100tb" / "selected_records.jsonl"
        ),
        "curated_1tb": audit_one_tb(
            args.root / "curated_1tb" / "selected_records.jsonl", args.one_tb_index
        ),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
