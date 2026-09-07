#!/usr/bin/env python3
"""Build matched logical-record feature slices for the data-quality ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


DEFAULT_RAW_ROOTS = (
    Path("/mnt/huawei_deepcad/siglip2_features"),
    Path("/mnt/huawei_deepcad/siglip2_features_3ch"),
    Path("/mnt/huawei_deepcad/siglip2_features_multich"),
    Path("/mnt/huawei_deepcad/siglip2_features_huge_multich"),
    Path("/mnt/huawei_deepcad/siglip2_features_highNA_file"),
    Path("/mnt/huawei_deepcad/siglip2_features_highNA_multi_ch245"),
    Path("/mnt/huawei_deepcad/siglip2_features_slfm"),
)


@dataclass(frozen=True)
class FeaturePart:
    feature_path: Path
    record_path: Path
    logical_records: int
    feature_rows: int
    feature_dim: int
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/huawei_deepcad/deduplication/"
            "data_quality_ablation_1m_20260901/features"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--inventory-workers", type=int, default=32)
    parser.add_argument("--raw-read-workers", type=int, default=16)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("raw_1pb", "dedup_100tb", "curated_1tb"),
        default=("raw_1pb", "dedup_100tb", "curated_1tb"),
    )
    parser.add_argument("--raw-roots", type=Path, nargs="+", default=DEFAULT_RAW_ROOTS)
    parser.add_argument(
        "--hundred-tb-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/deepcad_100t/tile_siglip_features_run_20260829"),
    )
    parser.add_argument(
        "--one-tb-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/siglip2_features_1tb_patched"),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def normalized_mean(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2:
        valid = np.linalg.norm(values, axis=1) > 0
        if not valid.any():
            raise ValueError("logical record has no nonzero feature vectors")
        vector = values[valid].mean(axis=0)
    else:
        flattened = values.reshape(-1, values.shape[-1])
        valid = np.linalg.norm(flattened, axis=1) > 0
        if not valid.any():
            raise ValueError("logical record has no nonzero feature vectors")
        vector = flattened[valid].mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("logical record produced an invalid feature vector")
    return vector / norm


def choose_global_indices(population: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if population < sample_size:
        raise ValueError(f"population {population:,} is smaller than sample {sample_size:,}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(population, size=sample_size, replace=False, shuffle=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def assignments_by_part(
    parts: list[FeaturePart], selected: np.ndarray
) -> Iterator[tuple[FeaturePart, np.ndarray, np.ndarray]]:
    prefix = 0
    output_start = 0
    for part in parts:
        end = prefix + part.logical_records
        left = int(np.searchsorted(selected, prefix, side="left"))
        right = int(np.searchsorted(selected, end, side="left"))
        if right > left:
            local = selected[left:right] - prefix
            output_indices = np.arange(output_start, output_start + len(local), dtype=np.int64)
            yield part, local, output_indices
            output_start += len(local)
        prefix = end
    if output_start != len(selected):
        raise RuntimeError(f"assigned {output_start:,} selections, expected {len(selected):,}")


def raw_record_path(feature_path: Path) -> Path | None:
    if feature_path.name == "all_features.npy":
        candidates = (feature_path.with_name("all_paths.txt"), feature_path.with_suffix(".txt"))
    else:
        candidates = (
            feature_path.with_name(feature_path.name.replace("features_", "valid_paths_").replace(".npy", ".txt")),
            feature_path.with_suffix(".txt"),
        )
    return next((path for path in candidates if path.is_file()), None)


def inspect_raw_part(item: tuple[Path, Path]) -> FeaturePart:
    root, feature_path = item
    record_path = raw_record_path(feature_path)
    if record_path is None:
        raise FileNotFoundError(f"raw feature file lacks a path manifest: {feature_path}")
    array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if array.ndim < 2 or array.shape[-1] <= 1:
        raise ValueError(f"unsupported feature shape {array.shape}: {feature_path}")
    return FeaturePart(
        feature_path=feature_path,
        record_path=record_path,
        logical_records=len(array),
        feature_rows=len(array),
        feature_dim=int(array.shape[-1]),
        source_name=str(root),
    )


def load_raw_inventory_cache(cache_path: Path, roots: Iterable[Path]) -> list[FeaturePart] | None:
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text())
    if payload.get("version") != 1 or payload.get("roots") != [str(root) for root in roots]:
        return None
    return [
        FeaturePart(
            feature_path=Path(row["feature_path"]),
            record_path=Path(row["record_path"]),
            logical_records=int(row["logical_records"]),
            feature_rows=int(row["feature_rows"]),
            feature_dim=int(row["feature_dim"]),
            source_name=row["source_name"],
        )
        for row in payload["parts"]
    ]


def inventory_raw(
    roots: Iterable[Path], workers: int, cache_path: Path
) -> list[FeaturePart]:
    roots = tuple(roots)
    cached = load_raw_inventory_cache(cache_path, roots)
    if cached is not None:
        print(f"RAW inventory cache: {cache_path} ({len(cached):,} parts)", flush=True)
        return cached
    candidates: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        feature_paths = sorted(root.rglob("features_*.npy"))
        feature_paths.extend(sorted(root.rglob("all_features.npy")))
        candidates.extend((root, path) for path in feature_paths)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        parts = list(executor.map(inspect_raw_part, candidates))
    if not parts:
        raise RuntimeError("no raw feature/path pairs found")
    atomic_json(
        cache_path,
        {
            "version": 1,
            "roots": [str(root) for root in roots],
            "parts": [
                {
                    "feature_path": str(part.feature_path),
                    "record_path": str(part.record_path),
                    "logical_records": part.logical_records,
                    "feature_rows": part.feature_rows,
                    "feature_dim": part.feature_dim,
                    "source_name": part.source_name,
                }
                for part in parts
            ],
        },
    )
    return parts


def inventory_grouped(root: Path, suffix: str, logical_field: str) -> list[FeaturePart]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    parts: list[FeaturePart] = []
    for status_path in sorted(root.glob("*.done.json")):
        status = json.loads(status_path.read_text())
        stem = status_path.name[: -len(".done.json")]
        feature_path = root / f"{stem}.features.npy"
        record_path = root / f"{stem}.records.jsonl"
        if not feature_path.is_file() or not record_path.is_file():
            continue
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        expected_rows_value = status.get("features", status.get("channel_features"))
        if expected_rows_value is None:
            raise KeyError(f"missing feature-row count in {status_path}")
        expected_rows = int(expected_rows_value)
        if len(array) != expected_rows:
            raise ValueError(f"status/feature mismatch for {feature_path}: {expected_rows} != {len(array)}")
        parts.append(
            FeaturePart(
                feature_path=feature_path,
                record_path=record_path,
                logical_records=int(status[logical_field]),
                feature_rows=expected_rows,
                feature_dim=int(array.shape[-1]),
                source_name=str(status.get("shard_path") or status.get("tar") or stem),
            )
        )
    if not parts:
        raise RuntimeError(f"no complete {suffix} feature parts found in {root}")
    return parts


class SliceWriter:
    def __init__(self, output_dir: Path, sample_size: int, feature_dim: int):
        if output_dir.exists() and any(output_dir.iterdir()):
            if (output_dir / ".complete").is_file():
                raise FileExistsError(f"slice is already complete: {output_dir}")
            raise RuntimeError(f"refusing non-empty incomplete directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.manifest_path = output_dir / "selected_records.jsonl"
        self.features_path = output_dir / "features.f16.npy"
        self.manifest = self.manifest_path.open("w", encoding="utf-8")
        self.features = np.lib.format.open_memmap(
            self.features_path, mode="w+", dtype=np.float16, shape=(sample_size, feature_dim)
        )
        self.written = 0

    def add(self, output_index: int, vector: np.ndarray, record: dict) -> None:
        if output_index != self.written:
            raise RuntimeError(f"nonsequential output index: {output_index} after {self.written}")
        self.features[output_index] = vector.astype(np.float16, copy=False)
        self.manifest.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        self.written += 1

    def close(self) -> None:
        self.features.flush()
        self.manifest.flush()
        os.fsync(self.manifest.fileno())
        self.manifest.close()
        del self.features


def load_raw_selections(
    assignment: tuple[FeaturePart, np.ndarray, np.ndarray]
) -> list[tuple[int, np.ndarray, dict]]:
    part, local_indices, output_indices = assignment
    selected_by_line = {int(local): int(output) for local, output in zip(local_indices, output_indices)}
    features = np.load(part.feature_path, mmap_mode="r", allow_pickle=False)
    rows: list[tuple[int, np.ndarray, dict]] = []
    max_line = int(local_indices[-1])
    with part.record_path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            output_index = selected_by_line.get(line_index)
            if output_index is not None:
                source_path = line.strip()
                if not source_path:
                    raise ValueError(f"empty source path at {part.record_path}:{line_index + 1}")
                rows.append(
                    (
                        output_index,
                        normalized_mean(features[line_index]),
                        {
                            "output_index": output_index,
                            "dataset": "raw_1pb",
                            "logical_key": stable_key("raw_1pb", part.feature_path, line_index, source_path),
                            "canonical_id": source_path,
                            "source_path": source_path,
                            "feature_file": str(part.feature_path),
                            "feature_row": line_index,
                        },
                    )
                )
            if line_index >= max_line:
                break
    if len(rows) != len(local_indices):
        raise RuntimeError(f"found {len(rows)} of {len(local_indices)} selected paths in {part.record_path}")
    return rows


def build_raw(
    parts: list[FeaturePart], selected: np.ndarray, writer: SliceWriter, workers: int
) -> None:
    assignments = list(assignments_by_part(parts, selected))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = executor.map(load_raw_selections, assignments)
        for part_index, rows in enumerate(results, start=1):
            for output_index, vector, record in rows:
                writer.add(output_index, vector, record)
            if part_index % 1000 == 0 or part_index == len(assignments):
                print(
                    f"RAW parts={part_index:,}/{len(assignments):,} records={writer.written:,}",
                    flush=True,
                )


def grouped_records(
    record_path: Path, id_field: str
) -> Iterator[tuple[int, int, object, dict]]:
    start = 0
    current_id: object | None = None
    first_record: dict | None = None
    row_index = 0
    with record_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            record = json.loads(line)
            identifier = record[id_field]
            if current_id is None:
                current_id = identifier
                first_record = record
                start = row_index
            elif identifier != current_id:
                assert first_record is not None
                yield start, row_index, current_id, first_record
                current_id = identifier
                first_record = record
                start = row_index
    if current_id is not None:
        assert first_record is not None
        yield start, row_index + 1, current_id, first_record


def build_grouped(
    dataset: str,
    parts: list[FeaturePart],
    selected: np.ndarray,
    writer: SliceWriter,
    id_field: str,
) -> None:
    assignments = list(assignments_by_part(parts, selected))
    for part_index, (part, local_indices, output_indices) in enumerate(assignments, start=1):
        selected_by_group = {int(local): int(output) for local, output in zip(local_indices, output_indices)}
        features = np.load(part.feature_path, mmap_mode="r", allow_pickle=False)
        logical_count = 0
        found = 0
        for row_start, row_end, identifier, first_record in grouped_records(part.record_path, id_field):
            output_index = selected_by_group.get(logical_count)
            if output_index is not None:
                record = dict(first_record)
                record.update(
                    {
                        "output_index": output_index,
                        "dataset": dataset,
                        "logical_key": str(identifier),
                        "feature_file": str(part.feature_path),
                        "feature_row_start": row_start,
                        "feature_row_end": row_end,
                        "source_container": part.source_name,
                    }
                )
                if dataset == "dedup_100tb":
                    record["canonical_id"] = "|".join(
                        str(record.get(field))
                        for field in ("source_path", "frame_idx", "y0", "y1", "x0", "x1")
                    )
                writer.add(output_index, normalized_mean(features[row_start:row_end]), record)
                found += 1
            logical_count += 1
        if logical_count != part.logical_records:
            raise RuntimeError(
                f"logical count mismatch for {part.record_path}: {logical_count} != {part.logical_records}"
            )
        if found != len(local_indices):
            raise RuntimeError(f"found {found} of {len(local_indices)} selected groups in {part.record_path}")
        if part_index % 100 == 0 or part_index == len(assignments):
            print(
                f"{dataset} parts={part_index:,}/{len(assignments):,} records={writer.written:,}",
                flush=True,
            )


def feature_dimension(parts: list[FeaturePart]) -> int:
    dimensions = {part.feature_dim for part in parts}
    if len(dimensions) != 1:
        raise ValueError(f"mixed feature dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def finish_slice(
    dataset: str,
    output_dir: Path,
    writer: SliceWriter,
    parts: list[FeaturePart],
    population: int,
    sample_size: int,
    seed: int,
    feature_dim: int,
) -> None:
    writer.close()
    if writer.written != sample_size:
        raise RuntimeError(f"wrote {writer.written:,} records, expected {sample_size:,}")
    summary = {
        "dataset": dataset,
        "sample_unit": "logical_record",
        "multi_channel_policy": "mean_l2_normalized_channel_features_then_l2_normalize",
        "selection": "uniform_without_replacement_over_sorted_complete_parts",
        "seed": seed,
        "population_records": population,
        "selected_records": sample_size,
        "feature_shape": [sample_size, feature_dim],
        "feature_dtype": "float16",
        "source_parts": len(parts),
        "source_feature_rows": sum(part.feature_rows for part in parts),
        "manifest_sha256": sha256_file(writer.manifest_path),
    }
    inventory_path = output_dir / "source_inventory.jsonl"
    with inventory_path.open("w", encoding="utf-8") as handle:
        for part in parts:
            handle.write(
                json.dumps(
                    {
                        "feature_path": str(part.feature_path),
                        "record_path": str(part.record_path),
                        "logical_records": part.logical_records,
                        "feature_rows": part.feature_rows,
                        "feature_dim": part.feature_dim,
                        "source_name": part.source_name,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary["source_inventory_sha256"] = sha256_file(inventory_path)
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / ".complete").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


def build_dataset(dataset: str, args: argparse.Namespace, seed: int) -> None:
    if dataset == "raw_1pb":
        parts = inventory_raw(
            args.raw_roots, args.inventory_workers, args.output_root / "raw_1pb_inventory.json"
        )
    elif dataset == "dedup_100tb":
        parts = inventory_grouped(args.hundred_tb_root, dataset, "manifest_items")
    else:
        parts = inventory_grouped(args.one_tb_root, dataset, "logical_samples")

    population = sum(part.logical_records for part in parts)
    selected = choose_global_indices(population, args.sample_size, seed)
    feature_dim = feature_dimension(parts)
    output_dir = args.output_root / dataset
    writer = SliceWriter(output_dir, args.sample_size, feature_dim)
    print(
        f"START dataset={dataset} population={population:,} sample={args.sample_size:,} "
        f"parts={len(parts):,} seed={seed}",
        flush=True,
    )
    try:
        if dataset == "raw_1pb":
            build_raw(parts, selected, writer, args.raw_read_workers)
        elif dataset == "dedup_100tb":
            build_grouped(dataset, parts, selected, writer, "tile_item_id")
        else:
            build_grouped(dataset, parts, selected, writer, "key")
        finish_slice(
            dataset, output_dir, writer, parts, population, args.sample_size, seed, feature_dim
        )
    except BaseException:
        if not writer.manifest.closed:
            writer.close()
        raise


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_offsets = {"raw_1pb": 0, "dedup_100tb": 1, "curated_1tb": 2}
    for dataset in args.datasets:
        build_dataset(dataset, args, args.seed + seed_offsets[dataset])


if __name__ == "__main__":
    main()
