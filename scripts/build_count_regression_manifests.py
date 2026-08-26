#!/usr/bin/env python3
"""Build fixed CoNIC and LIVECell image-level cell-count regression manifests."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


CONIC_SPLIT_SEED = 42
CONIC_NUM_FOLDS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_stats(rows: list[dict]) -> dict[str, float | int]:
    values = np.asarray([float(row["cell_count"]) for row in rows], dtype=np.float64)
    return {
        "rows": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
    }


def conic_group_folds(count_rows: list[dict[str, str]], patch_rows: list[dict[str, str]]) -> np.ndarray:
    """Assign source-image groups to target-stratified folds within each source."""
    targets = np.asarray(
        [sum(int(value) for value in row.values()) for row in count_rows],
        dtype=np.float64,
    )
    groups = np.asarray([row["patch_info"].rsplit("-", 1)[0] for row in patch_rows])
    sources = np.asarray([group.split("_", 1)[0] for group in groups])
    folds = np.full(len(count_rows), -1, dtype=np.int64)

    for source_offset, source in enumerate(sorted(set(sources))):
        source_indices = np.flatnonzero(sources == source)
        source_targets = targets[source_indices]
        quantile_edges = np.quantile(source_targets, [0.2, 0.4, 0.6, 0.8])
        target_bins = np.digitize(source_targets, quantile_edges, right=True)
        splitter = StratifiedGroupKFold(
            n_splits=CONIC_NUM_FOLDS,
            shuffle=True,
            random_state=CONIC_SPLIT_SEED + source_offset,
        )
        for fold, (_, fold_local_indices) in enumerate(
            splitter.split(
                np.zeros(len(source_indices)),
                target_bins,
                groups[source_indices],
            )
        ):
            folds[source_indices[fold_local_indices]] = fold

    if np.any(folds < 0):
        raise RuntimeError("Failed to assign every CoNIC patch to a fold")
    for group in set(groups):
        if len(set(folds[groups == group])) != 1:
            raise RuntimeError(f"CoNIC source image {group} crosses folds")
    return folds


def build_conic(benchmark_root: Path) -> dict:
    source = benchmark_root / "segmentation/conic/extracted"
    with (source / "counts.csv").open(newline="") as handle:
        count_rows = list(csv.DictReader(handle))
    with (source / "patch_info.csv").open(newline="") as handle:
        patch_rows = list(csv.DictReader(handle))
    if len(count_rows) != len(patch_rows):
        raise ValueError("CoNIC counts.csv and patch_info.csv have different lengths")
    folds = conic_group_folds(count_rows, patch_rows)

    cell_types = list(count_rows[0])
    rows = []
    for index, (source_row, patch_row) in enumerate(zip(count_rows, patch_rows, strict=True)):
        typed_counts = {name: int(source_row[name]) for name in cell_types}
        fold = int(folds[index])
        split = "test" if fold == 0 else "val" if fold == 1 else "train"
        source_image = patch_row["patch_info"].rsplit("-", 1)[0]
        rows.append({
            "split": split,
            "fold": fold,
            "image_index": index,
            "source": source_image.split("_", 1)[0],
            "source_image": source_image,
            "cell_count": sum(typed_counts.values()),
            **typed_counts,
        })
    output = benchmark_root / "Regression/CoNIC_Cell_Count/conic_cell_count.csv"
    write_csv(
        output,
        [
            "split",
            "fold",
            "image_index",
            "source",
            "source_image",
            "cell_count",
            *cell_types,
        ],
        rows,
    )
    split_sources = {
        split: sorted({row["source"] for row in rows if row["split"] == split})
        for split in ("train", "val", "test")
    }
    if any(len(sources) != 5 for sources in split_sources.values()):
        raise RuntimeError(f"A CoNIC split is missing a source dataset: {split_sources}")
    return {
        "source_counts": str(source / "counts.csv"),
        "source_counts_sha256": sha256(source / "counts.csv"),
        "source_patch_info": str(source / "patch_info.csv"),
        "source_patch_info_sha256": sha256(source / "patch_info.csv"),
        "target": "sum of the six official counts for nuclei intersecting the central 224x224 region",
        "input_crop": "central 224x224 pixels of each 256x256 patch",
        "split_protocol": (
            "source-image-grouped, source-and-count-stratified 10-fold split; "
            "folds 2-9 train, fold 1 validation, fold 0 test"
        ),
        "split_seed": CONIC_SPLIT_SEED,
        "sources_by_split": split_sources,
        "manifest": str(output),
        "manifest_sha256": sha256(output),
        "splits": {
            split: target_stats([row for row in rows if row["split"] == split])
            for split in ("train", "val", "test")
        },
    }


def build_livecell(benchmark_root: Path) -> dict:
    source = benchmark_root / "segmentation/LIVECell/LIVECell_dataset_2021"
    rows = []
    source_hashes = {}
    for split in ("train", "val", "test"):
        annotation = source / f"annotations/LIVECell/livecell_coco_{split}.json"
        source_hashes[split] = sha256(annotation)
        with annotation.open() as handle:
            data = json.load(handle)
        counts = Counter(int(ann["image_id"]) for ann in data.get("annotations", []))
        image_dir = "livecell_test_images" if split == "test" else "livecell_train_val_images"
        image_root = source / "images" / image_dir
        for image in data.get("images", []):
            image_id = int(image["id"])
            image_path = image_root / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            rows.append({
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path.relative_to(source)),
                "cell_type": image["file_name"].split("_", 1)[0],
                "cell_count": int(counts.get(image_id, 0)),
            })
        del data, counts
        gc.collect()
    output = benchmark_root / "Regression/LIVECell_Cell_Count/livecell_cell_count.csv"
    write_csv(
        output,
        ["split", "image_id", "image_path", "cell_type", "cell_count"],
        rows,
    )
    return {
        "source_annotations": str(source / "annotations/LIVECell"),
        "source_annotation_sha256": source_hashes,
        "target": "number of official COCO instances in each full image",
        "input_geometry": "full image, mean-color letterboxed to a square before model resize",
        "split_protocol": "official LIVECell train/validation/test split",
        "manifest": str(output),
        "manifest_sha256": sha256(output),
        "cell_types": sorted({row["cell_type"] for row in rows}),
        "splits": {
            split: target_stats([row for row in rows if row["split"] == split])
            for split in ("train", "val", "test")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    args = parser.parse_args()
    root = Path(args.benchmark_root)
    metadata = {
        "protocol_version": "v1",
        "conic": build_conic(root),
        "livecell": build_livecell(root),
    }
    metadata_path = root / "Regression/count_regression_manifest_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
