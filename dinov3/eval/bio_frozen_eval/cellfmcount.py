"""CellFMCount official DAPI scope with an explicit trainval validation holdout."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, validate_records
from .datasets import _pad_to_square, _to_rgb_uint8


CELLFM_RELATIVE_ROOT = "ood/ood_regression/datasets/CellFMCount/extracted"
METADATA_FIELDS = {"id", "cell_count", "cell_type", "staining", "objective", "markers", "set"}


def build_cellfm_manifest(benchmark_root, seed=0):
    """Preserve released DAPI test IDs and select 10% of trainval for validation."""
    root = Path(benchmark_root) / CELLFM_RELATIVE_ROOT
    metadata = root / "metadata.csv"
    with metadata.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != METADATA_FIELDS:
            raise ValueError(f"Unexpected CellFMCount metadata header: {reader.fieldnames}")
        rows = list(reader)
    records, all_ids = [], set()
    for row in rows:
        sample_id = row["id"]
        if not sample_id.isdigit() or sample_id in all_ids:
            raise ValueError(f"Invalid or duplicate CellFMCount image ID: {sample_id!r}")
        all_ids.add(sample_id)
        if row["staining"] != "DAPI":
            continue
        source_split = row["set"]
        if source_split not in ("trainval", "test"):
            raise ValueError(f"Unknown CellFMCount source split: {source_split!r}")
        try:
            target = float(row["cell_count"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid cell count for {sample_id}") from error
        if not np.isfinite(target) or target < 0 or target != int(target):
            raise ValueError(f"Invalid cell count for {sample_id}: {target}")
        image_path = root / "img" / f"{sample_id}.tiff"
        annotation = root / "ground_truth" / f"{sample_id}.csv"
        if not image_path.is_file() or not annotation.is_file():
            raise ValueError(f"Missing image or coordinate annotation for {sample_id}")
        with Image.open(image_path) as image:
            if image.n_frames != 1:
                raise ValueError(f"Unexpected multiframe CellFMCount image: {sample_id}")
            width, height = image.size
            image.load()
        with annotation.open(newline="") as handle:
            points = csv.DictReader(handle)
            if points.fieldnames != ["X", "Y"]:
                raise ValueError(f"Invalid coordinate header for {sample_id}: {points.fieldnames}")
            point_rows = list(points)
        if len(point_rows) != int(target):
            raise ValueError(f"Count mismatch for {sample_id}: metadata={int(target)}, coordinates={len(point_rows)}")
        out_of_bounds = 0
        for point in point_rows:
            try:
                x, y = float(point["X"]), float(point["Y"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid coordinate for {sample_id}") from error
            if not np.isfinite([x, y]).all():
                raise ValueError(f"Nonfinite coordinate for {sample_id}: {(x, y)}")
            # Count regression preserves source dots, including imprecise edge clicks.
            out_of_bounds += int(not (0 <= x <= width and 0 <= y <= height))
        records.append({"sample_id": sample_id, "path": str(image_path.relative_to(root)),
                        "target": int(target), "group": sample_id,
                        "source_split": source_split, "split": "test" if source_split == "test" else "train",
                        "cell_type": row["cell_type"], "objective": row["objective"],
                        "out_of_bounds_coordinates": out_of_bounds,
                        "image_sha256": file_digest(image_path),
                        "annotation_sha256": file_digest(annotation)})
    trainval_ids = sorted((r["sample_id"] for r in records if r["source_split"] == "trainval"), key=int)
    if len(trainval_ids) < 2:
        raise ValueError("CellFMCount DAPI trainval needs at least two image IDs")
    val_count = max(1, int(np.ceil(0.1 * len(trainval_ids))))
    val_ids = set(np.random.default_rng(seed).permutation(trainval_ids)[:val_count].tolist())
    for record in records:
        if record["sample_id"] in val_ids:
            record["split"] = "val"
    records.sort(key=lambda r: int(r["sample_id"]))
    stats = validate_records(records)
    manifest = {"dataset": "cellfmcount-dapi-count", "version": 1, "seed": seed,
                "task": "regression", "classification": "OFFICIAL",
                "scope": "DAPI only", "official_split": "released metadata trainval/test IDs",
                "validation_classification": "PROPOSED_BY_US",
                "validation_split": "10% trainval IDs (ceil), sorted numerically then numpy default_rng(seed) permutation",
                "grouping": "image ID; specimen/donor IDs unavailable and biological grouping unverified",
                "metric": "mae", "preprocessing": "existing _to_rgb_uint8 minmax followed by full-FOV mean-color square padding",
                "license": {"id": "cc-by-sa-4.0", "source": "https://zenodo.org/api/records/17088532"},
                "license_status": "AVAILABLE_CCBYSA",
                "readiness": "READY_SCALAR_COUNT_RESIDUAL_BIOLOGICAL_GROUP_UNKNOWN",
                "metadata_sha256": file_digest(metadata), "stats": stats, "records": records,
                "out_of_bounds_coordinates": sum(r["out_of_bounds_coordinates"] for r in records),
                "sources": ["https://arxiv.org/html/2511.19351v1", "https://github.com/NRT-D4/CellFMCount"],
                "limitations": ["No assertion of specimen-disjoint or donor-disjoint splits.",
                                "Original filenames absent from local metadata; biological grouping remains unverified.",
                                "Frozen scalar count probe, not the paper's fine-tuned density-map model.",
                                "Finite out-of-bounds edge annotations retained for source scalar counts and recorded; not localization supervision."]}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


class CellFMCountDataset:
    def __init__(self, benchmark_root, manifest, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}")
        validate_records(manifest["records"])
        self.root = Path(benchmark_root) / CELLFM_RELATIVE_ROOT
        self.records = [record for record in manifest["records"] if record["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            rgb = Image.fromarray(_to_rgb_uint8(np.asarray(image)))
        return _pad_to_square(rgb), float(record["target"]), record["sample_id"]
