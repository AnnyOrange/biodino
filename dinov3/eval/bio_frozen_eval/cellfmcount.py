"""CellFMCount official DAPI scope with an explicit trainval validation holdout."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, pixel_digest, validate_records
from .datasets import _pad_to_square, _to_rgb_uint8


CELLFM_RELATIVE_ROOT = "ood/ood_regression/datasets/CellFMCount/extracted"
METADATA_FIELDS = {"id", "cell_count", "cell_type", "staining", "objective", "markers", "set"}
OFFICIAL_ARCHIVE_MD5 = "e87d6247e6459268f5cf4535ec25e709"


def audit_cellfm_source(benchmark_root):
    """Compare every released member and decoded duplicate against the official ZIP."""
    root = Path(benchmark_root) / CELLFM_RELATIVE_ROOT
    archive_path = root.parent / "raw" / "cellfmcount.zip"
    md5 = hashlib.md5()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            md5.update(chunk)
    official_md5 = OFFICIAL_ARCHIVE_MD5
    if md5.hexdigest() != official_md5:
        raise ValueError("CellFMCount ZIP differs from official Zenodo 17088532 checksum")
    with (root / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records, pixel_groups = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.read("dataset/metadata.csv") != (root / "metadata.csv").read_bytes():
            raise ValueError("CellFMCount extracted metadata differs from official ZIP")
        for row in rows:
            sample_id = row["id"]
            record = dict(row)
            for folder, extension in (("img", "tiff"), ("ground_truth", "csv")):
                relative = f"{folder}/{sample_id}.{extension}"
                local_sha = file_digest(root / relative)
                archive_bytes = archive.read("dataset/" + relative)
                archive_sha = hashlib.sha256(archive_bytes).hexdigest()
                if local_sha != archive_sha:
                    raise ValueError(f"CellFMCount extracted member differs from ZIP: {relative}")
                record["image_sha256" if folder == "img" else "annotation_sha256"] = local_sha
                record["image_source_path" if folder == "img" else "annotation_source_path"] = "dataset/" + relative
                if folder == "ground_truth":
                    points = csv.DictReader(io.StringIO(archive_bytes.decode("utf-8-sig")))
                    if points.fieldnames != ["X", "Y"]:
                        raise ValueError(f"Invalid source coordinate header: {relative}")
                    point_rows = list(points)
                    record["source_annotation_count"] = len(point_rows)
                    if len(point_rows) != int(row["cell_count"]):
                        raise ValueError(f"Source metadata/count mismatch: {sample_id}")
            record["image_pixel_sha256"] = pixel_digest(root / "img" / f"{sample_id}.tiff")
            records.append(record)
            pixel_groups.setdefault(record["image_pixel_sha256"], []).append(record)
    duplicates = []
    for pixel_sha, group in sorted(pixel_groups.items()):
        if len(group) < 2:
            continue
        conflicting = len({r["cell_count"] for r in group}) > 1
        duplicates.append({"image_pixel_sha256": pixel_sha,
                           "root_cause": "DUPLICATE_WITH_CONFLICTING_SOURCE_LABELS" if conflicting else "OTHER",
                           "byte_identical": len({r["image_sha256"] for r in group}) == 1,
                           "members": sorted(group, key=lambda r: int(r["id"])),
                           "extracted_members_equal_official_archive": True,
                           "physical_image_multi_count_definition_verified": False})
    return {"dataset": "CellFMCount", "official_archive_url": "https://zenodo.org/records/17088532/files/cellfmcount.zip",
            "official_archive_md5": official_md5, "archive_md5_verified": True,
            "metadata_sha256": file_digest(root / "metadata.csv"), "all_records": records,
            "source_rows": len(rows), "duplicate_groups": duplicates,
            "source_annotation_rule": "Coordinates and cell_count are compared by numerical image ID; archive bytes preserved, no local join/conversion.",
            "original_filename_status": "Absent from released seven-column metadata; no original XML/source filename mapping provided.",
            "limitations": ["Conflicting annotations originate in checksum-verified official release; available assets cannot distinguish author-side conversion from source annotation errors.",
                            "Paper says original filenames and duplicate removal exist; released metadata does not expose those identities, and observed duplicates contradict complete deduplication."]}


def _deduplicate_records(records):
    grouped, excluded = {}, []
    for record in records:
        grouped.setdefault(record["image_pixel_sha256"], []).append(record)
    kept = []
    for pixel_sha, group in sorted(grouped.items()):
        if len({r["target"] for r in group}) > 1:
            excluded.extend({**r, "reason": "DUPLICATE_WITH_CONFLICTING_SOURCE_LABELS"} for r in group)
            continue
        # Do not move released test observations into training or count repeats twice.
        ordered = sorted(group, key=lambda r: (r["source_split"] != "test", int(r["sample_id"])))
        kept.append(ordered[0])
        excluded.extend({**r, "reason": "SAME_TARGET_PIXEL_DUPLICATE", "retained_sample_id": ordered[0]["sample_id"]}
                        for r in ordered[1:])
    return kept, sorted(excluded, key=lambda r: int(r["sample_id"]))


def build_cellfm_manifest(benchmark_root, seed=0, duplicate_policy="quarantine", source_audit=None):
    """Preserve released DAPI test IDs and select 10% of trainval for validation."""
    root = Path(benchmark_root) / CELLFM_RELATIVE_ROOT
    metadata = root / "metadata.csv"
    if source_audit is not None and (not source_audit.get("archive_md5_verified")
                                    or source_audit["metadata_sha256"] != file_digest(metadata)):
        raise ValueError("Source audit does not verify current CellFMCount metadata")
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
                        "image_pixel_sha256": pixel_digest(image_path),
                        "annotation_sha256": file_digest(annotation)})
    if source_audit is not None:
        audited = {r["id"]: r for r in source_audit["all_records"]}
        for record in records:
            expected = audited.get(record["sample_id"])
            if expected is None or any(record[key] != expected[key] for key in
                                       ("image_sha256", "image_pixel_sha256", "annotation_sha256")):
                raise ValueError(f"Source audit is stale for CellFMCount image/annotation: {record['sample_id']}")
    if duplicate_policy not in {"quarantine", "error"}:
        raise ValueError(f"Unknown CellFMCount duplicate policy: {duplicate_policy}")
    excluded = []
    if duplicate_policy == "quarantine":
        records, excluded = _deduplicate_records(records)
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
    manifest = {"dataset": "cellfmcount-dapi-count", "version": 2, "seed": seed,
                "task": "regression", "classification": "PROPOSED_BY_US",
                "scope": "DAPI only", "official_split": "released metadata trainval/test IDs",
                "validation_classification": "PROPOSED_BY_US",
                "validation_split": "10% trainval IDs (ceil), sorted numerically then numpy default_rng(seed) permutation",
                "grouping": "deduplicated decoded image content; specimen/donor IDs unavailable and biological grouping unverified",
                "metric": "mae", "preprocessing": "existing _to_rgb_uint8 minmax followed by full-FOV mean-color square padding",
                "license": {"id": "cc-by-sa-4.0", "source": "https://zenodo.org/api/records/17088532"},
                "license_status": "AVAILABLE_CCBYSA",
                "readiness": "READY_SCALAR_COUNT_RESIDUAL_BIOLOGICAL_GROUP_UNKNOWN",
                "metadata_sha256": file_digest(metadata), "stats": stats, "records": records,
                "duplicate_policy": duplicate_policy, "excluded_duplicates": excluded,
                "duplicate_handling_rule": "Quarantine all DAPI records sharing decoded pixels with conflicting counts; equal-count duplicates retain smallest released test ID if present, otherwise smallest trainval ID. Never average or choose conflicting labels.",
                "source_audit_sha256": digest(source_audit) if source_audit is not None else None,
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
