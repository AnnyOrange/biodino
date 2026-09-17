"""Explicit, versioned scalar-regression candidates; no random image fallback."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import _pad_to_square, _to_rgb_uint8


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def pixel_digest(path):
    """Hash decoded pixels, including shape/dtype, independently of image encoding."""
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected single-frame image: {path}")
        pixels = np.asarray(image)
        result = hashlib.sha256()
        result.update(str((pixels.shape, pixels.dtype.str)).encode("ascii"))
        result.update(np.ascontiguousarray(pixels).tobytes())
        return result.hexdigest()


def image_digests(path):
    """Read once for encoded and decoded identity; suitable for small crop files."""
    data = Path(path).read_bytes()
    encoded = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected single-frame image: {path}")
        pixels = np.asarray(image)
        decoded = hashlib.sha256()
        decoded.update(str((pixels.shape, pixels.dtype.str)).encode("ascii"))
        decoded.update(np.ascontiguousarray(pixels).tobytes())
    return encoded, decoded.hexdigest()


def validate_records(records):
    if not records:
        raise ValueError("Empty dataset manifest")
    seen = set()
    groups = {split: set() for split in ("train", "val", "test")}
    contents = {}
    pixel_contents = {}
    for record in records:
        split, group, sample = record["split"], record["group"], record["sample_id"]
        if split not in groups or not group or sample in seen:
            raise ValueError(f"Invalid split/group or duplicate sample: {record}")
        seen.add(sample)
        groups[split].add(group)
        if not np.isfinite(record["target"]):
            raise ValueError(f"Nonfinite target for {sample}")
        image_sha = record["image_sha256"]
        if image_sha in contents:
            raise ValueError(f"Duplicate image content: {sample} and {contents[image_sha]}")
        contents[image_sha] = sample
        pixel_sha = record.get("image_pixel_sha256")
        if pixel_sha is not None:
            if pixel_sha in pixel_contents:
                raise ValueError(f"Duplicate decoded image content: {sample} and {pixel_contents[pixel_sha]}")
            pixel_contents[pixel_sha] = sample
    for split, split_groups in groups.items():
        if not split_groups:
            raise ValueError(f"Empty {split} split")
        for other in groups:
            if split != other and split_groups & groups[other]:
                raise ValueError(f"Group leakage: {split}/{other}")
    return {split: {"samples": sum(r["split"] == split for r in records),
                    "groups": len(groups[split])} for split in groups}


def build_idcia_manifest(benchmark_root, seed=0):
    """Hold out complete treatment conditions, including both paired channels."""
    root = Path(benchmark_root) / "ood/ood_regression/datasets/IDCIA/extracted/IDCIA"
    images = sorted((root / "images").rglob("*.tiff"))
    annotations = {}
    for path in sorted((root / "ground_truth").rglob("*.csv")):
        key = path.stem.casefold()
        if key in annotations:
            raise ValueError(f"Ambiguous annotation {key}")
        annotations[key] = path
    records, excluded = [], []
    for image in images:
        fields = image.stem.split("_")
        if len(fields) != 8 or fields[2] not in ("A", "B", "C", "D"):
            raise ValueError(f"Unrecognized IDCIA filename {image.name}")
        if "." in fields[4]:
            excluded.append(image.name)
            continue
        annotation = annotations[image.stem.casefold()]
        with annotation.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"X", "Y"}.issubset(reader.fieldnames):
                raise ValueError(f"Unrecognized coordinate annotation {annotation}")
            points = list(reader)
        for point in points:
            if not all(np.isfinite(float(point[key])) for key in ("X", "Y")):
                raise ValueError(f"Invalid coordinate in {annotation}")
        records.append({"sample_id": image.stem, "path": str(image.relative_to(root)),
                        "target": len(points), "group": fields[1].casefold() + ":" + fields[2],
                        "fov": "_".join(fields[:5]).casefold(),
                        "image_sha256": file_digest(image),
                        "annotation_sha256": file_digest(annotation)})
    group_names = sorted({record["group"] for record in records})
    if len(group_names) != 4:
        raise ValueError(f"Expected four IDCIA condition groups, found {group_names}")
    ordered = np.random.default_rng(seed).permutation(group_names).tolist()
    assignments = {group: "train" if i < 2 else ("val" if i == 2 else "test")
                   for i, group in enumerate(ordered)}
    for record in records:
        record["split"] = assignments[record["group"]]
    stats = validate_records(records)
    source_splits = {}
    for split in ("train", "val", "test"):
        with (root / f"{split}.csv").open(newline="") as handle:
            names = [row[0] for row in csv.reader(handle) if row and row[0]]
        source_splits[split] = {"sha256": file_digest(root / f"{split}.csv"),
                                "samples": len(names)}
    return {"dataset": "idcia-condition-count", "version": 1, "seed": seed,
            "classification": "PROPOSED_BY_US", "task": "regression",
            "grouping": "celltype + treatment condition across dates/markers/FOVs/channels",
            "group_assignments": assignments, "stats": stats, "source_splits": source_splits,
            "excluded_subfields": excluded, "records": records,
            "limitations": ["Four condition groups only; one validation and one test condition.",
                            "No donor/specimen IDs: not a donor-heldout benchmark.",
                            "Frozen-backbone count regression, not official density-map counting."]}


class CandidateCountDataset:
    def __init__(self, benchmark_root, manifest, split):
        self.root = Path(benchmark_root) / "ood/ood_regression/datasets/IDCIA/extracted/IDCIA"
        self.records = [r for r in manifest["records"] if r["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            # Retain high-bit-depth contrast and the full FOV for whole-image counts.
            rgb = Image.fromarray(_to_rgb_uint8(np.asarray(image)))
        return _pad_to_square(rgb), float(record["target"]), record["sample_id"]
