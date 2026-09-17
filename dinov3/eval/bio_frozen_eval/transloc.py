"""Recover published Transloc ratios and require verifiable original image IDs.

The Zenodo StyleGAN release is complete but irreversibly renames crops and
publishes binary conditions, not regression targets. Its numerical image
order must never be joined to the ChAda-ViT CSV row order.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, validate_records
from .datasets import _pad_to_square, _to_rgb_uint8
from .opencell import _fetch


TRANSLOC_RELATIVE_ROOT = "Regression/Transloc"
SPLIT_BASE = "https://raw.githubusercontent.com/nicoboou/chadavit/main/src/data/splits/translocation/"
ZENODO_SOURCE = "https://zenodo.org/api/records/8287453"
ORIGINAL_NAME = re.compile(r"^(\d{6})_([A-P]\d{2})_(\d+)_(\d+)_(\d+)\.png$")


def inspect_source_workbook(path):
    """Inspect the original paper's numeric source data without Excel packages."""
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as handle:
        strings = ["".join(node.itertext()) for node in ET.fromstring(handle.read("xl/sharedStrings.xml")).findall("s:si", namespace)]
        rows = ET.fromstring(handle.read("xl/worksheets/sheet3.xml")).findall("s:sheetData/s:row", namespace)
        counts = Counter()
        for row in rows[1:]:
            cells = {}
            for cell in row.findall("s:c", namespace):
                value = cell.findtext("s:v", namespaces=namespace)
                cells[re.sub(r"\d", "", cell.attrib["r"])] = strings[int(value)] if cell.attrib.get("t") == "s" else value
            if cells.get("A") == "REAL" and cells.get("C") not in (None, "NaN"):
                counts[cells["B"]] += 1
        return {"dataset_sheet": "Figure 1.D Raw Data", "real_ratio_counts": dict(counts),
                "target_header": "green cell/nucleus ratio",
                "crop_identity_strings_present": any(".png" in value or "170721" in value or "img000" in value for value in strings),
                "mapping_available": False,
                "limitation": "Only data-type, condition, numeric ratio columns; no image/cell/FOV identities. Row order is not an authoritative mapping."}


def recover_transloc_metadata(benchmark_root):
    root = Path(benchmark_root) / TRANSLOC_RELATIVE_ROOT
    metadata = root / "analyse_256x256"
    metadata.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        name = f"single_ratio_images_{split}.csv"
        if not (metadata / name).exists():
            (metadata / name).write_bytes(_fetch(SPLIT_BASE + name))
    sources = root / "official_sources"
    sources.mkdir(exist_ok=True)
    checks = {
        "zenodo_record.json": ZENODO_SOURCE,
        "chadavit_custom_datasets.py": "https://raw.githubusercontent.com/nicoboou/chadavit/main/src/data/custom_datasets.py",
        "phenexplain_make_datasetjson.py": "https://raw.githubusercontent.com/biocompibens/phenexplain/master/make_datasetjson.py",
        "phenexplain_README.md": "https://raw.githubusercontent.com/biocompibens/phenexplain/master/README.md",
        "chadavit_issues.json": "https://api.github.com/repos/nicoboou/chadavit/issues?state=all&per_page=100",
        "chadavit_issue5_comments.json": "https://api.github.com/repos/nicoboou/chadavit/issues/5/comments",
        "chadavit_repository_tree.json": "https://api.github.com/repos/nicoboou/chadavit/git/trees/main?recursive=1",
        "biophenics_repository_tree.json": "https://api.github.com/repos/biocompibens/BioPhenics_app/git/trees/main?recursive=1",
        "phenexplain_source_data.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-42124-6/MediaObjects/41467_2023_42124_MOESM3_ESM.xlsx",
    }
    for name, url in checks.items():
        if not (sources / name).exists():
            (sources / name).write_bytes(_fetch(url))
    rows = read_ratio_rows(root)
    zenodo = json.loads((sources / "zenodo_record.json").read_text())
    archive_info = next(row for row in zenodo["files"] if row["key"] == "translocation_256.zip")
    archive_path = root / "archives" / "translocation_256.zip"
    archive = {"official_size": archive_info["size"], "official_checksum": archive_info["checksum"]}
    if archive_path.exists():
        result = hashlib.md5()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                result.update(chunk)
        archive["local_size"] = archive_path.stat().st_size
        archive["local_checksum"] = "md5:" + result.hexdigest()
        archive["matches_official"] = (archive["local_size"] == archive["official_size"] and
                                       archive["local_checksum"] == archive["official_checksum"])
        with zipfile.ZipFile(archive_path) as handle:
            annotations = json.loads(handle.read("dataset.json"))
            archive["annotation_keys"] = sorted(annotations)
            archive["image_records"] = len(annotations["labels"])
            archive["source_target_values"] = dict(Counter(str(row[1]) for row in annotations["labels"]))
            archive["original_names_present"] = any(ORIGINAL_NAME.fullmatch(Path(name).name) for name in handle.namelist())
    present = sum((root / row["path"]).is_file() for row in rows)
    source_groups = {split: {row["group"] for row in rows if row["source_split"] == split}
                     for split in ("train", "val")}
    result = {"dataset": "Transloc", "continuous_metadata_recovered": True,
              "ratio_rows": dict(Counter(row["source_split"] for row in rows)),
              "ratio_treatment_counts": dict(Counter(row["category"] for row in rows)),
              "original_source_crop_names_recovered": len(rows), "matched_original_images": present,
              "original_wells": len({row["group"] for row in rows}),
              "original_fovs": len({row["fov"] for row in rows}),
              "published_train_val_overlapping_wells": len(source_groups["train"] & source_groups["val"]),
              "archive": archive, "license": zenodo["metadata"]["license"]["id"],
              "root_cause": "OTHER: complete official StyleGAN release discards original crop identities and continuous ratios; not incomplete download or parser failure",
              "handling": "use only original_named 256x256/<cat>/<image_name>; reject numerical-order or category-derived ratio joins",
              "checked_sources": checks,
              "original_paper_source_data": inspect_source_workbook(sources / "phenexplain_source_data.xlsx"),
              "remaining_requirement": None if present == len(rows) else "author-provided original-name crop release or authoritative source-name mapping (not present in either official public repository or Zenodo metadata/archive)"}
    return result


def read_ratio_rows(root):
    root = Path(root)
    rows, seen = [], set()
    for split in ("train", "val"):
        path = root / "analyse_256x256" / f"single_ratio_images_{split}.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["cat", "image_name", "ratio"]:
                raise ValueError(f"Unexpected official Transloc header: {reader.fieldnames}")
            for row in reader:
                match = ORIGINAL_NAME.fullmatch(row["image_name"])
                if not match or row["cat"] not in ("0", "5", "10", "20"):
                    raise ValueError(f"Unknown official Transloc crop identity: {row}")
                day, well, field, x, y = match.groups()
                if row["image_name"] in seen:
                    raise ValueError(f"Duplicate original Transloc crop: {row['image_name']}")
                seen.add(row["image_name"])
                target = float(row["ratio"])
                if not np.isfinite(target) or target < 0:
                    raise ValueError(f"Invalid nuclear-translocation ratio: {row}")
                rows.append({"sample_id": row["image_name"], "path": f"256x256/{row['cat']}/{row['image_name']}",
                             "target": target, "group": f"{day}:{well}", "fov": f"{day}:{well}:{field}",
                             "category": row["cat"], "source_split": split, "cell_xy": [int(x), int(y)]})
    return sorted(rows, key=lambda row: row["sample_id"])


def build_transloc_manifest(benchmark_root, seed=0):
    """Proposed well-held-out regrouping of the published ratio-bearing crops."""
    root = Path(benchmark_root) / TRANSLOC_RELATIVE_ROOT
    records = read_ratio_rows(root)
    if len(records) != 10571:
        raise ValueError(f"Expected 10571 released ratio-bearing crops, found {len(records)}")
    missing = [row["path"] for row in records if not (root / row["path"]).is_file()]
    if missing:
        raise ValueError(f"Transloc has recovered ratios but missing {len(missing)} original-identity images; first={missing[0]}. Anonymous StyleGAN numbering is not a mapping.")
    groups = defaultdict(set)
    group_category = {}
    for row in records:
        if row["group"] in group_category and group_category[row["group"]] != row["category"]:
            raise ValueError(f"One biological well has several treatment categories: {row['group']}")
        group_category[row["group"]] = row["category"]
        groups[row["category"]].add(row["group"])
    rng, assignments = np.random.default_rng(seed), {}
    for category in sorted(groups, key=int):
        ordered = rng.permutation(sorted(groups[category])).tolist()
        if len(ordered) < 3:
            raise ValueError(f"Insufficient well groups for category {category}")
        heldout = max(1, int(np.ceil(0.15 * len(ordered))))
        for i, group in enumerate(ordered):
            assignments[group] = "test" if i < heldout else ("val" if i < 2 * heldout else "train")
    for row in records:
        path = root / row["path"]
        with Image.open(path) as image:
            image.load()
            if image.size != (256, 256) or image.mode != "RGB":
                raise ValueError(f"Unexpected Transloc image layout: {path}: {image.size}/{image.mode}")
            row["pixel_sha256"] = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        row["image_sha256"] = file_digest(path)
        row["split"] = assignments[row["group"]]
    if len({row["pixel_sha256"] for row in records}) != len(records):
        raise ValueError("Duplicate Transloc pixels require source-level adjudication")
    manifest = {"dataset": "transloc-ratio-well-heldout", "version": 1, "task": "regression",
                "classification": "PROPOSED_BY_US", "seed": seed, "records": records,
                "stats": validate_records(records), "metric": "r2", "secondary_metrics": ["mae", "pearson"],
                "split": "per-treatment sorted acquisition-day/well groups; default_rng(seed) permutation; ceil(15%) test then ceil(15%) val, remaining train",
                "grouping": "acquisition day/well, all FOV/cell crops together",
                "preprocessing": "original 256x256 RGB channels; existing encoder resize/normalize",
                "source_splits": {split: file_digest(root / 'analyse_256x256' / f'single_ratio_images_{split}.csv') for split in ('train', 'val')},
                "sources": [SPLIT_BASE, ZENODO_SOURCE], "license": "CC-BY-4.0",
                "limitations": ["Proposed well-disjoint split replaces released leaking cell-crop train/val split.",
                                "Single acquisition day; no patient/specimen IDs are provided.",
                                "Ridge/OLS task is separate from the paper's trained SGD linear head."]}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


class TranslocDataset:
    def __init__(self, benchmark_root, manifest, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split}")
        validate_records(manifest["records"])
        self.root = Path(benchmark_root) / TRANSLOC_RELATIVE_ROOT
        self.records = [row for row in manifest["records"] if row["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["path"]) as image:
            rgb = Image.fromarray(_to_rgb_uint8(np.asarray(image)))
        return _pad_to_square(rgb), float(row["target"]), row["sample_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--recover-metadata", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = recover_transloc_metadata(args.benchmark_root) if args.recover_metadata else build_transloc_manifest(args.benchmark_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
