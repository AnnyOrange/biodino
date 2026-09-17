"""Leakage-aware manifests for released CytoImageNet and Allen cell crops.

These are explicitly proposed protocols, not replacements silently advertised as
the original papers' benchmark splits. Pixel auditing is required for readiness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, image_digests
from .datasets import load_flattened_multichannel_image


CYTO_ROOT = "Representation/CytoImageNet"
ALLEN_ROOT = "Classification/CHAMMI"


def _ordered(groups, seed):
    return sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())


def validate_grouped_manifest(manifest):
    records = manifest["records"]
    if not records:
        raise ValueError("Empty grouped manifest")
    seen, groups, pixels, fovs = set(), {}, {}, {}
    stats = {}
    for record in records:
        sample = record["sample_id"]
        if sample in seen:
            raise ValueError(f"Duplicate sample ID: {sample}")
        seen.add(sample)
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if record["group"] in groups and groups[record["group"]] != split:
            raise ValueError(f"Source group leakage: {record['group']}")
        groups[record["group"]] = split
        if record.get("fov"):
            fov = record["fov"]
            if fov in fovs and fovs[fov] != split:
                raise ValueError(f"Source FOV leakage: {fov}")
            fovs[fov] = split
        if not np.isfinite(record["target"]):
            raise ValueError(f"Invalid target: {sample}")
        pixel = record.get("image_pixel_sha256")
        if pixel is not None:
            if pixel in pixels:
                raise ValueError(f"Duplicate decoded pixels: {sample}/{pixels[pixel]}")
            pixels[pixel] = sample
    for split in ("train", "val", "test"):
        selected = [r for r in records if r["split"] == split]
        if not selected:
            raise ValueError(f"Empty {split}")
        stats[split] = {"samples": len(selected), "groups": len({r["group"] for r in selected})}
        if manifest["task"] == "classification":
            if {r["target"] for r in selected} != set(range(len(manifest["classes"]))):
                raise ValueError(f"Incomplete class coverage: {split}")
    return stats


def _audit_and_deduplicate(records, image_root, audit_pixels):
    if not audit_pixels:
        return records, [], {"completed": False, "files_checked": 0}
    pixel_groups = defaultdict(list)
    # Bounded batches avoid allocating one future per crop in the 890k-image release.
    with ThreadPoolExecutor(max_workers=16) as executor:
        for start in range(0, len(records), 1000):
            batch = records[start:start + 1000]
            results = executor.map(image_digests, (image_root / r["path"] for r in batch))
            for record, (encoded, decoded) in zip(batch, results):
                record["image_sha256"], record["image_pixel_sha256"] = encoded, decoded
                pixel_groups[decoded].append(record)
            if (start + len(batch)) % 10000 == 0:
                print(f"pixel audit: {start + len(batch)}/{len(records)}", flush=True)
    kept, excluded = [], []
    for pixel, members in sorted(pixel_groups.items()):
        if len({r["target"] for r in members}) > 1:
            excluded.extend({**r, "reason": "CONFLICTING_TARGET_PIXEL_DUPLICATE"} for r in members)
        else:
            ordered = sorted(members, key=lambda r: (r.get("source_split") != "Task_two", r["sample_id"]))
            kept.append(ordered[0])
            excluded.extend({**r, "reason": "SAME_TARGET_PIXEL_DUPLICATE", "retained_sample_id": ordered[0]["sample_id"]}
                            for r in ordered[1:])
    return kept, excluded, {"completed": True, "files_checked": len(records),
                            "duplicate_groups": sum(len(g) > 1 for g in pixel_groups.values())}


def build_cytoimagenet_manifest(benchmark_root, seed=0, audit_pixels=True):
    root = Path(benchmark_root) / CYTO_ROOT
    metadata = root / "metadata" / "metadata.csv"
    records, original_groups = [], defaultdict(set)
    with metadata.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["idx"] or not row["label"]:
                raise ValueError("Missing CytoImageNet source idx/label")
            relative = Path(row["path"]).name + "/" + row["filename"]
            if not audit_pixels and not (root / "extracted" / relative).is_file():
                raise ValueError(f"Missing CytoImageNet file: {relative}")
            original_groups[row["idx"]].add(row["dset"])
            records.append({"sample_id": relative, "path": relative, "group": row["idx"],
                            "label": row["label"], "target": 0, "source_split": row["dset"],
                            "source_dataset": row["dir_name"]})
    # Populate labels before deduplication, otherwise conflicting weak labels go unnoticed.
    all_classes = sorted({r["label"] for r in records})
    targets = {label: i for i, label in enumerate(all_classes)}
    for record in records:
        record["target"] = targets[record["label"]]
    records, excluded, content_audit = _audit_and_deduplicate(records, root / "extracted", audit_pixels)
    label_groups = {label: set() for label in all_classes}
    group_labels = defaultdict(set)
    for record in records:
        label_groups[record["label"]].add(record["group"])
        group_labels[record["group"]].add(record["label"])
    if any(len(labels) != 1 for labels in group_labels.values()):
        raise ValueError("CytoImageNet source idx assigned multiple labels; cannot stratify independently")
    insufficient = {label: len(groups) for label, groups in label_groups.items() if len(groups) < 3}
    classes = sorted(set(label_groups) - insufficient.keys())
    classes_to_idx = {label: i for i, label in enumerate(classes)}
    assignments = {}
    for label in classes:
        groups = _ordered(label_groups[label], seed)
        n_holdout = max(1, int(np.ceil(0.1 * len(groups))))
        if 2 * n_holdout >= len(groups):
            n_holdout = 1
        for i, group in enumerate(groups):
            assignments[group] = "test" if i < n_holdout else "val" if i < 2 * n_holdout else "train"
    kept = []
    for record in records:
        if record["label"] in insufficient:
            excluded.append({**record, "reason": "FEWER_THAN_THREE_SOURCE_IMAGES_FOR_CLASS"})
        else:
            record["target"] = classes_to_idx[record["label"]]
            record["split"] = assignments[record["group"]]
            kept.append(record)
    manifest = {"dataset": "cytoimagenet-source-grouped", "version": 1, "task": "classification",
                "classification": "PROPOSED_BY_US", "seed": seed, "classes": classes,
                "records": sorted(kept, key=lambda r: r["sample_id"]), "excluded_records": excluded,
                "grouping": "Released idx (original source image); every scale/crop of one source remains together",
                "split_definition": "Per-class source groups ordered by SHA256(seed:idx); ceil(10%) test, ceil(10%) validation, remainder training; minimum one group per split; classes with <3 source groups excluded explicitly",
                "metric": "top1_accuracy", "preprocessing": "Released grayscale PNG converted to RGB; encoder spatial transform specified by authoritative machine-readable selection config; no upstream channel reconstruction",
                "aggregation": "micro image accuracy primary, macro class accuracy secondary",
                "content_audit": content_audit, "metadata_sha256": file_digest(metadata),
                "official_source_idx_split_leakage": sum(len(splits) > 1 for splits in original_groups.values()),
                "excluded_class_source_group_counts": insufficient,
                "sources": ["https://github.com/stan-hua/CytoImageNet", "https://github.com/stan-hua/CytoImageNet/blob/master/scripts/model_pretraining.py"],
                "limitations": ["Original image idx does not guarantee donor/plate/well disjointness; those upstream mappings are not in released metadata.",
                                "Weak labels may overlap semantically. This is not the author's 894-class row-random reproduction.",
                                "Pretraining corpus overlap must be checked independently before interpreting as OOD."]}
    manifest["stats"] = validate_grouped_manifest(manifest)
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def build_allen_morphology_manifest(benchmark_root, seed=0, audit_pixels=True):
    root = Path(benchmark_root) / ALLEN_ROOT
    metadata = root / "Allen" / "enriched_meta.csv"
    records, source_fovs = [], defaultdict(set)
    with metadata.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source_split = row["train_test_split"]
            if source_split not in {"Train", "Task_one", "Task_two"}:
                raise ValueError(f"Unknown CHAMMI Allen split: {source_split}")
            target = float(row["cell_volume"])
            if not np.isfinite(target) or target < 0:
                raise ValueError(f"Invalid Allen volume for {row['CellId']}")
            if not row["WellId"] or not row["FOVId"] or not row["PlateId"]:
                raise ValueError("Missing Allen acquisition identity")
            if not audit_pixels and not (root / row["file_path"]).is_file():
                raise ValueError(f"Missing Allen crop: {row['file_path']}")
            source_fovs[row["FOVId"]].add(source_split)
            records.append({"sample_id": row["CellId"], "path": row["file_path"],
                            "target": float(np.log1p(target)), "raw_cell_volume": target,
                            "group": row["PlateId"] + ":" + row["WellId"],
                            "fov": row["FOVId"], "plate": row["PlateId"], "well": row["WellId"],
                            "source_split": source_split, "channel_width": int(float(row["channel_width"]))})
    records, excluded, content_audit = _audit_and_deduplicate(records, root, audit_pixels)
    test_groups = {r["group"] for r in records if r["source_split"] == "Task_two"}
    trainval_groups = {r["group"] for r in records if r["source_split"] in {"Train", "Task_one"}}
    if trainval_groups & test_groups:
        raise ValueError("CHAMMI Task_two has wells in training pool; preserve test by redesigning split explicitly")
    ordered = _ordered(trainval_groups, seed)
    val_groups = set(ordered[:max(1, int(np.ceil(0.1 * len(ordered))))])
    for record in records:
        record["split"] = "test" if record["group"] in test_groups else "val" if record["group"] in val_groups else "train"
    manifest = {"dataset": "allen-cell-volume-well-grouped", "version": 1, "task": "regression",
                "classification": "PROPOSED_BY_US", "seed": seed,
                "grouping": "PlateId:WellId; all FOVs and cell crops of one well kept together",
                "split_definition": "Released CHAMMI Task_two retained as shifted-distribution test; merge Train and Task_one into training pool, then SHA256(seed:PlateId:WellId) order with ceil(10%) whole wells for validation",
                "metric": "mae", "target_transform": "log1p(cell_volume)",
                "preprocessing": "Existing flattened multichannel loader; independent 1st/99th percentile clipping per channel; preserve released Allen WTC-11 three channels using existing channel-adaptive route",
                "aggregation": "micro cell MAE primary; well-macro MAE and R2/Spearman secondary",
                "records": sorted(records, key=lambda r: int(r["sample_id"])), "excluded_records": excluded,
                "metadata_sha256": file_digest(metadata), "content_audit": content_audit,
                "official_source_fov_split_leakage": sum(len(splits) > 1 for splits in source_fovs.values()),
                "sources": ["https://github.com/chaudatascience/channel_adaptive_models", "https://doi.org/10.5281/zenodo.7988357", "https://www.allencell.org/terms-of-use.html"],
                "limitations": ["Task_two is a shifted-structure holdout, not an official volume-regression benchmark.",
                                "Well-disjoint is not plate-disjoint or donor-disjoint; hiPSC cells are a shared cell-line resource.",
                                "Allen research/noncommercial usage terms apply; software license does not license images."]}
    manifest["stats"] = validate_grouped_manifest(manifest)
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


class GroupedBenchmarkDataset:
    def __init__(self, benchmark_root, manifest, split):
        if manifest["dataset"] not in {"allen-cell-volume-well-grouped", "cytoimagenet-source-grouped"}:
            raise ValueError(f"Unsupported grouped benchmark: {manifest['dataset']}")
        validate_grouped_manifest(manifest)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.manifest = manifest
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.root = Path(benchmark_root) / (ALLEN_ROOT if manifest["task"] == "regression" else CYTO_ROOT + "/extracted")
        if manifest["task"] == "classification":
            self.classes = manifest["classes"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["path"]
        if self.manifest["task"] == "regression":
            image = load_flattened_multichannel_image(path, record["channel_width"], p_low=1, p_high=99)
            target = float(record["target"])
        else:
            with Image.open(path) as source:
                image = source.convert("RGB")
            target = int(record["target"])
        return image, target, record["sample_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dataset", choices=["cytoimagenet", "allen"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-pixel-audit", action="store_true")
    args = parser.parse_args()
    build = build_cytoimagenet_manifest if args.dataset == "cytoimagenet" else build_allen_morphology_manifest
    manifest = build(args.benchmark_root, seed=args.seed, audit_pixels=not args.skip_pixel_audit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"dataset": manifest["dataset"], "stats": manifest["stats"],
                      "manifest_sha256": manifest["manifest_sha256"], "content_audit": manifest["content_audit"]}))


if __name__ == "__main__":
    main()
