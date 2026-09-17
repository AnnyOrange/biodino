"""Official OpenCell projection acquisition and proposed protein-held-out task.

This is not the Cytoself single-cell benchmark. All FOVs of each tagged
protein are kept together; multi-localizing proteins are excluded explicitly.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, validate_records
from .datasets import _pad_to_square


OPENCELL_RELATIVE_ROOT = "external_benchmarks_20260901/OpenCell_projections"
BUCKET = "https://czb-opencell.s3.us-west-2.amazonaws.com"
ANNOTATION_URL = ("https://raw.githubusercontent.com/czbiohub-sf/2021-opencell-figures/"
                  "master/data/2021-09-29-public-annotations-flat.csv")
DOCUMENTATION = "https://opencell.sf.czbiohub.org/download"
LICENSE_SOURCE = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/czb-opencell.yaml"
FILENAME = re.compile(r"^OC-FOV_(.+)_(ENSG\d+)_(CID\d+)_(FID\d+)_proj\.tif$")
EXPECTED_PROJECTIONS = 6301


def _fetch(url):
    if any(os.environ.get(key) for key in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")) and shutil.which("curl"):
        # Reuse the repo's dataset-download transport; urllib CONNECT can stall
        # on institutional proxies even when curl transfers the same URL.
        for attempt in range(4):
            try:
                return subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--connect-timeout", "10", "--max-time", "90", url],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    # Some institutional networks intermittently blackhole raw.githubusercontent.
    # GitHub's content API serves the same versioned source with base64 content.
    if url.startswith("https://raw.githubusercontent.com/"):
        owner, repo, ref, path = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?" + urlencode({"ref": ref})
        try:
            payload = json.loads(_fetch(endpoint))
            if payload.get("encoding") == "base64":
                return base64.b64decode(payload["content"])
        except (OSError, HTTPException, ValueError, KeyError):
            pass
    for attempt in range(4):
        try:
            with urlopen(url, timeout=30) as handle:
                return handle.read()
        except (OSError, HTTPException, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def list_official_projections():
    """Enumerate S3 exactly, including pagination, without AWS credentials."""
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects, continuation = [], None
    while True:
        query = {"list-type": "2", "prefix": "microscopy/raw/", "max-keys": 1000}
        if continuation:
            query["continuation-token"] = continuation
        page = ET.fromstring(_fetch(BUCKET + "/?" + urlencode(query)))
        for node in page.findall("s:Contents", namespace):
            key = node.findtext("s:Key", namespaces=namespace)
            if not key.endswith("_proj.tif"):
                continue
            match = FILENAME.fullmatch(Path(key).name)
            if not match:
                raise ValueError(f"Unknown official projection filename: {key}")
            gene, protein, cell_line, fov = match.groups()
            objects.append({"path": key, "bytes": int(node.findtext("s:Size", namespaces=namespace)),
                            "etag": node.findtext("s:ETag", namespaces=namespace).strip('"'),
                            "protein": protein, "gene": gene, "cell_line": cell_line, "fov": fov})
        if page.findtext("s:IsTruncated", namespaces=namespace) == "false":
            break
        continuation = page.findtext("s:NextContinuationToken", namespaces=namespace)
        if not continuation:
            raise ValueError("Truncated S3 response without continuation token")
    objects.sort(key=lambda row: row["path"])
    if len({row["fov"] for row in objects}) != len(objects):
        raise ValueError("Official projection listing contains duplicate FOV IDs")
    return objects


def select_proteins(root, source, seed=0, min_proteins_per_class=10, max_proteins_per_class=16):
    if min_proteins_per_class < 3 or (max_proteins_per_class is not None and max_proteins_per_class < 3):
        raise ValueError("At least three proteins per class are needed for three splits")
    major = defaultdict(set)
    with (Path(root) / "annotations.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ensg_id", "target_name", "annotation_name", "annotation_grade"]:
            raise ValueError("Unknown official OpenCell annotation header")
        for row in reader:
            if int(row["annotation_grade"]) == 3:
                major[row["ensg_id"]].add(row["annotation_name"])
    available = {row["protein"] for row in source["objects"]}
    proteins_by_class = defaultdict(list)
    for protein, labels in major.items():
        if protein in available and len(labels) == 1:
            proteins_by_class[next(iter(labels))].append(protein)
    selected = {}
    for label in sorted(proteins_by_class):
        proteins = proteins_by_class[label]
        if len(proteins) < min_proteins_per_class:
            continue
        ordered = sorted(proteins, key=lambda protein: (hashlib.sha256(f"{seed}:{protein}".encode()).hexdigest(), protein))
        selected[label] = sorted(ordered if max_proteins_per_class is None else ordered[:max_proteins_per_class])
    if not selected:
        raise ValueError("No eligible major-localization classes")
    return selected


def download_opencell(benchmark_root, workers=64, seed=0, max_proteins_per_class=16):
    """Acquire every FOV in an explicit bounded protein subset (or full mode)."""
    root = Path(benchmark_root) / OPENCELL_RELATIVE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "official_projection_objects.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
    else:
        objects = list_official_projections()
        if len(objects) != EXPECTED_PROJECTIONS:
            raise ValueError(f"Official dataset changed: expected {EXPECTED_PROJECTIONS}, found {len(objects)}")
        source = {"dataset": "OpenCell", "source": BUCKET, "documentation": DOCUMENTATION,
                  "license": "CC-BY-SA-4.0", "license_source": LICENSE_SOURCE,
                  "expected_fovs": EXPECTED_PROJECTIONS, "objects": objects}
        source_path.write_text(json.dumps(source, indent=2) + "\n")
    for name, url in (("annotations.csv", ANNOTATION_URL), ("license-source.yaml", LICENSE_SOURCE)):
        if not (root / name).exists():
            (root / name).write_bytes(_fetch(url))
    selected = select_proteins(root, source, seed=seed, max_proteins_per_class=max_proteins_per_class)
    selected_proteins = {protein for proteins in selected.values() for protein in proteins}
    objects = [row for row in source["objects"] if row["protein"] in selected_proteins]
    scope = {"selection": "lowest sha256(str(seed)+':'+ENSG) within each eligible localization class",
             "seed": seed, "max_proteins_per_class": max_proteins_per_class,
             "full_official_fovs": len(source["objects"]), "expected_selected_fovs": len(objects),
             "selected_proteins_by_class": selected, "objects": objects}
    (root / "selected_projection_objects.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(f"OpenCell selected {len(selected_proteins)} proteins/{len(objects)} FOVs from official {len(source['objects'])} FOVs", flush=True)

    def acquire(row):
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            content = path.read_bytes()
        else:
            content = _fetch(BUCKET + "/" + quote(row["path"]))
        if len(content) != row["bytes"] or hashlib.md5(content).hexdigest() != row["etag"]:
            raise ValueError(f"Official S3 size/MD5 mismatch: {row['path']}")
        if not path.exists():
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        return row["fov"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(acquire, row) for row in objects]
        for count, future in enumerate(as_completed(futures), start=1):
            future.result()
            if count % 100 == 0 or count == len(objects):
                print(f"OpenCell verified {count}/{len(objects)} selected projections", flush=True)
    return source


def read_projection(path):
    import tifffile

    with tifffile.TiffFile(path) as handle:
        if len(handle.series) != 1 or handle.series[0].axes != "CYX":
            raise ValueError(f"Unexpected OpenCell TIFF axes: {path}")
        image = handle.asarray()
    if image.shape != (2, 600, 600) or image.dtype != np.uint16:
        raise ValueError(f"Unexpected OpenCell image shape/dtype: {path}: {image.shape}/{image.dtype}")
    return image


def projection_rgb(image):
    """Protein -> red, Hoechst -> green, blue=0; independent per-FOV minmax."""
    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError("OpenCell requires two CYX channels")
    channels = []
    for index in (1, 0):
        values = image[index].astype(np.float32)
        low, high = values.min(), values.max()
        channels.append(np.zeros(values.shape, np.uint8) if high == low else
                        np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8))
    return Image.fromarray(np.stack([*channels, np.zeros_like(channels[0])], axis=-1))


def build_opencell_manifest(benchmark_root, seed=0, min_proteins_per_class=10, max_proteins_per_class=16):
    root = Path(benchmark_root) / OPENCELL_RELATIVE_ROOT
    source = json.loads((root / "official_projection_objects.json").read_text())
    if source["expected_fovs"] != EXPECTED_PROJECTIONS or len(source["objects"]) != EXPECTED_PROJECTIONS:
        raise ValueError("Incomplete official OpenCell source listing")
    proteins_by_class = select_proteins(root, source, seed, min_proteins_per_class, max_proteins_per_class)
    classes = sorted(proteins_by_class)
    assignments, targets = {}, {}
    rng = np.random.default_rng(seed)
    for target, label in enumerate(classes):
        proteins = rng.permutation(sorted(proteins_by_class[label])).tolist()
        heldout = max(1, int(np.ceil(0.15 * len(proteins))))
        for index, protein in enumerate(proteins):
            assignments[protein] = "test" if index < heldout else ("val" if index < 2 * heldout else "train")
            targets[protein] = target
    records = []
    all_fovs, all_hashes = set(), set()
    for row in source["objects"]:
        if row["protein"] not in assignments:
            continue
        match = FILENAME.fullmatch(Path(row["path"]).name)
        if not match or match.groups() != (row["gene"], row["protein"], row["cell_line"], row["fov"]):
            raise ValueError(f"Inconsistent official OpenCell source identities: {row}")
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Missing/incomplete official OpenCell projection: {path}")
        image = read_projection(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != row["etag"]:
            raise ValueError(f"OpenCell source checksum mismatch: {row['path']}")
        raw_hash = file_digest(path)
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if row["fov"] in all_fovs or pixel_hash in all_hashes:
            raise ValueError(f"Duplicate FOV/pixels: {row['path']}")
        all_fovs.add(row["fov"])
        all_hashes.add(pixel_hash)
        records.append({"sample_id": row["fov"], "path": row["path"], "group": row["protein"],
                        "protein": row["protein"], "cell_line": row["cell_line"],
                        "split": assignments[row["protein"]], "target": targets[row["protein"]],
                        "image_sha256": raw_hash, "pixel_sha256": pixel_hash})
    stats = validate_records(records)
    manifest = {"dataset": "opencell-major-localization-protein-heldout", "version": 1,
                "classification": "PROPOSED_BY_US", "task": "classification", "seed": seed,
                "classes": classes, "records": records, "stats": stats,
                "data_complete": True, "selected_projections_verified": len(all_fovs),
                "full_official_fovs_listed": len(source["objects"]),
                "expected_selected_fovs": sum(row["protein"] in assignments for row in source["objects"]),
                "max_proteins_per_class": max_proteins_per_class, "min_proteins_per_class": min_proteins_per_class,
                "selected_proteins_by_class": proteins_by_class,
                "protein_selection": "lowest sha256(str(seed)+':'+ENSG) per eligible class; selection precedes split",
                "scope": f"all FOV projections of selected proteins with exactly one grade3 localization; eligible classes have >={min_proteins_per_class} proteins",
                "split": "per-class sorted ENSG IDs; default_rng(seed) permutation; ceil(15%) test then ceil(15%) val, remainder train",
                "grouping": "stable ENSG protein ID; all cell lines/FOVs for one protein together",
                "metric": "balanced_accuracy", "aggregation": "FOV-level; protein-average accuracy secondary",
                "preprocessing": "600x600 full-FOV; independent channel minmax; protein red, Hoechst green, blue zero; existing encoder resize",
                "annotation_sha256": file_digest(root / "annotations.csv"),
                "source_manifest_sha256": file_digest(root / "official_projection_objects.json"),
                "license": "CC-BY-SA-4.0", "sources": [DOCUMENTATION, ANNOTATION_URL, LICENSE_SOURCE],
                "limitations": ["Not an official supervised benchmark or Cytoself single-cell protocol.",
                                "Multi-major-localizing and insufficient-protein classes excluded explicitly.",
                                "Acquisition batch metadata unavailable; batch-disjointness is not asserted."]}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


class OpenCellDataset:
    def __init__(self, benchmark_root, manifest, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split}")
        validate_records(manifest["records"])
        self.root = Path(benchmark_root) / OPENCELL_RELATIVE_ROOT
        self.records = [row for row in manifest["records"] if row["split"] == split]
        self.classes = manifest["classes"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return _pad_to_square(projection_rgb(read_projection(self.root / row["path"]))), int(row["target"]), row["sample_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="all eligible proteins instead of bounded max16/class subset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.download:
        download_opencell(args.benchmark_root, args.workers, max_proteins_per_class=None if args.full else 16)
    manifest = build_opencell_manifest(args.benchmark_root, max_proteins_per_class=None if args.full else 16)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": manifest["dataset"], "stats": manifest["stats"], "classes": manifest["classes"]}))


if __name__ == "__main__":
    main()
