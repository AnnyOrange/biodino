#!/usr/bin/env python3
"""Prepare deterministic image/label caches for the external-4 quick screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image
from scipy import sparse


DATA = Path("/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901")
DEFAULT_OUT = Path("outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910/cache")
CTC_2D = (
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7",
    "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DH-SIM+",
    "Fluo-N2DL-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC",
)


def stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def rgb224(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > 3:
        array = array.max(axis=0)
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        array = array.max(axis=0)
    if array.ndim == 2:
        finite = array[np.isfinite(array)]
        if finite.size:
            lo, hi = np.percentile(finite, (1, 99))
            array = np.clip((array.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1)
        else:
            array = np.zeros_like(array, dtype=np.float32)
        array = np.repeat((array * 255).astype(np.uint8)[..., None], 3, axis=2)
    else:
        array = array[..., :3]
        if array.dtype != np.uint8:
            finite = array[np.isfinite(array)]
            lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0, 1)
            array = (np.clip((array.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(array).resize((224, 224), Image.Resampling.BILINEAR).convert("RGB"))


def write_cache(folder: Path, images: list[np.ndarray], labels: np.ndarray, metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "images.npy", np.stack(images).astype(np.uint8))
    np.save(folder / "labels.npy", np.asarray(labels))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def evenly(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n, dtype=int)]


def prepare_ctc(out: Path) -> None:
    images, labels, rows = [], [], []
    root = DATA / "CellTrackingChallenge" / "training_zips"
    for domain in CTC_2D:
        archive = root / f"{domain}.zip"
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for sequence, split in (("01", "train"), ("02", "test")):
                raw = sorted(n for n in names if n.startswith(f"{domain}/{sequence}/t") and n.endswith(".tif"))
                pairs = []
                for name in raw:
                    frame = Path(name).stem.removeprefix("t")
                    mask = f"{domain}/{sequence}_GT/TRA/man_track{frame}.tif"
                    if mask in names:
                        pairs.append((name, mask, frame))
                for name, mask, frame in evenly(pairs, 4):
                    raw_im = tifffile.imread(io.BytesIO(zf.read(name)))
                    mask_im = tifffile.imread(io.BytesIO(zf.read(mask)))
                    ids = np.unique(mask_im)
                    count = int(ids.size) - int(ids.size > 0 and ids[0] == 0)
                    images.append(rgb224(raw_im))
                    labels.append(float(count))
                    rows.append({"domain": domain, "sequence": sequence, "split": split, "frame": frame})
    write_cache(out / "ctc", images, np.asarray(labels, np.float32), {
        "protocol_id": "ctc_2d_count_proxy_v1", "rows": rows,
        "note": "2-D frozen-feature cell-count proxy; not official CTC TRA/SEG",
    })


def decode(values) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(values).reshape(-1)]


def read_h5ad_rows(path: Path, wanted_barcodes: list[str], genes: list[str]) -> np.ndarray:
    with h5py.File(path) as f:
        obs = decode(f["obs"]["_index"][:])
        var = decode(f["var"]["_index"][:])
        obs_map = {v: i for i, v in enumerate(obs)}
        var_map = {v: i for i, v in enumerate(var)}
        row_idx = np.asarray([obs_map[v] for v in wanted_barcodes], dtype=np.int64)
        gene_idx = np.asarray([var_map[v] for v in genes], dtype=np.int64)
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            values = x[row_idx][:, gene_idx]
        else:
            # Reading the complete CSR payload for every HEST sample needlessly
            # scans tens of GB.  Only fetch the sparse spans for selected spots.
            indptr = x["indptr"][:]
            wanted = {int(col): j for j, col in enumerate(gene_idx)}
            values = np.zeros((len(row_idx), len(gene_idx)), dtype=np.float32)
            for out_row, source_row in enumerate(row_idx):
                lo, hi = int(indptr[source_row]), int(indptr[source_row + 1])
                indices = x["indices"][lo:hi]
                data = x["data"][lo:hi]
                for col, value in zip(indices, data):
                    target_col = wanted.get(int(col))
                    if target_col is not None:
                        values[out_row, target_col] = value
    return np.log1p(np.asarray(values, dtype=np.float32))


def hest_split(task: Path, split: str, genes: list[str], cap: int):
    records = sorted(csv.DictReader((task / "splits" / f"{split}_0.csv").open()), key=lambda row: row["sample_id"])
    # The one-hour screen uses one deterministic sample from each official
    # split.  The full patient/sample matrix is reserved for canonical HEST.
    records = records[:1]
    chosen = []
    quota = max(1, math.ceil(cap / len(records)))
    for row in records:
        patch_path = task / row["patches_path"]
        with h5py.File(patch_path) as f:
            bars = decode(f["barcode"][:quota])
        # Contiguous spots keep HDF5/NFS reads bounded; sample IDs are sorted
        # and the quota spreads the fixed budget across split members.
        chosen.extend(("", row, idx, barcode) for idx, barcode in enumerate(bars[:quota]))
    chosen = chosen[:cap]
    by_sample = defaultdict(list)
    for _, row, idx, barcode in chosen:
        by_sample[row["sample_id"]].append((row, idx, barcode))
    images, ys, meta = [], [], []
    for sample_id, entries in sorted(by_sample.items()):
        entries = sorted(entries, key=lambda item: item[1])
        row = entries[0][0]
        idxs = [x[1] for x in entries]
        bars = [x[2] for x in entries]
        with h5py.File(task / row["patches_path"]) as f:
            if idxs == list(range(idxs[0], idxs[-1] + 1)):
                patch_images = f["img"][idxs[0]:idxs[-1] + 1]
            else:
                patch_images = f["img"][idxs]
        expr = read_h5ad_rows(task / row["expr_path"], bars, genes)
        images.extend(rgb224(x) for x in patch_images)
        ys.extend(expr)
        meta.extend({"task": task.name, "split": split, "sample_id": sample_id, "barcode": b} for b in bars)
    return images, ys, meta


def prepare_hest(out: Path, cap: int = 64) -> None:
    root = DATA / "HEST_benchmark"
    images, labels, rows, task_genes = [], [], [], {}
    tasks = sorted(p for p in root.iterdir() if (p / "splits/train_0.csv").exists() and (p / "splits/test_0.csv").exists())
    for task in tasks:
        gene_path = task / "var_50genes.json"
        if not gene_path.exists():
            gene_path = task / "mean_50genes.json"
        genes = json.loads(gene_path.read_text())["genes"][:50]
        task_genes[task.name] = genes
        for split in ("train", "test"):
            ims, ys, meta = hest_split(task, split, genes, cap)
            images.extend(ims); labels.extend(ys); rows.extend(meta)
    write_cache(out / "hest", images, np.asarray(labels, np.float32), {
        "protocol_id": "hest_fold0_64_v1", "rows": rows, "task_genes": task_genes,
        "note": "Official fold 0 with deterministic fixed spot budget per task/split",
    })


def normalize_channel(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1)


class RxReader:
    def __init__(self, root: Path):
        import pyarrow.parquet as pq
        self.pq = pq
        self.shards = sorted((root / "data").glob("train-*.parquet"))
        self.files = {}
        self.groups: OrderedDict[tuple[int, int], dict] = OrderedDict()

    def file(self, shard: int):
        if shard not in self.files:
            self.files[shard] = self.pq.ParquetFile(self.shards[shard])
        return self.files[shard]

    def group(self, shard: int, group: int):
        key = (shard, group)
        if key not in self.groups:
            self.groups[key] = self.file(shard).read_row_group(group, columns=["__key__", "jp2"]).to_pydict()
            while len(self.groups) > 12:
                self.groups.popitem(last=False)
        self.groups.move_to_end(key)
        return self.groups[key]

    def read(self, record: dict) -> np.ndarray:
        parquet = self.file(int(record["shard"]))
        offset, group = int(record["row"]), 0
        while offset >= parquet.metadata.row_group(group).num_rows:
            offset -= parquet.metadata.row_group(group).num_rows
            group += 1
        arrays = []
        local = offset
        for channel in range(1, 7):
            table = self.group(int(record["shard"]), group)
            expected = f"{record['key']}_{channel}"
            if table["__key__"][local] != expected:
                raise RuntimeError(f"RxRx3 channel mismatch: {table['__key__'][local]} != {expected}")
            payload = table["jp2"][local]["bytes"]
            with Image.open(io.BytesIO(payload)) as image:
                arrays.append(np.asarray(image).astype(np.float32, copy=False))
            local += 1
            if local == len(table["__key__"]) and channel < 6:
                group += 1; local = 0
        return np.stack(arrays)


def choose_rx_records(records: list[dict], genes_cap: int):
    by_gene = defaultdict(list)
    for row in records:
        if int(stable_key(row["gene"])[:8], 16) % 5 != 0:
            by_gene[row["gene"]].append(row)
    gallery, query = [], []
    for gene in sorted(by_gene):
        rows = sorted(by_gene[gene], key=lambda r: (int(r["plate"]), r["well_id"]))
        plates = sorted({int(r["plate"]) for r in rows})
        if len(plates) < 2:
            continue
        split = max(1, len(plates) // 2)
        g = next((r for r in rows if int(r["plate"]) in set(plates[:split])), None)
        q = next((r for r in rows if int(r["plate"]) in set(plates[split:])), None)
        if g and q:
            gallery.append(g); query.append(q)
        if len(gallery) >= genes_cap:
            break
    if len(gallery) != len(query) or not gallery:
        raise RuntimeError("RxRx3 selection failed")
    return gallery, query


def prepare_rxrx3(out: Path, genes_cap: int = 128) -> None:
    root = DATA / "RxRx3_core"
    index_path = Path("outputs/02_eval_runs/rxrx3_input_layer_screen/rxrx3_crispr_query_guide_index.jsonl")
    records = [json.loads(line) for line in index_path.open()]
    gallery, query = choose_rx_records(records, genes_cap)
    reader = RxReader(root)
    images, labels, rows = [], [], []
    for split, selected in (("gallery", gallery), ("query", query)):
        for record in selected:
            six = reader.read(record)
            norm = np.stack([normalize_channel(x) for x in six])
            rgb = np.stack(((norm[0] + norm[1]) / 2, (norm[2] + norm[3]) / 2, (norm[4] + norm[5]) / 2), axis=-1)
            images.append(rgb224((rgb * 255).astype(np.uint8)))
            labels.append(record["gene"])
            rows.append({"split": split, "gene": record["gene"], "plate": record["plate"], "well_id": record["well_id"]})
    write_cache(out / "rxrx3", images, np.asarray(labels), {
        "protocol_id": "rxrx3_plate_disjoint_128_v1", "rows": rows,
        "note": "CRISPR query-guide, plate-disjoint gallery/query, compact3 mapping",
    })


def crop_with_pad(image: Image.Image, box: tuple[int, int, int, int], size: int = 224) -> np.ndarray:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = size // 2
    canvas = Image.new("RGB", (size, size))
    src = (max(0, cx-half), max(0, cy-half), min(image.width, cx+half), min(image.height, cy+half))
    tile = image.crop(src).convert("RGB")
    canvas.paste(tile, (max(0, half-cx), max(0, half-cy)))
    return np.asarray(canvas)


def prepare_midog(out: Path, cap_per_split: int = 256) -> None:
    root = DATA / "MIDOGpp"
    database = json.loads((root / "repository/databases/MIDOG++.json").read_text())
    split_rows = list(csv.DictReader((root / "repository/datasets_xvalidation.csv").open(), delimiter=";"))
    split_by_slide = {int(r["Slide"]): r for r in split_rows}
    image_by_id = {int(x["id"]): x for x in database["images"]}
    candidates = defaultdict(list)
    for ann in database["annotations"]:
        image_info = image_by_id[int(ann["image_id"])]
        slide = int(Path(image_info["file_name"]).stem)
        split_info = split_by_slide.get(slide)
        image_path = root / "images" / image_info["file_name"]
        expected_rgb_bytes = int(image_info["width"]) * int(image_info["height"]) * 3
        # Six locally present TIFFs are visibly truncated (far smaller than a
        # single uncompressed RGB plane) and fail both PIL and tifffile reads.
        locally_complete = image_path.exists() and image_path.stat().st_size >= expected_rgb_bytes
        if split_info and locally_complete and split_info["Dataset"] in ("train", "test"):
            label = 1 if int(ann["category_id"]) == 1 else 0
            key = stable_key(f"{split_info['Dataset']}/{label}/{ann['id']}")
            candidates[(split_info["Dataset"], label)].append((key, ann, image_info, split_info))
    chosen = []
    each = cap_per_split // 2
    for split in ("train", "test"):
        for label in (0, 1):
            chosen.extend(sorted(candidates[(split, label)], key=lambda x: x[0])[:each])
    by_image = defaultdict(list)
    for item in chosen:
        by_image[item[2]["file_name"]].append(item)
    images, labels, rows = [], [], []
    for file_name, entries in sorted(by_image.items()):
        image_path = root / "images" / file_name
        try:
            with Image.open(image_path) as opened:
                opened.load()
                wsi = opened.convert("RGB")
        except (OSError, ValueError):
            array = tifffile.imread(image_path)
            if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                array = np.moveaxis(array, 0, -1)
            wsi = Image.fromarray(array).convert("RGB")
        try:
            for _, ann, info, split_info in entries:
                images.append(crop_with_pad(wsi, tuple(map(int, ann["bbox"]))))
                labels.append(1 if int(ann["category_id"]) == 1 else 0)
                rows.append({"split": split_info["Dataset"], "slide": int(Path(file_name).stem),
                             "tumor": split_info["Tumor"], "scanner": split_info["Scanner"], "annotation_id": ann["id"]})
        finally:
            wsi.close()
    write_cache(out / "midogpp", images, np.asarray(labels, np.int64), {
        "protocol_id": "midogpp_candidate_256_v1", "rows": rows,
        "note": "Balanced annotated-candidate proxy on official slide train/test split; not whole-slide detection",
    })


def validate(out: Path) -> dict:
    report = {}
    for name in ("ctc", "hest", "rxrx3", "midogpp"):
        folder = out / name
        images = np.load(folder / "images.npy", mmap_mode="r")
        labels = np.load(folder / "labels.npy", mmap_mode="r")
        meta = json.loads((folder / "metadata.json").read_text())
        assert len(images) == len(labels) == len(meta["rows"]), (name, images.shape, labels.shape, len(meta["rows"]))
        assert images.shape[1:] == (224, 224, 3) and images.dtype == np.uint8
        leakage = False
        if name == "hest":
            train = {(r["task"], r["sample_id"]) for r in meta["rows"] if r["split"] == "train"}
            test = {(r["task"], r["sample_id"]) for r in meta["rows"] if r["split"] == "test"}
            leakage = bool(train & test)
        elif name == "rxrx3":
            by_gene = defaultdict(lambda: {"gallery": set(), "query": set()})
            for row in meta["rows"]:
                by_gene[row["gene"]][row["split"]].add(row["plate"])
            leakage = any(v["gallery"] & v["query"] for v in by_gene.values())
        elif name == "midogpp":
            train = {r["slide"] for r in meta["rows"] if r["split"] == "train"}
            test = {r["slide"] for r in meta["rows"] if r["split"] == "test"}
            leakage = bool(train & test)
        elif name == "ctc":
            leakage = any((r["split"] == "train") != (r["sequence"] == "01") for r in meta["rows"])
        assert not leakage, f"{name}: split leakage detected"
        report[name] = {"n": len(images), "label_shape": list(labels.shape), "protocol_id": meta["protocol_id"], "leakage": leakage}
    (out / "cache_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", choices=("ctc", "hest", "rxrx3", "midogpp", "all", "validate"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    funcs = {"ctc": prepare_ctc, "hest": prepare_hest, "rxrx3": prepare_rxrx3, "midogpp": prepare_midog}
    selected = funcs if args.only == "all" else ({} if args.only == "validate" else {args.only: funcs[args.only]})
    for name, func in selected.items():
        print(f"[prepare] {name}", flush=True)
        func(args.output)
    if args.only in ("all", "validate"):
        print(json.dumps(validate(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
