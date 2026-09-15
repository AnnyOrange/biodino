#!/usr/bin/env python3
"""Build the fixed all-eligible-gene RxRx3-core formal retrieval cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901/RxRx3_core")
INDEX = ROOT / "outputs/02_eval_runs/rxrx3_input_layer_screen/rxrx3_crispr_query_guide_index.jsonl"
OUTPUT = ROOT / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
PROTOCOL = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"


def rgb224(array: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(array).resize((224, 224), Image.Resampling.BILINEAR).convert("RGB"))


def normalize_channel(array: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(array, (1, 99))
    return np.clip((array.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1)


def write_cache(folder: Path, images: list[np.ndarray], labels: np.ndarray, metadata: dict) -> None:
    np.save(folder / "images.npy", np.stack(images).astype(np.uint8))
    np.save(folder / "labels.npy", np.asarray(labels))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


class RxReader:
    def __init__(self, root: Path):
        import pyarrow.parquet as pq
        self.pq = pq
        self.shards = sorted((root / "data").glob("train-*.parquet"))
        self.files: dict[int, object] = {}
        self.groups: OrderedDict[tuple[int, int], dict] = OrderedDict()

    def file(self, shard: int):
        if shard not in self.files:
            self.files[shard] = self.pq.ParquetFile(self.shards[shard])
        return self.files[shard]

    def group(self, shard: int, group: int) -> dict:
        key = (shard, group)
        if key not in self.groups:
            self.groups[key] = self.file(shard).read_row_group(group, columns=["__key__", "jp2"]).to_pydict()
            while len(self.groups) > 12:
                self.groups.popitem(last=False)
        self.groups.move_to_end(key)
        return self.groups[key]

    def read(self, record: dict) -> np.ndarray:
        shard = int(record["shard"])
        parquet = self.file(shard)
        offset, group = int(record["row"]), 0
        while offset >= parquet.metadata.row_group(group).num_rows:
            offset -= parquet.metadata.row_group(group).num_rows
            group += 1
        arrays, local = [], offset
        for channel in range(1, 7):
            table = self.group(shard, group)
            expected = f"{record['key']}_{channel}"
            if table["__key__"][local] != expected:
                raise RuntimeError(f"RxRx3 channel mismatch: {table['__key__'][local]} != {expected}")
            with Image.open(io.BytesIO(table["jp2"][local]["bytes"])) as image:
                arrays.append(np.asarray(image).astype(np.float32, copy=False))
            local += 1
            if local == len(table["__key__"]) and channel < 6:
                group += 1
                local = 0
        return np.stack(arrays)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_gene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_gene[str(record["gene"])].append(record)
    gallery, query = [], []
    for gene in sorted(by_gene):
        rows = sorted(by_gene[gene], key=lambda r: (int(r["plate"]), str(r["well_id"]), str(r["key"])))
        plates = sorted({int(r["plate"]) for r in rows})
        if len(plates) < 2:
            continue
        split = max(1, len(plates) // 2)
        gallery_plates, query_plates = set(plates[:split]), set(plates[split:])
        gallery.append(next(row for row in rows if int(row["plate"]) in gallery_plates))
        query.append(next(row for row in rows if int(row["plate"]) in query_plates))
    if len(gallery) != len(query) or not gallery:
        raise RuntimeError("eligible-gene selection failed")
    return gallery, query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.index.open()]
    gallery, query = select(records)

    manifest_path = args.output / "split_manifest.jsonl"
    manifest_rows = []
    for split_name, selected in (("gallery", gallery), ("query", query)):
        for record in selected:
            manifest_rows.append({
                "split": split_name, "gene": record["gene"], "well_id": record["well_id"],
                "plate": int(record["plate"]), "key": record["key"],
                "shard": int(record["shard"]), "row": int(record["row"]),
            })
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))
    manifest_sha = sha256(manifest_path)

    reader = RxReader(args.data_root)
    images, labels = [], []
    for index, row in enumerate(manifest_rows, 1):
        six = reader.read(row)
        norm = np.stack([normalize_channel(channel) for channel in six])
        rgb = np.stack(
            ((norm[0] + norm[1]) / 2, (norm[2] + norm[3]) / 2, (norm[4] + norm[5]) / 2), axis=-1
        )
        images.append(rgb224((rgb * 255).astype(np.uint8)))
        labels.append(row["gene"])
        if index == 1 or index % 100 == 0:
            print(f"[rxrx3 formal cache] {index}/{len(manifest_rows)}", flush=True)

    write_cache(args.output, images, np.asarray(labels), {
        "protocol_id": PROTOCOL, "rows": manifest_rows,
        "manifest_sha256": manifest_sha,
        "source_index_sha256": sha256(args.index),
        "metadata_sha256": sha256(args.data_root / "metadata_rxrx3_core.csv"),
        "input_mapping": "compact3_pairmean_after_per_channel_p01_p99",
        "selection": "all genes with >=2 plates; one deterministic gallery and query well per gene",
    })
    report = {
        "status": "VALID_COMPLETE", "protocol_id": PROTOCOL,
        "eligible_genes": len(gallery), "gallery_wells": len(gallery), "query_wells": len(query),
        "manifest_sha256": manifest_sha, "manifest": str(manifest_path),
        "same_well_overlap": bool({r["well_id"] for r in manifest_rows if r["split"] == "gallery"} &
                                  {r["well_id"] for r in manifest_rows if r["split"] == "query"}),
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
