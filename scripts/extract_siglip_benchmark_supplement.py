#!/usr/bin/env python3
"""Resumable, multi-GPU SigLIP feature extraction for benchmark supplements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def key_for(records: list[dict]) -> str:
    return hashlib.sha1("\n".join(json.dumps(r, sort_keys=True) for r in records).encode()).hexdigest()[:16]


def load_records(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    excluded = set(args.exclude_dataset)
    for root in args.queue_root:
        for manifest in sorted(Path(root).rglob("*.jsonl")):
            for line in manifest.open():
                row = json.loads(line)
                if row.get("dataset") not in excluded:
                    rows.append(row)
    for root_text in args.scan_root:
        root = Path(root_text)
        dataset = root.name
        if dataset in excluded:
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(p), "dataset": dataset})
    for spec in args.array:
        path_text, array_key = spec.rsplit(":", 1)
        path = Path(path_text)
        data = np.load(path, mmap_mode="r")
        arr = data[array_key] if hasattr(data, "files") else data
        dataset = path.parent.name
        for i in range(len(arr)):
            rows.append({"path": str(path), "array_key": array_key, "array_sample_index": i, "dataset": dataset})
    # A queue can contain repeated paths through different benchmark aliases.
    unique = {json.dumps(r, sort_keys=True): r for r in rows}
    return [unique[k] for k in sorted(unique)]


def as_rgb(row: dict) -> Image.Image:
    path = Path(row["path"])
    if "array_sample_index" in row:
        data = np.load(path, mmap_mode="r")
        arr = data[row.get("array_key")] if hasattr(data, "files") else data
        image = np.asarray(arr[int(row["array_sample_index"])])
        while image.ndim > 3:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.dtype != np.uint8:
            lo, hi = np.percentile(image, (1, 99))
            image = (np.clip((image.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    return Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--queue-root", action="append", default=[])
    ap.add_argument("--scan-root", action="append", default=[])
    ap.add_argument("--array", action="append", default=[])
    ap.add_argument("--exclude-dataset", action="append", default=[])
    ap.add_argument("--worker-index", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=2048)
    args = ap.parse_args()

    rows = load_records(args)[args.worker_index::args.workers]
    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(args.model, local_files_only=True).eval().to(device)
    out = Path(args.out) / f"worker_{args.worker_index:02d}"
    for start in range(0, len(rows), args.shard_size):
        chunk = rows[start:start + args.shard_size]
        shard = out / f"{start:09d}_{key_for(chunk)}"
        done = shard / "done.json"
        if done.exists():
            continue
        vectors, kept = [], []
        for b in range(0, len(chunk), args.batch_size):
            valid, images = [], []
            for row in chunk[b:b + args.batch_size]:
                try:
                    images.append(as_rgb(row)); valid.append(row)
                except Exception as exc:
                    print(f"skip {row['path']}: {exc}", flush=True)
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                vectors.append(model(pixel_values=inputs).pooler_output.float().cpu().numpy())
            kept.extend(valid)
        if not vectors:
            continue
        tmp = shard.with_name(shard.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "features.npy", np.concatenate(vectors).astype(np.float16))
        with (tmp / "records.jsonl").open("w") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")
        (tmp / "done.json").write_text(json.dumps({"features": len(kept), "dim": 1152, "dtype": "float16"}))
        shard.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(shard)
        print(json.dumps({"worker": args.worker_index, "shard": str(shard), "features": len(kept)}), flush=True)


if __name__ == "__main__":
    main()
