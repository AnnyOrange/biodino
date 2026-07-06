#!/usr/bin/env python3
"""Pack the ID weak train-only collection into packed WebDataset tar shards.

The output format matches the packed DINOv3 loader contract: each sample has
``<key>.meta.json`` plus ``<key>.ch<N>.tif`` channel members.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
import time
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageSequence


def tar_entry_bytes(payload_size: int) -> int:
    return 512 + int(math.ceil(payload_size / 512.0) * 512)


def safe_key_part(text: str, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def make_key(index: int, row: dict[str, str]) -> str:
    digest = hashlib.sha1(row["rel_path"].encode("utf-8")).hexdigest()[:10]
    dataset = safe_key_part(row.get("dataset", "dataset"), 48)
    sample = safe_key_part(row.get("sample_id", "sample"), 48)
    return f"idweak_{index:09d}_{dataset}_{sample}_{digest}"


def encode_tiff_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    try:
        image.save(out, format="TIFF", compression="tiff_deflate")
    except Exception:
        out = io.BytesIO()
        image.save(out, format="TIFF")
    return out.getvalue()


def image_to_channel_tiffs(path: Path) -> tuple[list[bytes], list[int]]:
    with Image.open(path) as img:
        # Use the first frame/page only; all collection entries are image samples.
        try:
            img = next(ImageSequence.Iterator(img)).copy()
        except Exception:
            img = img.copy()

    mode = img.mode
    if mode in {"RGB", "RGBA"}:
        rgb = img.convert("RGB")
        bands = list(rgb.split())
    elif mode in {"L", "I", "I;16", "I;16B", "I;16L", "F"}:
        bands = [img]
    elif mode == "LA":
        bands = [img.getchannel(0)]
    else:
        # Palette, CMYK, YCbCr, and uncommon modes are normalized to RGB.
        bands = list(img.convert("RGB").split())

    channel_bytes = [encode_tiff_bytes(band) for band in bands]
    width, height = img.size
    return channel_bytes, [len(channel_bytes), height, width]


class TarShardWriter:
    def __init__(self, output_dir: Path, prefix: str, max_count: int, max_size: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_count = max_count
        self.max_size = max_size
        self.shard_index = -1
        self.tar: tarfile.TarFile | None = None
        self.path: Path | None = None
        self.current_count = 0
        self.current_size = 0
        self.total_count = 0
        self.shards: list[dict[str, int | str]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self) -> None:
        self.close_current()
        self.shard_index += 1
        self.path = self.output_dir / f"{self.prefix}-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(self.path, "w")
        self.current_count = 0
        self.current_size = 0
        print(f"[pack-idweak] open {self.path}", flush=True)

    def close_current(self) -> None:
        if self.tar is None or self.path is None:
            return
        self.tar.close()
        size = self.path.stat().st_size
        self.shards.append({"path": str(self.path), "samples": self.current_count, "bytes": size})
        print(
            f"[pack-idweak] close {self.path.name} samples={self.current_count} size={size}",
            flush=True,
        )
        self.tar = None
        self.path = None

    def write_sample(self, key: str, members: list[tuple[str, bytes]]) -> None:
        sample_bytes = sum(tar_entry_bytes(len(data)) for _, data in members)
        if (
            self.tar is None
            or self.current_count >= self.max_count
            or (self.current_count > 0 and self.current_size + sample_bytes > self.max_size)
        ):
            self._open_next()
        assert self.tar is not None
        for suffix, data in members:
            name = f"{key}.{suffix}"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            self.tar.addfile(info, io.BytesIO(data))
        self.current_count += 1
        self.total_count += 1
        self.current_size += sample_bytes

    def close(self) -> None:
        self.close_current()


def iter_manifest(manifest: Path) -> Iterable[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=Path("/mnt/huawei_deepcad/tasks/id_weak_train_collection"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="id_weak_train")
    parser.add_argument("--max-count", type=int, default=5000)
    parser.add_argument("--max-size", type=int, default=1024**3)
    parser.add_argument("--limit", type=int, default=0, help="debug only: stop after N samples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.collection_root.resolve()
    manifest = args.manifest or root / "metadata" / "manifest.csv"
    output_dir = args.output_dir or root / "wds"

    existing = sorted(output_dir.glob(f"{args.prefix}-*.tar")) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already has {len(existing)} {args.prefix}-*.tar shards; use --overwrite"
        )
    if existing and args.overwrite:
        for path in existing:
            path.unlink()

    started = time.time()
    failures: list[dict[str, str]] = []
    by_dataset: dict[str, int] = {}
    by_channels: dict[str, int] = {}
    total_payload = 0

    writer = TarShardWriter(output_dir, args.prefix, args.max_count, args.max_size)
    try:
        for idx, row in enumerate(iter_manifest(manifest)):
            if args.limit and idx >= args.limit:
                break
            image_path = root / row["rel_path"]
            key = make_key(idx, row)
            try:
                channel_tiffs, chw = image_to_channel_tiffs(image_path)
            except Exception as exc:
                failures.append({"rel_path": row.get("rel_path", ""), "error": repr(exc)})
                continue

            meta = {
                "id": key,
                "dataset_name": row.get("dataset", ""),
                "task": row.get("task", ""),
                "split": row.get("split", ""),
                "label": row.get("label", ""),
                "collection_sample_id": row.get("sample_id", ""),
                "available_channels": list(range(1, len(channel_tiffs) + 1)),
                "source_channel_count": len(channel_tiffs),
                "source_image_shape": chw,
                "patch_shape": [chw[1], chw[2]],
                "original_shape": None,
                "original_path": row.get("source", ""),
                "source_sample_id": key,
                "collection_rel_path": row.get("rel_path", ""),
                "collection_storage": row.get("storage", ""),
                "kept_as_full_image": True,
                "leakage_policy": "train-only ID weak collection; OOD/val/test excluded upstream",
            }
            members = [(f"ch{i}.tif", data) for i, data in enumerate(channel_tiffs, 1)]
            meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            members.append(("meta.json", meta_bytes))
            writer.write_sample(key, members)

            dataset = row.get("dataset", "")
            by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
            by_channels[str(len(channel_tiffs))] = by_channels.get(str(len(channel_tiffs)), 0) + 1
            total_payload += sum(len(data) for _, data in members)
            if writer.total_count == 1 or writer.total_count % 10000 == 0:
                print(
                    f"[pack-idweak] written={writer.total_count} failures={len(failures)} last={row.get('rel_path','')}",
                    flush=True,
                )
    finally:
        writer.close()

    summary = {
        "collection_root": str(root),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "prefix": args.prefix,
        "samples_written": writer.total_count,
        "failures": len(failures),
        "failure_examples": failures[:20],
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_source_channels": dict(sorted(by_channels.items(), key=lambda kv: int(kv[0]))),
        "total_payload_bytes": total_payload,
        "tar_shards": writer.shards,
        "tar_shard_count": len(writer.shards),
        "elapsed_seconds": time.time() - started,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pack-idweak] summary {summary_path}", flush=True)
    if failures:
        fail_path = output_dir / f"{args.prefix}_failures.jsonl"
        with fail_path.open("w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[pack-idweak] failures {fail_path}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
