#!/usr/bin/env python3
"""Rebuild an exact random WebDataset subset from its selection manifest."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def load_manifest(path: Path, expected_samples: int):
    by_source: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row["key"]
            source = row["source_shard"]
            if key in by_source[source]:
                raise ValueError(f"duplicate sample key in {source}: {key}")
            by_source[source][key] = tuple(key + suffix for suffix in row["members"])
    count = sum(len(rows) for rows in by_source.values())
    if count != expected_samples:
        raise ValueError(f"manifest has {count} samples, expected {expected_samples}")
    return by_source


class ShardWriter:
    def __init__(self, output_dir: Path, expected_samples: int, expected_shards: int):
        self.output_dir = output_dir
        self.expected_samples = expected_samples
        self.expected_shards = expected_shards
        self.samples_per_shard = math.ceil(expected_samples / expected_shards)
        self.shard_index = -1
        self.samples_in_shard = 0
        self.total_samples = 0
        self.total_members = 0
        self.current_sample = None
        self.tar: tarfile.TarFile | None = None

    def _open_next(self) -> None:
        if self.tar is not None:
            self.tar.close()
        self.shard_index += 1
        path = self.output_dir / f"filtered_mixed_train_w00-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(path, "w", format=tarfile.PAX_FORMAT)
        self.samples_in_shard = 0

    def add(self, sample_key: str, member: tarfile.TarInfo, source: tarfile.TarFile) -> None:
        if sample_key != self.current_sample:
            if self.tar is None or self.samples_in_shard >= self.samples_per_shard:
                self._open_next()
            self.current_sample = sample_key
            self.samples_in_shard += 1
            self.total_samples += 1
        assert self.tar is not None
        fileobj = source.extractfile(member) if member.isfile() else None
        self.tar.addfile(member, fileobj)
        self.total_members += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def main() -> None:
    args = parse_args()
    marker = args.output_dir / ".transfer_complete"
    if marker.exists():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing:
        raise RuntimeError(f"refusing non-empty incomplete output directory: {args.output_dir}")

    selected = load_manifest(args.manifest, args.expected_samples)
    expected_members = sum(len(names) for rows in selected.values() for names in rows.values())
    writer = ShardWriter(args.output_dir, args.expected_samples, args.expected_shards)
    try:
        for index, source_name in enumerate(sorted(selected), start=1):
            source_path = args.source_root / source_name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            sample_members = selected[source_name]
            member_to_key = {
                member_name: key
                for key, member_names in sample_members.items()
                for member_name in member_names
            }
            found = set()
            with tarfile.open(source_path, "r:") as source:
                for member in source:
                    key = member_to_key.get(member.name)
                    if key is None:
                        continue
                    writer.add(key, member, source)
                    found.add(member.name)
            missing = set(member_to_key) - found
            if missing:
                preview = sorted(missing)[:3]
                raise RuntimeError(f"{source_name} is missing {len(missing)} selected members: {preview}")
            if index % 20 == 0 or index == len(selected):
                print(
                    f"sources={index}/{len(selected)} samples={writer.total_samples}/{args.expected_samples}",
                    flush=True,
                )
    finally:
        writer.close()

    shard_count = writer.shard_index + 1
    if writer.total_samples != args.expected_samples:
        raise RuntimeError(f"wrote {writer.total_samples} samples, expected {args.expected_samples}")
    if writer.total_members != expected_members:
        raise RuntimeError(f"wrote {writer.total_members} members, expected {expected_members}")
    if shard_count != args.expected_shards:
        raise RuntimeError(f"wrote {shard_count} shards, expected {args.expected_shards}")

    shutil.copy2(args.manifest, args.output_dir / "selected_samples.jsonl")
    summary = {
        "source_root": str(args.source_root),
        "samples": writer.total_samples,
        "members": writer.total_members,
        "shards": shard_count,
        "samples_per_shard_limit": writer.samples_per_shard,
    }
    (args.output_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    marker.touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
