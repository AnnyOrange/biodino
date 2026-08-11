#!/usr/bin/env python3
"""Rebuild nested random WebDataset subsets in one source-shard scan."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubsetSpec:
    label: str
    manifest: Path
    output_dir: Path
    expected_samples: int
    expected_shards: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--subset",
        action="append",
        nargs=5,
        metavar=("LABEL", "MANIFEST", "OUTPUT_DIR", "EXPECTED_SAMPLES", "EXPECTED_SHARDS"),
        required=True,
        help="Repeat for each nested subset to build.",
    )
    return parser.parse_args()


def parse_specs(rows: list[list[str]]) -> list[SubsetSpec]:
    specs = [
        SubsetSpec(label, Path(manifest), Path(output), int(samples), int(shards))
        for label, manifest, output, samples, shards in rows
    ]
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate subset labels: {labels}")
    return specs


def load_manifest(spec: SubsetSpec) -> dict[str, dict[str, tuple[str, ...]]]:
    by_source: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    with spec.manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row["key"]
            source = row["source_shard"]
            if key in by_source[source]:
                raise ValueError(f"{spec.label}: duplicate sample key in {source}: {key}")
            by_source[source][key] = tuple(key + suffix for suffix in row["members"])
    count = sum(len(rows) for rows in by_source.values())
    if count != spec.expected_samples:
        raise ValueError(f"{spec.label}: manifest has {count} samples, expected {spec.expected_samples}")
    return by_source


class ShardWriter:
    def __init__(self, spec: SubsetSpec):
        self.spec = spec
        self.samples_per_shard = math.ceil(spec.expected_samples / spec.expected_shards)
        self.shard_index = -1
        self.samples_in_shard = 0
        self.total_samples = 0
        self.total_members = 0
        self.current_sample: str | None = None
        self.tar: tarfile.TarFile | None = None

    def _open_next(self) -> None:
        if self.tar is not None:
            self.tar.close()
        self.shard_index += 1
        path = self.spec.output_dir / f"filtered_mixed_train_w00-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(path, "w", format=tarfile.PAX_FORMAT)
        self.samples_in_shard = 0

    def add(self, sample_key: str, member: tarfile.TarInfo, payload: bytes | None) -> None:
        if sample_key != self.current_sample:
            if self.tar is None or self.samples_in_shard >= self.samples_per_shard:
                self._open_next()
            self.current_sample = sample_key
            self.samples_in_shard += 1
            self.total_samples += 1
        assert self.tar is not None
        fileobj = io.BytesIO(payload) if payload is not None else None
        self.tar.addfile(copy.copy(member), fileobj)
        self.total_members += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def finish(self, expected_members: int, source_root: Path) -> dict[str, object]:
        shard_count = self.shard_index + 1
        if self.total_samples != self.spec.expected_samples:
            raise RuntimeError(
                f"{self.spec.label}: wrote {self.total_samples} samples, expected {self.spec.expected_samples}"
            )
        if self.total_members != expected_members:
            raise RuntimeError(
                f"{self.spec.label}: wrote {self.total_members} members, expected {expected_members}"
            )
        if shard_count != self.spec.expected_shards:
            raise RuntimeError(
                f"{self.spec.label}: wrote {shard_count} shards, expected {self.spec.expected_shards}"
            )
        shutil.copy2(self.spec.manifest, self.spec.output_dir / "selected_samples.jsonl")
        summary = {
            "label": self.spec.label,
            "source_root": str(source_root),
            "samples": self.total_samples,
            "members": self.total_members,
            "shards": shard_count,
            "samples_per_shard_limit": self.samples_per_shard,
        }
        (self.spec.output_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (self.spec.output_dir / ".transfer_complete").touch()
        return summary


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.subset)
    selections = {spec.label: load_manifest(spec) for spec in specs}
    expected_members = {
        spec.label: sum(len(names) for rows in selections[spec.label].values() for names in rows.values())
        for spec in specs
    }

    for spec in specs:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        if any(spec.output_dir.iterdir()):
            raise RuntimeError(f"refusing non-empty output directory: {spec.output_dir}")

    writers = {spec.label: ShardWriter(spec) for spec in specs}
    source_names = sorted({source for selection in selections.values() for source in selection})
    try:
        for source_index, source_name in enumerate(source_names, start=1):
            source_path = args.source_root / source_name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            targets_by_member: dict[str, list[tuple[str, str]]] = defaultdict(list)
            expected_by_label: dict[str, set[str]] = {}
            for spec in specs:
                rows = selections[spec.label].get(source_name, {})
                expected = set()
                for key, member_names in rows.items():
                    for member_name in member_names:
                        targets_by_member[member_name].append((spec.label, key))
                        expected.add(member_name)
                expected_by_label[spec.label] = expected

            found_by_label = {spec.label: set() for spec in specs}
            with tarfile.open(source_path, "r:") as source:
                for member in source:
                    targets = targets_by_member.get(member.name)
                    if not targets:
                        continue
                    payload = source.extractfile(member).read() if member.isfile() else None
                    for label, key in targets:
                        writers[label].add(key, member, payload)
                        found_by_label[label].add(member.name)

            for spec in specs:
                missing = expected_by_label[spec.label] - found_by_label[spec.label]
                if missing:
                    raise RuntimeError(
                        f"{spec.label}: {source_name} is missing {len(missing)} members: {sorted(missing)[:3]}"
                    )
            if source_index % 10 == 0 or source_index == len(source_names):
                progress = " ".join(
                    f"{label}={writer.total_samples}/{writer.spec.expected_samples}"
                    for label, writer in writers.items()
                )
                print(f"sources={source_index}/{len(source_names)} {progress}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    summaries = [
        writers[spec.label].finish(expected_members[spec.label], args.source_root) for spec in specs
    ]
    print(json.dumps(summaries, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
