#!/usr/bin/env python3
"""Select exactly 300k + 700k WDS samples, then materialize both components.

Reservoir sampling is uniform over image records in each complete source pool.
The manifest phase scans tar headers only and can resume from a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path


SOURCES = (
    ("old_1tb_300k", Path("/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle"), 300_000),
    ("new_4tb_700k", Path("/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle"), 700_000),
)
MEMBER_NAME = re.compile(r"^(.+)(\.ch[0-9]+\.tif|\.meta\.json)$")


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def select_component(name: str, source: Path, wanted: int, out: Path, seed: int, checkpoint_every: int) -> Path:
    manifest = out / f"{name}.selected_samples.jsonl"
    if manifest.exists():
        print(f"{name}: manifest already complete: {manifest}", flush=True)
        return manifest

    shards = sorted(source.glob("filtered_mixed_train*.tar"))
    if not shards:
        raise RuntimeError(f"No source tar files in {source}")
    inventory = hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in shards).encode()).hexdigest()
    checkpoint = out / f"{name}.reservoir.pkl"
    rng = random.Random(seed)
    reservoir: list[tuple[str, str, tuple[str, ...]]] = []
    seen = start = 0
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        if state["inventory"] != inventory or state["wanted"] != wanted or state["seed"] != seed:
            raise RuntimeError(f"Source inventory or selection parameters changed for {name}")
        start, seen, reservoir = state["next_shard"], state["seen"], state["reservoir"]
        rng.setstate(state["rng_state"])
        print(f"{name}: resuming at shard {start}/{len(shards)}, seen={seen}", flush=True)

    for index in range(start, len(shards)):
        shard = shards[index]
        current_key: str | None = None
        current_suffixes: list[str] = []
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                match = MEMBER_NAME.fullmatch(member.name)
                if match is None:
                    raise RuntimeError(f"Unknown member name in {shard}: {member.name}")
                key, suffix = match.groups()
                if current_key is not None and key != current_key:
                    raise RuntimeError(f"Incomplete or non-contiguous sample in {shard}: {current_key}")
                current_key = key
                current_suffixes.append(suffix)
                if suffix != ".meta.json":
                    continue
                if len(current_suffixes) < 2 or len(set(current_suffixes)) != len(current_suffixes):
                    raise RuntimeError(f"Incomplete or duplicate members in {shard}: {key}")
                seen += 1
                item = (shard.name, key, tuple(current_suffixes))
                if len(reservoir) < wanted:
                    reservoir.append(item)
                else:
                    slot = rng.randrange(seen)
                    if slot < wanted:
                        reservoir[slot] = item
                current_key = None
                current_suffixes = []
        if current_key is not None:
            raise RuntimeError(f"Unfinished final sample in {shard}: {current_key}")
        if (index + 1) % checkpoint_every == 0 or index + 1 == len(shards):
            save_checkpoint(checkpoint, {
                "inventory": inventory, "wanted": wanted, "seed": seed,
                "next_shard": index + 1, "seen": seen,
                "reservoir": reservoir, "rng_state": rng.getstate(),
            })
            print(f"{name}: indexed {index + 1}/{len(shards)} shards; pool={seen:,}; selected={len(reservoir):,}", flush=True)

    if seen < wanted or len(set(reservoir)) != wanted:
        raise RuntimeError(f"{name}: invalid selection: pool={seen}, unique selected={len(set(reservoir))}")
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w") as handle:
        for shard_name, key, suffixes in sorted(reservoir):
            handle.write(json.dumps({"source_shard": shard_name, "key": key, "members": suffixes}, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, manifest)
    summary = {
        "component": name, "source_root": str(source), "pool_samples": seen,
        "selected_samples": wanted, "seed": seed, "source_shards": len(shards),
        "inventory_sha256": inventory, "selection": "uniform reservoir without replacement over meta.json image records",
        "manifest": str(manifest),
    }
    (out / f"{name}.selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifests = []
    for component_index, (name, source, wanted) in enumerate(SOURCES):
        manifest = select_component(name, source, wanted, args.output_root, args.seed + component_index, args.checkpoint_every)
        manifests.append((name, source, wanted, manifest))
    if args.select_only:
        return

    rebuilder = Path(__file__).with_name("rebuild_random_wds_subset_from_manifest.py")
    for name, source, wanted, manifest in manifests:
        target = args.output_root / name
        command = [
            sys.executable, "-u", str(rebuilder), "--source-root", str(source),
            "--manifest", str(manifest), "--output-dir", str(target),
            "--expected-samples", str(wanted), "--expected-shards", str(wanted // 2000),
        ]
        print("materializing", name, "into", target, flush=True)
        subprocess.run(command, check=True)

    summary = {
        "recipe": {"old_1tb": 0.3, "new_4tb": 0.7},
        "total_samples": 1_000_000,
        "component_outputs": {name: str(args.output_root / name) for name, _, _, _ in manifests},
        "elapsed_seconds": time.time() - started,
    }
    (args.output_root / "extraction_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
