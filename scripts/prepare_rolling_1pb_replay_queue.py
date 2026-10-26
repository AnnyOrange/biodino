#!/usr/bin/env python3
"""Stage the first five verified 1PB WDS batches for an isolated replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def prepare(source_queue: Path, replay_queue: Path) -> None:
    ready_root = replay_queue / "ready"
    ready_root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 6):
        name = f"batch-{number:06d}"
        source = source_queue / "completed" / name
        source_ready = source / "READY.json"
        raw_ready = source_ready.read_bytes()
        report = json.loads(raw_ready)
        shards = sorted((source / "shards").glob("*.tar"))
        if (report.get("status") != "PASS" or report.get("samples") != 499712 or
                len(shards) != report.get("shard_count") or len(shards) != 256):
            raise ValueError(f"Source batch is incomplete: {source}")
        if any((replay_queue / state / name).exists() for state in ("ready", "active", "completed")):
            print(f"EXISTS {name}", flush=True)
            continue
        staging = replay_queue / ".building" / name
        target_shards = staging / "shards"
        target_shards.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            target = target_shards / shard.name
            try:
                os.link(shard, target)
            except FileExistsError:
                if not shard.samefile(target):
                    raise RuntimeError(f"Replay shard differs from source: {target}")
        shutil.copyfile(source_ready, staging / "READY.json")
        (staging / "REPLAY_SOURCE.json").write_text(json.dumps({
            "source_batch": str(source),
            "ready_sha256": hashlib.sha256(raw_ready).hexdigest(),
            "source_shards_hardlinked": len(shards),
            "original_world_size": 8,
            "replay_world_size": 4,
            "exact_weight_recovery": False,
        }, indent=2) + "\n")
        os.replace(staging, ready_root / name)
        print(f"READY {name} shards={len(shards)} samples={report['samples']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--replay-queue", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_queue, args.replay_queue)


if __name__ == "__main__":
    main()
