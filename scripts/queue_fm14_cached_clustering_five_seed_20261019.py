#!/usr/bin/env python3
"""Run disjoint FM14 cached-clustering model shards on CPU workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_fm14_cached_clustering_five_seed_20261019 import MODELS, ROOT, RUN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard index out of range")
    failures = []
    for index, model in enumerate(MODELS):
        if index % args.shards != args.shard:
            continue
        print(f"[run] {model}", flush=True)
        code = subprocess.run([sys.executable,
                               str(ROOT / "scripts/run_fm14_cached_clustering_five_seed_20261019.py"),
                               "--model", model], cwd=ROOT, check=False).returncode
        if code:
            failures.append({"model": model, "exit_code": code})
    status = RUN / f"cached_five_seed_shard{args.shard}.json"
    status.write_text(json.dumps({"shard": args.shard, "shards": args.shards,
                                  "ended": datetime.now(timezone.utc).isoformat(),
                                  "failures": failures}, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
