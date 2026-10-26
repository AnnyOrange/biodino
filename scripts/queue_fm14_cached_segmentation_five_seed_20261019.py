#!/usr/bin/env python3
"""Run disjoint FM14 cached dense-probe shards on one physical GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from run_fm14_cached_segmentation_five_seed_20261019 import OUT, ROOT


def free_memory_mib(gpu_index: int) -> int:
    text = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return int(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=7000)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard index out of range")
    cells = json.loads((OUT / "audit.json").read_text())["cells"]
    failures = []
    for index, cell in enumerate(cells):
        if index % args.shards != args.shard:
            continue
        directory = OUT / "results" / cell["model"] / cell["dataset"] / cell["split_protocol"]
        if all((directory / f"E{budget}" / f"seed{seed}" / "results.json").is_file()
               for budget in (20, 50) for seed in (3, 4)):
            print(f"[skip] cell{index}", flush=True)
            continue
        while free_memory_mib(args.gpu_index) < args.min_free_mib:
            print(f"[wait] cell{index}: insufficient free GPU memory", flush=True)
            time.sleep(30)
        print(f"[run] cell{index} {cell['model']}/{cell['dataset']}/{cell['split_protocol']}", flush=True)
        code = subprocess.run([sys.executable,
                               str(ROOT / "scripts/run_fm14_cached_segmentation_five_seed_20261019.py"),
                               "--cell-index", str(index)], cwd=ROOT, check=False).returncode
        if code:
            failures.append({"cell_index": index, "exit_code": code})
            print(f"[failed] cell{index}: {code}", flush=True)
    status = OUT / f"shard{args.shard}_status.json"
    status.write_text(json.dumps({"shard": args.shard, "shards": args.shards,
                                  "ended": datetime.now(timezone.utc).isoformat(),
                                  "failures": failures}, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
