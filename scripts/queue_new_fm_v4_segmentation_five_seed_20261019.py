#!/usr/bin/env python3
"""Resume five-seed dense probes for one new FM across the v4 datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dinov3.eval.bio_frozen_eval.external_fm_protocol import PANNUKE_PROTOCOLS, SEGMENTATION


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("distillpath", "morphem", "genbio_pathfm", "openpath")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    args = parser.parse_args()
    cells = [(dataset, protocol) for dataset in SEGMENTATION
             for protocol in (PANNUKE_PROTOCOLS if dataset == "pannuke" else (None,))]
    failures = []
    for dataset, protocol in cells:
        command = [sys.executable, str(ROOT / "scripts/run_new_fm_v4_segmentation_20261016.py"),
                   "--model", args.model, "--dataset", dataset, "--all-probes",
                   "--extract-batch-size", str(args.extract_batch_size)]
        if protocol:
            command.extend(("--split-protocol", protocol))
        print(f"[run] {args.model}/{dataset}/{protocol or 'default'}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            failures.append({"dataset": dataset, "split_protocol": protocol,
                             "exit_code": completed.returncode})
            print(f"[failed] {dataset}/{protocol}: {completed.returncode}", flush=True)
    status = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / args.model / "segmentation_five_seed_queue.json"
    status.write_text(json.dumps({"model": args.model, "ended": datetime.now(timezone.utc).isoformat(),
                                  "failures": failures}, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
