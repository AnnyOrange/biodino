#!/usr/bin/env python3
"""Resume new-FM detection seeds 3-4 after the BBBC038 seed-3 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("distillpath", "morphem", "genbio_pathfm", "openpath")
DATASETS = ("bbbc038", "conic", "livecell")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    output = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / args.model
    failures = []
    for dataset in DATASETS:
        for seed in (3, 4):
            if dataset == "bbbc038" and seed == 3:
                continue
            result = output / "detection" / dataset / f"seed{seed}" / "component_result.json"
            if result.is_file():
                print(f"[skip] {args.model}/{dataset}/seed{seed}", flush=True)
                continue
            command = [sys.executable, str(ROOT / "scripts/run_new_fm_v4_detection_20261016.py"),
                       "--model", args.model, "--dataset", dataset, "--seed", str(seed)]
            print(f"[run] {args.model}/{dataset}/seed{seed}", flush=True)
            code = subprocess.run(command, cwd=ROOT, check=False).returncode
            if code:
                failures.append({"dataset": dataset, "seed": seed, "exit_code": code})
                print(f"[failed] {args.model}/{dataset}/seed{seed}: {code}", flush=True)
    status = output / "detection_five_seed_queue.json"
    status.write_text(json.dumps({"model": args.model, "ended": datetime.now(timezone.utc).isoformat(),
                                  "failures": failures}, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
