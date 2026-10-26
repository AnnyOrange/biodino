#!/usr/bin/env python3
"""Resume FM14 detection seeds 3-4 for an explicit model slice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from queue_fm14_detection_multiseed_20261016 import DATASETS, MODELS, ROOT


def free_memory_mib() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(visible) != 1 or not visible[0].isdigit():
        raise ValueError("Exactly one physical GPU index must be visible")
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={visible[0]}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return int(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(3, 4), default=(3, 4))
    parser.add_argument("--min-free-mib", type=int, default=8000)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    output = ROOT / "outputs/02_eval_runs/fm14_detection_multiseed_20261016"
    failures = []
    for model in args.models:
        for dataset in DATASETS:
            for seed in args.seeds:
                result = output / model / dataset / f"seed{seed}" / "component_result.json"
                if result.is_file():
                    print(f"[skip] {model}/{dataset}/seed{seed}", flush=True)
                    continue
                while free_memory_mib() < args.min_free_mib:
                    print(f"[wait] {model}/{dataset}/seed{seed}: low free GPU memory", flush=True)
                    time.sleep(30)
                print(f"[run] {model}/{dataset}/seed{seed}", flush=True)
                code = subprocess.run([
                    sys.executable, str(ROOT / "scripts/run_fm14_detection_multiseed_20261016.py"),
                    "--model", model, "--dataset", dataset, "--seed", str(seed),
                    "--output-root", str(output),
                ], cwd=ROOT, check=False).returncode
                if code:
                    failures.append({"model": model, "dataset": dataset,
                                     "seed": seed, "exit_code": code})
                    print(f"[failed] {model}/{dataset}/seed{seed}: {code}", flush=True)
    status = output / f"five_seed_{args.tag}_status.json"
    status.write_text(json.dumps({"models": args.models, "ended": datetime.now(timezone.utc).isoformat(),
                                  "failures": failures}, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
