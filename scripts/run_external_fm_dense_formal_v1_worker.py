#!/usr/bin/env python3
"""Sequential, restartable worker for formal-v1 external-FM CoNIC/PanNuke jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONIC_PROTOCOL = "official-baseline-fold0-nested-v1"
PANNUKE_PROTOCOLS = (
    "pannuke-fold1-train-fold2-val-fold3-test",
    "pannuke-fold2-train-fold1-val-fold3-test",
    "pannuke-fold3-train-fold2-val-fold1-test",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--metric-python", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = [("conic", CONIC_PROTOCOL)] + [("pannuke", p) for p in PANNUKE_PROTOCOLS]
    status_rows = []
    environment = os.environ.copy()
    vendor_paths = [
        "/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311",
        "/mnt/huawei_deepcad/benchmark_model/_vendor",
        "/mnt/huawei_deepcad/benchmark_model",
        str(REPO),
    ]
    environment["PYTHONPATH"] = os.pathsep.join(vendor_paths + [environment.get("PYTHONPATH", "")])
    environment["KERAS_BACKEND"] = "torch"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")

    for model in args.models:
        for dataset, split_protocol in jobs:
            manifest = output_root / dataset / split_protocol / model / "formal_v1_manifest.json"
            if manifest.is_file():
                try:
                    payload = json.loads(manifest.read_text())
                except Exception:
                    payload = {}
                if payload.get("status") == "VALIDATION_PENDING" and Path(payload.get("result_path", "")).is_file():
                    status_rows.append({"model": model, "dataset": dataset, "split_protocol": split_protocol, "status": "cached"})
                    continue
            command = [
                sys.executable,
                str(REPO / "scripts" / "run_external_fm_dense_formal_v1.py"),
                "--model", model,
                "--dataset", dataset,
                "--split-protocol", split_protocol,
                "--output-root", str(output_root),
                "--metric-python", args.metric_python,
            ]
            log_path = log_root / f"{model}__{dataset}__{split_protocol}.log"
            with log_path.open("w") as log:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                process = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status = "ok" if process.returncode == 0 else f"failed({process.returncode})"
            status_rows.append({
                "model": model,
                "dataset": dataset,
                "split_protocol": split_protocol,
                "status": status,
                "log": str(log_path),
            })
            (log_root / f"worker_{os.getpid()}_status.json").write_text(json.dumps(status_rows, indent=2) + "\n")
    failed = [row for row in status_rows if str(row["status"]).startswith("failed")]
    print(json.dumps({"jobs": len(status_rows), "failed": len(failed), "rows": status_rows}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
