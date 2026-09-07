#!/usr/bin/env python3
"""Snapshot complete ID scalar metrics for the two DINOv3 H+/16 checkpoints.

The alpha sweep already evaluated checkpoint 0 (official DINOv3) and checkpoint
100 (H+ BioDINO) with the same frozen scalar protocol.  The earlier Fig. 3
source retained only one retrieval/clustering metric, so this utility copies
the complete JSON results into a versioned local source file for plotting.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    "/data_2/suxin/runs/h100_hplus_7b_sigreg_ab_tuning_20260725/"
    "hplus_nosigreg/alpha_tune_e15_2gpu/eval_scalar"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/04_figures/fig3_representation_20260812/sources/hplus_s6_scalar_metrics_complete.json"
)
RETRIEVAL_DATASETS = ["crc-val-he-7k", "lc25000", "nct-crc-he-100", "nct-crc-he-1k"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="suxin-8H100-1")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Keep the remote query self-contained so the local output records exactly
    # the JSON files that supplied every reported scalar.
    remote_code = f'''import json
from pathlib import Path
root = Path({args.root!r})
retrieval_datasets = {RETRIEVAL_DATASETS!r}
models = {{"biodino_hplus": "100", "dinov3_hplus_official": "0"}}
result = {{"protocol": "frozen ID scalar evaluation", "root": str(root), "models": {{}}}}
for name, checkpoint in models.items():
    retrieval = {{}}
    paths = []
    for dataset in retrieval_datasets:
        path = root / "bio_retrieval" / dataset / checkpoint / "last_result.json"
        retrieval[dataset] = json.loads(path.read_text())
        paths.append(str(path))
    regression_path = root / "bio_regression" / "bbbc005" / checkpoint / "last_result.json"
    result["models"][name] = {{
        "checkpoint": checkpoint,
        "retrieval_clustering": retrieval,
        "regression": json.loads(regression_path.read_text()),
        "source_paths": paths + [str(regression_path)],
    }}
print(json.dumps(result, indent=2, sort_keys=True))'''
    remote_command = "python3 -c " + shlex.quote(remote_code)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", args.host, remote_command],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
