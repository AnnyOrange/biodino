#!/usr/bin/env python3
"""Wait for all three full frozen banks, then run the six-arm CPU probe once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_fusion_v2'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['dermamnist', 'bbbc005', 'pneumoniamnist', 'bloodmnist'])
    parser.add_argument('--timeout-hours', type=float, default=18)
    args = parser.parse_args()
    dataset = args.dataset
    output = ROOT / 'fusion' / f'{dataset}.json'
    if output.exists():
        print(f'Already complete: {output}', flush=True)
        return 0
    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        summaries = [ROOT / 'frozen' / dataset / role / 'summary.csv' for role in 'EML']
        if all(p.is_file() and p.stat().st_size > 0 for p in summaries):
            import csv
            rows = [list(csv.DictReader(p.open(newline=''))) for p in summaries]
            if any(len(row) != 1 or row[0]['error'] for row in rows):
                print(f'FAILED_INPUT: one baseline summary is missing or errored: {summaries}', flush=True)
                return 1
            env = os.environ.copy()
            env.update(PYTHONPATH=str(SNAPSHOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_classification',
                   '--dataset', dataset, '--feature-root', str(ROOT / 'frozen'), '--output', str(output)]
            print(json.dumps({'command': cmd, 'cwd': str(SNAPSHOT), 'source_manifest': str(ROOT / 'fusion_v2_manifest.json')}), flush=True)
            return subprocess.call(cmd, cwd=SNAPSHOT, env=env)
        time.sleep(45)
    print(f'TIMEOUT: full E/M/L baseline trio unavailable after {args.timeout_hours}h: {dataset}', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
