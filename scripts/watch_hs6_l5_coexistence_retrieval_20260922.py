#!/usr/bin/env python3
"""Run matched retrieval/clustering five-arm readout when all frozen banks finish."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_retrieval_v3'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['crc-val-he-7k', 'nct-crc-he-1k', 'nct-crc-he-100', 'lc25000'])
    p.add_argument('--timeout-hours', type=float, default=18)
    args = p.parse_args()
    result = ROOT / 'fusion' / f'retrieval_{args.dataset}.json'
    if result.exists():
        print(f'Already done: {result}', flush=True)
        return 0
    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        paths = [ROOT / 'retrieval' / args.dataset / role / 'summary.csv' for role in 'EML']
        if all(f.is_file() and f.stat().st_size for f in paths):
            rows = [list(csv.DictReader(f.open(newline=''))) for f in paths]
            if any(len(v) != 1 or v[0]['error'] for v in rows):
                print(f'FAILED_INPUT: retrieval summaries absent/error: {paths}', flush=True)
                return 1
            env = os.environ.copy()
            env.update(PYTHONPATH=str(SNAPSHOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_retrieval',
                   '--dataset', args.dataset, '--feature-root', str(ROOT / 'retrieval'), '--output', str(result)]
            print(json.dumps({'command': cmd, 'cwd': str(SNAPSHOT),
                              'source_manifest': str(ROOT / 'retrieval_v3_manifest.json')}), flush=True)
            return subprocess.call(cmd, cwd=SNAPSHOT, env=env)
        time.sleep(45)
    print(f'TIMEOUT: {args.dataset} did not finish matched E/M/L', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
