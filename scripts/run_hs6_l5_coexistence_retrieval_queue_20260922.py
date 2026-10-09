#!/usr/bin/env python3
"""Bounded v4 within-set retrieval/clustering E/M/L + fusion on 3090-qi."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SOURCE = ROOT / 'source_snapshot_retrieval_v3'
LAUNCHER = SOURCE / 'scripts/launch_hs6_l5_coexistence_retrieval_20260922.sh'
ALLOWED = {'nct-crc-he-1k', 'nct-crc-he-100', 'crc-val-he-7k', 'lc25000'}


def log(msg):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {msg}', flush=True)


def admitted():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()[0]
    used, total = [int(x.strip()) for x in line.split(',')]
    return used / total < .60 and total - used >= 8192


def baseline_done(dataset, role):
    file = ROOT / 'retrieval' / dataset / role / 'summary.csv'
    if not file.exists():
        return False
    with file.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1 or rows[0]['error'] or rows[0]['dataset'] != dataset:
        raise ValueError(f'Existing retrieval baseline is incomplete or invalid: {file}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--max-hours', type=float, default=20)
    args = p.parse_args()
    if not set(args.datasets) <= ALLOWED or len(set(args.datasets)) != len(args.datasets):
        raise ValueError(f'Unapproved within-set retrieval queue: {args.datasets}')
    deadline = time.monotonic() + args.max_hours * 3600
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(SOURCE), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for dataset in args.datasets:
        for role in 'EML':
            if baseline_done(dataset, role):
                log(f'[skip] {dataset}/{role}')
                continue
            while not admitted():
                if time.monotonic() > deadline:
                    log(f'GPU_ADMISSION_TIMEOUT: {dataset}/{role}')
                    return 2
                time.sleep(45)
            log(f'[launch] {dataset}/{role}')
            path = ROOT / 'logs' / f'queue_retrieval_{dataset}_{role}_{os.uname().nodename}.log'
            if path.exists():
                raise FileExistsError(path)
            with path.open('w') as f:
                status = subprocess.run(['bash', str(LAUNCHER), role, dataset], cwd=SOURCE, env=env,
                                        stdout=f, stderr=subprocess.STDOUT, check=False).returncode
            if status or not baseline_done(dataset, role):
                log(f'FAILED_PROTOCOL_OR_RESOURCE {dataset}/{role}, rc={status} log={path}')
                return 1
        output = ROOT / 'fusion' / f'retrieval_{dataset}.json'
        if not output.exists():
            cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_retrieval',
                   '--dataset', dataset, '--feature-root', str(ROOT / 'retrieval'), '--output', str(output)]
            log(f'[fusion] {dataset}: {cmd}')
            with (ROOT / 'logs' / f'queue_retrieval_fusion_{dataset}.log').open('w') as f:
                status = subprocess.run(cmd, cwd=SOURCE, env=env,
                                        stdout=f, stderr=subprocess.STDOUT, check=False).returncode
            if status or not output.exists():
                log(f'FAILED_FUSION {dataset} rc={status}')
                return 1
        log(f'[done] {dataset} E/M/L + E+L/M+L; retrieval PCA requires disjoint calibration')
    log('QUEUE_COMPLETED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
