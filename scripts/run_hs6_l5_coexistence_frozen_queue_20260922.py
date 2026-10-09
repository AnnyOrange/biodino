#!/usr/bin/env python3
"""Bounded resumable v4 frozen E/M/L + matched fusion queue on one 3090."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
EXTRACT = ROOT / 'source_snapshot' / 'scripts' / 'launch_hs6_l5_coexistence_pilot_20260922.sh'
FUSION_SOURCE = ROOT / 'source_snapshot_lc_v7'


def log(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def gpu_room() -> tuple[int, int]:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()[0]
    return tuple(int(part.strip()) for part in output.split(','))


def baseline_complete(dataset: str, role: str):
    path = ROOT / 'frozen' / dataset / role / 'summary.csv'
    if not path.exists():
        return False
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1 or rows[0].get('dataset') != dataset or rows[0].get('error'):
        raise ValueError(f'Existing baseline needs manual audit, refusing overwrite: {path}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--max-hours', type=float, default=20)
    args = p.parse_args()
    proto = json.loads((FUSION_SOURCE / 'Evaluation Rules/protocol_v4.json').read_text())
    allowed = {name for section in ('tier_a', 'tier_b', 'union_extension')
               for task in ('classification', 'regression') for name in proto[section].get(task, [])}
    if any(dataset not in allowed for dataset in args.datasets) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError(f'Dataset list must be distinct v4 classification/regression: {args.datasets}')
    if not (ROOT / 'lc_v7_manifest.json').is_file():
        raise FileNotFoundError('Pinned v7 fusion source manifest absent')
    deadline = time.monotonic() + args.max_hours * 3600
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(FUSION_SOURCE),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for dataset in args.datasets:
        if time.monotonic() >= deadline:
            log(f'QUEUE_TIME_LIMIT before {dataset}; remaining need manual continuation')
            return 2
        for role in 'EML':
            if baseline_complete(dataset, role):
                log(f'[skip] {dataset}/{role} baseline done')
                continue
            while True:
                used, total = gpu_room()
                if used / total < .60 and total - used >= 8192:
                    break
                if time.monotonic() >= deadline:
                    log(f'GPU_ADMISSION_TIMEOUT {dataset}/{role}, used={used}/{total} MiB')
                    return 2
                time.sleep(45)
            command = ['bash', str(EXTRACT), role, dataset]
            log(f'[launch] {dataset}/{role} gpu_used={used}/{total} command={command}')
            output = ROOT / 'logs' / f'queue_{dataset}_{role}_{os.uname().nodename}.log'
            if output.exists():
                raise FileExistsError(f'Unclaimed prior queue log: {output}')
            with output.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT / 'source_snapshot',
                                        env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode or not baseline_complete(dataset, role):
                log(f'FAILED_PROTOCOL_OR_RESOURCE {dataset}/{role} code={result.returncode}: {output}')
                return 1
        result_file = ROOT / 'fusion' / f'{dataset}.json'
        if result_file.exists():
            log(f'[skip] fusion {result_file}')
            continue
        cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_classification',
               '--dataset', dataset, '--feature-root', str(ROOT / 'frozen'), '--output', str(result_file)]
        log(f'[fusion] {dataset} cwd={FUSION_SOURCE}, source_manifest={ROOT / "lc_v7_manifest.json"}')
        with (ROOT / 'logs' / f'queue_fusion_{dataset}_{os.uname().nodename}.log').open('w') as stream:
            process = subprocess.run(cmd, cwd=FUSION_SOURCE, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, check=False)
        if process.returncode or not result_file.exists():
            log(f'FAILED_FUSION {dataset} code={process.returncode}')
            return 1
        log(f'[done] {dataset} matched E/M/L and six-arm CPU fusion')
    log('QUEUE_COMPLETED all assigned READY datasets')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
