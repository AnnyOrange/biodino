#!/usr/bin/env python3
"""Pinned v4 frozen E/M/L extraction and six-arm CPU probe on a chosen GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path('/mnt/huawei_deepcad/dinov3')
ROOT = WORKSPACE / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
RUN = WORKSPACE / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'
SOURCE = ROOT / 'source_snapshot'
FUSION = ROOT / 'source_snapshot_lc_v7'
PYTHON = Path('/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
STEPS = {'E': 12687, 'M': 20007, 'L': 29279}


def report(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def admitted(index: int) -> tuple[bool, int, int]:
    data = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
    if len(data) != 1:
        raise RuntimeError(f'Expected exactly one physical GPU: {index}')
    used, total = (int(x.strip()) for x in data[0].split(','))
    return used / total < .60 and total - used >= 8192, used, total


def complete(dataset: str, role: str) -> bool:
    path = ROOT / 'frozen' / dataset / role / 'summary.csv'
    if not path.exists():
        return False
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1 or rows[0].get('dataset') != dataset or rows[0].get('error'):
        raise ValueError(f'Prior incomplete baseline requires audit: {path}')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--host-tag', choices=('qi3090', 'local5090'), required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--max-hours', type=float, default=20)
    args = parser.parse_args()
    if not 0 <= args.gpu_index < 8 or len(args.datasets) != len(set(args.datasets)):
        parser.error('Physical GPU must be in 0..7; datasets must be distinct')
    protocol = json.loads((SOURCE / 'Evaluation Rules/protocol_v4.json').read_text())
    allowed = {name for section in ('tier_a', 'tier_b', 'union_extension')
               for family in ('classification', 'regression') for name in protocol[section].get(family, [])}
    if not set(args.datasets) <= allowed:
        parser.error(f'Only protocol-v4 frozen classification/regression allowed: {args.datasets}')
    if not (ROOT / 'lc_v7_manifest.json').is_file() or not PYTHON.is_file():
        raise FileNotFoundError('Pinned fusion source or evaluation environment missing')
    deadline = time.monotonic() + args.max_hours * 3600
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu_index), PYTHONPATH=str(SOURCE),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report(f'LAUNCH_PROVENANCE physical_gpu={args.gpu_index} host={args.host_tag} '
           f'queue_sha256={source_hash} extractor_snapshot={SOURCE} fusion_manifest={ROOT / "lc_v7_manifest.json"}')
    for dataset in args.datasets:
        if time.monotonic() > deadline:
            report(f'QUEUE_TIME_LIMIT before {dataset}')
            return 2
        for role, update in STEPS.items():
            if complete(dataset, role):
                report(f'[skip] {dataset}/{role}: matching summary exists')
                continue
            while True:
                ready, used, total = admitted(args.gpu_index)
                if ready:
                    break
                if time.monotonic() > deadline:
                    report(f'GPU_ADMISSION_TIMEOUT gpu={args.gpu_index} dataset={dataset}/{role} {used}/{total}')
                    return 2
                time.sleep(45)
            checkpoint = RUN / f'eval/training_{update}/teacher_checkpoint.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            dest = ROOT / 'frozen' / dataset / role
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_classification',
                   '--checkpoint', str(checkpoint), '--train-config', str(RUN / 'config.yaml'),
                   '--benchmark-root', '/mnt/huawei_deepcad/benchmark', '--datasets', dataset,
                   '--output-dir', str(dest), '--model-name', f'hs6_l5_{role}_{update}',
                   '--batch-size', '64', '--num-workers', '2', '--n-last-blocks', '1',
                   '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
                   '--resolution-protocol', 'best', '--split-protocol', 'current', '--save-paths']
            log = ROOT / 'logs' / f'queue_gpu_{args.host_tag}_{args.gpu_index}_{dataset}_{role}.log'
            report(f'[launch] {dataset}/{role} physical_gpu={args.gpu_index} used={used}/{total} log={log}')
            with log.open('x') as stream:
                status = subprocess.run(cmd, cwd=SOURCE, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False).returncode
            if status or not complete(dataset, role):
                report(f'FAILED_PROTOCOL_OR_RESOURCE {dataset}/{role} code={status}: {log}')
                return 1
        result_file = ROOT / 'fusion' / f'{dataset}.json'
        if not result_file.exists():
            env['PYTHONPATH'] = str(FUSION)
            cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_classification',
                   '--dataset', dataset, '--feature-root', str(ROOT / 'frozen'), '--output', str(result_file)]
            log = ROOT / 'logs' / f'queue_gpu_{args.host_tag}_{args.gpu_index}_{dataset}_fusion.log'
            report(f'[fusion] {dataset} pinned_source={ROOT / "lc_v7_manifest.json"} log={log}')
            with log.open('x') as stream:
                status = subprocess.run(cmd, cwd=FUSION, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False).returncode
            if status or not result_file.exists():
                report(f'FAILED_FUSION {dataset} code={status}: {log}')
                return 1
            env['PYTHONPATH'] = str(SOURCE)
        report(f'[done] {dataset} paired frozen E/M/L and CPU six-arm fusion')
    report('QUEUE_COMPLETED all assigned READY datasets')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
