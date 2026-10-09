#!/usr/bin/env python3
"""Launch v3-only, formal-split E/M/L spatial single-teacher baselines on one GPU.

These are paired-source prerequisites for six-arm spatial coexistence; this
launcher does not infer missing E+L/M+L/PCA scores from the single baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path('/mnt/huawei_deepcad/dinov3')
CAMP = REPO / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
RUN = REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'
SOURCE = CAMP / 'source_snapshot'
PYTHON = Path('/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
STEPS = {'E': ('early', 12687), 'M': ('middle', 20007), 'L': ('late', 29279)}
DATASETS = {'conic', 'livecell', 'multimodal_cellseg', 'pannuke', 'tissuenet'}


def report(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def room(gpu: int) -> bool:
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    used, total = (int(item.strip()) for item in output.strip().split(','))
    return used / total < .60 and total - used >= 8192


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', required=True, type=int)
    parser.add_argument('--host-tag', required=True, choices=('local5090', 'qi3090'))
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    parser.add_argument('--roles', nargs='+', choices=tuple(STEPS), required=True)
    parser.add_argument('--max-hours', type=float, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.gpu_index < 8 or len(args.roles) != len(set(args.roles)):
        parser.error('Physical GPU must be in 0..7 and roles unique')
    if not (CAMP / 'launch_manifest.json').exists():
        raise FileNotFoundError('Source/checkpoint launch manifest is missing')
    if not (SOURCE / 'dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py').is_file():
        raise FileNotFoundError('Pinned segmentation pipeline missing')
    deadline = time.monotonic() + args.max_hours * 3600
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES=str(args.gpu_index),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    report(f'PROVENANCE launcher_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} '
           f'pinned_source={SOURCE} physical_gpu={args.gpu_index} '
           f'host={args.host_tag} split=formal-v1 budgets=20,50 seeds=0,1,2')
    for role in args.roles:
        label, step = STEPS[role]
        checkpoint = RUN / f'eval/training_{step}/teacher_checkpoint.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        command = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline',
                   '--datasets', args.dataset, '--checkpoint-file', str(checkpoint),
                   '--checkpoint-id', str(step), '--train-config', str(RUN / 'config.yaml'),
                   '--protocol', 'best', '--dataset-split-protocol', 'formal-v1',
                   '--feature-batch-size', '32', '--feature-num-workers', '2',
                   '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
                   '--probe-epoch-grid', '20', '50', '--probe-seeds', '0', '1', '2',
                   '--probe-batch-size', '32', '--probe-lr', '0.001', '--probe-weight-decay', '0.0001',
                   '--probe-eval-every', '1', '--probe-num-workers', '2', '--gpu', str(args.gpu_index),
                   '--cache-root', str(CAMP / 'segmentation/remaining_cache'),
                   '--output-root', str(CAMP / 'segmentation/remaining_results'),
                   '--run-name', f'hs6_l5_{label}']
        if args.dry_run:
            command.append('--dry-run')
            report(f'DRY_RUN {role}/{args.dataset} cmd={command}')
            return subprocess.run(command, env=env, cwd=SOURCE, check=False).returncode
        while not room(args.gpu_index):
            if time.monotonic() > deadline:
                report(f'GPU_ADMISSION_TIMEOUT {role}/{args.dataset} gpu={args.gpu_index}')
                return 2
            time.sleep(45)
        if time.monotonic() > deadline:
            report(f'QUEUE_TIME_LIMIT before {role}/{args.dataset}')
            return 2
        log = CAMP / 'logs' / f'queue_seg_{args.host_tag}_gpu{args.gpu_index}_{args.dataset}_{role}.log'
        report(f'[launch] {role}/{args.dataset} physical_gpu={args.gpu_index} log={log}')
        with log.open('x') as stream:
            result = subprocess.run(command, env=env, cwd=SOURCE, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            report(f'FAILED_SPATIAL_SINGLE role={role} dataset={args.dataset} '
                   f'code={result.returncode} log={log}')
            return 1
        report(f'[done] {role}/{args.dataset} baseline; paired spatial fusion still requires verified banks')
    report('QUEUE_COMPLETED assigned spatial baselines')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
