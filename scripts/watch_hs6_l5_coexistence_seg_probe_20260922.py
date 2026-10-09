#!/usr/bin/env python3
"""Run a full E20/E50 × three-seed v3 Cellpose head on one fused spatial bank."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_dense_v4'


def admitted():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        text=True, timeout=15
    ).strip().splitlines()[0]
    used, total = (int(x.strip()) for x in output.split(','))
    return used / total < 0.60 and total - used >= 8192


def complete(path: Path, budget: int, seed: int):
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    meta = data['_meta']
    if meta.get('probe_epochs') == budget and meta.get('seed') == seed and \
            meta.get('probe_eval_every') == 1 and meta.get('test_evaluations') == 1 and 'test' in data:
        return True
    raise ValueError(f'Incomplete/conflicting result requires manual audit: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm', choices=('E+L', 'M+L', 'PCA_E+L', 'unit_E', 'unit_M', 'unit_L'))
    args = p.parse_args()
    variant = ('dense_controls_v8' if args.arm.startswith('unit_') else
               'dense_pca_v5' if args.arm == 'PCA_E+L' else 'dense_v4')
    variant_sources = {'dense_controls_v8': ('source_snapshot_dense_controls_v8', 'dense_controls_v8_manifest.json'),
                       'dense_pca_v5': ('source_snapshot_dense_pca_v5', 'dense_pca_v5_manifest.json'),
                       'dense_v4': ('source_snapshot_dense_v4', 'dense_v4_manifest.json')}
    source_snapshot = ROOT / variant_sources[variant][0]
    source_manifest = ROOT / variant_sources[variant][1]
    cache = ROOT / 'segmentation' / 'fused_cache' / args.arm
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if all((cache / f'{split}.npz').exists() and (cache / f'{split}.identity.json').exists()
               for split in ('train', 'val', 'test')):
            break
        time.sleep(45)
    else:
        print(f'TIMEOUT: fused Cellpose banks missing for {args.arm}', flush=True)
        return 2
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source_snapshot), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for budget in (20, 50):
        for seed in (0, 1, 2):
            out = ROOT / 'segmentation' / 'fused_results' / args.arm / f'budget{budget}' / f'seed{seed}' / 'cellpose'
            if complete(out / 'results.json', budget, seed):
                continue
            while not admitted():
                if time.monotonic() > deadline:
                    print(f'GPU_ADMISSION_TIMEOUT: {args.arm} budget={budget} seed={seed}', flush=True)
                    return 2
                time.sleep(45)
            cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_segmentation.linear_probe',
                   '--dataset', 'cellpose', '--use-cached-features',
                   '--train-cache', str(cache / 'train.npz'), '--val-cache', str(cache / 'val.npz'),
                   '--test-cache', str(cache / 'test.npz'), '--output-dir', str(out),
                   '--epochs', str(budget), '--batch-size', '32', '--lr', '0.001',
                   '--weight-decay', '0.0001', '--num-workers', '2', '--eval-every', '1',
                   '--seed', str(seed)]
            print(json.dumps({'command': cmd, 'cwd': str(source_snapshot), 'source': str(source_manifest)}), flush=True)
            if subprocess.call(cmd, cwd=source_snapshot, env=env):
                print(f'FAILED_PROTOCOL_OR_RESOURCE: {args.arm} budget={budget} seed={seed}', flush=True)
                return 1
            if not complete(out / 'results.json', budget, seed):
                return 1
    print(f'COMPLETE: {args.arm} Cellpose full E20/E50 seed grid', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
