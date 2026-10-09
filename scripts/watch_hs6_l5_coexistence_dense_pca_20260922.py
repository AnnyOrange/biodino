#!/usr/bin/env python3
"""Fit a Cellpose patch PCA on train-only E+L after fused banks are finalized."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_dense_pca_v5'
SOURCE = ROOT / 'segmentation' / 'fused_cache' / 'E+L'
OUTPUT = ROOT / 'segmentation' / 'fused_cache' / 'PCA_E+L'


def ready(split):
    path = SOURCE / f'{split}.npz'
    if not path.is_file() or not path.with_suffix('.identity.json').is_file():
        return False
    try:
        with zipfile.ZipFile(path) as bank:
            return 'features.npy' in bank.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def main():
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if all(ready(split) for split in ('train', 'val', 'test')):
            break
        time.sleep(45)
    else:
        print('TIMEOUT: train/val/test E+L spatial caches never became ready', flush=True)
        return 2
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SNAPSHOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_segmentation.coexistence_dense_pca',
           '--input-dir', str(SOURCE), '--output-dir', str(OUTPUT)]
    print(f'[pca] source={ROOT / "dense_pca_v5_manifest.json"} command={cmd}', flush=True)
    return subprocess.call(cmd, cwd=SNAPSHOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
