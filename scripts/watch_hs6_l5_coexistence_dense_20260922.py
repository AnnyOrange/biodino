#!/usr/bin/env python3
"""Wait for nine v4 Cellpose spatial banks; create two matched fusion caches."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_dense_v4'


def ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return 'features.npy' in z.namelist() and 'inst_maps.npy' in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def main():
    sys.path.insert(0, str(SNAPSHOT))
    from dinov3.eval.bio_segmentation.coexistence_dense import _input_path

    output = ROOT / 'segmentation' / 'fused_cache'
    if all((output / arm / f'{split}.npz').exists()
           for arm in ('E+L', 'M+L') for split in ('train', 'val', 'test')):
        print('Already fused', flush=True)
        return 0
    paths = [_input_path(ROOT / 'segmentation' / 'cache', role, split)
             for role in 'EML' for split in ('train', 'val', 'test')]
    deadline = time.monotonic() + 18 * 3600
    while time.monotonic() < deadline:
        if all(ready(p) for p in paths):
            env = os.environ.copy()
            env.update(PYTHONPATH=str(SNAPSHOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_segmentation.coexistence_dense',
                   '--input-root', str(ROOT / 'segmentation' / 'cache'), '--output-root', str(output)]
            print(f'[fuse] ready=9 source={ROOT / "dense_v4_manifest.json"} command={cmd}', flush=True)
            return subprocess.call(cmd, cwd=SNAPSHOT, env=env)
        time.sleep(45)
    print('TIMEOUT: nine valid E/M/L Cellpose banks not available', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
