#!/usr/bin/env python3
"""After matched BBBC038 baselines, run two spatial fusion detection proxies."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SNAPSHOT = ROOT / 'source_snapshot_detection_v6'
TRAIN = Path('/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907')


def gpu_room():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()[0]
    used, total = [int(x.strip()) for x in line.split(',')]
    return used / total < .60 and total - used >= 8192


def check_baselines():
    paths = [ROOT / 'detection' / 'bbbc038' / role / 'results_bio_detection.json' for role in 'EML']
    if not all(p.exists() for p in paths):
        return False
    for p in paths:
        data = json.loads(p.read_text())
        if data.get('dataset') != 'bbbc038' or data.get('batch_size') != 8 or data.get('epochs') != 5 \
                or not math.isfinite(float(data['test_patch_f1'])):
            raise ValueError(f'Mismatched or invalid baseline: {p}')
    return True


def main():
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if check_baselines():
            break
        time.sleep(45)
    else:
        print('TIMEOUT: E/M/L BBBC038 proxy baseline missing', flush=True)
        return 2
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SNAPSHOT), CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for arm in ('E+L', 'M+L'):
        out = ROOT / 'detection' / 'bbbc038' / arm
        result = out / 'results_bio_detection.json'
        if result.exists():
            raise FileExistsError(f'Refusing to overwrite a preexisting fusion: {result}')
        while not gpu_room():
            if time.monotonic() > deadline:
                print(f'GPU_ADMISSION_TIMEOUT: {arm}', flush=True)
                return 2
            time.sleep(45)
        cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_detection.run_coexistence_detection',
               '--arm', arm, '--run-root', str(TRAIN), '--output-dir', str(out)]
        print(json.dumps({'command': cmd, 'cwd': str(SNAPSHOT),
                          'source_manifest': str(ROOT / 'detection_v6_manifest.json')}), flush=True)
        if subprocess.call(cmd, cwd=SNAPSHOT, env=env):
            print(f'FAILED_PROTOCOL_OR_RESOURCE: BBBC038/{arm}', flush=True)
            return 1
    print('COMPLETE: BBBC038 E/M/L plus E+L and M+L', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
