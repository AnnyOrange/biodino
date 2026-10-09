#!/usr/bin/env python3
"""Run matching normalized single-checkpoint BBBC038 detection controls."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SOURCE = ROOT / 'source_snapshot_detection_controls_v9'
TRAIN = Path('/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907')


def gpu_room():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()[0]
    used, total = [int(x.strip()) for x in line.split(',')]
    return used / total < .60 and total - used >= 8192


def main():
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        paths = [ROOT / 'detection/bbbc038' / arm / 'results_bio_detection.json' for arm in ('E+L', 'M+L')]
        if all(p.exists() and 'test_patch_f1' in json.loads(p.read_text()) for p in paths):
            break
        time.sleep(45)
    else:
        print('TIMEOUT: normalized controls require two completed BBBC038 fusion proxies', flush=True)
        return 2
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for role in 'EML':
        arm = f'unit_{role}'
        output = ROOT / 'detection/bbbc038' / arm
        if (output / 'results_bio_detection.json').exists():
            raise FileExistsError(output / 'results_bio_detection.json')
        while not gpu_room():
            if time.monotonic() > deadline:
                print(f'GPU_ADMISSION_TIMEOUT: {arm}', flush=True)
                return 2
            time.sleep(45)
        cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_detection.run_coexistence_detection',
               '--arm', arm, '--run-root', str(TRAIN), '--output-dir', str(output)]
        print(f'[launch] arm={arm} source={ROOT / "detection_controls_v9_manifest.json"} command={cmd}', flush=True)
        if subprocess.call(cmd, cwd=SOURCE, env=env):
            print(f'FAILED_PROTOCOL_OR_RESOURCE: {arm}', flush=True)
            return 1
    print('COMPLETE: BBBC038 unit-normalized single-checkpoint controls', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
