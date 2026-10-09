#!/usr/bin/env python3
"""One nonduplicated official RxRx1 cross-experiment frozen teacher on a physical GPU."""
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
TRAIN = REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'
SOURCE = CAMP / 'source_snapshot_retrieval_v3'
PYTHON = Path('/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
ROLES = {'E': 12687, 'M': 20007, 'L': 29279}


def stamp(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def memory() -> tuple[int, int]:
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:', 'MemAvailable:')):
            key, value = line.split(':', 1)
            values[key] = int(value.strip().split()[0])
    return values['MemAvailable'], values['MemTotal']


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=tuple(ROLES), required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--host-tag', choices=('qi3090', 'local5090'), required=True)
    p.add_argument('--max-hours', type=float, default=20)
    args = p.parse_args()
    if args.gpu_index not in range(8) or not (CAMP / 'retrieval_v3_manifest.json').is_file():
        p.error('GPU index 0..7 and pinned retrieval v3 manifest required')
    dest = CAMP / 'retrieval/rxrx1-cross' / args.role
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Unclaimed prior RxRx1 output: {dest}')
    checkpoint = TRAIN / f'eval/training_{ROLES[args.role]}/teacher_checkpoint.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stamp(f'PROVENANCE runner_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} '
          f'source={CAMP / "retrieval_v3_manifest.json"} role={args.role} '
          f'gpu={args.gpu_index} official_rxrx1_core_manifest, threshold_gpu_ram=80%')
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        raw = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index),
                                       '--query-gpu=memory.used,memory.total',
                                       '--format=csv,noheader,nounits'], text=True, timeout=15)
        used, total = (int(x.strip()) for x in raw.strip().split(','))
        available, mem_total = memory()
        # Reserve 4 GiB GPU and 8 GiB host memory per independently started test.
        if (used + 4096) / total <= .80 and (available - 8 * 1024 * 1024) / mem_total >= .20:
            break
        stamp(f'WAIT_RAM gpu={used}/{total}MiB host_available={available}/{mem_total}KiB')
        time.sleep(45)
    else:
        stamp('GPU_OR_RAM_ADMISSION_TIMEOUT')
        return 2
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu_index), PYTHONPATH=str(SOURCE),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    dest.mkdir(parents=True, exist_ok=False)
    command = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_retrieval_clustering',
               '--checkpoint', str(checkpoint), '--train-config', str(TRAIN / 'config.yaml'),
               '--benchmark-root', '/mnt/huawei_deepcad/benchmark', '--datasets', 'rxrx1-cross',
               '--output-dir', str(dest), '--model-name', f'hs6_l5_{args.role}_{ROLES[args.role]}',
               '--batch-size', '64', '--num-workers', '2', '--autocast-dtype', 'bf16',
               '--n-last-blocks', '1', '--channel-policy', 'auto', '--channel-tta-samples', '8',
               '--metric-device', 'cpu', '--seed', '0']
    stamp(f'[launch] gpu={args.gpu_index} host_ram_available={available}/{mem_total}KiB '
          f'gpu_memory_used={used}/{total}MiB')
    result = subprocess.run(command, cwd=SOURCE, env=env, check=False)
    if result.returncode or not (dest / 'summary.csv').is_file():
        stamp(f'FAILED_PROTOCOL_OR_RESOURCE role={args.role} code={result.returncode}')
        return 1
    stamp(f'[done] RxRx1 {args.role} source data/features locked; paired fusion still pending')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
