#!/usr/bin/env python3
"""Pinned HPA query/gallery E/M/L extraction on an admitted physical GPU."""
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


def log(s: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {s}', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--max-hours', type=float, default=20)
    args = p.parse_args()
    if not 0 <= args.gpu_index < 8 or not (CAMP / 'retrieval_v3_manifest.json').is_file():
        p.error('Expect physical GPU 0..7 and pinned v3 retrieval source')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu_index), PYTHONPATH=str(SOURCE),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    log(f'PROVENANCE launcher_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} '
        f'source={CAMP / "retrieval_v3_manifest.json"} gpu={args.gpu_index} '
        'dataset=hpa-subcellular protocol=locked-query-gallery')
    deadline = time.monotonic() + args.max_hours * 3600
    for role, step in (('E', 12687), ('M', 20007), ('L', 29279)):
        dest = CAMP / 'retrieval/hpa-subcellular' / role
        summary = dest / 'summary.csv'
        if summary.is_file() and summary.stat().st_size > 0:
            log(f'PRIOR_OUTPUT_NEEDS_AUDIT {role} {summary}; no overwrite or implicit skip')
            return 1
        ckpt = TRAIN / f'eval/training_{step}/teacher_checkpoint.pth'
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        while True:
            raw = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index),
                                           '--query-gpu=memory.used,memory.total',
                                           '--format=csv,noheader,nounits'], text=True, timeout=15)
            used, total = (int(x.strip()) for x in raw.strip().split(','))
            if used / total < .60 and total - used >= 8192:
                break
            if time.monotonic() > deadline:
                log(f'GPU_ADMISSION_TIMEOUT {role} gpu={args.gpu_index}')
                return 2
            time.sleep(45)
        cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_retrieval_clustering',
               '--checkpoint', str(ckpt), '--train-config', str(TRAIN / 'config.yaml'),
               '--benchmark-root', '/mnt/huawei_deepcad/benchmark',
               '--datasets', 'hpa-subcellular', '--output-dir', str(dest),
               '--model-name', f'hs6_l5_{role}_{step}', '--batch-size', '64',
               '--num-workers', '2', '--autocast-dtype', 'bf16', '--n-last-blocks', '1',
               '--channel-policy', 'auto', '--channel-tta-samples', '8',
               '--metric-device', 'cpu', '--seed', '0']
        dest.mkdir(parents=True, exist_ok=True)
        task_log = CAMP / 'logs' / f'queue_retrieval_hpa_local5090_gpu{args.gpu_index}_{role}.log'
        log(f'[launch] {role} gpu={args.gpu_index} log={task_log}')
        with task_log.open('x') as stream:
            rc = subprocess.run(cmd, env=env, cwd=SOURCE, stdout=stream,
                                stderr=subprocess.STDOUT, check=False).returncode
        if rc or not summary.is_file():
            log(f'FAILED_PROTOCOL_OR_RESOURCE {role} code={rc} log={task_log}')
            return 1
        log(f'[done] {role} HPA baselines; paired query/gallery fusion not yet run')
    log('QUEUE_COMPLETED HPA E/M/L prerequisites')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
