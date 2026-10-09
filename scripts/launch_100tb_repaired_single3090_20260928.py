#!/usr/bin/env python3
"""Launch the repaired global-pool 100TB 1M HS6-L run on one RTX 3090."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/huawei_blm/random_1pb_100tb_global_v3/100tb_final_1m_repaired_20260928')
BASE = ROOT / 'outputs/01_training_runs/HS6_L_random100tb1m_robust_biosafe256_gb1024_lr1e4_e1_seed0_ddp_b128acc4_2xA100deepcad_20260922/config.yaml'
PY = '/home/inspur/anaconda3/envs/dinov3/bin/python'
OUT = ROOT / 'outputs/01_training_runs/HS6_L_100tb_global_repaired1m_biosafe256_gb1024_e1_single3090_20260928'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stats-log', type=Path)
    p.add_argument('--mean', type=float, nargs=3)
    p.add_argument('--std', type=float, nargs=3)
    args = p.parse_args()
    if not (args.smoke or (args.stats_log and args.mean and args.std)):
        p.error('Final training requires --stats-log, --mean and --std from robust training decode')
    if bool(args.mean) != bool(args.std):
        p.error('--mean and --std must be paired')
    final = json.loads((DATA / 'finalization.json').read_text())
    shards = sorted(DATA.glob('filtered_mixed_train*.tar'))
    assert final['samples'] == 1_000_000 and final['wds_shards'] == len(shards) == 500
    assert final['last_selected_priority'] == 999999
    assert all(x.stat().st_size > 0 for x in shards)
    assert (DATA / 'pre_robust_rgb_stats_1m.json').is_file() and BASE.is_file()
    if args.stats_log:
        content = args.stats_log.read_text()
        assert 'ROBUST_TRAIN_TENSOR' in content and 'samples=' in content
    out = OUT.with_name(OUT.name + '_smoke') if args.smoke else OUT
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    overrides = {
        'train.dataset_path': f'packwds_robust:{DATA}/filtered_mixed_train*.tar::pct=1,99',
        'train.batch_size_per_gpu': 8,
        'optim.gradient_accumulation_steps': 1 if args.smoke else 128,
        'train.num_workers': 2,
        'train.wds_deterministic_resampling': True,
        'train.checkpointing': True,
        'train.checkpointing_full': True,
        'train.checkpointing_blocks': 24,
        'train.pin_memory': False,
        'train.prefetch_factor': 1,
        'train.compile': False,
        'train.OFFICIAL_EPOCH_LENGTH': 977,
        'train.max_updates': 1 if args.smoke else None,
        'evaluation.eval_period_iterations': 0 if args.smoke else 977,
        'checkpointing.period': 977 if args.smoke else 100,
        'checkpointing.max_to_keep': 2,
        'optim.epochs': 1,
        'train.seed': 0,
    }
    if args.mean:
        overrides['crops.rgb_mean'] = args.mean
        overrides['crops.rgb_std'] = args.std
    def value(x):
        return x if isinstance(x, str) else json.dumps(x, separators=(',', ':'))
    cmd = [PY, '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port=32928',
           'dinov3/train/train.py', '--config-file', str(BASE), '--output-dir', str(out),
           '--no-resume', '--seed', '0'] + [f'{k}={value(v)}' for k, v in overrides.items()]
    manifest = {
        'purpose': 'repaired global storage pool 100TB 1M training; smoke only' if args.smoke else 'repaired global storage pool 100TB 1M training, 1 nominal epoch',
        'data_root': str(DATA), 'data_finalization': final,
        'finalization_sha256': sha(DATA / 'finalization.json'),
        'pre_robust_stats_sha256': sha(DATA / 'pre_robust_rgb_stats_1m.json'),
        'robust_stats_log': str(args.stats_log) if args.stats_log else None,
        'robust_stats_sha256': sha(args.stats_log) if args.stats_log else None,
        'base_config': str(BASE), 'base_config_sha256': sha(BASE),
        'command': cmd, 'overrides': overrides,
        'effective_optimizer_batch': 8 * (1 if args.smoke else 128),
        'target_updates': 1 if args.smoke else 977,
        'note': 'The WDS loader resamples shards, so 1 epoch means 977 optimizer updates (~1M image visits), not exactly one visit to each distinct image.',
    }
    (out / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_IB_DISABLE='1',
               NCCL_P2P_DISABLE='1')
    with (out / 'console.log').open('w', buffering=1) as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / 'exit.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    print(json.dumps({'output': str(out), 'returncode': result.returncode}), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
