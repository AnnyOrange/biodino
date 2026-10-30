#!/usr/bin/env python3
"""Run matched two-GPU 1PB/100TB seed repeats on the frozen source pools."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = Path('/mnt/liyixin22/seed_repeats_1pb_100tb_20261028')
BASE_CONFIG = Path('/mnt/liyixin22/100tb/config.yaml')
RAW_INDEX = Path('/mnt/liyixin22/100tb_2gpu_deepcad67_20261026/stream_index_full_pool_2rank')
PYTHONS = {'1pb': '/home/lxy/miniconda3/envs/dinov3/bin/python',
           '100tb': '/home/deepcad/anaconda3/envs/dinov3/bin/python'}


def source(dataset):
    if dataset == '1pb':
        manifest = json.loads((ROOT / '1pb_pool_manifest.json').read_text())
        assignment = Path(manifest['assignment'])
        if (manifest['sample_count'] < 80000 * 1024 or not assignment.is_file() or
                len(list((ROOT / '1pb_pool').glob('batch-*'))) != manifest['unique_batches']):
            raise ValueError('Frozen 1PB inventory is incomplete')
        return manifest['dataset_path'], assignment
    assignment = json.loads((RAW_INDEX / 'assignment.json').read_text())
    if assignment['status'] != 'PASS' or assignment['world_size'] != 2:
        raise ValueError('Audited 100TB two-rank index is unavailable')
    return 'raw100tb_full_pool_robust:' + str(RAW_INDEX), None


def repack_finite_audits(output, checkpoint, old_command, new_microbatch):
    old_microbatch = int(next(item.split('=', 1)[1] for item in old_command
                              if item.startswith('train.batch_size_per_gpu=')))
    old_accumulation = int(next(item.split('=', 1)[1] for item in old_command
                                if item.startswith('optim.gradient_accumulation_steps=')))
    if old_microbatch * old_accumulation != new_microbatch * (512 // new_microbatch):
        raise ValueError('Batch change would alter the number of consumed samples per update')
    expected_lines = (checkpoint + 1) * old_accumulation
    staged = []
    try:
        for rank in range(2):
            audit = output / f'consumed_sample_keys_rank{rank:02d}.jsonl'
            backup = output / f'consumed_sample_keys_rank{rank:02d}_before_batch_change_{checkpoint}.jsonl'
            temporary = audit.with_suffix('.batch_change_tmp')
            if backup.exists() or temporary.exists():
                raise ValueError(f'Audit batch transition already exists: {audit}')
            written, buffered = 0, []
            with audit.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as dest:
                for index in range(expected_lines):
                    line = source.readline()
                    if not line:
                        raise ValueError(f'{audit} ends before old batch {index}')
                    keys = json.loads(line)
                    if len(keys) != old_microbatch or len(set(keys)) != len(keys):
                        raise ValueError(f'{audit} has malformed old batch {index}')
                    buffered.extend(keys)
                    while len(buffered) >= new_microbatch:
                        dest.write(json.dumps(buffered[:new_microbatch], separators=(',', ':')) + '\n')
                        buffered = buffered[new_microbatch:]
                        written += 1
            if buffered or written != (checkpoint + 1) * (512 // new_microbatch):
                raise ValueError(f'{audit} cannot be regrouped into the new microbatch')
            staged.append((audit, backup, temporary))
        for audit, backup, _ in staged:
            os.link(audit, backup)
        for audit, _, temporary in staged:
            os.replace(temporary, audit)
    finally:
        for _, _, temporary in staged:
            temporary.unlink(missing_ok=True)


def run(dataset, seed, gpu_pair, master_port, microbatch_override=None):
    dataset_path, assignment = source(dataset)
    output = ROOT / f'{dataset}_seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    microbatch = microbatch_override or (64 if dataset == '100tb' else 32)
    if 512 % microbatch:
        raise ValueError('Two-rank effective batch 1024 requires microbatch to divide 512')
    accumulation = 512 // microbatch
    command = [PYTHONS[dataset], '-m', 'torch.distributed.run',
               '--nproc_per_node=2', f'--master_port={master_port}',
               'dinov3/train/train.py', '--config-file', str(BASE_CONFIG),
               '--output-dir', str(output), '--seed', str(seed),
               'train.seed=' + str(seed), 'train.dataset_path=' + dataset_path,
               f'train.batch_size_per_gpu={microbatch}',
               f'optim.gradient_accumulation_steps={accumulation}',
               'train.max_updates=80000', 'checkpointing.period=500',
               'checkpointing.max_to_keep=null']
    if dataset == '1pb':
        command.append('train.num_workers=1')
    plan = dict(dataset=dataset, seed=seed, gpu_pair=gpu_pair,
                microbatch_per_gpu=microbatch, gradient_accumulation_steps=accumulation,
                effective_global_batch=1024, target_updates=80000,
                initialization=str(BASE_CONFIG), source=dataset_path,
                frozen_assignment=str(assignment) if assignment else None,
                command=command, host=socket.gethostname())
    plan_path = output / 'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous['command'] != command:
            batch_options = ('train.batch_size_per_gpu=', 'optim.gradient_accumulation_steps=')
            without_batch = lambda items: [item for item in items if not item.startswith(batch_options)]
            if without_batch(previous['command']) != without_batch(command):
                raise ValueError(f'Existing run differs beyond batch configuration: {output}')
            checkpoints = [int(path.name) for path in (output / 'ckpt').glob('[0-9]*')
                           if path.name.isdigit() and (path / 'checkpoint.pth').is_file()]
            if not checkpoints:
                raise ValueError(f'Cannot change batch before first complete checkpoint: {output}')
            transition_step = max(checkpoints)
            archive = output / f'plan_before_batch_change_{transition_step}.json'
            if archive.exists():
                raise ValueError(f'Batch transition already recorded at {transition_step}: {output}')
            if dataset == '1pb':
                repack_finite_audits(output, transition_step, previous['command'], microbatch)
            archive.write_text(json.dumps(previous, indent=2) + '\n')
            resolved = output / 'config.yaml'
            if resolved.is_file():
                shutil.copy2(resolved, output / f'config_before_batch_change_{transition_step}.yaml')
            plan['configuration_history'] = previous.get('configuration_history', []) + [
                dict(checkpoint=transition_step, changed_unix=time.time(),
                     prior_microbatch=previous.get('microbatch_per_gpu'),
                     new_microbatch=microbatch, prior_command=previous['command'])]
        else:
            plan['configuration_history'] = previous.get('configuration_history', [])
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_pair,
               PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True',
               MALLOC_MMAP_THRESHOLD_='131072', MALLOC_ARENA_MAX='4')
    if assignment:
        env['DQ_FINITE_SHARD_ASSIGNMENT'] = str(assignment)
        env['DQ_FINITE_FAST_SKIP_AUDIT_DIR'] = str(output)
        env['DQ_FINITE_RESUME_SAMPLES_PER_SHARD'] = '1952'
        env['DQ_FINITE_RESUME_SLACK_SAMPLES'] = '50000'
    else:
        env['DQ_STREAM_FAILURE_LOG_DIR'] = str(output)
    status_path = output / 'status.json'
    failures = 0
    prior_step = -1
    while True:
        checkpoints = [int(p.name) for p in (output / 'ckpt').glob('[0-9]*')
                       if p.name.isdigit() and (p / 'checkpoint.pth').is_file()]
        latest = max(checkpoints, default=-1)
        if latest >= 79999:
            status = 'COMPLETE'
            code = 0
            break
        if latest > prior_step:
            failures = 0
            prior_step = latest
        status_path.write_text(json.dumps(dict(status='RUNNING', seed=seed,
                                               latest_checkpoint=latest,
                                               updated_unix=time.time()), indent=2) + '\n')
        with (output / 'console.log').open('a', buffering=1) as log:
            log.write(f'LAUNCH seed={seed} latest_checkpoint={latest} time={time.time()}\n')
            code = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code == 0:
            checkpoints = [int(p.name) for p in (output / 'ckpt').glob('[0-9]*')
                           if p.name.isdigit() and (p / 'checkpoint.pth').is_file()]
            latest = max(checkpoints, default=-1)
            if latest >= 79999:
                status = 'COMPLETE'
                break
            code = 1
        failures += 1
        if failures >= 3:
            status = 'FAILED_AFTER_THREE_ATTEMPTS'
            break
        time.sleep(120)
    status_path.write_text(json.dumps(dict(status=status, returncode=code,
                                           latest_checkpoint=latest,
                                           updated_unix=time.time()), indent=2) + '\n')
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('1pb', '100tb'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--gpu-pair', required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--microbatch-per-gpu', type=int, choices=(32, 64, 128, 256, 512))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    dataset_path, assignment = source(args.dataset)
    if not BASE_CONFIG.is_file() or not Path(PYTHONS[args.dataset]).is_file():
        raise FileNotFoundError('Training configuration or environment is unavailable')
    if args.preflight:
        microbatch = args.microbatch_per_gpu or (64 if args.dataset == '100tb' else 32)
        print(json.dumps(dict(dataset=args.dataset, seeds=args.seeds,
                              data=dataset_path, assignment=str(assignment),
                              gpu_pair=args.gpu_pair, microbatch_per_gpu=microbatch,
                              gradient_accumulation_steps=512 // microbatch,
                              effective_global_batch=1024), indent=2))
        return
    for seed in args.seeds:
        if run(args.dataset, seed, args.gpu_pair, args.master_port, args.microbatch_per_gpu):
            raise SystemExit(f'{args.dataset} seed {seed} stopped after repeated failures')


if __name__ == '__main__':
    main()
