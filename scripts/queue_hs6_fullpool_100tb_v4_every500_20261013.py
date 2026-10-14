#!/usr/bin/env python3
"""Continuously evaluate every 500-update checkpoint from the 100TB full-pool run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


REPO = Path('/mnt/huawei_deepcad/dinov3')
DATA = Path('/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009')
RUN = REPO / 'outputs/01_training_runs/HS6_L_quality_long_100tb_fullpool_ddp_gb1024_mb128_ga2_40k_4xa100_deepcad_20261010'
CONTROL = REPO / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
BASE = CONTROL
EVAL_BASE = CONTROL / 'campaigns'
EVAL_PYTHON = sys.executable
EXISTING_7999 = REPO / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_ck7999_partial_20261012/status.json'
STEPS = tuple(range(499, 40_000, 500))
SOURCE_ARCHIVE = REPO / 'outputs/03_source_archives/100tb_fullpool_v4_20261013/source.tar.gz'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def register_campaign(host: str, gpus: list[int]) -> None:
    import run_shared_frozen_stage1_20260917 as source_queue
    manifest_path = CONTROL / 'campaign_manifest.json'
    snapshot = source_queue.current_source_snapshot()
    snapshot['git_commit'] = os.environ.get('DINOV3_GIT_COMMIT', 'UNKNOWN')
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(f'Missing canonical source archive: {SOURCE_ARCHIVE}')
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        archive_sha = sha256(SOURCE_ARCHIVE)
        if manifest.get('source_snapshot_sha256') != snapshot['sha256'] or \
                manifest.get('source_archive_sha256') != archive_sha:
            revisions = list(manifest.get('prelaunch_source_revisions', []))
            revisions.append(dict(
                source_snapshot_sha256=manifest.get('source_snapshot_sha256'),
                source_archive_sha256=manifest.get('source_archive_sha256'),
                status='PRELAUNCH_HASH_DRIFT_NO_GPU_WORKER_STARTED', recorded_unix=time.time()))
            manifest.update(source_snapshot=snapshot, source_snapshot_sha256=snapshot['sha256'],
                            source_archive=str(SOURCE_ARCHIVE), source_archive_sha256=archive_sha,
                            git_commit=snapshot['git_commit'], prelaunch_source_revisions=revisions)
    else:
        manifest = dict(
            protocol_id='bio-eval-union-v4',
            scope='HS6 100TB full-pool; checkpoints every 500 optimizer updates; 38 shared + 9 extension + MoNuSeg 30/7/14',
            train_run=str(RUN), train_target_optimizer_updates=40_000,
            checkpoint_steps=list(STEPS), existing_ck7999_report=str(EXISTING_7999),
            evaluation_root=str(EVAL_BASE), git_commit=snapshot['git_commit'],
            git_status='Worktree code and rules are fully hash-recorded; exact source archive is included below',
            source_snapshot=snapshot, source_snapshot_sha256=snapshot['sha256'],
            source_archive=str(SOURCE_ARCHIVE), source_archive_sha256=sha256(SOURCE_ARCHIVE),
            protocol_sha256=sha256(REPO / 'Evaluation Rules/protocol_v4.json'),
            environment_requirements={'torch':'2.10.0','torchvision':'0.25.0','numpy':'2.4.3',
                                      'scipy':'1.17.1','scikit-learn':'1.8.0','Pillow':'12.1.1'},
            workers={}, created_unix=time.time())
    import torch
    env = dict(hostname=platform.node(), gpus=gpus, python=sys.executable,
               python_version=platform.python_version(), torch=torch.__version__,
               torchvision=importlib.metadata.version('torchvision'),
               numpy=importlib.metadata.version('numpy'), scipy=importlib.metadata.version('scipy'),
               sklearn=importlib.metadata.version('scikit-learn'), pillow=importlib.metadata.version('Pillow'),
               transformers=importlib.util.find_spec('transformers') is not None,
               timm=importlib.util.find_spec('timm') is not None,
               safetensors=importlib.util.find_spec('safetensors') is not None,
               cuda_runtime=torch.version.cuda)
    manifest.setdefault('workers', {})[host] = env
    save(manifest_path, manifest)


def checkpoint_ready(step: int) -> bool:
    optimizer = RUN / f'ckpt/{step}/checkpoint.pth'
    metrics = RUN / 'raw_loss_metrics.jsonl'
    return optimizer.is_file() and optimizer.stat().st_size > 3_000_000_000 and metrics.is_file()


def completed(step: int) -> bool:
    if step == 7999 and EXISTING_7999.is_file():
        status = json.loads(EXISTING_7999.read_text())
        if status.get('shared_valid_complete') == status.get('shared_expected') == 38 and \
                status.get('extension_valid_complete') == status.get('extension_expected') == 9 and \
                status.get('monuseg') == 'VALID_COMPLETE':
            return True
    campaign = EVAL_BASE / f'100tb_ck{step}'
    status = campaign / 'driver_status.json'
    return status.is_file() and json.loads(status.read_text()).get('state') == 'COMPLETE'


def evaluate(step: int, gpus: list[int], target_per_gpu: int, host: str) -> None:
    campaign = EVAL_BASE / f'100tb_ck{step}'
    label = f'long_fullpool500_100tb_ck{step}'
    (BASE / 'logs').mkdir(parents=True, exist_ok=True)
    (CONTROL / 'logs').mkdir(parents=True, exist_ok=True)
    if completed(step):
        return
    if not checkpoint_ready(step):
        raise FileNotFoundError(f'Checkpoint {step} is not complete yet')
    teacher = RUN / f'eval/training_{step}/teacher_checkpoint.pth'
    if not teacher.is_file():
        command = [EVAL_PYTHON, '-u', str(REPO / 'scripts/export_hs6_fullpool_teacher_from_ckpt_20261013.py'),
                   '--run', str(RUN), '--step', str(step)]
        subprocess.run(command, cwd=REPO, check=True)
    if not (campaign / 'prepared.json').is_file():
        subprocess.run([EVAL_PYTHON, '-u', str(REPO / 'scripts/prepare_hs6_long_v4_id_20261009.py'),
                        '--arm', '100tb', '--step', str(step), '--train-root', str(RUN),
                        '--eval-base', str(EVAL_BASE), '--label-tag', 'fullpool500', '--allow-partial'],
                       cwd=REPO, check=True)
    status = campaign / 'driver_status.json'
    if status.is_file() and json.loads(status.read_text()).get('state') == 'COMPLETE':
        return
    command = [EVAL_PYTHON, '-u', str(REPO / 'scripts/run_data_quality_v4_id_20261002.py'),
               '--arm', label, '--eval-root', str(campaign), '--gpus', *map(str, gpus),
               '--target-per-gpu', str(target_per_gpu)]
    log = CONTROL / 'logs' / f'{host}_ck{step}.log'
    with log.open('a', buffering=1) as stream:
        stream.write(json.dumps(dict(event='START', checkpoint=step, host=host, gpus=gpus,
                                     target_per_gpu=target_per_gpu, time_unix=time.time())) + '\n')
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(json.dumps(dict(event='EXIT', checkpoint=step, returncode=result.returncode,
                                     time_unix=time.time())) + '\n')
    if result.returncode:
        raise RuntimeError(f'V4 driver exited {result.returncode}; see {log}')
    if not status.is_file() or json.loads(status.read_text()).get('state') != 'COMPLETE':
        raise RuntimeError(f'V4 campaign did not validate all phases: {campaign}')


def worker(args: argparse.Namespace) -> None:
    register_campaign(args.host, args.gpus)
    BASE.mkdir(parents=True, exist_ok=True)
    state = CONTROL / 'workers' / f'{args.host}.json'
    attempts: dict[int, int] = {}
    while True:
        did_work = False
        for step in STEPS:
            if completed(step) or not checkpoint_ready(step):
                continue
            claim = BASE / 'claims' / f'ck{step}'
            claim.parent.mkdir(parents=True, exist_ok=True)
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            did_work = True
            save(claim / 'owner.json', dict(host=args.host, pid=os.getpid(),
                                            checkpoint=step, time_unix=time.time()))
            save(state, dict(host=args.host, gpus=args.gpus, state='EVALUATING',
                             checkpoint=step, python=EVAL_PYTHON, time_unix=time.time()))
            try:
                if not completed(step):
                    evaluate(step, args.gpus, args.target_per_gpu, args.host)
                attempts.pop(step, None)
                save(state, dict(host=args.host, gpus=args.gpus, state='LAST_COMPLETE',
                                 checkpoint=step, time_unix=time.time()))
            except Exception as exc:
                attempts[step] = attempts.get(step, 0) + 1
                save(CONTROL / 'failures' / f'ck{step}_{args.host}_{attempts[step]}.json',
                     dict(host=args.host, checkpoint=step, attempt=attempts[step],
                          error=f'{type(exc).__name__}: {exc}', time_unix=time.time()))
                if attempts[step] >= 3:
                    save(state, dict(host=args.host, gpus=args.gpus, state='SKIPPING_AFTER_ERRORS',
                                     checkpoint=step, time_unix=time.time()))
                else:
                    time.sleep(300)
            finally:
                (claim / 'owner.json').unlink(missing_ok=True)
                claim.rmdir()
            break
        if not did_work:
            remaining = [step for step in STEPS if not completed(step)]
            save(state, dict(host=args.host, gpus=args.gpus, state='WAITING_FOR_CHECKPOINTS',
                             latest_optimizer_checkpoint=max((int(p.parent.name) for p in (RUN / 'ckpt').glob('*/checkpoint.pth')), default=None),
                             remaining=len(remaining), time_unix=time.time()))
            if args.exit_when_done and not remaining:
                return
            time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--host', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--target-per-gpu', type=int, default=5)
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--poll-seconds', type=int, default=180)
    parser.add_argument('--exit-when-done', action='store_true')
    args = parser.parse_args()
    global EVAL_PYTHON
    EVAL_PYTHON = args.python_bin
    if not args.worker:
        raise SystemExit('Run with --worker on an evaluation host')
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('GPU ids must be unique')
    worker(args)


if __name__ == '__main__':
    main()
