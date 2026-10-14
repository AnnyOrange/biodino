#!/usr/bin/env python3
"""Add and run the exact MoNuSeg 30/7/14 V4 cells for the 20TB checkpoints."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
PAIRED = REPO / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
FULL = REPO / 'outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012'
MONU = REPO / 'outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929'
SOURCE = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')
BENCHMARK = Path('/mnt/huawei_deepcad/benchmark')
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
GPU = 4
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs() -> tuple[dict, dict]:
    campaign = json.loads((MONU / 'campaign_manifest.json').read_text())
    source = json.loads((SOURCE / 'source_snapshot.json').read_text())
    if source != campaign['source_snapshot']:
        raise RuntimeError('Frozen MoNuSeg evaluator snapshot differs from the main campaign')
    for rel, expected in source['files'].items():
        path = SOURCE / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Frozen evaluator source hash mismatch: {path}')
    dataset = next(d for d in campaign['datasets'] if d['dataset'] == 'monuseg')
    if dataset['counts'] != dict(train=30, val=7, test=14):
        raise RuntimeError(f'Unexpected MoNuSeg split counts: {dataset["counts"]}')
    for path, expected in dataset['source_hashes'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f'MoNuSeg source hash mismatch: {path}')
    return campaign, dataset


def prepare() -> None:
    source_campaign, dataset = verify_inputs()
    paired = json.loads((PAIRED / 'campaign_manifest.json').read_text())
    if len(paired['checkpoint_assets']) != 4:
        raise RuntimeError('Expected exactly four campaign checkpoints')
    (FULL / 'tasks').mkdir(parents=True, exist_ok=True)
    (FULL / 'claims').mkdir(parents=True, exist_ok=True)
    (FULL / 'cells').mkdir(parents=True, exist_ok=True)
    tasks = []
    for asset in paired['checkpoint_assets']:
        for key in ('path', 'config'):
            if not Path(asset[key]).is_file():
                raise FileNotFoundError(asset[key])
        arm = f"{asset['arm']}_ck{asset['checkpoint_id']}"
        task_id = f'{arm}__segmentation__monuseg__primary-last__formal-static-v1'
        task = dict(id=task_id, arm=arm, family='segmentation', dataset='monuseg',
                    priority=4, order=0, asset=asset, monuseg_spec=dataset,
                    monuseg_source=source_campaign['source_snapshot'],
                    expected_memory_mib=18000, host='5090-zxr', gpu=GPU)
        claim = FULL / 'claims' / task_id
        claim.mkdir(exist_ok=True)
        claim_status = claim / 'status.json'
        if not claim_status.exists():
            save(claim_status, dict(state='RESERVED_FOR_EXACT_MONUSEG', host='5090-zxr',
                                    gpu=GPU, task=task_id, time=time.time()))
        target = FULL / 'tasks' / f'{task_id}.json'
        if target.exists() and json.loads(target.read_text()) != task:
            raise RuntimeError(f'Existing task differs: {target}')
        save(target, task)
        tasks.append(task)
    manifest_path = FULL / 'campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['extension_executions'] = len(list((FULL / 'tasks').glob('*.json')))
    manifest['monuseg'] = 'QUEUED_EXACT_30_7_14'
    manifest['monuseg_protocol'] = dict(
        counts=dataset['counts'], split_identity_sha256=dataset['split_identity_sha256'],
        split_protocol_id=dataset['split_protocol_id'],
        source_snapshot_sha256=source_campaign['source_snapshot']['sha256'],
        evaluator_files=len(source_campaign['source_snapshot']['files']),
        task_ids=[t['id'] for t in tasks])
    manifest['runtime_hashes'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    save(manifest_path, manifest)
    marker = FULL / 'FULL_V4_WORKERS_LAUNCHED.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        record['full_tasks'] = manifest['extension_executions']
        record['monuseg'] = 'QUEUED_EXACT_30_7_14'
        save(marker, record)
    print(json.dumps(dict(tasks=[t['id'] for t in tasks], counts=dataset['counts'],
                          split_identity=dataset['split_identity_sha256']), indent=2))


def worker(task_ids: set[str] | None = None, gpu: int = GPU) -> None:
    campaign, dataset = verify_inputs()
    os.environ.update(THREADS)
    os.environ['DINOV3_CODE_ROOT'] = str(SOURCE)
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'scripts')]
    fleet = load_module('exact_monu_fleet_20260929', SOURCE / 'scripts/run_retest_fleet_20260918.py')
    fill = load_module('full_v4_fill', REPO / 'scripts/run_v2_progress_fleet_fill_20261006.py')
    resource = load_module('resource_monitor', REPO / 'scripts/run_deepcad_method_20260927.py')
    paired = json.loads((PAIRED / 'campaign_manifest.json').read_text())
    expected = paired['numerical_environment']
    import importlib.metadata
    for package, version in expected.items():
        if importlib.metadata.version(package).split('+')[0] != version:
            raise RuntimeError(f'Numerical environment mismatch: {package}')
    tasks = [json.loads(p.read_text()) for p in sorted((FULL / 'tasks').glob('*__segmentation__monuseg__*.json'))]
    if len(tasks) != 4:
        raise RuntimeError(f'Expected four MoNuSeg tasks, found {len(tasks)}')
    lock = (FULL / f'workers/monuseg_exact_5090-zxr_gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cards = resource.gpu_info()
    if cards[gpu]['total'] - cards[gpu]['used'] < 20000:
        raise RuntimeError(f'GPU {gpu} lacks 20 GiB free: {cards[gpu]}')
    for task in tasks:
        key, asset = task['id'], task['asset']
        if task_ids is not None and key not in task_ids:
            continue
        status_path = FULL / 'claims' / key / 'status.json'
        state = json.loads(status_path.read_text())
        if state.get('state') == 'VALID_COMPLETE':
            continue
        if state.get('state') not in ('RESERVED_FOR_EXACT_MONUSEG', 'INTERRUPTED', 'FAILED'):
            raise RuntimeError(f'Task is owned by another worker: {key}: {state}')
        cell = FULL / 'cells' / key
        cell.mkdir(parents=True, exist_ok=True)
        record = fill.checkpoint_record(PAIRED, asset)
        invocation = dict(checkpoint=record, git_commit=campaign['source_snapshot']['git_commit'],
                          source_snapshot_sha256=campaign['source_snapshot']['sha256'],
                          numerical_environment=expected, host='5090-zxr', gpu=gpu,
                          started_unix=time.time(), batch_size=32)
        fleet_task = dict(key=key, asset=asset, dataset=dataset)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SOURCE),
                   DINOV3_ROOT=str(SOURCE), DINOV3_CODE_ROOT=str(SOURCE))
        command = fleet.command(fleet_task, cell, str(BENCHMARK))
        invocation.update(command=command, task=task)
        save(cell / 'invocation_manifest.json', invocation)
        save(status_path, dict(state='RUNNING', host='5090-zxr', gpu=gpu,
                               task=key, pid=None, time=time.time()))
        log_path = FULL / 'logs' / f'{key}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            proc = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            save(status_path, dict(state='RUNNING', host='5090-zxr', gpu=gpu,
                                   task=key, pid=proc.pid, time=time.time()))
            rc = proc.wait()
        if rc:
            save(status_path, dict(state='FAILED', host='5090-zxr', gpu=gpu,
                                   task=key, returncode=rc, end=time.time()))
            raise RuntimeError(f'{key} exited {rc}; see {log_path}')
        result = fleet.validate(fleet_task, cell, invocation)
        if result.get('status') != 'VALID_COMPLETE':
            raise RuntimeError(f'{key} failed validation: {result}')
        save(status_path, dict(state='VALID_COMPLETE', host='5090-zxr', gpu=gpu,
                               task=key, returncode=0, validation_report=str(cell / 'validation_report.json'),
                               end=time.time()))
        print('VALID_COMPLETE', key, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'worker'))
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--task-id', action='append')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        worker(set(args.task_id) if args.task_id else None, args.gpu)
