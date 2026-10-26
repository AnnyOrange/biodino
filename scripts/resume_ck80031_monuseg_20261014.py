#!/usr/bin/env python3
"""Retry the frozen ck80031 MoNuSeg task after its source snapshot is restored."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/v2_full_v4_ck80031_20261008')
KEY = 'cls_slow2_20tb_ck80031__segmentation__monuseg__primary-last__formal-static-v1'
SOURCE = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')


def save(path, value):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    task = json.loads((ROOT / 'tasks' / f'{KEY}.json').read_text())
    pinned = json.loads((SOURCE / 'source_snapshot.json').read_text())
    assert pinned == task['monuseg_source']
    for name, expected in pinned['files'].items():
        assert sha(SOURCE / name) == expected, name
    for name, expected in task['monuseg_spec']['source_hashes'].items():
        assert sha(Path(name)) == expected, name

    claim = ROOT / 'claims' / KEY
    old = json.loads((claim / 'status.json').read_text())
    assert old['state'] == 'FAILED' and 'No such file or directory' in old['reason']
    history = ROOT / 'claims' / 'retry_history' / '20261014'
    history.mkdir(parents=True, exist_ok=True)
    shutil.copytree(claim, history / KEY, dirs_exist_ok=False)

    spec = importlib.util.spec_from_file_location('monufleet', SOURCE / 'scripts/run_retest_fleet_20260918.py')
    fleet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fleet)
    cell = ROOT / 'cells' / KEY
    command = fleet.command(dict(key=KEY, asset=task['asset'], dataset=task['monuseg_spec']),
                            cell, '/mnt/huawei_deepcad/benchmark')
    invocation = json.loads((claim / 'invocation.json').read_text())
    invocation.update(host=os.uname().nodename, gpu=0, started_unix=time.time(), command=command)
    save(claim / 'invocation.json', invocation)
    state = dict(state='RUNNING', host=os.uname().nodename, gpu=0, task=KEY,
                 time=time.time(), pid=os.getpid())
    save(claim / 'status.json', state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(SOURCE),
               DINOV3_ROOT=str(SOURCE), DINOV3_CODE_ROOT=str(SOURCE),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1')
    try:
        result = subprocess.run(command, cwd=SOURCE, env=env)
        if result.returncode:
            raise RuntimeError(f'evaluator exit {result.returncode}')
        report = fleet.validate(dict(key=KEY, asset=task['asset'], dataset=task['monuseg_spec']),
                                cell, invocation)
        assert report['status'] == 'VALID_COMPLETE'
        state.update(state='VALID_COMPLETE', end=time.time())
    except Exception as exc:
        state.update(state='FAILED', reason=str(exc), end=time.time())
        save(claim / 'status.json', state)
        raise
    save(claim / 'status.json', state)


if __name__ == '__main__':
    main()
