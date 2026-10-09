#!/usr/bin/env python3
"""Run one locked 37-pool MoNuSeg E20/E50 comparison cell."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/02_eval_runs/monuseg37_comparison_20260929'
TASKS = ROOT / 'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923/tasks'
LOCK = ROOT / 'outputs/02_eval_inputs/monuseg37_v4_20260929/manifest.json'
EXPECTED = '743d8fe6a7dec80dcc0c699cfc173349f83328c3a9fda8a9422fcf6727c66f68'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if hashlib.sha256(LOCK.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('MoNuSeg37 identity lock changed')
    task = json.loads((TASKS / f'seg_primary__monuseg__{args.arm}.json').read_text())
    command = task['cmd'][:]
    destination = BASE / args.arm
    output = destination / 'results'
    cache = destination / 'cache'
    for flag, value in [('--output-root', str(output)), ('--cache-root', str(cache)),
                        ('--run-name', f'{args.arm}_monuseg37_locked')]:
        command[command.index(flag)+1] = value
    command[command.index('--gpu')+1] = str(args.gpu)
    assert command[command.index('--probe-epoch-grid')+1:command.index('--probe-seeds')] == ['20','50']
    assert command[command.index('--probe-seeds')+1:command.index('--probe-batch-size')] == ['0','1','2']
    assert command[command.index('--probe-batch-size')+1] == '32'
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), DINOV3_MONUSEG_LEGACY='1',
               PYTHONPATH=task['pythonpath'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    info = dict(protocol='bio-eval-union-v4/monuseg37-comparison-20260929',
                lock_sha256=EXPECTED, arm=args.arm, gpu=args.gpu, command=command,
                start_utc=now(), source_task=str(TASKS / f'seg_primary__monuseg__{args.arm}.json'))
    (destination/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    (destination/'status.json').write_text(json.dumps(dict(state='RUNNING',start=now()),indent=2)+'\n')
    with (destination/'run.log').open('a') as log:
        result = subprocess.run(command,cwd=task['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    paths = list(output.glob(f'*/budget*/seed*/monuseg/*/results.json'))
    accepted = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = data['_meta']
        if meta.get('used_train_samples') == 30 and meta.get('probe_batch_size') == 32:
            accepted.append((meta.get('probe_epochs'), meta.get('seed'), path))
    expected = {(budget,seed) for budget in (20,50) for seed in (0,1,2)}
    good = result.returncode == 0 and {(b,s) for b,s,_ in accepted} == expected
    status = dict(state='DONE' if good else 'FAILED',end=now(),returncode=result.returncode,
                  accepted_results=[str(p) for _,_,p in accepted])
    (destination/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
    if not good:
        sys.exit(1)


if __name__ == '__main__':
    main()
