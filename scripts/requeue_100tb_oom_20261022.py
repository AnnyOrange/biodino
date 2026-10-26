#!/usr/bin/env python3
"""Archive verified, inactive 100TB V4 OOM claims so they can be retried."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=(41499, 41999), required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    shared = ROOT / f'campaigns/100tb_ck{args.step}/shared'
    state = shared / '_state'
    workers = json.loads((state / 'workers/client14.json').read_text())
    active = set(workers['own_active'])
    processes = subprocess.check_output(['ps', '-eo', 'args='], text=True)
    results = []
    for failure in sorted((state / 'failed_resource').glob('*.json')):
        key = failure.stem
        claim = state / 'claims' / key
        report = shared / 'cells' / key / 'validation_report.json'
        log = shared / 'cells' / key / 'run.log'
        if not claim.is_dir() or not report.is_file() or not log.is_file():
            continue
        owner = json.loads((claim / 'owner.json').read_text())
        validation = json.loads(report.read_text())
        oom = 'out of memory' in log.read_text(errors='replace').lower()
        task_running = any(key in line and 'dinov3.eval.' in line
                           for line in processes.splitlines())
        if (owner['host'] != 'client14' or validation.get('status') != 'FAILED'
                or not oom or key in active or task_running
                or (state / 'done' / f'{key}.json').exists()):
            raise RuntimeError(f'Unsafe to requeue {key}')
        destination = ROOT / 'claims_oom_20261022' / f'100tb_ck{args.step}' / key
        if args.apply:
            if destination.exists():
                raise FileExistsError(destination)
            destination.mkdir(parents=True)
            for source in (report, log, shared / 'cells' / key / 'failed_result.json'):
                if source.is_file():
                    shutil.copy2(source, destination / source.name)
            os.rename(claim, destination / 'claim')
            os.rename(failure, destination / failure.name)
            running = state / 'running' / f'{key}.json'
            if running.exists():
                os.rename(running, destination / 'running.json')
        results.append(key)
    print(json.dumps(dict(step=args.step, apply=args.apply, requeued=results,
                          time_unix=time.time())))


if __name__ == '__main__':
    main()
