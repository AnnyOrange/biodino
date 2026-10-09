#!/usr/bin/env python3
"""Run the cpu2-interrupted CHAMMI portion on qi after qi's first three queues."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3')
CAMP = ROOT / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
FIRST = ('bbbc048-cellcycle', 'chestmnist', 'octmnist')
NEXT = ('chammi-cp-task2', 'chammi-cp-task3', 'chammi-hpa-task1', 'chammi-hpa-task2')


def report(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def main() -> int:
    if not (CAMP / 'lc_v7_manifest.json').is_file():
        raise FileNotFoundError('Pinned fusion source manifest missing')
    for dataset in NEXT:
        for role in 'EML':
            log = CAMP / 'logs' / f'queue_{dataset}_{role}_{os.uname().nodename}.log'
            if log.exists():
                raise FileExistsError(f'Preexisting queue log needs audit: {log}')
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        statuses = []
        for dataset in FIRST:
            log = CAMP / 'logs' / f'queue_three_qi3090_{dataset}.log'
            tail = log.read_text()[-1600:] if log.exists() else ''
            if 'FAILED_PROTOCOL_OR_RESOURCE' in tail or 'FAILED_FUSION' in tail or 'QUEUE_TIME_LIMIT' in tail:
                report(f'STOP: first queue failed ({dataset}); manual audit required')
                return 1
            statuses.append('QUEUE_COMPLETED all assigned READY datasets' in tail)
        if all(statuses):
            break
        time.sleep(60)
    else:
        report('STOP: qi first three queues did not all complete within 12h')
        return 2
    report(f'first queues complete; now serial qi GPU0 continuation: {NEXT}')
    command = [sys.executable, '-B', str(ROOT / 'scripts/run_hs6_l5_coexistence_frozen_queue_20260922.py'),
               '--datasets', *NEXT, '--max-hours', '20']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    output = CAMP / 'logs/queue_three_qi3090_chammi_continuation.log'
    with output.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    report(f'continuation exited code={result.returncode}; log={output}')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
