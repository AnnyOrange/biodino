#!/usr/bin/env python3
"""Record and validate the isolated ck80031 CTC execution on cpu18."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/v2_full_v4_ck80031_20261008')
KEY = 'ctc__native__cls_slow2_20tb_ck80031'
PID = int(sys.argv[1])


def save(path, value):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def running():
    status = Path(f'/proc/{PID}/stat')
    return status.exists() and status.read_text().split(') ')[1][0] != 'Z'


def main():
    task = json.loads((ROOT / 'tasks' / f'{KEY}.json').read_text())
    assert running(), f'CTC pid {PID} is not running'
    claim = ROOT / 'claims' / KEY
    claim.mkdir(exist_ok=False)
    command = ['/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python', '-u',
               '/mnt/huawei_deepcad/dinov3/scripts/run_method_native_ctc_canonical_v4_20261013.py',
               '--arm', task['arm'], '--checkpoint', task['asset']['path'],
               '--config', task['asset']['config'], '--step', task['asset']['checkpoint_id'],
               '--output', str(ROOT / 'ctc' / task['arm'])]
    save(claim / 'invocation.json', dict(task=task, command=command,
                                        host=os.uname().nodename, gpu=0, pid=PID,
                                        checkpoint=task['asset']['path'],
                                        config=task['asset']['config'],
                                        amendment='read-only Git ref lookup for py-ctcmetrics',
                                        monitored_unix=time.time()))
    state = dict(state='RUNNING', host=os.uname().nodename, gpu=0,
                 task=KEY, pid=PID, time=time.time())
    save(claim / 'status.json', state)
    while running():
        time.sleep(30)

    sys.path.insert(0, '/mnt/huawei_deepcad/dinov3/scripts')
    import run_method_v4_fleet_20260928 as validator
    try:
        valid, reason = validator.validate(task)
        if not valid:
            raise RuntimeError(reason)
        state.update(state='VALID_COMPLETE', reason=reason, end=time.time())
    except Exception as exc:
        state.update(state='FAILED', reason=str(exc), end=time.time())
        save(claim / 'status.json', state)
        raise
    save(claim / 'status.json', state)


if __name__ == '__main__':
    main()
