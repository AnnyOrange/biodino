#!/usr/bin/env python3
"""Run full-v4 extensions after paired components settle and keep report files current."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
PAIRED = REPO / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
FULL = REPO / 'outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012'
PY = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
SINGLE = ('cpu1', 'cpu2', 'cpu5', 'cpu8', 'cpu9', 'cpu10', 'cpu11', 'cpu12', 'cpu15')
LOCAL_5090 = '5090-zxr'


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def terminal_component_tasks():
    manifest = load(PAIRED / 'campaign_manifest.json')
    done = PAIRED / '_state/done'
    failed = PAIRED / '_state/failed_resource'
    for task in manifest.get('tasks', []):
        key = task['key']
        if (done / f'{key}.json').exists() or (failed / f'{key}.json').exists():
            continue
        claim = load(PAIRED / '_state/claims' / key / 'status.json')
        if claim.get('state') in ('FAILED', 'INVALID_PROTOCOL'):
            continue
        return False
    return not any((PAIRED / '_state/running').glob('*.json'))


def full_tasks_terminal():
    for task_path in (FULL / 'tasks').glob('*.json'):
        task = load(task_path)
        state = load(FULL / 'claims' / task_path.stem / 'status.json').get('state')
        if state not in ('VALID_COMPLETE', 'FAILED'):
            return False
    return True


def launch_workers():
    hosts = [(host, [0], 2) for host in SINGLE] + [(LOCAL_5090, list(range(8)), 48)]
    for host, gpus, jobs in hosts:
        log = FULL / 'logs' / f'{host}.worker.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        args = [PY, str(REPO / 'scripts/prepare_20tb_resampling_cls_v4_20261012.py'),
                'worker-full', '--host', host, '--gpus', *map(str, gpus),
                '--max-jobs', str(jobs), '--ram-reserve', '96']
        command = ' '.join(__import__('shlex').quote(x) for x in args)
        if host == LOCAL_5090:
            process = subprocess.Popen(args, cwd=REPO, stdout=log.open('a'),
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(host=host, gpus=gpus, pid=str(process.pid),
                          returncode=None, stderr='', started=time.time(), placement='local')
        else:
            remote = (f'cd {REPO}; mkdir -p {FULL}/logs; '
                      f'nohup {command} > {log} 2>&1 < /dev/null & echo $!')
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, remote],
                                    text=True, capture_output=True, timeout=20)
            record = dict(host=host, gpus=gpus, pid=result.stdout.strip(),
                          returncode=result.returncode, stderr=result.stderr[-1000:], started=time.time())
        (FULL / 'workers').mkdir(exist_ok=True)
        (FULL / 'workers' / f'{host}.launcher.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    log = FULL / 'watcher.log'
    FULL.mkdir(parents=True, exist_ok=True)
    launched = (FULL / 'FULL_V4_WORKERS_LAUNCHED.json').exists()
    with log.open('a') as stream:
        stream.write(f'{time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())} watcher_start\n')
        while True:
            try:
                subprocess.run([sys.executable, str(REPO / 'scripts/summarize_20tb_resampling_cls_v4_20261012.py')],
                               cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            except Exception as exc:
                stream.write(f'report_update_failed {exc!r}\n'); stream.flush()
            if not launched and terminal_component_tasks():
                launch_workers()
                marker = dict(launched_unix=time.time(), paired_tasks_settled=True,
                              hosts=list(SINGLE) + [LOCAL_5090], full_tasks=48,
                              monuseg='BLOCKED_MISSING_FROZEN_20260929_SOURCE_SNAPSHOT')
                (FULL / 'FULL_V4_WORKERS_LAUNCHED.json').write_text(json.dumps(marker, indent=2) + '\n')
                launched = True
                stream.write('full_v4_workers_launched\n'); stream.flush()
            if launched and full_tasks_terminal():
                try:
                    subprocess.run([sys.executable, str(REPO / 'scripts/summarize_20tb_resampling_cls_v4_20261012.py')],
                                   cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
                    stream.write('all_queued_full_v4_tasks_terminal\n'); stream.flush()
                except Exception as exc:
                    stream.write(f'final_report_update_failed {exc!r}\n'); stream.flush()
                return
            time.sleep(180)


if __name__ == '__main__':
    main()
