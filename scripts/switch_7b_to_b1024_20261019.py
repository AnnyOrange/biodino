#!/usr/bin/env python3
"""Preserve the 288-batch run at its next checkpoint, then start a fresh 1024-batch baseline."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


REPORT = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/l20tb_domain_7b_baseline_20261018')
STATE = REPORT / 'seven_b_batch1024_switch_state.json'
HOST = '5090-hxw-xzj'
PYTHON = '/home/xzj/miniconda3/envs/dinov3/bin/python'
REPO = '/home/xzj/biodino_7b_20261015'
PAUSE = '/home/xzj/biodino_l20tb_domain_snapshot_20261018/scripts/pause_hxw_7b_for_l_20261019.py'
SUPERVISOR = f'{REPO}/scripts/resume_7b_20tb_20261018.py'
OLD = '/home/xzj/storage/merged/vit7b_20tb_recovered249_20261018'
NEW = '/home/xzj/storage/merged/vit7b_20tb_hs6_nogram_b1024_from249_20261019'
ORIGINAL_CONFIG = '/home/xzj/storage/merged/vit7b_20tb_route2_4x5090hxw_20261015/config.yaml'
TEACHER = ('/home/xzj/storage/merged/vit7b_20tb_route2_4x5090hxw_20261015'
           '/eval/training_249/sharded_teacher_checkpoint')


def save(state):
    state.update(pid=os.getpid(), updated_unix=time.time())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def remote(args, timeout=180):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           HOST, shlex.join(map(str, args))], check=True,
                          capture_output=True, text=True, timeout=timeout).stdout.strip()


def pause_old(state):
    if state.get('old_paused'):
        return True
    args = [PYTHON, PAUSE, '--run', OLD]
    info = json.loads(remote(args))
    min_step = 749 if info['supervisor_active'] or info['train_active'] else 624
    if info['checkpoint']['step'] < min_step or (info['saving'] and
                                                (info['supervisor_active'] or info['train_active'])):
        state.update(status='WAITING_FOR_OLD_CHECKPOINT', latest_old=info['checkpoint']['step'],
                     required_old=min_step)
        save(state)
        return False
    info = json.loads(remote(args + ['--min-step', str(min_step), '--pause',
                                     '--batch-change'], timeout=300))
    if not info['ready'] or info['supervisor_active'] or info['train_active']:
        raise RuntimeError('Old 7B training did not stop at a complete checkpoint')
    state.update(old_paused=True, old_checkpoint=info['checkpoint'],
                 status='OLD_288_BATCH_PRESERVED')
    save(state)
    return True


def launch_new(state):
    if state.get('new_supervisor_pid'):
        return
    marker = f'{REPO}/github_synced_b1024_20261019.json'
    sync = json.loads(remote(['cat', marker]))
    if not sync.get('verified') or sync.get('effective_global_batch') != 1024:
        raise RuntimeError('GitHub-sourced 7B code has not been verified')
    remote(['mkdir', '-p', NEW])
    command = [PYTHON, '-u', SUPERVISOR, '--repo', REPO,
               '--original-config', ORIGINAL_CONFIG, '--teacher', TEACHER,
               '--output', NEW, '--python', PYTHON, '--gpus', '0,1,2,3',
               '--batch-size', '8', '--accum-steps', '32', '--checkpoint-period', '25']
    logfile = f'{NEW}/supervisor.log'
    shell = f'nohup setsid {shlex.join(command)} > {shlex.quote(logfile)} 2>&1 < /dev/null & echo $!'
    pid = int(subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, shell], check=True,
                             capture_output=True, text=True, timeout=30).stdout.strip())
    time.sleep(5)
    remote(['kill', '-0', str(pid)])
    state.update(new_supervisor_pid=pid, new_output=NEW,
                 status='NEW_1024_BATCH_SUPERVISOR_STARTED')
    save(state)


def verify_new(state):
    recovery = json.loads(remote(['cat', f'{NEW}/recovery_state.json']))
    if recovery.get('global_batch') != 1024 or recovery.get('batch_size_per_gpu') != 8 or \
            recovery.get('accumulation_steps') != 32:
        raise RuntimeError('New 7B supervisor did not apply the required batch')
    if recovery.get('status') in ('STOPPED_AFTER_THREE_FAILURES', 'STOPPED_LOW_DISK'):
        raise RuntimeError(f"New 7B supervisor stopped: {recovery['status']}")
    state.update(status='NEW_1024_BATCH_CONFIRMED',
                 new_recovery_status=recovery.get('status'))
    save(state)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'seven_b_batch1024_switch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {
            'reason': 'User corrected the 7B baseline to effective batch 1024 on 2026-10-19',
            'status': 'WAITING_FOR_OLD_CHECKPOINT',
        }
        while True:
            try:
                if pause_old(state):
                    launch_new(state)
                    verify_new(state)
                    return
            except Exception as exc:
                state.update(status='RETRY', last_error=repr(exc),
                             failures=state.get('failures', 0) + 1)
                save(state)
            time.sleep(60)


if __name__ == '__main__':
    main()
