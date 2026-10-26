#!/usr/bin/env python3
"""Pause 7B at a complete checkpoint and start the two approved L resumes."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from transfer_approved_l20tb_to_hxw_20261019 import (
    COMPLEMENT, DOMAIN, HOST, REPORT, STATE as TRANSFER_STATE,
)


STATE = REPORT / 'approved_l_activation_state.json'
SNAPSHOT = '/home/xzj/biodino_hxw_sync_20261019'
PYTHON = '/home/xzj/miniconda3/envs/dinov3/bin/python'
PAUSE = f'{SNAPSHOT}/scripts/pause_hxw_7b_for_l_20261019.py'
SEVEN_B_RUN = '/home/xzj/storage/merged/vit7b_20tb_hs6_nogram_b1024_from249_20261019'
PREPARE = f'{SNAPSHOT}/scripts/prepare_hxw_l20tb_resume_20261019.py'
SUPERVISOR = f'{SNAPSHOT}/scripts/launch_l20tb_domain_20261018.py'


def save(state):
    state.update(pid=os.getpid(), updated_unix=time.time())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def remote(args, timeout=900):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             HOST, shlex.join(map(str, args))], capture_output=True,
                            text=True, check=True, timeout=timeout)
    return result.stdout.strip()


def ready(*names):
    if not TRANSFER_STATE.exists():
        return False
    assets = json.loads(TRANSFER_STATE.read_text()).get('assets', {})
    return all(assets.get(name, {}).get('status') == 'VERIFIED' for name in names)


def pause_7b(state):
    if state.get('seven_b_paused'):
        return
    if 'minimum_7b_checkpoint' not in state:
        info = json.loads(remote([PYTHON, PAUSE, '--run', SEVEN_B_RUN]))
        checkpoint = info['checkpoint']
        recent = (time.time() - checkpoint['metadata_mtime'] < 300 or
                  not (info['supervisor_active'] or info['train_active']))
        state['minimum_7b_checkpoint'] = checkpoint['step'] if recent else checkpoint['step'] + 1
        state['status'] = 'WAITING_FOR_FRESH_7B_CHECKPOINT'
        save(state)
    info = json.loads(remote([PYTHON, PAUSE, '--run', SEVEN_B_RUN, '--min-step',
                              str(state['minimum_7b_checkpoint']), '--pause'], timeout=300))
    if not info['ready']:
        return
    if info['supervisor_active'] or info['train_active']:
        raise RuntimeError('7B remained active after the pause request')
    state.update(seven_b_paused=True, seven_b_checkpoint=info['checkpoint'],
                 status='7B_PAUSED_FOR_L')
    save(state)


def start_branch(state, key, output, source_step, gpus, port, bio5tb=None):
    if state.get(key) == 'LAUNCHED':
        return
    config_args = [PYTHON, PREPARE, '--source-config', f'{output}/source_candidate.yaml',
                   '--output-dir', output, '--resume-step', str(source_step)]
    if bio5tb:
        config_args += ['--bio5tb', bio5tb]
    state.update(status=f'PREPARING_{key.upper()}')
    save(state)
    remote(config_args, timeout=300)
    command = [PYTHON, '-u', SUPERVISOR, '--snapshot', SNAPSHOT, '--output', output,
               '--python', PYTHON, '--gpus', gpus, '--port', str(port),
               '--config', f'{output}/candidate_hxw.yaml']
    logfile = f'{output}/supervisor.log'
    shell = f'nohup setsid {shlex.join(command)} > {shlex.quote(logfile)} 2>&1 < /dev/null & echo $!'
    pid = int(subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, shell], check=True,
                             capture_output=True, text=True, timeout=30).stdout.strip())
    time.sleep(5)
    remote(['kill', '-0', str(pid)], timeout=30)
    state.update({key: 'LAUNCHED', f'{key}_supervisor_pid': pid,
                  f'{key}_output': output, 'status': f'{key.upper()}_LAUNCHED'})
    save(state)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'approved_l_activation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {
            'approval': 'User approved the three L source assets and L priority over 7B on 2026-10-19',
            'status': 'WAITING_FOR_COMPLEMENT_ASSETS',
        }
        while True:
            try:
                if ready('bio5tb_teacher', 'complement_49775'):
                    pause_7b(state)
                    if state.get('seven_b_paused'):
                        start_branch(state, 'complement', COMPLEMENT, 49775, '4,5,6,7', 29725,
                                     f'{COMPLEMENT}/teachers/bio5tb_19519.pth')
                if state.get('seven_b_paused') and ready('domain_52703'):
                    start_branch(state, 'domain', DOMAIN, 52703, '0,1,2,3', 29726)
                if state.get('complement') == 'LAUNCHED' and state.get('domain') == 'LAUNCHED':
                    state['status'] = 'BOTH_L_SUPERVISORS_LAUNCHED'
                    save(state)
                    return
                save(state)
            except Exception as exc:
                state.update(status='RETRY', last_error=repr(exc),
                             failures=state.get('failures', 0) + 1)
                save(state)
            time.sleep(60)


if __name__ == '__main__':
    main()
