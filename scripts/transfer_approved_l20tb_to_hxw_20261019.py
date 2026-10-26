#!/usr/bin/env python3
"""Transfer only the three L assets approved by the user on 2026-10-19."""

import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


ROOT = Path('/mnt/huawei_deepcad/dinov3')
REPORT = ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018'
STATE = REPORT / 'approved_l_transfer_state.json'
HOST = '5090-hxw-xzj'
MERGED = '/home/xzj/storage/merged'
COMPLEMENT = f'{MERGED}/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019'
DOMAIN = f'{MERGED}/l20tb_domain_recoverability_from47823_len19520_hxw_20261019'
RUNS = ROOT / 'outputs/01_training_runs'
ASSETS = (
    ('bio5tb_teacher',
     RUNS / 'HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/eval/training_19519/teacher_checkpoint.pth',
     f'{COMPLEMENT}/teachers/bio5tb_19519.pth', 1402008175,
     'd3ac51352dff7f387f62aa9c03ea087937ff6cfebcc58210287afb9d08899c10'),
    ('complement_49775',
     RUNS / 'l20tb_complement_bio5tb_from47823_len19520_qi_20261018/ckpt/49775/checkpoint.pth',
     f'{COMPLEMENT}/ckpt/49775/checkpoint.pth', 7118413459,
     'e16c1394b4ac97f6ec1e962bb1c14f29a82df53d116756066e1adbf81d528121'),
    ('domain_52703',
     RUNS / 'l20tb_domain_recoverability_from47823_len19520_20261018/ckpt/52703/checkpoint.pth',
     f'{DOMAIN}/ckpt/52703/checkpoint.pth', 7067957195,
     '3b3505f83d2f1a4b2a5ae9b3950a65ec26b3792530905845944c68364ef168cf'),
)
STATE_LOCK = threading.Lock()


def _save_unlocked(state):
    state.update(pid=os.getpid(), updated_unix=time.time())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def save(state):
    with STATE_LOCK:
        _save_unlocked(state)


def update_asset(state, name, **fields):
    with STATE_LOCK:
        state['assets'].setdefault(name, {}).update(fields)
        _save_unlocked(state)


def remote(*args, timeout=300):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           HOST, shlex.join(args)], check=True, capture_output=True,
                          text=True, timeout=timeout).stdout.strip()


def remote_digest(path):
    output = remote('sha256sum', path, timeout=1800)
    return output.split()[0]


def transfer(name, source, destination, size, expected, state):
    if source.stat().st_size != size:
        raise RuntimeError(f'Source size changed: {source}')
    remote('mkdir', '-p', str(Path(destination).parent))
    for attempt in range(1, 6):
        update_asset(state, name, status='TRANSFERRING', source=str(source),
                     destination=destination, expected_bytes=size, attempt=attempt)
        try:
            result = subprocess.run(['rsync', '-a', '--partial', '--append-verify',
                                     '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                     str(source), f'{HOST}:{destination}'],
                                    capture_output=True, text=True, timeout=12 * 3600)
        except subprocess.TimeoutExpired as exc:
            update_asset(state, name, status='RETRY', error=repr(exc))
            time.sleep(min(60 * attempt, 300))
            continue
        if result.returncode == 0:
            actual_size = int(remote('stat', '-c', '%s', destination))
            actual_hash = remote_digest(destination)
            if actual_size != size or actual_hash != expected:
                update_asset(state, name, status='HASH_MISMATCH',
                             actual_bytes=actual_size, actual_sha256=actual_hash)
                raise RuntimeError(f'Hash mismatch for {name}')
            update_asset(state, name, status='VERIFIED', sha256=actual_hash,
                         verified_unix=time.time())
            return
        update_asset(state, name, status='RETRY', error=result.stderr[-1200:])
        time.sleep(min(60 * attempt, 300))
    raise RuntimeError(f'Five transfer attempts failed for {name}')


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'approved_l_transfer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {
            'approval': 'User approved exactly the two L full optimizer checkpoints and one 5TB teacher on 2026-10-19',
            'assets': {},
        }
        pending = [asset for asset in ASSETS
                   if state['assets'].get(asset[0], {}).get('status') != 'VERIFIED']
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(transfer, *asset, state) for asset in pending]
            for future in as_completed(futures):
                future.result()
        state['status'] = 'COMPLETE'
        save(state)


if __name__ == '__main__':
    main()
