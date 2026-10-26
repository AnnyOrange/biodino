#!/usr/bin/env python3
"""Resume the 100TB full-pool DDP run after an unexpected trainer exit."""

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3')
RUN = ROOT / 'outputs/01_training_runs/HS6_L_quality_long_100tb_fullpool_ddp_gb1024_mb128_ga2_40k_4xa100_deepcad_20261010'
EVAL = ROOT / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
EXISTING_7999 = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_every500_v4_20261019/ck7999/status.json'
PYTHON = '/home/deepcad/anaconda3/envs/dinov3/bin/python'
COMMAND = [
    PYTHON, '-u', str(ROOT / 'scripts/launch_hs6_long_100tb_deepcad4_40k_20261010.py'),
    '--resume', '--full-pool', '--target-updates', '80000',
    '--full-pool-microbatch', '128', '--gpu-group', '0,3,4,7',
    '--master-port', '31883',
]


def trainer_running():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if (b'torch.distributed.run' in arguments
                and b'--master_port=31883' in arguments
                and str(RUN).encode() in arguments):
            return True
    return False


def completed_updates():
    checkpoint = RUN / 'ckpt/79999/checkpoint.pth'
    if checkpoint.is_file() and checkpoint.stat().st_size > 3_000_000_000:
        return 80000
    metrics = RUN / 'raw_loss_metrics.jsonl'
    with metrics.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(max(0, end - 4096))
        lines = stream.read().splitlines()
    return json.loads(lines[-1])['optimizer_updates_completed'] if lines else 0


def old_evaluations_complete():
    for step in range(499, 41_000, 500):
        if step == 7999:
            path = EXISTING_7999
            expected = lambda status: (status.get('shared_valid_complete') == 38
                                       and status.get('extension_valid_complete') == 9
                                       and status.get('monuseg') == 'VALID_COMPLETE')
        else:
            path = EVAL / f'campaigns/100tb_ck{step}/driver_status.json'
            expected = lambda status: status.get('state') == 'COMPLETE'
        try:
            if not expected(json.loads(path.read_text())):
                return False
        except (OSError, ValueError):
            return False
    return True


def train_cards_idle():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.splitlines()}
    return all(used[gpu] < 1024 for gpu in (0, 3, 4, 7))


def record(event, **details):
    row = {'time_unix': time.time(), 'event': event, **details}
    with (RUN / 'watch_80k_events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def main():
    with (RUN / 'watch_80k.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        missing_checks = 0
        waiting_for_eval = False
        while True:
            updates = completed_updates()
            if updates >= 80000:
                record('COMPLETE', updates=updates)
                return
            if trainer_running():
                missing_checks = 0
                time.sleep(300)
                continue
            if not old_evaluations_complete():
                if not waiting_for_eval:
                    record('WAIT_EVAL_BACKLOG', updates=updates,
                           checkpoint_steps=82, last_checkpoint=40999)
                    waiting_for_eval = True
                time.sleep(180)
                continue
            if not train_cards_idle():
                time.sleep(60)
                continue
            if waiting_for_eval:
                record('EVAL_BACKLOG_COMPLETE', updates=updates)
                waiting_for_eval = False
            missing_checks += 1
            if missing_checks < 2:
                time.sleep(60)
                continue
            record('RESTART', updates=updates, command=COMMAND)
            with (RUN / 'watch_80k_launcher.log').open('a', buffering=1) as log:
                result = subprocess.run(COMMAND, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT)
            record('EXIT', returncode=result.returncode,
                   updates=completed_updates())
            missing_checks = 0
            time.sleep(600)


if __name__ == '__main__':
    main()
