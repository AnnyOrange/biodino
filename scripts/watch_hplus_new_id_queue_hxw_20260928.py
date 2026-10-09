#!/usr/bin/env python3
"""Restart the H+ ID queue if its process or heartbeat disappears."""
import fcntl
from pathlib import Path
import os
import subprocess
import time

ADMIN = Path('/data/hs6_hplus_5tb_eval_20260921/new_checkpoint_queue_20260927')
POOL = '/data/hs6_hplus_5tb_eval_20260921/bin/run_hplus_new_id_queue_hxw_20260927.py'
PYTHON = '/home/xzj/eval_envs/hs6_protocol_v2/bin/python'
COMMAND = [PYTHON, '-u', POOL, '--gpus', *(str(gpu) for gpu in range(8)), '--slots', '6']


def pool_pids():
    result = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if len(args) >= 3 and args[2].decode() == POOL:
                result.append(int(proc.name))
        except (OSError, UnicodeDecodeError):
            pass
    return result


def main():
    ADMIN.mkdir(parents=True, exist_ok=True)
    with (ADMIN / 'watchdog.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            pids = pool_pids()
            heartbeat = ADMIN / 'status.json'
            stale = heartbeat.exists() and time.time() - heartbeat.stat().st_mtime > 60
            if stale:
                for pid in pids:
                    try: os.kill(pid, 15)
                    except ProcessLookupError: pass
                time.sleep(2)
                pids = pool_pids()
            if not pids:
                with (ADMIN / 'pool.log').open('ab') as log:
                    subprocess.Popen(COMMAND, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                print('RESTART_POOL', time.time(), 'stale' if stale else 'missing', flush=True)
            time.sleep(15)


if __name__ == '__main__':
    main()
