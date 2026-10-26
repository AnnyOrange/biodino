#!/usr/bin/env python3
"""Minute-level watchdog for the two 20TB controllers; never kills evaluations."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3')
BASE = ROOT / 'outputs/02_eval_runs/l20tb_selected29_fill_20261021'
PYTHON = '/home/lxy/miniconda3/bin/python'


def main():
    if (BASE / 'STOP_AUTOMATION').exists():
        return
    with (BASE / 'controller_watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reports = []
        for script, heartbeat, log in [
            ('manage_20tb_selected29_fill_20261021.py', 'status.json', 'controller.log'),
            ('accelerate_20tb_selected29_20261021.py', 'capacity_supervision.json', 'capacity_controller.log'),
        ]:
            pids = []
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:parts = path.read_bytes().split(b'\0')
                except OSError:continue
                if any(p.endswith(script.encode()) for p in parts) and b'watch' in parts:
                    pids.append(int(path.parent.name))
            status = BASE / heartbeat
            age = time.time() - status.stat().st_mtime if status.exists() else None
            # A controller may be waiting on a bounded hash/plot/sync; ten
            # minutes without a heartbeat is outside its normal work cycle.
            if pids and age is not None and age > 600:
                for pid in pids:
                    try:os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:pass
                reports.append(dict(controller=script,status='RESTART_PENDING_STALE',pids=pids,heartbeat_age=age))
                continue
            if not pids:
                with (BASE / log).open('a') as stream:
                    child = subprocess.Popen([PYTHON, '-u', str(ROOT / 'scripts' / script), 'watch'],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                reports.append(dict(controller=script,status='STARTED',pid=child.pid))
            else:reports.append(dict(controller=script,status='ALIVE',pids=pids,heartbeat_age=age))
        target = BASE / 'controller_watchdog.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_unix=time.time(),controllers=reports),indent=2)+'\n')
        temporary.replace(target)


if __name__ == '__main__':
    main()
