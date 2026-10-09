#!/usr/bin/env python3
"""Stop our deepcad evaluation controllers and single-rank Adaptive for DDP migration."""
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/huawei_deepcad/dinov3')
REPORT = ROOT / 'outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_ddp4_20260929/quiesce.json'

EXPECTED = {
    995802: 'scripts/run_deepcad_coexist_20260928.py',
    1001187: 'scripts/run_deepcad_gpu6_v4_20260928.py',
    1014188: 'watch_regression.py',
    1238562: 'torch.distributed.run',
    3234275: 'scripts/run_monuseg37_comparison_20260929.py',
    3244479: 'scripts/run_monuseg37_comparison_20260929.py',
    3244480: 'scripts/run_monuseg37_comparison_20260929.py',
    3244481: 'scripts/run_monuseg37_comparison_20260929.py',
    3244482: 'scripts/run_monuseg37_comparison_20260929.py',
}


def cmd(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def children(pid):
    result = []
    for task in Path('/proc').glob('[0-9]*'):
        try:
            stat = (task / 'stat').read_text().split(') ', 1)[1].split()
            if int(stat[1]) == pid:
                result.append(int(task.name))
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
    return result


def descendants(pid):
    found = []
    for child in children(pid):
        found.extend(descendants(child))
        found.append(child)
    return found


def stop_tree(pid, expected):
    line = cmd(pid)
    if not line:
        return {'pid': pid, 'state': 'already_gone'}
    if expected not in line:
        raise RuntimeError(f'PID {pid} changed identity: {line[:180]}')
    processes = descendants(pid) + [pid]
    for child in processes:
        try:
            os.kill(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return {'pid': pid, 'command': line, 'descendants': processes[:-1]}


def main():
    for pid, expected in EXPECTED.items():
        line = cmd(pid)
        if line and expected not in line:
            raise RuntimeError(f'PID {pid} changed identity: {line[:180]}')
    # Snapshot the two managers' evaluation children before stopping the managers.
    eval_roots = [p for p in children(995802) if p != 1238562] + children(1001187)
    records = []
    for pid in (995802, 1001187, 1014188):
        records.append(stop_tree(pid, EXPECTED[pid]) if pid == 1014188 else
                       stop_controller_only(pid, EXPECTED[pid]))
    for pid in eval_roots:
        line = cmd(pid)
        if line and '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python' in line:
            records.append(stop_tree(pid, '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'))
    for pid in (3234275, 3244479, 3244480, 3244481, 3244482):
        records.append(stop_tree(pid, EXPECTED[pid]))
    time.sleep(3)
    records.append(stop_tree(1238562, EXPECTED[1238562]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                                  'reason': 'user requested four-card Adaptive DDP training',
                                  'stopped': records}, indent=2) + '\n')
    print(json.dumps({'report': str(REPORT), 'stopped_roots': [r['pid'] for r in records]}))


def stop_controller_only(pid, expected):
    line = cmd(pid)
    if line:
        if expected not in line:
            raise RuntimeError(f'PID {pid} changed identity: {line[:180]}')
        os.kill(pid, signal.SIGTERM)
    return {'pid': pid, 'command': line, 'controller_only': True}


if __name__ == '__main__':
    main()
