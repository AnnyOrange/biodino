#!/usr/bin/env python3
"""Start a persistent shared-queue worker on an allowed resident GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
PY = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
sys.path.insert(0, str(ROOT / 'scripts'))
from unified_paper_gpu_policy import excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-label', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--slot', required=True)
    args = parser.parse_args()
    if args.host_label == '5090-lyx-xr' or excluded(args.host_label, args.gpu):
        parser.error('GPU reserved by user')
    queue = OUT / args.queue
    if queue.parent != OUT or not queue.is_file() or not json.loads(queue.read_text()).get('tasks'):
        parser.error('Queue missing or empty')
    if not Path(PY).is_file():
        parser.error('Evaluation Python is unavailable on this host')
    ident = f'{args.host_label}_gpu{args.gpu}_slot{args.slot}'
    state = OUT / 'states' / f'worker_{ident}.json'
    if state.exists():
        try:
            pid = json.loads(state.read_text())['pid']
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'unified_paper_queue.py' in cmd and args.slot.encode() in cmd:
                print(json.dumps({'status': 'already_running', 'pid': pid, 'state': str(state)}))
                return
        except (KeyError, FileNotFoundError, json.JSONDecodeError):
            pass
    log = OUT / 'logs' / f'worker_{ident}.log'
    command = [PY, '-u', str(ROOT / 'scripts/unified_paper_queue.py'),
               '--host-label', args.host_label, '--gpu', str(args.gpu),
               '--slot', args.slot, '--queue-file', str(queue)]
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print(json.dumps({'status': 'started', 'host': args.host_label,
                      'gpu': args.gpu, 'pid': process.pid, 'queue': str(queue),
                      'state': str(state), 'log': str(log)}))


if __name__ == '__main__':
    main()
