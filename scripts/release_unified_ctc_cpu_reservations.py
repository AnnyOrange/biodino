#!/usr/bin/env python3
"""Release GPU loading reservations after verified final CTC inference."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
DATA = ROOT / 'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def verified_cpu_only(pid, reservation, job, data):
    try:
        process = Path(f'/proc/{pid}')
        started = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
        command = process.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        if started != reservation['start_ticks']:
            return False, 'process identity changed'
        if 'run_unified_paper_ctc.py' not in command:
            return False, 'not a CTC evaluation child'
        if not all(str(arg) in command for arg in job['args']):
            return False, 'invocation differs from queued task'
        invocation = read(Path(job['result']).parent / 'invocation_manifest.json')
        domains = invocation['dataset']['test_domains']
        last_domain = domains[-1]
        fold = invocation['dataset']['fold']
        frames = len(data['domains'][last_domain]['test_frames'])
        log = Path(job['log']).read_text(errors='replace')
        current_attempt = log[log.rfind('\nSTART '):]
        marker = f'[fold {fold}] {last_domain} frame={frames}/{frames}'
        if marker not in current_attempt and f'[fold {fold}] reuse domain {last_domain}' not in current_attempt:
            return False, 'final test-domain inference is not verified'
        return True, marker
    except (KeyError, OSError, IndexError, ValueError) as error:
        return False, repr(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    gpu_rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = dict((int(index.strip()), value.strip()) for index, value in
                (line.split(',', 1) for line in gpu_rows.splitlines()))[args.gpu]
    jobs = {task['key']: task for name in ('ctc_queue.json', 'remote_queue.json')
            for task in read(OUT / name).get('tasks', [])
            if task.get('task') == 'cell_tracking'}
    data = read(DATA)
    path = OUT / 'gpu_reservations' / f'{uuid}.json'
    lock_path = path.with_suffix('.lock')
    observations = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(path)
        reservations = state.get('reservations', {})
        for pid, reservation in list(reservations.items()):
            job = jobs.get(reservation.get('task'))
            if job is None:
                continue
            ready, evidence = verified_cpu_only(int(pid), reservation, job, data)
            observations.append(dict(pid=int(pid), task=job['key'],
                                     ready=ready, evidence=evidence))
            if ready and args.apply:
                del reservations[pid]
                process = OUT / 'gpu_reservations' / f'process_{os.uname().nodename}_{pid}.json'
                record = read(process)
                if record:
                    record.update(status='cpu_only', updated=time.time(),
                                  release_evidence=evidence)
                    temporary = process.with_suffix('.tmp')
                    temporary.write_text(json.dumps(record, indent=2) + '\n')
                    temporary.replace(process)
        if args.apply and any(item['ready'] for item in observations):
            state['updated'] = time.time()
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(path)
    print(json.dumps(dict(host=os.uname().nodename, gpu=args.gpu, apply=args.apply,
                          observations=observations), indent=2))


if __name__ == '__main__':
    main()
