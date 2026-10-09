#!/usr/bin/env python3
"""Claim-based worker for the weight-space baseline campaign (one worker = one GPU slot).

Tasks are claimed atomically (mkdir on the shared campaign root), so any number
of workers on any host may run concurrently; each task is executed once. Before
launching, the worker requires: GPU memory used < 60% and >= 8 GiB free,
at most --max-per-gpu campaign processes on this physical GPU, and (for
segmentation) >= 4 GiB /tmp, >= 80 GiB available RAM and <= --max-seg-per-host
segmentation pipelines on this host. Launches on one GPU are serialized by a
lock with a settle delay so admission reads real memory. Protocol/batch/size
are never lowered; failures are recorded as FAILED and the worker stops after
two consecutive failures or when the campaign-wide FAILED count reaches 6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import ROOT, THREADS, task_done  # noqa: E402

HOST = socket.gethostname()  # replaced by --host-tag in main(): hostnames are not unique across sites


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def gpu_query(index: int) -> dict:
    out = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20).strip().splitlines()
    if len(out) != 1:
        raise RuntimeError(f'expected exactly one GPU row for index {index}: {out}')
    used, total, util = (int(x.strip()) for x in out[0].split(','))
    return {'used_mib': used, 'total_mib': total, 'util': util}


def ram_available_gib() -> float:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return 0.0


def tmp_free_gib() -> float:
    return shutil.disk_usage('/tmp').free / 2**30


def status_path(task_id: str) -> Path:
    return ROOT / 'claims' / task_id / 'status.json'


def read_status(task_id: str) -> dict | None:
    p = status_path(task_id)
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def running_here(gpu: int | None = None, family: str | None = None) -> int:
    n = 0
    for st in (ROOT / 'claims').glob('*/status.json'):
        try:
            s = json.loads(st.read_text())
        except Exception:
            continue
        if s.get('state') != 'RUNNING' or s.get('host') != HOST:
            continue
        if gpu is not None and s.get('gpu') != gpu:
            continue
        if family is not None and s.get('family') != family:
            continue
        try:
            os.kill(int(s['pid']), 0)
        except Exception:
            continue
        n += 1
    return n


def failed_total() -> int:
    n = 0
    for st in (ROOT / 'claims').glob('*/status.json'):
        try:
            if json.loads(st.read_text()).get('state') == 'FAILED':
                n += 1
        except Exception:
            pass
    return n


def acquire_gpu_lock(gpu: int, timeout_s: float) -> bool:
    lock = ROOT / 'locks' / f'{HOST}_gpu{gpu}'
    t0 = time.monotonic()
    while True:
        try:
            lock.mkdir()
            (lock / 'owner').write_text(f'{os.getpid()} {now()}\n')
            return True
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > 900:  # stale lock (holder died)
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.monotonic() - t0 > timeout_s:
                return False
            time.sleep(5)


def release_gpu_lock(gpu: int) -> None:
    shutil.rmtree(ROOT / 'locks' / f'{HOST}_gpu{gpu}', ignore_errors=True)


def telemetry(gpu: int, task_id: str, pid: int | None, phase: str) -> None:
    rec = {'utc': now(), 'host': HOST, 'gpu': gpu, 'task': task_id, 'pid': pid, 'phase': phase,
           'ram_available_gib': round(ram_available_gib(), 1), 'load1': os.getloadavg()[0],
           'tmp_free_gib': round(tmp_free_gib(), 1)}
    try:
        rec.update(gpu_query(gpu))
    except Exception as exc:  # telemetry must never kill the worker
        rec['gpu_error'] = str(exc)
    with (ROOT / 'node_telemetry' / f'{HOST}.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host-tag', required=True)
    ap.add_argument('--gpu-index', type=int, required=True, help='physical GPU index on this host')
    ap.add_argument('--worker-id', type=int, required=True)
    ap.add_argument('--families', nargs='*', default=None, help='restrict to these task families')
    ap.add_argument('--max-per-gpu', type=int, default=5, help='cap on campaign processes per physical GPU')
    ap.add_argument('--preexisting', type=int, default=0, help='other-campaign processes already on this GPU')
    ap.add_argument('--max-seg-per-host', type=int, default=2)
    ap.add_argument('--max-hours', type=float, default=40)
    ap.add_argument('--start-delay', type=float, default=0)
    ap.add_argument('--settle-seconds', type=float, default=75)
    args = ap.parse_args()
    global HOST
    HOST = args.host_tag

    tasks = json.loads((ROOT / 'tasks.json').read_text())['tasks']
    tasks.sort(key=lambda t: t['order'])
    self_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    deadline = time.monotonic() + args.max_hours * 3600
    log(f'WORKER_PROVENANCE host_tag={HOST} hostname={socket.gethostname()} gpu={args.gpu_index} worker={args.worker_id} '
        f'worker_sha256={self_hash} tasks_sha256={hashlib.sha256((ROOT / "tasks.json").read_bytes()).hexdigest()} '
        f'families={args.families} max_per_gpu={args.max_per_gpu} preexisting={args.preexisting}')
    if args.start_delay:
        time.sleep(args.start_delay)
    consecutive_failures = 0
    completed = 0
    while time.monotonic() < deadline:
        if failed_total() >= 6:
            log('CAMPAIGN_FAILED_CAP_REACHED (>=6 FAILED tasks); refusing new claims')
            return 3
        claimed = None
        for task in tasks:
            if args.families and task['family'] not in args.families:
                continue
            if (ROOT / 'claims' / task['id']).exists():
                continue
            if task.get('heavy'):
                if tmp_free_gib() < 4 or ram_available_gib() < 80:
                    continue
                if running_here(family='segmentation') >= args.max_seg_per_host:
                    continue
            ok, _ = task_done(task['done'])
            if ok:
                # Output already valid (e.g. produced before a crash); register without running.
                try:
                    (ROOT / 'claims' / task['id']).mkdir()
                    status_path(task['id']).write_text(json.dumps(
                        {'state': 'DONE', 'note': 'PRE_EXISTING_VALID_OUTPUT', 'host': HOST, 'utc': now()}) + '\n')
                except FileExistsError:
                    pass
                continue
            try:
                (ROOT / 'claims' / task['id']).mkdir()
            except FileExistsError:
                continue
            claimed = task
            break
        if claimed is None:
            log('NO_CLAIMABLE_TASKS; worker exits')
            return 0
        task = claimed
        status_path(task['id']).write_text(json.dumps({'state': 'CLAIMED', 'host': HOST, 'gpu': args.gpu_index,
                                                       'worker': args.worker_id, 'utc': now()}) + '\n')
        # ---- admission ----
        admitted = False
        wait_start = time.monotonic()
        while time.monotonic() < deadline:
            g = gpu_query(args.gpu_index)
            here = running_here(gpu=args.gpu_index) + args.preexisting
            mem_ok = g['used_mib'] / g['total_mib'] < 0.60 and g['total_mib'] - g['used_mib'] >= 8192
            if mem_ok and here < args.max_per_gpu and acquire_gpu_lock(args.gpu_index, 120):
                g = gpu_query(args.gpu_index)  # re-read under the lock
                if g['used_mib'] / g['total_mib'] < 0.60 and g['total_mib'] - g['used_mib'] >= 8192:
                    admitted = True
                    break
                release_gpu_lock(args.gpu_index)
            if time.monotonic() - wait_start > 6 * 3600:
                break
            time.sleep(45)
        if not admitted:
            log(f'GPU_ADMISSION_TIMEOUT {task["id"]} gpu={args.gpu_index}; releasing claim')
            shutil.rmtree(ROOT / 'claims' / task['id'], ignore_errors=True)
            return 2
        # ---- launch ----
        env = os.environ.copy()
        env.update(THREADS)
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
        env['PYTHONPATH'] = task['pythonpath']
        cmd = [c.replace('{GPU}', str(args.gpu_index)) for c in task['cmd']]
        log_path = ROOT / 'logs' / f'{task["id"]}__{HOST}_gpu{args.gpu_index}.log'
        if log_path.exists():
            log_path = ROOT / 'logs' / f'{task["id"]}__{HOST}_gpu{args.gpu_index}_{int(time.time())}.log'
        Path(task['cmd'][task['cmd'].index('--output-dir') + 1]).mkdir(parents=True, exist_ok=True) \
            if '--output-dir' in task['cmd'] else None
        with log_path.open('x') as stream:
            stream.write(f'# {now()} host={HOST} gpu={args.gpu_index} worker={args.worker_id}\n# cwd={task["cwd"]}\n'
                         f'# PYTHONPATH={task["pythonpath"]}\n# pinned_manifest={task["pinned_manifest"]}\n'
                         f'# checkpoint_sha256={task["checkpoint_sha256"]}\n# cmd={json.dumps(cmd)}\n')
            stream.flush()
            proc = subprocess.Popen(cmd, cwd=task['cwd'], env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        status_path(task['id']).write_text(json.dumps(
            {'state': 'RUNNING', 'host': HOST, 'gpu': args.gpu_index, 'worker': args.worker_id, 'pid': proc.pid,
             'family': task['family'], 'dataset': task['dataset'], 'arm': task['arm'], 'start_utc': now(),
             'log': str(log_path), 'cmd': cmd, 'cwd': task['cwd'], 'gpu_at_launch': g,
             'checkpoint_sha256': task['checkpoint_sha256']}) + '\n')
        log(f'[launch] {task["id"]} pid={proc.pid} gpu={args.gpu_index} used={g["used_mib"]}/{g["total_mib"]} log={log_path}')
        telemetry(args.gpu_index, task['id'], proc.pid, 'start')
        time.sleep(args.settle_seconds)
        release_gpu_lock(args.gpu_index)
        last_tel = time.monotonic()
        while proc.poll() is None:
            time.sleep(30)
            if time.monotonic() - last_tel >= 1800:
                telemetry(args.gpu_index, task['id'], proc.pid, 'running')
                last_tel = time.monotonic()
        rc = proc.returncode
        ok, why = task_done(task['done'])
        state = 'DONE' if (rc == 0 and ok) else 'FAILED'
        st = json.loads(status_path(task['id']).read_text())
        st.update(state=state, rc=rc, done_check=why, end_utc=now())
        status_path(task['id']).write_text(json.dumps(st) + '\n')
        telemetry(args.gpu_index, task['id'], proc.pid, state)
        if state == 'DONE':
            completed += 1
            consecutive_failures = 0
            log(f'[done] {task["id"]} rc={rc} ({why})')
        else:
            consecutive_failures += 1
            log(f'FAILED_PROTOCOL_OR_RESOURCE {task["id"]} rc={rc} done_check={why} log={log_path}')
            if consecutive_failures >= 2:
                log('two consecutive failures on this worker; stopping (fail closed)')
                return 1
    log(f'WORKER_TIME_LIMIT completed={completed}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
