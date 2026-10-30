"""Run existing component workers with shared GPU reservations; preserve evaluator recipes."""
import argparse
import atexit
import fcntl
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_screening45_8k_20261024 as admission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['rolling', 'shared'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host-label', choices=['deepcad', '5090-zxr'], required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--slot', required=True)
    a = p.parse_args()
    if a.host_label == 'deepcad' and a.gpu not in ['1', '4', '6', '7']:
        p.error('deepcad permits only GPUs 1, 4, 6 and 7')
    ceiling = .35 if a.host_label == 'deepcad' else .80
    maximum = 2 if a.host_label == 'deepcad' else 5
    ident = f'{a.host_label}_gpu{a.gpu}_{a.slot}'
    state = ROOT / 'outputs/02_eval_runs/eval_migration_20261027'
    lock = (state / (ident + '.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    worker_pid = os.getpid()
    worker_ticks = admission.start_ticks(worker_pid)
    cards = admission.gpu_state()
    gpu = cards[a.gpu]
    path = admission.reservation_dir(ROOT) / (gpu['uuid'] + '.json')
    selected = {}
    mutex = threading.RLock()
    cached_card = {}
    checked_at = 0.0
    reported_at = 0.0

    def report(**kwargs):
        admission.save(state / (ident + '.json'), dict(pid=worker_pid, start_ticks=worker_ticks,
            host=a.host_label, hostname=socket.gethostname(), gpu=a.gpu, campaign=str(a.output),
            updated=time.time(), ceiling=ceiling, **kwargs))

    def guard(g, actual, task):
        nonlocal cached_card, checked_at, reported_at
        expected = int(task['dataset'].get('reserve_mib') or max(
            18432 if task['dataset']['image_size'] >= 384 else 4096,
            int(task['asset'].get('reserve_mib') or 0)))
        # Avoid an NVML/proc scan for every oversized item in a large queue.
        # This cache can only defer work; actual admission still rechecks under lock.
        if time.monotonic() - checked_at > 5:
            cached_card = admission.gpu_state()[a.gpu]
            checked_at = time.monotonic()
        if cached_card['used'] + expected > cached_card['total'] * ceiling:
            if time.monotonic() - reported_at > 5:
                report(status='WAIT_CAPACITY', task=task['key'],
                       used_mib=cached_card['used'], expected_peak_mib=expected)
                reported_at = time.monotonic()
            return False
        with mutex, path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            card = admission.gpu_state()[a.gpu]
            peers = admission.live_reservations(path)
            peers.pop(str(worker_pid), None)
            processes = admission.proc_table()
            # Each reservation is either a real evaluator or an admitted loading slot.
            active = set(peers)
            for pid in card['apps']:
                args = processes.get(pid, {}).get('args', [])
                if any('dinov3.eval.' in s or 'run_unified_paper_' in s or '/fm/scripts/run_task.py' in s for s in args):
                    active.add(str(pid))
            # Count a reservation and its CUDA child once.
            for pid in peers:
                descendants = admission.descendants(int(pid), processes) - {int(pid)}
                active.difference_update(str(x) for x in descendants)
            extra = sum(max(0, r['expected_peak_mib'] - card['apps'].get(int(pid), 0)) for pid, r in peers.items())
            fits = card['used'] + extra + expected <= card['total'] * ceiling and len(active) < maximum
            if fits:
                selected.clear()
                selected.update(task=task['key'], full_peak_mib=expected, admitted_at=time.time())
                peers[str(worker_pid)] = dict(pid=worker_pid, start_ticks=worker_ticks,
                    host=socket.gethostname(), task=task['key'], expected_peak_mib=expected,
                    full_peak_mib=expected, updated=time.time())
                admission.save(path, dict(reservations=peers, updated=time.time()))
            report(status='ADMITTED' if fits else 'WAIT_CAPACITY', task=task['key'],
                   used_mib=card['used'], other_reserved_mib=extra, expected_peak_mib=expected,
                   actual_or_loading_tests=len(active))
            return fits

    def release():
        with path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            peers = admission.live_reservations(path)
            peers.pop(str(worker_pid), None)
            admission.save(path, dict(reservations=peers, updated=time.time()))
    atexit.register(release)

    def heartbeat():
        while True:
            try:
                with mutex:
                    processes = admission.proc_table()
                    children = admission.descendants(worker_pid, processes) - {worker_pid}
                    card = admission.gpu_state()[a.gpu]
                    used = sum(m for pid,m in card['apps'].items() if pid in children)
                    with path.with_suffix('.lock').open('a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        peers = admission.live_reservations(path)
                        r = peers.get(str(worker_pid))
                        if r:
                            if not children and time.time()-selected.get('admitted_at',0)>60:
                                peers.pop(str(worker_pid),None)
                            else:
                                r.update(expected_peak_mib=max(0,r['full_peak_mib']-used),
                                         descendant_used_mib=used,updated=time.time())
                            admission.save(path,dict(reservations=peers,updated=time.time()))
                    if children:
                        report(status='EVALUATING' if used else 'LOADING_OR_CPU_STAGE',
                               task=selected.get('task'),child_pids=sorted(children),gpu_used_mib=used,
                               total_gpu_used_mib=card['used'])
            except Exception as e:
                print('RESERVATION_HEARTBEAT',repr(e),flush=True)
            time.sleep(10)
    threading.Thread(target=heartbeat,daemon=True).start()
    if a.kind == 'rolling':
        snapshot = ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/1pb_rolling_selected29_v5_20261022/source_snapshot/scripts'
        sys.path.insert(0,str(snapshot))
        fleet=importlib.import_module('run_retest_fleet_20260918')
        queue=importlib.import_module('rolling_1pb_eval_queue')
        fleet.queue=queue
        os.environ['ROLLING_1PB_MAX_DENSE_PER_GPU']='1'
    else:
        fleet=importlib.import_module('run_retest_fleet_20260918')
    args=SimpleNamespace(output=a.output.resolve(),host=ident,gpus=[int(a.gpu)],
        target_per_gpu=maximum,max_host_jobs=1,max_global_jobs=160,task_family='mixed',
        admission_guard=guard,shared_admission_lock=state/'legacy_admission.lock')
    report(status='VERIFYING_PROTOCOL')
    fleet.worker(args)
    report(status='WORKER_EXITED')


if __name__=='__main__':
    main()
