#!/usr/bin/env python3
"""Fill lyx resident 5TB dense evaluations toward 75% VRAM, with up to twelve slots/GPU.
Reuse the recorded evaluator and protocols; extend arm registration only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import time

E = Path('/data/xuzijing/hs6_l5_v2_recovery_eval_20260930')
BASE = E / 'bin/run_v2_dense_queue_lyx_20261004.py'
SEG = E / 'bin/run_v2_v4_segmentation_lyx_dynamic_20261004.py'
ARMS = ('global', 'global_local', 'global_cls', 'global_cls_w03', 'global_slow', 'global_w03', 'global_cls_early',
        'global_cls_slow', 'global_cls_slow2', 'global_cls_w3', 'global_cls_slow2_w3', 'global_cls_slow2_w03', 'gram_ext')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if '--campaign' in sys.argv:
        module = load(SEG, 'resident_segment')
        for arm in ARMS:
            module.CAMPAIGNS['v2_' + arm] = (E / arm, {29767 + 488 * i for i in range(12)})
        module.main()
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(6)))
    p.add_argument('--per-gpu', type=int, default=12)
    p.add_argument('--memory-target', type=float, default=.75)
    p.add_argument('--min-ram-gib', type=float, default=80)
    p.add_argument('--max-host-jobs', type=int, default=12)
    a = p.parse_args()
    if not 1 <= a.per_gpu <= 12:
        p.error('Use one to twelve slots per GPU')
    base = load(BASE, 'resident_dense')
    runner = Path(__file__).resolve()
    log_root = E / 'logs/dense_fill_20261006'
    log_root.mkdir(parents=True, exist_ok=True)
    state = log_root / 'status.json'
    manifest = log_root / ('scheduler_manifest_' + digest(runner)[:12] + '.json')
    hashes = {str(path): digest(path) for path in (BASE, SEG, runner)}
    if manifest.exists() and json.loads(manifest.read_text())['source_sha256'] != hashes:
        raise RuntimeError('Scheduler source changed since launch')
    base.atomic(manifest, dict(source_sha256=hashes, arms=ARMS, arguments=vars(a),
        protocol='existing union-v4 dense components; batch/splits/layers unchanged'))
    processes = ''
    def foreign(task):
        arm, point, family, ds = task
        if family == 'detection':
            return ('--output-dir ' + str(base.det_cell(arm, point, ds)) + ' ') in processes
        dataset = 'pannuke' if ds in base.PANNUKE else ds
        pattern = re.escape(f'--campaign v2_{arm} --point {point} --dataset {dataset} --gpu ') + r'\d+'
        if ds in base.PANNUKE:
            pattern += re.escape(' --fold ' + base.PANNUKE[ds])
        return re.search(pattern, processes) is not None
    base.SEG_RUNNER = runner
    def reserve(task):
        # Low-resolution B8 detection / 256px segmentation leave room beside training.
        if task[2] == 'detection' or task[3] in ('conic', 'tissuenet') or task[3].startswith('pannuke/'):
            return 6000
        return 18000 if task[3] == 'monuseg' else 8500
    active, attempts, cooldown = {}, {}, {}
    while True:
        processes = base.subprocess.check_output(['ps', '-eo', 'args'], text=True)
        for key, job in list(active.items()):
            rc = job['child'].poll()
            if rc is None:
                continue
            ok = rc == 0 and base.valid(job['task'])
            print('DONE' if ok else 'FAILED', key, 'rc', rc, flush=True)
            if not ok:
                attempts[key] = attempts.get(key, 0) + 1
                tail = job['log'].read_text(errors='replace')[-16000:]
                if 'out of memory' in tail.lower():
                    cooldown[job['gpu']] = time.time() + 300
            del active[key]
        pending = []
        for task in base.tasks(ARMS):
            arm, point, family, ds = task
            if family == 'segmentation' and ds == 'monuseg':
                continue  # 30/7/14 amendment has its own fingerprinted queue.
            key = f"{arm}__{point}__{family}__{ds.replace('/', '_')}"
            if key in active or attempts.get(key, 0) >= 2:
                continue
            if not (E / arm / 'adapters' / str(point) / 'checkpoint.pth').is_file():
                continue
            if not base.valid(task) and not foreign(task):
                pending.append(task)
        rows = base.subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        cards = [tuple(map(int, line.split(','))) for line in rows.splitlines()]
        counts = {}
        resident_counts = {}
        for line in processes.splitlines():
            if '--campaign v2_' in line:
                match = re.search(r' --gpu (\d+)(?: |$)', line)
                if match:
                    g = int(match.group(1))
                    resident_counts[g] = resident_counts.get(g, 0) + 1
        for gpu in a.gpus:
            running = [j for j in active.values() if j['gpu'] == gpu]
            counts[gpu] = max(len(running), resident_counts.get(gpu, 0))
            used, total = cards[gpu]
            if sum(resident_counts.values()) >= a.max_host_jobs or used / total >= a.memory_target or counts[gpu] >= a.per_gpu or cooldown.get(gpu, 0) > time.time():
                continue
            loading_vram = sum(reserve(j['task']) for j in running if time.time() - j['started'] < 60)
            loading_ram = sum(base.ram_gib(j['task']) for j in active.values() if time.time() - j['started'] < 120)
            for task in pending:
                if total - used - loading_vram < reserve(task) + 2048:
                    continue
                if base.mem_available_gib() - loading_ram - base.ram_gib(task) < a.min_ram_gib:
                    continue
                job = base.launch(task, gpu, log_root)
                active[job['key']] = job
                pending.remove(task)
                break
        base.atomic(state, dict(utc=base.dt.datetime.now(base.dt.timezone.utc).isoformat(),
            active=[dict(key=j['key'], gpu=j['gpu'], pid=j['child'].pid) for j in active.values()],
            pending=len(pending), failed_twice=[k for k,v in attempts.items() if v >= 2],
            gpu_memory=cards, under_70_percent=[g for g in a.gpus if cards[g][0]/cards[g][1] < .70]))
        time.sleep(10)

if __name__ == '__main__':
    main()
