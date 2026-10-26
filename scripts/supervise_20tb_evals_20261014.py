#!/usr/bin/env python3
"""Keep authorized 20TB evaluation workers alive and report fleet utilization."""
import concurrent.futures
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
OUT = REPO / 'outputs/00_reports/20tb_eval_watch_20261014'
ROUTE = REPO / 'outputs/02_eval_runs/route2_all_checkpoints_v4_20261012'
PAIR = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
DISTILL = REPO / 'outputs/02_eval_runs/v2_20tb_local_distill_20261014'
PY = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'


def ssh(host, remote, timeout=18):
    port = []
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
               *port, host, remote]
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, '', f'SSH timed out after {timeout}s')


def worker(kind, host, name, command, log, *, queue='route', cwd=REPO):
    return dict(kind=kind, host=host, name=name, command=command, log=str(log),
                queue=queue, cwd=str(cwd))


def specs():
    route = []
    for host in ('cpu2', 'cpu8', 'cpu9', 'cpu11', 'cpu20'):
        slots = 4 if host == 'cpu20' else 2
        command = (f'{PY} -u scripts/queue_all_route2_checkpoints_v4_20261012.py '
                   f'worker-component --host {host} --gpus 0 --target-per-gpu {slots} '
                   f'--max-jobs {slots}')
        route.append(worker('component', host, host, command,
                            ROUTE / 'logs' / f'{host}.component.log'))
    for host in ('cpu2', 'cpu8', 'cpu9', 'cpu11', 'cpu20'):
        slots = 1 if host == 'cpu20' else 3
        name = f'{host}-frozen'
        command = (f'{PY} -u scripts/run_v2_progress_fleet_20261006.py worker '
                   f'--output {ROUTE / "components"} --host {name} --gpus 0 '
                   f'--target-per-gpu {slots} --memory-target 0.85 '
                   f'--max-host-jobs {slots} --max-global-jobs 160 '
                   '--task-family frozen')
        route.append(worker('component_frozen', host, name, command,
                            ROUTE / 'logs' / f'{name}.component.log'))
    for host in ('cpu10', 'cpu15'):
        command = (f'{PY} -u scripts/queue_all_route2_checkpoints_v4_20261012.py '
                   f'worker-full --host {host} --gpus 0 --max-jobs 2')
        route.append(worker('full', host, host, command,
                            ROUTE / 'logs' / f'{host}.full.log'))
    command = f'{PY} -u scripts/run_route2_v4_3090qi_healthy_gpus_20261014.py --max-jobs 12'
    route.append(worker('full', '3090-qi', '3090-qi-healthy', command,
                        ROUTE / 'logs/3090-qi.healthy-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_healthy_gpus_20261014.py '
               '--max-jobs 12 --host 3090-qi-extra --gpus 2 3 4 6')
    route.append(worker('full', '3090-qi', '3090-qi-extra', command,
                        ROUTE / 'logs/3090-qi.extra-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_healthy_gpus_20261014.py '
               '--max-jobs 6 --host 3090-qi-classification --gpus 0 3 6 '
               '--family classification')
    route.append(worker('full', '3090-qi', '3090-qi-classification', command,
                        ROUTE / 'logs/3090-qi.classification-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_remapped_worker_20261015.py '
               '--max-jobs 10 --host 3090-qi-remapped --gpus 3 5')
    route.append(worker('full', '3090-qi', '3090-qi-remapped', command,
                        ROUTE / 'logs/3090-qi.remapped-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_remapped_worker_20261015.py '
               '--max-jobs 10 --host 3090-qi-remapped-b --gpus 3 5')
    route.append(worker('full', '3090-qi', '3090-qi-remapped-b', command,
                        ROUTE / 'logs/3090-qi.remapped-b-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_remapped_worker_20261015.py '
               '--max-jobs 15 --host 3090-qi-remapped-c --gpus 0 1 5')
    route.append(worker('full', '3090-qi', '3090-qi-remapped-c', command,
                        ROUTE / 'logs/3090-qi.remapped-c-full.log'))
    command = (f'{PY} -u scripts/run_route2_v4_3090qi_remapped_worker_20261015.py '
               '--max-jobs 15 --host 3090-qi-remapped-d --gpus 0 1 6')
    route.append(worker('full', '3090-qi', '3090-qi-remapped-d', command,
                        ROUTE / 'logs/3090-qi.remapped-d-full.log'))
    for host in ('cpu1', 'cpu5', 'cpu12'):
        command = (f'{PY} -u scripts/run_v2_progress_fleet_20261006.py worker '
                   f'--output {PAIR} --host {host} --gpus 0 --target-per-gpu 5 '
                   '--max-host-jobs 5 --max-global-jobs 30')
        route.append(worker('paired', host, host, command,
                            PAIR / 'logs' / f'{host}.worker.absolute_20261014.log', queue='paired'))
    command = (f'{PY} -u scripts/run_v2_progress_fleet_20261006.py worker '
               f'--output {DISTILL} --host cpu7 --gpus 0 --target-per-gpu 1 '
               '--max-host-jobs 1 --max-global-jobs 1')
    route.append(worker('distill', 'cpu7', 'cpu7', command,
                        DISTILL / 'cpu7.worker.log', queue='distill'))
    lyx_root = Path('/data/xuzijing/cooldown_20261013_eval/20tb_cooldown')
    lyx_script = '/data/xuzijing/cooldown_20261013_eval/bin/keep_cooldown_eval_slot_busy_lyx.sh'
    classification = 'classification_a,classification_b,classification_c,classification_d'
    lyx_slots = ((3, 'c1', classification), (3, 'r1', 'regression,retrieval'),
                 (4, 'c2', classification), (4, 'd1', 'detection'),
                 (5, 's1', 'segmentation_a,segmentation_b'),
                 (5, 's2', 'segmentation_c,segmentation_d'),
                 (5, 'c3', classification), (5, 'c4', classification),
                 (3, 'c5', classification), (4, 'c6', classification),
                 (6, 'c7', classification), (7, 'c8', classification),
                 (6, 'c9', classification))
    for gpu, slot, lanes in lyx_slots:
        command = f'bash {lyx_script} 20tb_cooldown 75151 10 {gpu} {slot} {lanes}'
        route.append(worker('lyx', '5090-lyx-xr', slot, command,
                            lyx_root / f'slot_{slot}_watch.log', cwd=lyx_root))
    hxw_root = Path('/home/xzj/storage/merged/cooldown_20261013')
    hxw_script = str(hxw_root / 'eval_root/bin/keep_cooldown_eval_slot_busy.sh')
    all_lanes = (classification + ',regression,retrieval,detection,'
                 'segmentation_a,segmentation_b,segmentation_c,segmentation_d')
    for arm, train, gpu_range in (
            ('20tb_distill_lw5', '20tb_distill_cooldown_len4880_lw5', range(4)),
            ('20tb_rebal_distill_lw2', '20tb_distill_cooldown_len4880_lw2_rebal', range(4, 8))):
        for gpu in gpu_range:
            snapshot = hxw_root / train / 'eval'
            for extra in range(4):
                slot = f'l{gpu}' if extra == 0 else f'l{gpu}x{extra}'
                command = (f'bash {hxw_script} {arm} {snapshot} 75151 10 '
                           f'{gpu} {slot} {all_lanes}')
                route.append(worker('hxw', '5090-hxw-xzj', slot, command,
                                    hxw_root / 'eval_root' / arm / f'slot_{slot}_watch.log',
                                    cwd=hxw_root / 'eval_root' / arm))
    return route


def probe(spec):
    if spec['kind'] in ('lyx', 'hxw'):
        pattern = '^' + spec['command']
        result = ssh(spec['host'], 'pgrep -f ' + shlex.quote(pattern))
        return dict(spec=spec, alive=result.returncode == 0,
                    pids=result.stdout.strip(), error=result.stderr.strip())
    token = (f'--host {spec["name"]}' if spec['kind'] in ('component', 'full', 'paired')
             else 'run_route2_v4_3090qi_healthy_gpus_20261014.py' if spec['name'] == '3090-qi-healthy'
             else f'--host {spec["name"]}')
    if spec['kind'] == 'full' and spec['host'] == '3090-qi':
        pattern = '^' + spec['command']
    elif spec['kind'] == 'component_frozen':
        pattern = '^' + spec['command']
    elif spec['kind'] == 'paired':
        pattern = '^' + PY + ' -u scripts/run_v2_progress_fleet_20261006.py worker .*' + token
    elif spec['kind'] == 'distill':
        pattern = '^' + PY + ' -u scripts/run_v2_progress_fleet_20261006.py worker .*' + token
    else:
        pattern = '^' + PY + ' -u scripts/queue_all_route2_checkpoints_v4_20261012.py .*' + token
    command = 'pgrep -f ' + shlex.quote(pattern)
    result = ssh(spec['host'], command)
    return dict(spec=spec, alive=result.returncode == 0,
                pids=result.stdout.strip(), error=result.stderr.strip())


def gpu_probe(host):
    if host == '3090-qi':
        query = ('for i in 0 2 3 4 5 6 7; do nvidia-smi -i "$i" '
                 '--query-gpu=index,memory.used,memory.total,utilization.gpu '
                 '--format=csv,noheader,nounits; done')
    else:
        query = ('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu '
                 '--format=csv,noheader,nounits')
    result = (subprocess.run(query, shell=True, capture_output=True, text=True, timeout=18)
              if host == 'local' else ssh(host, query))
    cards = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                index, used, total, util = [int(part.strip()) for part in line.split(',')]
                cards.append(dict(index=index, used_mib=used, total_mib=total,
                                  memory_fraction=round(used / total, 3), utilization=util))
            except (ValueError, ZeroDivisionError):
                continue
    return host, cards, result.stderr.strip()


def queue_open(kind):
    if kind == 'paired':
        return not (PAIR / '_state/PAUSED.json').exists()
    if kind == 'distill':
        return not (DISTILL / '_state/PAUSED.json').exists()
    return True


def hxw_memory_guard():
    root = '/home/xzj/storage/merged/cooldown_20261013/eval_root/20tb_rebal_distill_lw2/results/_state'
    result = ssh('5090-hxw-xzj', "awk '/MemAvailable:/ {print $2}' /proc/meminfo")
    if result.returncode:
        return f'hxw-xzj: RAM probe unavailable: {result.stderr.strip()}'
    try:
        available_gib = int(result.stdout.strip()) / 1024**2
    except ValueError:
        return 'hxw-xzj: RAM probe returned invalid data'
    if available_gib < 8:
        return f'hxw-xzj: only {available_gib:.1f} GiB RAM available; extra eval is held'
    if available_gib < 32:
        return None
    claim = f'{root}/claims/ckpt_75639__classification_a.lock'
    archive = f'{root}/attempt_archive_20261014_low_memory/ckpt_75639__classification_a.lock'
    command = (f'if test -d {claim} && ! test -e {archive} '
               f'&& ! kill -0 584299 2>/dev/null; then '
               f'mkdir -p {root}/attempt_archive_20261014_low_memory && '
               f'mv {claim} {archive} && echo requeued; fi')
    moved = ssh('5090-hxw-xzj', command)
    if moved.returncode:
        return f'hxw-xzj: deferred requeue failed: {moved.stderr.strip()}'
    if moved.stdout.strip() == 'requeued':
        return 'hxw-xzj: ck75639 classification_a returned to queue after RAM recovery'
    return None


def progress():
    inventory = json.loads((ROUTE / 'extensions/FULL_V4_INVENTORY_20261007.json').read_text())
    counts = {}
    for cell in inventory['cells']:
        counts[cell['state']] = counts.get(cell['state'], 0) + 1
    return dict(route=counts,
                paired_done=len(list((PAIR / '_state/done').glob('*.json'))),
                distill_done=len(list((DISTILL / '_state/done').glob('*.json'))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_restart = {}
        low_since = {}
        previous_progress = None
        last_progress_time = time.time()
        while True:
            start = time.time()
            workers = specs()
            hosts = sorted({spec['host'] for spec in workers} | {'local'})
            with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
                worker_rows = list(pool.map(probe, workers))
                gpu_rows = list(pool.map(gpu_probe, hosts))
            gpu = {host: cards for host, cards, _ in gpu_rows}
            alerts = [f'{host}: {error}' for host, cards, error in gpu_rows if not cards]
            hxw_guard = hxw_memory_guard()
            if hxw_guard:
                alerts.append(hxw_guard)
            for row in worker_rows:
                spec = row['spec']
                if row['alive'] or not queue_open(spec['queue']):
                    continue
                key = spec['name'] + ':' + spec['kind']
                if row['error'] or not gpu.get(spec['host']):
                    alerts.append(f'{key}: probe unavailable: {row["error"] or "GPU query failed"}')
                    continue
                if start - last_restart.get(key, 0) < 600:
                    alerts.append(f'{key}: stopped; restart cooldown active')
                    continue
                remote = (f'cd {shlex.quote(spec["cwd"])} || exit; '
                          f'nohup {spec["command"]} '
                          f'>> {shlex.quote(spec["log"])} 2>&1 < /dev/null &')
                result = ssh(spec['host'], remote)
                last_restart[key] = start
                alerts.append(f'{key}: restart {"submitted" if result.returncode == 0 else result.stderr.strip()}')
            for host, cards in gpu.items():
                minimum = 0.80 if host in ('5090-lyx-xr', '5090-hxw-xzj') else 0.60
                for card in cards:
                    key = f'{host}:{card["index"]}'
                    if card['memory_fraction'] > minimum:
                        low_since.pop(key, None)
                    else:
                        low_since.setdefault(key, start)
                        if start - low_since[key] > 600:
                            alerts.append(f'{key}: at or below {minimum:.0%} VRAM for over 10 minutes')
            try:
                current = progress()
                if current != previous_progress:
                    previous_progress = current
                    last_progress_time = start
                elif start - last_progress_time > 1200:
                    alerts.append('No completed evaluation change for over 20 minutes')
            except Exception as exc:
                current = dict(error=str(exc))
                alerts.append('Progress read failed: ' + str(exc))
            report = dict(updated_unix=start, utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(start)),
                          progress=current, workers=worker_rows, gpus=gpu, alerts=alerts)
            tmp = OUT / f'.status.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            tmp.replace(OUT / 'status.json')
            print(report['utc'], 'route_done', current.get('route', {}).get('VALID_COMPLETE'),
                  'paired_done', current.get('paired_done'), 'distill_done', current.get('distill_done'),
                  'alerts', len(alerts), flush=True)
            time.sleep(max(1, 60 - (time.time() - start)))


if __name__ == '__main__':
    main()
