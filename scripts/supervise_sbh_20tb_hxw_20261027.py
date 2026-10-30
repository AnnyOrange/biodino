#!/usr/bin/env python3
"""Supervise the user-authorized S+/B/H+ comparison without changing its recipe."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] if stat[0] != 'Z' and (p / 'cmdline').read_bytes() else None
    except OSError:
        return None


def gpu_state():
    out = {}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    for line in raw.splitlines():
        i, uuid, total, used, util = [s.strip() for s in line.split(',')]
        out[int(i)] = dict(uuid=uuid, total=int(total), used=int(used), utilization=int(util), apps={})
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    for line in raw.splitlines():
        uuid, pid, used = [s.strip() for s in line.split(',')]
        if pid.isdigit() and used.isdigit():
            for g in out.values():
                if uuid == g['uuid']:
                    g['apps'][int(pid)] = int(used)
    return out


def run_processes(output):
    result = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if b'--output-dir' in args and args[args.index(b'--output-dir') + 1].decode() == output:
                if any(x.endswith(b'dinov3/train/train.py') for x in args):
                    result[int(p.name)] = identity(p.name)
        except (OSError, IndexError):
            pass
    return result


def tail_metric(output):
    for name in ['raw_loss_metrics.jsonl', 'training_metrics.json']:
        p = Path(output) / name
        try:
            with p.open('rb') as f:
                f.seek(max(0, f.seek(0, 2) - 65536))
                lines = f.read().splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    return row
        except OSError:
            pass
    return {}


def latest_checkpoint(output):
    paths = [p for p in (Path(output) / 'ckpt').glob('*/checkpoint.pth')
             if p.parent.name.isdigit() and p.stat().st_size > 1_000_000]
    return max(paths, key=lambda p: int(p.parent.name), default=None)


def available_ram_mib():
    return int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:')).split()[1]) // 1024


def update_reservation(directory, gpu, own_pids, peak, active):
    path = directory / (gpu['uuid'] + '.json')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        peers = {k: v for k, v in read(path, {}).get('reservations', {}).items()
                 if identity(k) == v.get('start_ticks') and identity(k) is not None}
        peers.pop(str(os.getpid()), None)
        other_extra = sum(max(0, v['expected_peak_mib'] - gpu['apps'].get(int(k), 0))
                          for k, v in peers.items())
        actual = sum(m for p, m in gpu['apps'].items() if p in own_pids)
        if active:
            peers[str(os.getpid())] = dict(pid=os.getpid(), start_ticks=identity(os.getpid()),
                host=socket.gethostname(), task='sbh_20tb_training_20261027', kind='training',
                expected_peak_mib=max(0, peak - actual), full_peak_mib=peak,
                descendant_used_mib=actual, updated=time.time())
        write(path, dict(reservations=peers, updated=time.time()))
    return other_extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    a = parser.parse_args()
    plan = read(a.campaign / 'plan.json')
    lock = (a.campaign / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reservations = Path(plan['reservation_dir'])
    reservations.mkdir(parents=True, exist_ok=True)
    state_path = a.campaign / 'status.json'
    state = read(state_path, dict(models={}))
    children = {}
    stopping = False

    def stop_handler(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    while True:
        capacity_policy = read(a.campaign / 'evaluation_capacity_20261030/policy.json', {})
        gpus = gpu_state()
        ram = available_ram_mib()
        disk = os.statvfs(a.campaign)
        free_gib = disk.f_bavail * disk.f_frsize / 2**30
        state.update(updated=time.time(), supervisor_pid=os.getpid(), start_ticks=identity(os.getpid()),
                     ram_available_mib=ram, disk_free_gib=free_gib, gpus=gpus)
        claimed = []
        for model in plan['models']:
            name, output = model['name'], model['output']
            row = state['models'].setdefault(name, dict(attempts=[], failures=0))
            if name in children and children[name].poll() is not None:
                row['attempts'][-1].update(returncode=children[name].returncode, finished=time.time())
                final_metrics = tail_metric(output)
                updates = final_metrics.get('optimizer_updates_completed', 0)
                if children[name].returncode == 0 and updates >= plan['target_updates']:
                    write(Path(output) / 'TRAINING_COMPLETE.json', dict(time=time.time(), updates=updates))
                row['failures'] += int(children[name].returncode != 0)
                del children[name]
            own = run_processes(output)
            if not own and name not in children and row['attempts'] and 'finished' not in row['attempts'][-1]:
                row['attempts'][-1].update(finished=time.time(), returncode='exited_while_supervisor_restarted')
                row['failures'] += 1
            ck = latest_checkpoint(output)
            step = int(ck.parent.name) + 1 if ck else 0
            if step > row.get('last_progress_checkpoint', 0):
                row.update(failures=0, last_progress_checkpoint=step)
            metrics = tail_metric(output)
            row.update(gpus=model['gpus'], live_pids=own, latest_checkpoint=str(ck) if ck else None,
                       checkpoint_updates=step, metrics=metrics)
            completed = step >= plan['target_updates'] or (Path(output) / 'TRAINING_COMPLETE.json').exists()
            blocked = (a.campaign / 'STOP_NEW_TRAINING').exists() or row['failures'] >= 3 or stopping
            active = not completed and (bool(own) or not blocked)
            if active:
                claimed += model['gpus']
            extra = {}
            for i in model['gpus']:
                actual = sum(m for p, m in gpus[i]['apps'].items() if p in own)
                peaks = row.setdefault('observed_peak_mib', {})
                peaks[str(i)] = max(peaks.get(str(i), 0), actual)
                floor = model['peak_reserve_mib']
                # A measured steady-state floor applies only after a settled run
                # has saved multiple checkpoints. Startup retains its larger budget.
                if (own and step >= 976 and row['attempts'] and
                        time.time() - row['attempts'][-1]['started'] >= 1800):
                    floor = capacity_policy.get('training_steady_floor_mib', {}).get(name, floor)
                peak = max(floor, math.ceil(peaks[str(i)] * 1.10))
                extra[i] = update_reservation(reservations, gpus[i], own,
                                               peak, active)
            if own:
                row['status'] = ('TRAINING' if metrics.get('optimizer_updates_completed', 0) > 0
                                 else 'FORWARD_BACKWARD' if metrics else 'LOADING')
                row['gpu_memory_mib'] = {i: sum(m for p, m in gpus[i]['apps'].items() if p in own)
                                         for i in model['gpus']}
                continue
            if completed:
                row['status'] = 'COMPLETE'
                continue
            if blocked:
                row['status'] = 'STOPPED_AFTER_FAILURES' if row['failures'] >= 3 else 'PAUSED'
                continue
            if not (a.campaign / 'PREFLIGHT_PASSED.json').exists():
                row['status'] = 'WAITING_FOR_PREFLIGHT'
                continue
            if free_gib < 160 or ram < 120 * 1024:
                row['status'] = 'WAITING_FOR_HOST_RESOURCES'
                continue
            if any(gpus[i]['used'] + extra[i] + model['peak_reserve_mib'] > .98 * gpus[i]['total']
                   for i in model['gpus']):
                row.update(status='WAITING_FOR_EXISTING_EVALUATIONS', other_reserved_mib=extra)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model['gpus'])),
                       PYTHONPATH=plan['snapshot'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MALLOC_ARENA_MAX='4',
                       MALLOC_MMAP_THRESHOLD_='131072', PYTORCH_ALLOC_CONF='expandable_segments:True',
                       TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
                       NCCL_DEBUG='WARN', TMPDIR=str(a.campaign / 'tmp'))
            for k in ['DINOV3_TEACHER_DOMAIN_METADATA', 'DINOV3_SPECIMEN_PAIR_PROB']:
                env.pop(k, None)
            Path(output).mkdir(parents=True, exist_ok=True)
            (a.campaign / 'tmp').mkdir(exist_ok=True)
            command = [plan['python'], '-m', 'torch.distributed.run', '--nnodes=1',
                       f'--nproc_per_node={len(model["gpus"])}', '--master_addr=127.0.0.1',
                       f'--master_port={model["port"]}', 'dinov3/train/train.py',
                       '--config-file', model['config'], '--output-dir', output, '--seed', '0']
            if ck is None:
                command += ['--no-resume']
            number = len(row['attempts']) + 1
            log = Path(output) / f'attempt_{number:03d}.log'
            with log.open('a') as stream:
                child = subprocess.Popen(command, cwd=plan['snapshot'], env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children[name] = child
            row['attempts'].append(dict(started=time.time(), pid=child.pid, command=command, log=str(log)))
            row.update(status='STARTING', train_pid=child.pid)
        write(a.campaign / 'gpu_claims.json', dict(pid=os.getpid(), start_ticks=identity(os.getpid()),
              gpus=claimed, updated=time.time(), reason='User requested S+/B/H+ training on eight HXW GPUs'))
        write(state_path, state)
        if stopping:
            # Leave live training children intact; reservations remain backed by this owner.
            if not any(run_processes(m['output']) for m in plan['models']):
                return
        time.sleep(15)


if __name__ == '__main__':
    main()
