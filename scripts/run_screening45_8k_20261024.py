#!/usr/bin/env python3
"""Persistent, per-dataset screening queue with bounded GPU/RAM admission."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def live(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def fingerprint_ok(repo, manifest):
    for rel, digest in manifest['source_sha256'].items():
        p = repo / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen canonical evaluator changed: {rel}')


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    out = {}
    for line in raw.splitlines():
        i, uuid, total, used, util = [x.strip() for x in line.split(',')]
        out[i] = dict(uuid=uuid, total=int(total), used=int(used), utilization=int(util), apps={})
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    for line in raw.splitlines():
        uuid, pid, used = [x.strip() for x in line.split(',')]
        for g in out.values():
            if g['uuid'] == uuid:
                try:
                    g['apps'][int(pid)] = int(used)
                except ValueError:
                    pass
    return out


def proc_table():
    out = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not args or b'python' not in args[0]:
                continue
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            env = (p / 'environ').read_bytes().split(b'\0')
            cuda = next((x.split(b'=', 1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')), '')
            out[int(p.name)] = dict(ppid=int(stat[1]), args=[x.decode(errors='replace') for x in args if x], cuda=cuda)
        except (OSError, ValueError, StopIteration):
            pass
    return out


def descendants(pid, processes):
    found = {int(pid)}
    while True:
        new = {p for p, v in processes.items() if v['ppid'] in found}
        if new <= found:
            return found
        found |= new


def available_ram_mib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) // 1024


def reservation_dir(repo):
    p = repo / 'outputs/02_eval_runs/unified_paper/gpu_reservations'
    p.mkdir(parents=True, exist_ok=True)
    return p


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def live_reservations(path):
    result = {}
    for pid, item in read(path, {}).get('reservations', {}).items():
        try:
            if start_ticks(pid) == item.get('start_ticks') and Path(f'/proc/{pid}/cmdline').read_bytes():
                result[pid] = item
        except OSError:
            pass
    return result


def reserve_gpu(args, task):
    directory = reservation_dir(args.repo)
    while True:
        g = gpu_state()[args.gpu]
        path = directory / (g['uuid'] + '.json')
        with (directory / (g['uuid'] + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            peers = live_reservations(path)
            extra = sum(max(0, p['expected_peak_mib'] - g['apps'].get(int(pid), 0)) for pid, p in peers.items())
            if g['used'] + extra + task['reserve_mib'] <= .80 * g['total']:
                record = dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()), host=socket.gethostname(),
                              task=task['key'], expected_peak_mib=task['reserve_mib'],
                              full_peak_mib=task['reserve_mib'], updated=time.time())
                peers[str(os.getpid())] = record
                save(path, dict(reservations=peers, updated=time.time()))
                return directory, path, record
        time.sleep(10)


def update_gpu_reservation(reservation, child, gpu, release=False):
    directory, path, record = reservation
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        peers = live_reservations(path)
        if release:
            peers.pop(str(os.getpid()), None)
        else:
            pids = descendants(child, proc_table())
            actual = sum(m for p, m in gpu_state()[gpu]['apps'].items() if p in pids)
            record.update(expected_peak_mib=max(0, record['full_peak_mib'] - actual),
                          descendant_used_mib=actual, updated=time.time())
            peers[str(os.getpid())] = record
        save(path, dict(reservations=peers, updated=time.time()))


def validate(task, root):
    directory = root / task['relative_output']
    family, dataset = task['family'], task['dataset']
    paths = []
    scores = {}
    if family != 'segmentation':
        p = directory / 'last_result.json'
        d = read(p)
        if not d or d.get('dataset') != dataset:
            raise ValueError(f'Missing or wrong dataset: {p}')
        if family == 'retrieval':
            # Canonical retrieval results omit batch/seed. Verify the actual
            # recorded invocation rather than treating absent fields as zero.
            invocation = read(root / task['relative_cell'] / 'invocation.json', {})
            command = invocation.get('command', [])
            expected = task['command']
            for flag, value in [('--batch-size', '64'), ('--seed', '0'),
                                ('--checkpoint', expected[expected.index('--checkpoint') + 1]),
                                ('--datasets', dataset)]:
                if flag not in command or command[command.index(flag) + 1] != value:
                    raise ValueError(f'Retrieval invocation mismatch: {flag}')
            if d.get('checkpoint') != command[command.index('--checkpoint') + 1]:
                raise ValueError('Retrieval checkpoint mismatch')
            if ('batch_size' in d and int(d['batch_size']) != 64) or ('seed' in d and int(d['seed']) != 0):
                raise ValueError('Retrieval result contradicts frozen invocation')
        elif int(d.get('batch_size', 0)) != 64 or int(d.get('seed', -1)) != 0:
            raise ValueError('Frozen batch/seed mismatch')
        if family == 'classification':
            metric = 'macro_auc' if dataset == 'chestmnist' else 'balanced_accuracy'
            scores[f'classification/{dataset}'] = float(d[metric])
        elif family == 'regression':
            if dataset == 'bbbc013':
                if d.get('split') != 'compound-log1p-leave-one-replicate-row-out' or int(d.get('n_folds',0)) != 8 or int(d.get('oof_samples',0)) != 96:
                    raise ValueError('BBBC013 requires two-compound, eight-fold OOF')
                macro = (float(d['wortmannin_r2']) + float(d['ly294002_r2'])) / 2
                value = float(d.get('compound_mean_r2', d['r2']))
                if not math.isclose(value, macro, abs_tol=1e-10):
                    raise ValueError('BBBC013 reported score differs from compound mean')
                scores[f'regression/{dataset}'] = value
            else:
                scores[f'regression/{dataset}'] = float(d['r2'])
        else:
            scores[f'retrieval/{dataset}'] = float(d['recall_at_1'])
            scores[f'clustering/{dataset}'] = float(d['nmi'])
        paths = [p]
    else:
        paths = list(directory.glob(f'*/{dataset}/{task["checkpoint"]}/results.json'))
        expected = 3 if dataset == 'pannuke' else 1
        if len(paths) != expected:
            raise ValueError(f'{dataset}: expected {expected} rotation result(s), found {len(paths)}')
        values = []
        for p in paths:
            d = read(p, {})
            meta = d.get('_meta', {})
            if (meta.get('probe_epochs'), meta.get('probe_batch_size'), meta.get('seed'), meta.get('test_evaluations')) != (50, 16, 0, 1):
                raise ValueError(f'E50/B16/seed0/test-once mismatch: {p}')
            values.append(float(d['test']['mDice']))
        scores[f'segmentation/{dataset}'] = sum(values) / len(values)
    if not all(math.isfinite(x) for x in scores.values()):
        raise ValueError('Nonfinite score')
    return dict(scores=scores, result_files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def clean_cache(task, root):
    # Only this task's private regenerable features, after its children exit.
    cell = root / task['relative_cell']
    if any(any(str(cell) in arg for arg in p['args']) for pid, p in proc_table().items() if pid != os.getpid()):
        return
    for name in ('cache', 'features'):
        for p in list(cell.rglob(name)):
            if p.is_symlink():
                manifest = read(root / 'campaign_manifest.json', {})
                base = manifest.get('private_cache_root')
                if base:
                    expected = Path(base) / task['key'] / name
                    if p.resolve() == expected.resolve() and expected.is_dir():
                        shutil.rmtree(expected)
                        p.unlink()
                continue
            if p.is_dir() and not p.is_symlink() and cell in p.parents:
                shutil.rmtree(p)


def run_one(args, manifest, task):
    root, repo = args.campaign, args.repo
    key = task['key']
    lockfile = root / '_state/locks' / (key + '.lock')
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        done = root / '_state/done' / (key + '.json')
        if done.exists():
            return
        running = root / '_state/running' / (key + '.json')
        status = dict(key=key, host=socket.gethostname(), pid=os.getpid(), gpu=args.gpu,
                      reserve_mib=task['reserve_mib'], ram_mib=task['ram_mib'], started=time.time())
        save(running, status)
        logpath = root / 'logs' / (key + '.log')
        logpath.parent.mkdir(parents=True, exist_ok=True)
        process = None
        reservation = None
        try:
            fingerprint_ok(repo, manifest)
            reservation = reserve_gpu(args, task)
            command = [s.replace('{ROOT}', str(root)).replace('{REPO}', str(repo)).replace('{PYTHON}', args.python)
                       .replace('{GPU}', args.gpu) for s in task['command']]
            env = os.environ.copy()
            for k in list(env):
                if k.startswith('DINOV3_') or k in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES'):
                    env.pop(k, None)
            env.update(manifest['runtime_env'])
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(repo), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=str(root / '_runtime/matplotlib'))
            cell = root / task['relative_cell']
            cell.mkdir(parents=True, exist_ok=True)
            if manifest.get('private_cache_root'):
                for name, link in [('cache', cell / 'cache'), ('features', cell / 'result/features')]:
                    target = Path(manifest['private_cache_root']) / task['key'] / name
                    target.mkdir(parents=True, exist_ok=True)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists():
                        link.symlink_to(target, target_is_directory=True)
                    if not link.is_symlink() or link.resolve() != target.resolve():
                        raise ValueError(f'Private cache mapping conflict: {link}')
            save(cell / 'invocation.json', dict(command=command, task=task, env=manifest['runtime_env'],
                                               protocol_id=manifest['protocol_id'], source_sha256=manifest['source_sha256']))
            with logpath.open('a') as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                def stop(signum, frame):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    raise InterruptedError('Task interrupted')
                signal.signal(signal.SIGTERM, stop)
                signal.signal(signal.SIGINT, stop)
                while process.poll() is None:
                    save(running, dict(status, child_pid=process.pid, heartbeat=time.time()))
                    update_gpu_reservation(reservation, process.pid, args.gpu)
                    time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f'Evaluator exit {process.returncode}; see {logpath}')
            report = validate(task, root)
            save(done, dict(status='VALID_COMPLETE', completed=time.time(), **report))
            clean_cache(task, root)
        except Exception as error:
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            failure = root / '_state/failures' / (key + '.json')
            previous = read(failure, {})
            save(failure, dict(status, error=repr(error), attempts=previous.get('attempts', 0) + 1,
                               failed_at=time.time(), log=str(logpath)))
            # Do not delete incomplete caches while any child may still hold them.
            if process is None or process.poll() is not None:
                clean_cache(task, root)
        finally:
            if reservation:
                update_gpu_reservation(reservation, process.pid if process else os.getpid(), args.gpu, release=True)
            running.unlink(missing_ok=True)


def supervise(args, manifest):
    root = args.campaign
    state = root / '_state'
    for name in ('workers', 'running', 'done', 'failures', 'locks'):
        (state / name).mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    lock = (state / 'workers' / (host + '.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    fingerprint_ok(args.repo, manifest)
    pending = manifest['tasks']
    report_path = state / 'workers' / (host + '.json')
    while not (root / 'STOP').exists():
        for p in (state / 'running').glob('*.json'):
            r = read(p, {})
            if r.get('host') == host and not live(r.get('pid')) and time.time() - r.get('started', 0) > 60:
                # The wrapper launches a separate group: preserve any live orphan.
                if not live(r.get('child_pid')):
                    p.unlink(missing_ok=True)
        done = {p.stem for p in (state / 'done').glob('*.json')}
        if len(done) == len(pending):
            save(report_path, dict(host=host, status='ALL_COMPLETE', completed=len(done), updated=time.time()))
            return
        try:
            gpus = gpu_state()
            processes = proc_table()
            ram = available_ram_mib()
            running = {p.stem: read(p, {}) for p in (state / 'running').glob('*.json')}
            ours = [r for r in running.values() if r.get('host') == host]
            owned_pids = set()
            for r in ours:
                owned_pids |= descendants(r['pid'], processes)
            admissions = {}
            reservations_root = reservation_dir(args.repo)
            for gpu in args.gpus:
                g = gpus[gpu]
                own = [r for r in ours if r.get('gpu') == gpu]
                reserved = 0
                for r in own:
                    pids = descendants(r['pid'], processes)
                    actual = sum(m for p, m in g['apps'].items() if p in pids)
                    reserved += max(0, r['reserve_mib'] - actual)
                other_loading = []
                shared = live_reservations(reservations_root / (g['uuid'] + '.json'))
                own_wrappers = {str(r['pid']) for r in own}
                shared_extra = sum(max(0, r['expected_peak_mib'] - g['apps'].get(int(pid), 0))
                                   for pid, r in shared.items() if pid not in own_wrappers)
                reserved += shared_extra
                # Count actual independent evaluator processes, not launchers or DataLoader copies.
                eval_pids = {p for p, v in processes.items() if ('-m' in v['args'] and any(
                    x.startswith(('dinov3.eval.bio_frozen_eval.run_', 'dinov3.eval.bio_segmentation.linear_probe',
                                  'dinov3.eval.bio_segmentation.extract_datasets')) for x in v['args']))
                    or any(Path(x).name.startswith('run_unified_paper_') for x in v['args'][:3])}
                for pid in eval_pids:
                    v = processes[pid]
                    if pid in owned_pids or v['ppid'] in eval_pids or pid in g['apps']:
                        continue
                    if str(pid) in shared:
                        continue
                    paper = read(reservations_root / f'process_{host}_{pid}.json', {})
                    try:
                        verified = paper.get('start_ticks') == start_ticks(pid)
                    except OSError:
                        verified = False
                    if verified and paper.get('status') in ['waiting_for_capacity', 'cpu_only', 'exited']:
                        continue
                    if v['cuda'] in (gpu, g['uuid']):
                        other_loading.append(pid)
                reserved += 3072 * len(other_loading)
                admissions[gpu] = dict(**g, extra_reserved_mib=reserved, own_tasks=len(own),
                                       external_loading_pids=other_loading, shared_reservation_mib=shared_extra)
            launched = []
            for task in pending:
                key = task['key']
                if key in done or key in running:
                    continue
                failure = read(state / 'failures' / (key + '.json'), {})
                if failure.get('attempts', 0) >= 3 or time.time() - failure.get('failed_at', 0) < 300:
                    continue
                if ram < args.minimum_free_ram_mib + task['ram_mib']:
                    continue
                options = []
                for gpu, g in admissions.items():
                    if g['own_tasks'] >= args.max_per_gpu:
                        continue
                    load = g['used'] + g['extra_reserved_mib']
                    if load + task['reserve_mib'] <= g['total'] * args.ceiling:
                        # Continue toward five tests/60%, with peak safety taking precedence.
                        count = len(g['apps']) + g['own_tasks'] + len(g['external_loading_pids'])
                        if load >= g['total'] * .70 and count >= 5:
                            continue
                        options.append((load / g['total'], gpu))
                if not options:
                    continue
                gpu = min(options)[1]
                # Exclusive reservation across shared-host supervisors.
                claimfile = state / 'locks' / (key + '.dispatch')
                try:
                    fd = os.open(claimfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
                except FileExistsError:
                    # Dispatch is only a short reservation; task flock/running owns long work.
                    if time.time() - claimfile.stat().st_mtime > 120:
                        claimfile.unlink(missing_ok=True)
                    continue
                os.close(fd)
                logpath = root / 'logs' / (key + '.wrapper.log')
                logpath.parent.mkdir(parents=True, exist_ok=True)
                with logpath.open('a') as log:
                    command = [sys.executable, str(Path(__file__).resolve()), '--repo', str(args.repo),
                               '--campaign', str(root), '--python', args.python, '--task', key, '--gpu', gpu]
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(state / 'running' / (key + '.json'), dict(key=key, host=host, pid=child.pid, gpu=gpu,
                     reserve_mib=task['reserve_mib'], ram_mib=task['ram_mib'], started=time.time()))
                admissions[gpu]['extra_reserved_mib'] += task['reserve_mib']
                admissions[gpu]['own_tasks'] += 1
                ram -= task['ram_mib']
                launched.append(dict(task=key, gpu=gpu, pid=child.pid))
                if len(launched) >= 8:
                    break
            save(report_path, dict(host=host, pid=os.getpid(), updated=time.time(), status='ACTIVE',
                completed=len(done), expected=len(pending), launched=launched, gpus=admissions,
                available_ram_mib=ram, own_running=len(ours),
                constraint='GPU_PEAK_OR_RAM_OR_EXTERNAL_LOADING' if not launched else None))
        except Exception as error:
            save(report_path, dict(host=host, pid=os.getpid(), updated=time.time(), status='AUDIT_ERROR', error=repr(error)))
        time.sleep(args.poll_seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--gpus', nargs='+', default=['0'])
    p.add_argument('--gpu', default='0')
    p.add_argument('--task')
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--max-per-gpu', type=int, default=5)
    p.add_argument('--ceiling', type=float, default=.80)
    p.add_argument('--minimum-free-ram-mib', type=int, default=32768)
    args = p.parse_args()
    args.repo, args.campaign = args.repo.resolve(), args.campaign.resolve()
    manifest = read(args.campaign / 'campaign_manifest.json')
    if not manifest:
        raise SystemExit('Missing campaign manifest')
    if args.task:
        run_one(args, manifest, next(t for t in manifest['tasks'] if t['key'] == args.task))
    else:
        supervise(args, manifest)


if __name__ == '__main__':
    main()
