#!/usr/bin/env python3
"""Run the unchanged screening evaluator with measured, bounded resource budgets."""
import argparse
import builtins
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


def load_native(repo):
    spec = importlib.util.spec_from_file_location('locked_screening', repo / 'scripts/run_screening45_8k_20261024.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--repo', type=Path, required=True)
    preliminary.add_argument('--campaign', type=Path, required=True)
    preliminary.add_argument('--gpu', default='0')
    preliminary.add_argument('--gpus', nargs='+', default=['0'])
    preliminary.add_argument('--task')
    args, _ = preliminary.parse_known_args()
    policy_path = args.campaign.parent / 'evaluation_capacity_20261030/policy.json'
    policy = json.loads(policy_path.read_text())
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    allowed = set(map(str, policy['gpus']))
    if (args.task and args.gpu not in allowed) or not set(args.gpus) <= allowed:
        raise ValueError('GPU outside the user-authorized evaluation set')
    native = load_native(args.repo)
    native.__file__ = __file__  # Dispatch children through this bounded wrapper.

    def best_fit(options, *values, **kwargs):
        # The locked dispatcher calls min(options) on (load/total, GPU) pairs.
        # Place small jobs on the fullest compatible card, preserving room for H+.
        if (isinstance(options, list) and options and not values and not kwargs and
                all(isinstance(item, tuple) and len(item) == 2 and
                    isinstance(item[0], float) and item[1] in allowed for item in options)):
            return max(options)
        return builtins.min(options, *values, **kwargs)

    native.min = best_fit
    original_read = native.read
    original_ram = native.available_ram_mib
    original_popen = native.subprocess.Popen
    original_run_one = native.run_one
    original_update = native.update_gpu_reservation
    reservations_dir = native.reservation_dir(args.repo)
    children = []

    def read(path, default=None):
        result = original_read(path, default)
        if Path(path) == args.campaign / 'campaign_manifest.json' and result:
            for task in result['tasks']:
                rule = policy.get('tasks', {}).get(task['arm'] + '/' + task['family'] + '/' + task['dataset'])
                if rule:
                    task.update(reserve_mib=rule['reserve_mib'], ram_mib=rule['ram_mib'],
                                allocator_mib=rule['allocator_mib'])
        if Path(path).parent == args.campaign / '_state/failures' and result:
            block = original_read(args.campaign / 'capacity_blocks' / Path(path).name, {})
            if block.get('policy_sha256') == policy_hash:
                result['attempts'] = 3
        return result

    native.read = read

    def cpu_used_mib(pid):
        # Proportional anonymous RSS avoids counting shared fork pages repeatedly.
        pids = native.descendants(pid, native.proc_table())
        total = 0
        for process in pids:
            try:
                lines = Path(f'/proc/{process}/smaps_rollup').read_text().splitlines()
                total += sum(int(line.split()[1]) for line in lines if line.startswith('Pss_Anon:'))
            except OSError:
                pass
        return total // 1024

    def ram_extra(exclude=None):
        peers = {}
        for path in reservations_dir.glob('GPU-*.json'):
            peers.update(native.live_reservations(path))
        return sum(max(0, record.get('ram_mib', 0) - cpu_used_mib(pid))
                   for pid, record in peers.items() if pid != str(exclude) and record.get('ram_mib'))

    def ram_available():
        children[:] = [child for child in children if child.poll() is None]
        state = original_read(args.campaign.parent / 'status.json', {})
        for model in state.get('models', {}).values():
            updates = model.get('metrics', {}).get('optimizer_updates_completed', 0)
            if updates and (updates % 488 >= 478 or (updates % 488 == 0 and model.get('checkpoint_updates', 0) < updates)):
                # Checkpoint consolidation is a large, transient host allocation.
                return 0
        return max(0, original_ram() - ram_extra())

    native.available_ram_mib = ram_available

    def reserve_gpu(run_args, task):
        if run_args.gpu not in allowed:
            raise ValueError('Unauthorized GPU')
        while True:
            with (reservations_dir / 'sbh_host_ram.lock').open('a') as host_lock:
                fcntl.flock(host_lock, fcntl.LOCK_EX)
                available = ram_available()
                gpu = native.gpu_state()[run_args.gpu]
                path = reservations_dir / (gpu['uuid'] + '.json')
                with path.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    peers = native.live_reservations(path)
                    extra = sum(max(0, item['expected_peak_mib'] - gpu['apps'].get(int(pid), 0))
                                for pid, item in peers.items())
                    if (available >= policy['minimum_free_ram_mib'] + task['ram_mib'] and
                            gpu['used'] + extra + task['reserve_mib'] <= .80 * gpu['total']):
                        record = dict(pid=os.getpid(), start_ticks=native.start_ticks(os.getpid()),
                                      host=socket.gethostname(), task=task['key'],
                                      expected_peak_mib=task['reserve_mib'], full_peak_mib=task['reserve_mib'],
                                      ram_mib=task['ram_mib'], updated=time.time())
                        peers[str(os.getpid())] = record
                        native.save(path, dict(reservations=peers, updated=time.time()))
                        return reservations_dir, path, record
            time.sleep(5)

    native.reserve_gpu = reserve_gpu

    def bounded_popen(*popen_args, **kwargs):
        env = kwargs.get('env')
        if env and env.get('HS6_EVAL_ALLOCATOR_MIB'):
            env = dict(env)
            runtime = Path(__file__).parent / 'sbh_eval_runtime'
            env['PYTHONPATH'] = str(runtime) + os.pathsep + env.get('PYTHONPATH', '')
            kwargs['env'] = env
        process = original_popen(*popen_args, **kwargs)
        if kwargs.get('start_new_session'):
            children.append(process)
        return process

    native.subprocess.Popen = bounded_popen

    def update(reservation, child, gpu, release=False):
        original_update(reservation, child, gpu, release=release)
        if release:
            return
        _, _, record = reservation
        if record.get('descendant_used_mib', 0) > record['full_peak_mib']:
            raise RuntimeError('Evaluation exceeded its reserved NVML memory budget')
        if original_ram() < policy['emergency_free_ram_mib']:
            raise RuntimeError('Host RAM safety floor reached; evaluation must release capacity')
        if cpu_used_mib(child) > record['ram_mib']:
            raise RuntimeError('Evaluation exceeded its reserved anonymous host RAM budget')

    native.update_gpu_reservation = update

    def run_one(run_args, manifest, task):
        allocator = task.get('allocator_mib', task['reserve_mib'] - 768)
        os.environ['HS6_EVAL_ALLOCATOR_MIB'] = str(allocator)
        os.environ['HS6_EVAL_MEMORY_REPORTS'] = str(args.campaign / 'memory_reports' / task['key'])
        native.save(args.campaign / 'capacity_records' / (task['key'] + '.json'), dict(
            task=task['key'], gpu=run_args.gpu, policy=str(policy_path),
            policy_sha256=policy_hash,
            reserve_mib=task['reserve_mib'], allocator_mib=allocator,
            ram_mib=task['ram_mib'], started=time.time(), evaluator_command=task['command']))
        result = original_run_one(run_args, manifest, task)
        failure_path = args.campaign / '_state/failures' / (task['key'] + '.json')
        failure = original_read(failure_path, {})
        if failure and not (args.campaign / '_state/done' / (task['key'] + '.json')).exists():
            error = failure.get('error', '')
            log = Path(failure.get('log', '/nonexistent'))
            tail = ''
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 8192))
                    tail = stream.read().decode(errors='replace')
            if 'Host RAM safety floor' in error:
                event = args.campaign / 'resource_preemptions' / f'{task["key"]}.{int(time.time())}.json'
                native.save(event, failure)
                failure.update(attempts=max(0, failure.get('attempts', 1) - 1), resource_wait=True)
                native.save(failure_path, failure)
            elif 'OutOfMemoryError' in tail or 'exceeded its reserved' in error:
                native.save(args.campaign / 'capacity_blocks' / (task['key'] + '.json'), dict(
                    policy_sha256=policy_hash, error=error, log=str(log), updated=time.time()))
        return result

    native.run_one = run_one
    native.main()


if __name__ == '__main__':
    main()
