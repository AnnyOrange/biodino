#!/usr/bin/env python3
"""Discover EMA exports, queue the locked screening protocol, and audit real progress."""
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from supervise_sbh_20tb_hxw_20261027 import identity, read, write


def matching_processes(script, campaign):
    matches = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if script.encode() in args and str(campaign).encode() in args and b'--task' not in args:
                if identity(p.name):
                    matches.append(int(p.name))
        except OSError:
            pass
    return matches


def launch(command, log):
    with log.open('a') as stream:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True).pid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args()
    root = a.campaign
    lock = (root / 'watchdog.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    plan = read(root / 'plan.json')
    supervisor = str(Path(plan['snapshot']) / 'scripts/supervise_sbh_20tb_hxw_20261027.py')
    if not matching_processes(supervisor, root) and not (root / 'STOP_NEW_TRAINING').exists():
        launch(['/usr/bin/python3', supervisor, '--campaign', str(root)], root / 'supervisor.log')

    repo = Path('/home/xzj/screening45_8k_20261024/repo')
    source = repo / 'outputs/02_eval_runs/screening45_8k_20261024/hxw/campaign_manifest.json'
    template_path = root / 'eval_template.json'
    if not template_path.exists():
        write(template_path, read(source))
    template = read(template_path)
    template_tasks = [t for t in template['tasks'] if t['arm'] == 'l20tb_domain' and t['checkpoint'] == 57095]
    assert len(template_tasks) == 41  # Four retrieval jobs also return four clustering scores.
    eval_root = root / 'evaluation'
    manifest_path = eval_root / 'campaign_manifest.json'
    manifest = read(manifest_path)
    if manifest is None:
        manifest = copy.deepcopy(template)
        manifest.update(tasks=[], checkpoint_assets=[], site='hxw_sbh_20tb_20261027',
                        user_authorization='2026-10-27 S+/B/H+ training with automatic evaluation of every export',
                        created=time.time(), private_cache_root=str(root / 'eval_cache'),
                        deployment_root=str(eval_root), historical_reference='L comparison, corrected screening45 E50 seed0; not full V4-ID',
                        aggregate_allowed=False)
        write(manifest_path, manifest)
    observations = read(root / 'checkpoint_observations.json', {})
    known = {(x['arm'], x['checkpoint']) for x in manifest['checkpoint_assets']}
    added = []
    for model in plan['models']:
        arm = '20tb_' + model['name']
        for weight in sorted(Path(model['output']).glob('eval/training_*/teacher_checkpoint.pth')):
            step = int(weight.parent.name.removeprefix('training_'))
            if (arm, step) in known:
                continue
            stat = weight.stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            previous = observations.get(str(weight))
            observations[str(weight)] = signature
            if previous != signature or time.time() - stat.st_mtime < 120:
                continue
            import torch
            state = torch.load(weight, map_location='cpu', mmap=True, weights_only=False)
            assert 'teacher' in state and any(k.startswith('backbone.') for k in state['teacher'])
            del state
            digest = hashlib.sha256()
            with weight.open('rb') as stream:
                for block in iter(lambda: stream.read(16 << 20), b''):
                    digest.update(block)
            asset = dict(arm=arm, checkpoint=step, path=str(weight), config=model['config'],
                         bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest(),
                         config_sha256=hashlib.sha256(Path(model['config']).read_bytes()).hexdigest())
            new_tasks = []
            for base in template_tasks:
                task = copy.deepcopy(base)
                key = f'{arm}_ck{step}__{base["family"]}__{base["dataset"]}'
                old_config = base['command'][base['command'].index('--train-config') + 1]
                replacements = {base['key']: key, old_config: model['config']}
                if '--checkpoint' in base['command']:
                    old_checkpoint = base['command'][base['command'].index('--checkpoint') + 1]
                    replacements[old_checkpoint] = str(weight)
                else:
                    old_adapter = base['command'][base['command'].index('--checkpoints-dir') + 1]
                    replacements[old_adapter] = '{ROOT}/adapters/' + arm
                command = []
                for value in base['command']:
                    for before, after in replacements.items():
                        value = value.replace(before, after)
                    if value == '57095':
                        value = str(step)
                    if value == 'dinov3-57095':
                        value = f'dinov3-{step}'
                    command.append(value)
                task.update(key=key, arm=arm, checkpoint=step, command=command,
                            relative_cell=base['relative_cell'].replace(base['key'], key),
                            relative_output=base['relative_output'].replace(base['key'], key))
                # Only admission estimates change; evaluator batches, seeds and fitting budgets are locked.
                if model['name'] == 'hplus':
                    task['reserve_mib'] = max(16000, int(task['reserve_mib'] * 2.2))
                new_tasks.append(task)
            # Segmentation discovers <iteration>/checkpoint.pth; links stay on HXW.
            adapter = eval_root / 'adapters' / arm / str(step)
            adapter.mkdir(parents=True, exist_ok=True)
            link = adapter / 'checkpoint.pth'
            if not link.exists():
                link.symlink_to(weight)
            assert link.resolve() == weight.resolve()
            manifest['checkpoint_assets'].append(asset)
            manifest['tasks'].extend(new_tasks)
            added.append([arm, step])
    write(root / 'checkpoint_observations.json', observations)
    if added:
        manifest['updated'] = time.time()
        write(manifest_path, manifest)
    capacity_policy = read(root / 'evaluation_capacity_20261030/policy.json', {})
    runner = capacity_policy.get('runner', str(repo / 'scripts/run_screening45_8k_20261024.py'))
    eval_gpus = capacity_policy.get('gpus', list(range(8)))
    supervisors = matching_processes(runner, eval_root)
    if added:
        # Only the dispatch loop is replaced; task wrappers retain their locks and reservations.
        for pid in supervisors:
            os.kill(pid, signal.SIGTERM)
        supervisors = []
    if manifest['tasks'] and not supervisors and not capacity_policy.get('deployment_pending'):
        launch([plan['python'], runner, '--repo', str(repo), '--campaign', str(eval_root),
                '--python', plan['python'], '--gpus', *map(str, eval_gpus), '--max-per-gpu', '5',
                '--ceiling', '0.80', '--minimum-free-ram-mib', '65536'], root / 'eval_supervisor.log')
    coverage = []
    for asset in manifest['checkpoint_assets']:
        tasks = [t for t in manifest['tasks'] if (t['arm'], t['checkpoint']) == (asset['arm'], asset['checkpoint'])]
        scores = {}
        for t in tasks:
            done = read(eval_root / '_state/done' / (t['key'] + '.json'), {})
            if done.get('status') == 'VALID_COMPLETE':
                scores.update(done.get('scores', {}))
        coverage.append(dict(model=asset['arm'], checkpoint=asset['checkpoint'], valid_metrics=len(scores),
                             expected_metrics=45, complete=len(scores) == 45))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=['model', 'checkpoint', 'valid_metrics', 'expected_metrics', 'complete'])
    writer.writeheader()
    writer.writerows(coverage)
    (root / 'eval_coverage.csv').write_text(out.getvalue())
    if capacity_policy.get('runner'):
        spec = importlib.util.spec_from_file_location('screening_capacity_audit', repo / 'scripts/run_screening45_8k_20261024.py')
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        devices, processes = audit.gpu_state(), audit.proc_table()
        running = [read(p, {}) for p in (eval_root / '_state/running').glob('*.json')]
        actual, waiting = {}, []
        for device in map(str, eval_gpus):
            gpu = devices[device]
            tasks = []
            for task in running:
                if str(task.get('gpu')) != device or not identity(task.get('pid')):
                    continue
                descendants = audit.descendants(task['pid'], processes)
                used = sum(memory for pid, memory in gpu['apps'].items() if pid in descendants)
                record = dict(key=task['key'], wrapper_pid=task['pid'], used_mib=used)
                if used:
                    tasks.append(record)
                else:
                    waiting.append(record)
            actual[device] = dict(used_mib=gpu['used'], total_mib=gpu['total'],
                                  actual_test_count=len(tasks), tests=tasks)
        write(root / 'evaluation_capacity_20261030/live_status.json', dict(
            updated=time.time(), allowed_gpus=eval_gpus, gpus=actual, waiting_or_cpu_tasks=waiting,
            ram_available_mib=audit.available_ram_mib(),
            fully_evaluated_checkpoints=sum(x['complete'] for x in coverage),
            valid_metrics=sum(x['valid_metrics'] for x in coverage),
            resource_exception=capacity_policy.get('per_gpu_constraint'),
            high_resolution_constraint=capacity_policy.get('hplus_high_resolution_constraint')))
    write(root / 'watchdog_status.json', dict(updated=time.time(), added=added,
          exported_checkpoints=len(coverage), fully_evaluated_checkpoints=sum(x['complete'] for x in coverage),
          queued_tasks=len(manifest['tasks']), resource_exception='Eight GPUs assigned to training; evaluation only within real free memory and live reservations. Five concurrent tests per GPU may not fit.'))


if __name__ == '__main__':
    main()
