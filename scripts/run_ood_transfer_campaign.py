#!/usr/bin/env python3
"""Resource-bounded suxin scheduler with validation-only recipe selection."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import read_json, save_json, sha256


def alive(job):
    try:
        pid = str(job['pid'])
        stat = Path('/proc', pid, 'stat').read_text().split(') ', 1)[1].split()
        cmd = Path('/proc', pid, 'cmdline').read_bytes()
        return stat[0] != 'Z' and stat[19] == job['start_ticks'] and (b'run_ood_full_finetune.py' in cmd or b'run_ood_frozen_transfer.py' in cmd)
    except (OSError, KeyError):
        return False


def select_recipes(root, cfg):
    for branch, spec in cfg['branches'].items():
        target = root / 'selected' / branch
        if (target / 'lock.json').exists():
            continue
        candidates = []
        for candidate in spec['candidates']:
            runs = [root / 'runs' / f"{candidate['name']}_seed{seed}" for seed in cfg['development_seeds']]
            if not all((run / 'complete.json').exists() for run in runs):
                break
            results = [read_json(run / 'complete.json') for run in runs]
            if any(r['identity']['recipe_sha256'] != sha256(candidate['recipe']) or r['test_used'] for r in results):
                raise ValueError('Development result identity differs')
            scores = [r['best']['validation']['selection_score'] for r in results]
            candidates.append(dict(**candidate, mean_validation=sum(scores)/len(scores), scores=scores,
                                   sources=[str(run / 'complete.json') for run in runs]))
        else:
            winner = max(candidates, key=lambda c: (c['mean_validation'], -c['resolution'], -c['lr']))
            save_json(target / 'recipe.json', read_json(winner['recipe']))
            save_json(target / 'lock.json', dict(recipe_sha256=sha256(target / 'recipe.json'), winner=winner,
                      candidates=candidates, test_used=False, criterion='mean best-validation across seeds101,102'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = Path(args.campaign).resolve()
    import psutil
    children = []
    with (root / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            cfg, caps = read_json(root / 'campaign.json'), read_json(root / 'caps.json')
            state_path = root / 'scheduler_state.json'
            states = read_json(state_path) if state_path.exists() else {}
            select_recipes(root, cfg)
            for child in children:
                child.poll()
            children = [p for p in children if p.returncode is None]
            test_jobs = [dict(job, id=job['id']+'_test', phase='test') for job in cfg['jobs'] if job['phase']=='formal']
            all_jobs = cfg.get('frozen_jobs', []) + cfg['jobs'] + test_jobs
            for job in all_jobs:
                record = states.setdefault(job['id'], dict(status='pending', attempts=0))
                marker = 'test_result.json' if job['phase']=='test' else 'complete.json'
                if (Path(job['output']) / marker).exists():
                    record['status'] = 'complete'
                elif record['status'] == 'running' and not alive(record):
                    record['status'] = 'failed'
            active = [s for s in states.values() if s['status'] == 'running' and alive(s)]
            gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                                                '--format=csv,noheader,nounits'], text=True)
            gpu_rows = [list(map(int, row.split(','))) for row in gpu_text.strip().splitlines()]
            occupied = {s['gpu'] for s in active}
            candidates = [gpu for gpu, total, used in gpu_rows if gpu in caps['gpus'] and gpu not in occupied
                          and total-used >= caps['gpu_peak_mib'] + total*caps['gpu_reserve_fraction']]
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=.25) / 100
            allowed = (cfg['enabled'] and not (root / 'PAUSE').exists() and candidates
                       and len(active) < caps['max_jobs'] and cpu <= caps['max_cpu_util']
                       and memory.available >= memory.total*caps['ram_reserve_fraction'] + (len(active)+1)*caps['ram_peak_gib']*1024**3
                       and shutil.disk_usage(root).free >= (caps['min_disk_free_gib'] + 12*(len(active)+1))*1024**3)
            if allowed:
                for job in all_jobs:
                    record = states[job['id']]
                    if record['status'] != 'pending':
                        continue
                    inputs = Path(cfg['inputs'])
                    frozen = job['phase'] == 'frozen'
                    testing = job['phase'] == 'test'
                    suffix = 'frozen' if frozen else 'test' if testing else 'development'
                    manifest = inputs / f"{job['task']}_{suffix}.json"
                    receipt = inputs / f"{job['task']}_{suffix}.verified.json"
                    if not (manifest.exists() and receipt.exists() and Path(cfg['checkpoint']).exists()):
                        continue
                    if job['phase'] == 'formal' and not Path(job['recipe_lock']).exists():
                        continue
                    if testing:
                        branch_runs = [j for j in cfg['jobs'] if j['branch']==job['branch'] and j['phase']=='formal']
                        if not caps.get('allow_test',False) or not all((Path(j['output'])/'complete.json').exists() for j in branch_runs):
                            continue
                    output = Path(job['output']); output.mkdir(parents=True, exist_ok=True)
                    data_root = cfg.get('task_data_roots',{}).get(job['task'],cfg['data_root'])
                    if frozen:
                        command = [cfg['python'], str(Path(__file__).with_name('run_ood_frozen_transfer.py')),
                                   '--manifest',str(manifest),'--input-receipt',str(receipt),
                                   '--checkpoint',cfg['checkpoint'],'--train-config',cfg['train_config'],
                                   '--data-root',data_root,'--output',str(output)]
                    elif testing:
                        command=[cfg['python'],str(Path(__file__).with_name('run_ood_full_finetune.py')),'test',
                                 '--trained-run',str(output),'--test-manifest',str(manifest),'--input-receipt',str(receipt),
                                 '--checkpoint',cfg['checkpoint'],'--train-config',cfg['train_config'],
                                 '--data-root',data_root,'--microbatch',str(job['microbatch'])]
                    else:
                        command = [cfg['python'], str(Path(__file__).with_name('run_ood_full_finetune.py')), 'train',
                                   '--recipe', job['recipe'], '--phase', job['phase'], '--seed', str(job['seed']),
                                   '--development-manifest', str(manifest), '--input-receipt', str(receipt),
                                   '--checkpoint', cfg['checkpoint'], '--train-config', cfg['train_config'],
                                   '--data-root', data_root, '--output', str(output), '--microbatch', str(job['microbatch']),
                                   '--min-disk-free-gib', str(caps['min_disk_free_gib'])]
                        if job['phase'] == 'formal':
                            command += ['--recipe-lock', job['recipe_lock']]
                    env = os.environ.copy()
                    gpu = candidates[0]
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                        env[name] = str(caps['threads'])
                    with (output / 'train.log').open('a') as log:
                        proc = subprocess.Popen(['nice', '-n', str(caps['nice']), *command], env=env,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append(proc)
                    ticks = Path('/proc', str(proc.pid), 'stat').read_text().split(') ', 1)[1].split()[19]
                    record.update(status='running', pid=proc.pid, start_ticks=ticks, gpu=gpu,
                                  attempts=record['attempts']+1, started=time.time(), command=command)
                    break
            save_json(state_path, states)
            save_json(root / 'live_status.json', dict(time=time.time(), enabled=cfg['enabled'], running=len(active),
                      complete=sum(s['status']=='complete' for s in states.values()),
                      failed=sum(s['status']=='failed' for s in states.values()), total=len(all_jobs),
                      cpu_util=cpu, memory_available_gib=memory.available/1024**3,
                      disk_free_gib=shutil.disk_usage(root).free/1024**3,
                      test_jobs_started=sum(states[j['id']]['attempts']>0 for j in test_jobs),
                      note='Separate test command only after all5 formal seeds of a branch finish and allow_test is enabled.'))
            if args.once:
                return
            time.sleep(15)


if __name__ == '__main__':
    main()
