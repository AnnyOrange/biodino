#!/usr/bin/env python3
"""Run the domain-conditioned L/20TB candidate with bounded checkpoint recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--gpus', default='3,4,5,7')
    p.add_argument('--port', type=int, default=29723)
    p.add_argument('--config', type=Path)
    p.add_argument('--min-free-mib', type=int, default=0,
                   help='0 requires idle GPUs; positive value permits co-location with admitted free memory')
    a = p.parse_args()
    lock = (a.output / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = a.config or a.output / 'candidate.yaml'
    cfg = OmegaConf.load(config)
    end = int(cfg.train.max_updates)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, PYTHONPATH=str(a.snapshot),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MALLOC_ARENA_MAX='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               NCCL_DEBUG='WARN', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
               DINOV3_TEACHER_DOMAIN_METADATA='1', DINOV3_SPECIMEN_PAIR_PROB='0')
    state = dict(status='STARTING', started=time.time(), attempts=[])
    state_path = a.output / 'supervisor_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())

    def save(**kwargs):
        state.update(kwargs, updated=time.time(), supervisor_pid=os.getpid())
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    failures, previous_start = 0, -1
    while failures < 3:
        checkpoints = sorted((p for p in (a.output/'ckpt').glob('*') if p.name.isdigit()
                              and (p/'checkpoint.pth').is_file()), key=lambda p: int(p.name))
        if not checkpoints:
            raise RuntimeError('Full source optimizer checkpoint is required')
        start = int(checkpoints[-1].name) + 1
        if start >= end:
            save(status='COMPLETE', checkpoint=str(checkpoints[-1]))
            return
        if start > previous_start:
            failures = 0
            previous_start = start
        occupied = subprocess.check_output(['nvidia-smi', '-i', a.gpus,
                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        memory = subprocess.check_output(['nvidia-smi', '-i', a.gpus,
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        admitted = (min(map(int, memory.splitlines())) >= a.min_free_mib
                    if a.min_free_mib else not occupied.strip())
        if not admitted:
            save(status='WAITING_FOR_GPUS', occupied=occupied.splitlines())
            time.sleep(30)
            continue
        cmd = [a.python, '-m', 'torch.distributed.run', '--nnodes=1',
               f'--nproc_per_node={len(a.gpus.split(","))}',
               '--master_addr=127.0.0.1', f'--master_port={a.port}', 'dinov3/train/train.py',
               '--config-file', str(config), '--output-dir', str(a.output), '--seed', '0',
               '--expected-start-iteration', str(start)]
        attempt = dict(number=len(state['attempts'])+1, start=start, started=time.time(), command=cmd)
        state['attempts'].append(attempt)
        logfile = a.output / f'attempt_{attempt["number"]:03d}.log'
        with logfile.open('a') as handle:
            proc = subprocess.Popen(cmd, cwd=a.snapshot, env=env, stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            save(status='TRAINING', train_pid=proc.pid, log=str(logfile))
            attempt.update(returncode=proc.wait(), finished=time.time())
        failures += 1
        save(status='CHECKING_CHECKPOINT', consecutive_failures=failures)
        time.sleep(30)
    save(status='STOPPED_AFTER_THREE_FAILURES')


if __name__ == '__main__':
    main()
