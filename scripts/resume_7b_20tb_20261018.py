#!/usr/bin/env python3
"""Recover the interrupted 20TB 7B run, then supervise bounded exact restarts.

The old iteration-249 optimizer was never committed. Bootstrap from its complete
teacher DCP, preserving all backbone/head tensors, with fresh AdamW state. Commit
one real update before continuing the original 3000-update schedule. Subsequent
restarts require a complete model+optimizer checkpoint from this new run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def validate_checkpoint(path, full=False):
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(path).read_metadata()
    keys = metadata.state_dict_metadata
    if full and not all(any(k.startswith(prefix) for k in keys) for prefix in
                        ('model.student.backbone.', 'model.teacher.backbone.', 'optimizer.state.')):
        raise ValueError(f'Incomplete model/optimizer inventory: {path}')
    if 'iteration' not in keys:
        raise ValueError(f'Missing iteration: {path}')
    sizes = {}
    for item in metadata.storage_data.values():
        filename = path / item.relative_path
        if filename not in sizes:
            sizes[filename] = filename.stat().st_size
        if sizes[filename] < item.offset + item.length:
            raise ValueError(f'Truncated DCP shard: {filename}')
    return dict(tensors=len(keys), bytes=sum(sizes.values()), shards=len(sizes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--original-config', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accum-steps', type=int, default=32)
    p.add_argument('--checkpoint-period', type=int, default=25)
    a = p.parse_args()
    world_size = len(a.gpus.split(','))
    if world_size * a.batch_size * a.accum_steps != 1024:
        p.error('The plain 7B baseline requires effective global batch 1024')
    if a.checkpoint_period < 1:
        p.error('Checkpoint period must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_inventory = validate_checkpoint(a.teacher)
    state_path = a.output / 'recovery_state.json'
    state = dict(status='STARTING', original_config=str(a.original_config), teacher=str(a.teacher),
                 initialization='Complete step249 EMA teacher to student and EMA; fresh optimizer and loss centers',
                 logical_start=250, max_updates=3000, global_batch=1024, world_size=world_size,
                 source_inventory=source_inventory, started=time.time(), attempts=[])
    if state_path.exists():
        state = json.loads(state_path.read_text())
    state.update(world_size=world_size, gpus=a.gpus, batch_size_per_gpu=a.batch_size,
                 accumulation_steps=a.accum_steps, global_batch=1024,
                 checkpoint_period=a.checkpoint_period)

    def save(**kwargs):
        state.update(kwargs, updated=time.time(), supervisor_pid=os.getpid())
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, PYTHONPATH=str(a.repo),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MALLOC_ARENA_MAX='2',
               PYTORCH_ALLOC_CONF='expandable_segments:True', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               NCCL_DEBUG='WARN', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
    env['DINOV3_DCP_SINGLE_FILE_PER_RANK'] = '0'
    failures, last_checkpoint = 0, -1
    while failures < 3:
        checkpoints = sorted((d for d in (a.output / 'ckpt').glob('*')
                              if d.is_dir() and d.name.isdigit() and (d/'.metadata').is_file()),
                             key=lambda d: int(d.name))
        latest = checkpoints[-1] if checkpoints else None
        if latest:
            inventory = validate_checkpoint(latest, full=True)
            start = int(latest.name) + 1
            if start > last_checkpoint:
                failures = 0
                last_checkpoint = start
            if start >= 3000:
                save(status='COMPLETE', last_checkpoint=str(latest), checkpoint_inventory=inventory)
                return
        else:
            start = 250
        occupied = subprocess.check_output(['nvidia-smi', '-i', a.gpus,
                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if occupied.strip():
            save(status='WAITING_FOR_GPUS', occupied_pids=occupied.splitlines())
            time.sleep(30)
            continue
        if shutil.disk_usage(a.output).free < 350 * 1024**3:
            save(status='STOPPED_LOW_DISK', required_free_gib=350)
            return
        bootstrap = latest is None
        end = 251 if bootstrap else 3000
        cmd = [a.python, '-m', 'torch.distributed.run', '--nnodes=1', f'--nproc_per_node={world_size}',
               '--master_addr=127.0.0.1', '--master_port=29721', 'dinov3/train/train.py',
               '--config-file', str(a.original_config), '--output-dir', str(a.output), '--seed', '0',
               '--expected-start-iteration', '0' if bootstrap else str(start)]
        if bootstrap:
            cmd.append('--no-resume')
        cmd += [f'train.max_updates={end}', f'train.batch_size_per_gpu={a.batch_size}', 'train.num_workers=1',
                'train.wds_shuffle_buffer=128', 'train.prefetch_factor=1',
                f'optim.gradient_accumulation_steps={a.accum_steps}',
                f'train.start_iteration_override={250 if bootstrap else "null"}',
                f'student.resume_from_teacher_chkpt={a.teacher if bootstrap else "null"}',
                f'evaluation.eval_period_iterations={1 if bootstrap else 250}',
                f'checkpointing.period={1 if bootstrap else a.checkpoint_period}',
                'checkpointing.max_to_keep=2']
        attempt = dict(number=len(state['attempts']) + 1, start=start, end=end, bootstrap=bootstrap,
                       started=time.time(), command=cmd)
        state['attempts'].append(attempt)
        logpath = a.output / f'attempt_{attempt["number"]:03d}.log'
        with logpath.open('a') as logfile:
            proc = subprocess.Popen(cmd, cwd=a.repo, env=env, stdin=subprocess.DEVNULL,
                                    stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            save(status='BOOTSTRAP_TEACHER249' if bootstrap else 'TRAINING', train_pid=proc.pid, log=str(logpath))
            while proc.poll() is None:
                ram = next(int(s.split()[1]) / 1024**2 for s in Path('/proc/meminfo').read_text().splitlines()
                           if s.startswith('MemAvailable:'))
                # DCP on the existing slow disks can take >45 minutes. Allow six
                # hours without log progress; there is no wall-time limit on training.
                if ram < 24 or time.time() - logpath.stat().st_mtime > 6 * 3600:
                    attempt['abort_reason'] = (f'available RAM {ram:.2f} GiB <24 GiB'
                        if ram < 24 else 'log unchanged for 6 hours')
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                save(available_ram_gib=ram)
                time.sleep(20)
            attempt.update(returncode=proc.wait(), finished=time.time())
        committed = a.output / 'ckpt' / str(end - 1)
        if attempt['returncode'] == 0 and committed.is_dir():
            validate_checkpoint(committed, full=True)
            save(status='BOOTSTRAP_COMMITTED' if bootstrap else 'COMPLETE', last_checkpoint=str(committed))
            if not bootstrap:
                return
        else:
            failures += 1
            save(status='RETRY_BACKOFF', consecutive_failures=failures)
            time.sleep(60)
    save(status='STOPPED_AFTER_THREE_FAILURES')


if __name__ == '__main__':
    main()
