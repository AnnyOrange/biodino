#!/usr/bin/env python3
"""Chain the live four-rank Adaptive run from ck29279 through ck30743."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_ddp4_20260929'
CHECKPOINT = OUTPUT / 'ckpt/29279/checkpoint.pth'
FINAL = OUTPUT / 'ckpt/30743/checkpoint.pth'


def now():
    return datetime.now(timezone.utc).isoformat()


def save_status(**fields):
    path = OUTPUT / 'continuation_status.json'
    path.write_text(json.dumps(dict(time_utc=now(), **fields), indent=2) + '\n')


def command_for(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true')
    args = parser.parse_args()
    launch = json.loads((OUTPUT / 'launch_manifest.json').read_text())
    first = json.loads((OUTPUT / 'launcher_status.json').read_text())
    pid = int(first['pid'])
    cmdline = command_for(pid)
    if not cmdline or '--master_port=32944' not in cmdline or str(OUTPUT) not in cmdline:
        raise RuntimeError('Live four-rank Adaptive PID does not match launch manifest')
    if args.daemon:
        with (OUTPUT / 'continuation_monitor.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                    cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        save_status(state='WATCHING', monitor_pid=proc.pid, current_train_pid=pid,
                    expected_handoff_step=29279, target_step=30743)
        print(json.dumps(dict(monitor_pid=proc.pid, train_pid=pid, target_step=30743)))
        return
    while command_for(pid) == cmdline:
        time.sleep(30)
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 3_000_000_000:
        save_status(state='FAILED', reason='initial training stopped before complete ck29279')
        return
    if FINAL.exists():
        save_status(state='DONE', reason='ck30743 already exists')
        return
    cmd = launch['command'][:]
    old = 'train.max_updates=29280'
    if cmd.count(old) != 1:
        raise RuntimeError('Initial command no longer has its expected stopping point')
    cmd[cmd.index(old)] = 'train.max_updates=30744'
    if cmd.count('compute_precision.distributed_mode=ddp') != 1 or cmd.count('train.batch_size_per_gpu=128') != 1:
        raise RuntimeError('DDP or per-rank batch changed unexpectedly')
    source = Path(launch['source'])
    original = json.loads((Path(launch['parent']) / 'launch_manifest.json').read_text())
    env = os.environ.copy()
    env.update(original['environment'], CUDA_VISIBLE_DEVICES='0,3,4,5',
               PYTHONPATH=str(source), OMP_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1', NCCL_NET='Socket',
               NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0')
    manifest = dict(time_utc=now(), source=str(source), command=cmd,
                    resume_checkpoint=str(CHECKPOINT), final_checkpoint=str(FINAL),
                    cuda_visible_devices='0,3,4,5', world_size=4,
                    effective_global_batch=1024)
    (OUTPUT / 'continuation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (OUTPUT / 'continuation.log').open('a') as log:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_status(state='LAUNCHED', train_pid=proc.pid, resume_step=29279,
                target_step=30743)
    print(json.dumps(dict(state='LAUNCHED', pid=proc.pid, resume_step=29279,
                          target_step=30743)), flush=True)


if __name__ == '__main__':
    main()
