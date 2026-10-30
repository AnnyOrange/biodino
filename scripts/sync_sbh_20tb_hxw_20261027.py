#!/usr/bin/env python3
"""Mirror training/evaluation records without transferring checkpoints or caches."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/01_training_runs/hs6_sbh_20tb_hxw_20261027'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'sync.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    command = ['rsync', '-a', '--exclude=/source/', '--exclude=/tmp/',
               '--exclude=/runs/*/ckpt/', '--exclude=/runs/*/eval/', '--exclude=/eval_cache/',
               '--exclude=*/cache/', '--exclude=*/features/', '--exclude=*/archived_cache/',
               '--exclude=*.pth', '--exclude=*.pt', '--exclude=*.safetensors',
               '--exclude=*.npy', '--exclude=*.npz', '--exclude=*.pkl',
               '--include=*/', '--include=*.json', '--include=*.yaml', '--include=*.jsonl',
               '--include=*.csv', '--include=*.md', '--include=*.log', '--exclude=*',
               '5090-hxw-xzj:/home/xzj/storage/merged/hs6_sbh_20tb_hxw_20261027/', str(OUT) + '/']
    result = subprocess.run(command, text=True, capture_output=True, timeout=240)
    record = dict(updated=time.time(), returncode=result.returncode,
                  stderr=result.stderr[-2000:], checkpoints_transferred=False, caches_transferred=False)
    temp = OUT / 'sync_status.json.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(OUT / 'sync_status.json')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
