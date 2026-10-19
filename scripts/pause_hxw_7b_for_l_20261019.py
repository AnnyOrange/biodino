#!/usr/bin/env python3
"""Pause the hxw 7B trainer only after validating a committed optimizer checkpoint."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

from torch.distributed.checkpoint import FileSystemReader


DEFAULT_RUN = Path('/home/xzj/storage/merged/vit7b_20tb_recovered249_20261018')


def validate(path):
    metadata = FileSystemReader(path).read_metadata()
    keys = metadata.state_dict_metadata
    for prefix in ('model.student.backbone.', 'model.teacher.backbone.', 'optimizer.state.'):
        if not any(key.startswith(prefix) for key in keys):
            raise RuntimeError(f'Incomplete {prefix} in {path}')
    if 'iteration' not in keys:
        raise RuntimeError(f'Missing iteration in {path}')
    sizes = {}
    for item in metadata.storage_data.values():
        file = path / item.relative_path
        sizes.setdefault(file, file.stat().st_size)
        if sizes[file] < item.offset + item.length:
            raise RuntimeError(f'Truncated shard {file}')
    return dict(step=int(path.name), bytes=sum(sizes.values()), shards=len(sizes),
                metadata_mtime=(path / '.metadata').stat().st_mtime)


def latest(run):
    paths = sorted((path for path in (run / 'ckpt').glob('*')
                    if path.name.isdigit() and (path / '.metadata').is_file()),
                   key=lambda path: int(path.name), reverse=True)
    for path in paths:
        try:
            return validate(path)
        except (OSError, RuntimeError):
            continue
    raise RuntimeError('No complete 7B optimizer checkpoint')


def matches(pid, required):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and required.encode() in path.read_bytes()


def wait_gone(pid, required, seconds):
    for _ in range(seconds):
        if not matches(pid, required):
            return True
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--min-step', type=int, default=0)
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--batch-change', action='store_true')
    args = parser.parse_args()
    run = args.run
    checkpoint = latest(run)
    state = json.loads((run / 'recovery_state.json').read_text())
    log = Path(state['log']).read_text()
    saving = log.rfind('Saving DCP checkpoint') > log.rfind('Saved DCP sharded checkpoint')
    supervisor_pid = state.get('supervisor_pid', -1)
    train_pid = state.get('train_pid', -1)
    supervisor_active = matches(supervisor_pid, 'resume_7b_20tb_20261018.py')
    train_active = matches(train_pid, 'torch.distributed.run')
    ready = checkpoint['step'] >= args.min_step and (not saving or not (supervisor_active or train_active))
    status = dict(checkpoint=checkpoint, saving=saving, supervisor_active=supervisor_active,
                  train_active=train_active, ready=ready, min_step=args.min_step)
    if not args.pause or not ready:
        print(json.dumps(status))
        return

    if supervisor_active:
        os.kill(supervisor_pid, signal.SIGTERM)
        if not wait_gone(supervisor_pid, 'resume_7b_20tb_20261018.py', 20):
            raise RuntimeError('7B supervisor did not exit; training was left running')
    if train_active and matches(train_pid, 'torch.distributed.run'):
        os.killpg(train_pid, signal.SIGTERM)
        if not wait_gone(train_pid, 'torch.distributed.run', 90):
            os.killpg(train_pid, signal.SIGKILL)
            if not wait_gone(train_pid, 'torch.distributed.run', 10):
                raise RuntimeError('7B process group did not exit')
    checkpoint = latest(run)
    reason = ('Correct 7B effective batch to 1024 after preserving the 288-batch run'
              if args.batch_change else 'User authorized L priority and 7B pause on 2026-10-19')
    status.update(checkpoint=checkpoint, paused_unix=time.time(),
                  reason=reason,
                  supervisor_active=False, train_active=False)
    state.update(status='PAUSED_FOR_BATCH_CORRECTION' if args.batch_change else 'PAUSED_FOR_L',
                 last_checkpoint=str(run / 'ckpt' / str(checkpoint['step'])),
                 train_pid=None, supervisor_pid=None, paused_unix=status['paused_unix'])
    state_temp = (run / 'recovery_state.json').with_suffix('.tmp')
    state_temp.write_text(json.dumps(state, indent=2) + '\n')
    state_temp.replace(run / 'recovery_state.json')
    record = run / ('paused_for_batch1024_20261019.json' if args.batch_change
                    else 'paused_for_l_20261019.json')
    temp = record.with_suffix('.tmp')
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(record)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
