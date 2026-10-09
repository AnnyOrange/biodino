#!/usr/bin/env python3
"""Resume H+ from full ck20983 and supervise eight-GPU checkpointed restarts."""

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

BASE = Path('/data/xuzijing/biodino/outputs/01_training_runs')
SOURCE_RUN = BASE / 'HS6_Hplus_5tb_no_fsdp_fromck13663_bs64_8x5090lyxxr_20260924'
ADMIN = Path('/data/xuzijing/biodino/outputs/auto_train_logs/hplus_supervisor_20260927')
LAUNCHER = ADMIN.parent / 'resume_hs6_hplus_5tb_ddp_8x5090lyxxr_20260924.sh'
REPO = Path('/data/xuzijing/biodino_hplus_ddp_20260922')
TARGET_UPDATES = 4098 * 15


def write_status(state, **extra):
    obj = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'state': state,
           'supervisor_pid': os.getpid(), **extra}
    temp = ADMIN / 'status.json.tmp'
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(ADMIN / 'status.json')


def log(message):
    print(dt.datetime.now(dt.timezone.utc).isoformat(), message, flush=True)


def memory():
    values = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    available = values['MemAvailable']
    return 1 - available / values['MemTotal'], available / 2**30


def gpu_pids():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip().isdigit()}


def latest(run):
    # The training checkpointer atomically renames the directory after saving.
    files = []
    for p in (run / 'ckpt').glob('*/checkpoint.pth'):
        if p.parent.name.isdigit() and p.stat().st_size > 8_000_000_000:
            files.append((int(p.parent.name), p))
    return max(files, default=(-1, None))


def progress(run):
    path = run / 'raw_loss_metrics.jsonl'
    try:
        with path.open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 16000))
            lines = f.read().splitlines()
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                return obj.get('optimizer_updates_completed', 0)
            except (ValueError, TypeError):
                continue
    except OSError:
        pass
    return 0


def own_processes(run):
    result = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
            decoded = [a.decode(errors='replace') for a in args]
            if '--output-dir' in decoded:
                index = decoded.index('--output-dir')
                if decoded[index + 1] == str(run) and 'dinov3/train/train.py' in decoded:
                    result.append(int(directory.name))
        except (OSError, IndexError):
            pass
    return result


def stop(child, run):
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        if child.poll() is not None and not own_processes(run):
            return
        time.sleep(2)
    # Only signal processes whose live argv still belongs to this exact run.
    for pid in own_processes(run):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=30)


def main():
    ADMIN.mkdir(parents=True, exist_ok=True)
    with (ADMIN / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_step, source = latest(SOURCE_RUN)
        if source_step != 20983 or source is None:
            raise RuntimeError(f'Expected full ck20983, found {source_step}')
        consecutive_failures = 0
        segment = 0
        while True:
            while gpu_pids() or memory()[0] > 0.65:
                write_status('WAITING_FOR_RESOURCES', checkpoint=source_step,
                             ram_available_gib=round(memory()[1], 2))
                time.sleep(15)
            # Preserve each segment's metric history, including unsaved updates.
            while True:
                run = BASE / f'HS6_Hplus_5tb_no_fsdp_fromck{source_step}_bs64_8x5090lyxxr_20260927_r{segment:03d}'
                segment += 1
                if not run.exists():
                    break
            destination = run / 'ckpt' / str(source_step) / 'checkpoint.pth'
            destination.parent.mkdir(parents=True)
            os.link(source, destination)
            env = dict(os.environ, OUTPUT_DIR=str(run), SOURCE_ITER=str(source_step))
            output_log = ADMIN / f'{run.name}.log'
            with output_log.open('ab') as stream:
                child = subprocess.Popen(['bash', str(LAUNCHER)], cwd=REPO, env=env,
                                         stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            log(f'START pid={child.pid} checkpoint={source_step} run={run}')
            (ADMIN / 'latest_run.txt').write_text(str(run) + '\n')
            trigger_step = None
            reason = 'unexpected_exit'
            resume_verified = False
            last_progress = 0
            last_progress_time = time.monotonic()
            while child.poll() is None:
                used, available_gib = memory()
                checkpoint_step, _ = latest(run)
                current_progress = progress(run)
                if current_progress > last_progress:
                    last_progress, last_progress_time = current_progress, time.monotonic()
                if not resume_verified:
                    content = output_log.read_text(errors='replace')
                    match = re.search(r'Loaded consolidated model checkpoint with (\d+) missing keys and (\d+) unexpected keys', content)
                    if match:
                        if tuple(map(int, match.groups())) != (0, 0):
                            stop(child, run)
                            write_status('CHECKPOINT_LOAD_ERROR', run=str(run), train_log=str(output_log))
                            raise RuntimeError('Incomplete model restore; refusing to continue training')
                        resume_verified = True
                state = 'RUNNING' if current_progress > source_step else 'STARTING'
                write_status(state if trigger_step is None else 'WAITING_FOR_RAM_CHECKPOINT',
                             train_pid=child.pid, run=str(run), checkpoint=checkpoint_step,
                             optimizer_updates_completed=current_progress,
                             resume_verified=resume_verified,
                             ram_used_fraction=round(used, 4), ram_available_gib=round(available_gib, 2),
                             train_log=str(output_log))
                if time.monotonic() - last_progress_time > 1800:
                    reason = 'no_progress_for_30_minutes'
                    stop(child, run)
                    break
                if used >= 0.75 and trigger_step is None:
                    trigger_step = checkpoint_step
                    log(f'RAM threshold {used:.1%}: wait for checkpoint after {trigger_step}')
                if used >= 0.85:
                    reason = 'ram_emergency_latest_checkpoint'
                    stop(child, run)
                    break
                if trigger_step is not None and checkpoint_step > trigger_step:
                    reason = 'ram_checkpoint_restart'
                    stop(child, run)
                    break
                time.sleep(15)
            stop(child, run)  # Reap lingering ranks before launching another group.
            completed = progress(run)
            new_step, new_source = latest(run)
            log(f'EXIT rc={child.returncode} reason={reason} updates={completed} checkpoint={new_step}')
            if child.returncode == 0 and completed >= TARGET_UPDATES:
                write_status('COMPLETE', run=str(run), optimizer_updates_completed=completed)
                return
            if new_step > source_step:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= 3:
                write_status('REPEATED_FAILURE', run=str(run), checkpoint=new_step,
                             returncode=child.returncode, train_log=str(output_log))
                raise RuntimeError('Three exits without a new checkpoint; stopping retry loop')
            source_step, source = new_step, new_source
            write_status('RESTARTING', checkpoint=source_step, reason=reason)
            time.sleep(30)


if __name__ == '__main__':
    main()
