"""Audit or remove stopped, orphaned, claim-free queue parents on this host.

No process is resumed. Dry-run is the default; --apply uses pidfds so PID reuse
cannot redirect a signal. Incomplete claims newer than a worker protect it;
empty claims predating its birth cannot have been created by that worker.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
QUEUE_SCRIPT = ROOT / 'scripts/unified_paper_queue.py'


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Some cluster Python builds omit the wrappers despite kernel support.
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('pidfd wrapper unavailable on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    return fd


def pidfd_kill(fd):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, signal.SIGKILL, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal')


def claim_owners(host):
    owners = set()
    incomplete = []
    for claim in (OUT / 'locks').glob('*.claimdir'):
        try:
            record = (json.loads(os.readlink(claim)) if claim.is_symlink()
                      else json.loads((claim / 'owner.json').read_text()))
        except FileNotFoundError:
            if not claim.exists() and not claim.is_symlink():
                continue
            incomplete.append(dict(path=str(claim), changed=claim.stat().st_ctime))
            continue
        if record.get('host') == host:
            owners.update(int(record[key]) for key in ('pid', 'child_pid')
                          if record.get(key))
    return owners, incomplete


def snapshot(pid, host):
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        return None
    argv = [os.fsdecode(arg) for arg in (proc / 'cmdline').read_bytes().split(b'\0') if arg]
    if not any(Path(arg).name == 'unified_paper_queue.py' for arg in argv):
        return None
    script = next(Path(arg) for arg in argv if Path(arg).name == 'unified_paper_queue.py')
    if not script.is_absolute():
        script = (proc / 'cwd').resolve() / script
    if script.resolve() != QUEUE_SCRIPT.resolve():
        return None
    if '--host-label' not in argv or argv[argv.index('--host-label') + 1] != host:
        return None
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    children = []
    for thread in (proc / 'task').iterdir():
        children.extend((thread / 'children').read_text().split())
    return dict(pid=pid, state=stat[0], ppid=int(stat[1]), start_ticks=stat[19],
                children=sorted(set(children)), argv=argv)


def eligible(record, owners):
    return (record is not None and record['state'] == 'T' and record['ppid'] == 1
            and not record['children'] and record['pid'] not in owners)


def clean(host, apply=False):
    owners, incomplete = claim_owners(host)
    boot_time = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                         if line.startswith('btime ')))
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        fd = None
        try:
            record = snapshot(int(proc.name), host)
            if record is None:
                continue
            record['action'] = 'eligible' if eligible(record, owners) else 'preserved'
            born = boot_time + int(record['start_ticks']) / ticks_per_second
            if any(claim['changed'] >= born - 1 for claim in incomplete):
                record['action'] = 'incomplete_claim_preserved'
            rows.append(record)
            if not apply or record['action'] != 'eligible':
                continue
            fd = pidfd_open(record['pid'])
            fresh = snapshot(record['pid'], host)
            if (not eligible(fresh, owners) or fresh['start_ticks'] != record['start_ticks']
                    or fresh['argv'] != record['argv']):
                record['action'] = 'changed_preserved'
                continue
            pidfd_kill(fd)
            record['action'] = 'killed_stopped_orphan'
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            if fd is not None:
                os.close(fd)
    return dict(host=host, applied=apply, updated=time.time(), workers=rows,
                incomplete_claims=incomplete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = clean(args.host, args.apply)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    from collections import Counter
    print(json.dumps(dict(report=str(args.report), actions=dict(Counter(
        row['action'] for row in report['workers'])))))


if __name__ == '__main__':
    main()
