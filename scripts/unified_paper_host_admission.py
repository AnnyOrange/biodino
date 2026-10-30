"""Host-local RAM reservations, retained through CPU postprocessing until exit.

Use /tmp rather than NFS for the lock. Account for old evaluators that predate
this module, and subtract only their *unallocated* expected peak from available
RAM. This is conservative admission, not a hard limit on existing processes.
"""
import fcntl
import json
import os
from pathlib import Path
import time
from unified_paper_memory import memory

GIB = 1024 ** 3
DIRECTORY = Path('/tmp') / f'dinov3-host-admission-{os.getuid()}'


def expected_peak(entry):
    return (96 if 'ctc' in entry else 32) * GIB


def evaluators():
    rows = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            entry = next((os.fsdecode(x).rsplit('/', 1)[-1] for x in args
                          if x.rsplit(b'/', 1)[-1].startswith(b'run_unified_paper_')
                          and x.endswith(b'.py')), None)
            if entry is None:
                continue
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            rows[int(proc.name)] = dict(ppid=int(stat[1]), start=stat[19],
                rss=int(stat[21]) * os.sysconf('SC_PAGE_SIZE'), entry=entry)
        except (OSError, ValueError):
            continue
    # Forked data-loader workers inherit argv; count each evaluation only once.
    return {pid: row for pid, row in rows.items() if row['ppid'] not in rows}


def budget(total, available, rows, reservations, pid, peak):
    floor = max(8 * GIB, total * .20)
    active = {p: r for p, r in rows.items() if p != pid and
              reservations.get(str(p), {}).get('status') != 'waiting'}
    outstanding = sum(max(0, reservations.get(str(p), {}).get('peak',
                         expected_peak(r['entry'])) - r['rss']) for p, r in active.items())
    own_rss = rows.get(pid, {}).get('rss', 0)
    fits = available - outstanding - max(0, peak - own_rss) >= floor
    return fits and len(active) < 32, dict(available_gib=round(available / GIB, 2),
        pending_peak_gib=round(outstanding / GIB, 2), requested_peak_gib=peak / GIB,
        floor_gib=round(floor / GIB, 2), active=len(active), max_active=32)


def launch_ready(job, pid=0):
    """Cheap parent-side prefilter; child still performs atomic admission."""
    rows = evaluators()
    path = DIRECTORY / 'reservations.json'
    prior = json.loads(path.read_text()) if path.exists() else {}
    live = {p: r for p, r in prior.items() if int(p) in rows and rows[int(p)]['start'] == r['start']}
    peak = max(expected_peak(job.get('entrypoint', '')), int(job.get('host_peak_mib', 0)) * 1024 ** 2)
    return budget(*memory(), rows, live, pid, peak)


def reserve_for_current_command(job):
    DIRECTORY.mkdir(mode=0o700, exist_ok=True)
    pid = os.getpid()
    start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    peak = max(expected_peak(job.get('entrypoint', '')), int(job.get('host_peak_mib', 0)) * 1024 ** 2)
    state_path = DIRECTORY / 'reservations.json'
    notice = 0
    while True:
        with (DIRECTORY / 'admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = evaluators()
            prior = json.loads(state_path.read_text()) if state_path.exists() else {}
            live = {p: r for p, r in prior.items() if int(p) in rows and
                    rows[int(p)]['start'] == r['start']}
            fits, details = budget(*memory(), rows, live, pid, peak)
            live[str(pid)] = dict(start=start, peak=peak, task=job['key'],
                                 status='admitted' if fits else 'waiting', updated=time.time())
            tmp = state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(live))
            tmp.replace(state_path)
        if fits:
            print('HOST RAM ADMITTED', job['key'], details, flush=True)
            return
        if time.monotonic() - notice > 60:
            print('HOST RAM WAIT', job['key'], details, flush=True)
            notice = time.monotonic()
        time.sleep(15)
