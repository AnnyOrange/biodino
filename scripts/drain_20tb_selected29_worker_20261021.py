#!/usr/bin/env python3
"""Finish and validate an existing evaluation while replacing its launcher."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--snapshot', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.snapshot / 'scripts'), str(a.snapshot)]
import run_l20tb_selected29_worker_20261019 as locked
q = locked.fleet.queue
root = a.output
state = root / '_state'
drain = json.loads((state / 'draining_old_worker.json').read_text())
manifest = json.loads((root / 'campaign_manifest.json').read_text())
q.verify_campaign_source(manifest)
child = Path('/proc', str(drain['child_pid']), 'cmdline')
while child.exists():
    try:
        cmd = child.read_bytes()
    except OSError:
        break
    if not cmd or not any(str(root / 'cells' / key).encode() in cmd for key in drain['tasks']):
        break
    time.sleep(5)
reports = []
for key in drain['tasks']:
    if not (state / 'done' / f'{key}.json').exists():
        task = copy.deepcopy(next(t for t in manifest['tasks'] if t['key'] == key))
        directory = root / 'cells' / key
        try:
            invocation = json.loads((directory / 'invocation_manifest.json').read_text())
            task['asset']['path'] = invocation['checkpoint']['path']
            report = locked.validate(task, directory, invocation)
            q.save(state / 'done' / f'{key}.json', report)
            reports.append(dict(task=key, status='VALID_COMPLETE'))
        except Exception as exc:
            archive = state / 'recovery' / f'drain_{int(time.time())}'
            archive.mkdir(parents=True, exist_ok=True)
            claim = state / 'claims' / key
            if claim.exists():
                shutil.move(str(claim), str(archive / key))
            reports.append(dict(task=key, status='REQUEUED', error=repr(exc)))
    (state / 'running' / f'{key}.json').unlink(missing_ok=True)
parent = Path('/proc', str(drain['launcher_pid']), 'cmdline')
if parent.exists() and b'lw5_selected29_fill_20261021/worker.py' in parent.read_bytes():
    os.kill(drain['launcher_pid'], signal.SIGKILL)
worker_file = state / 'workers/hxw:lw5frozen.json'
worker = json.loads(worker_file.read_text())
worker.update(status='RETIRED', own_active=[], updated_unix=time.time())
q.save(worker_file, worker)
q.save(state / 'activity/hxw:lw5frozen.json', dict(host='hxw:lw5frozen', phase='RETIRED', updated_unix=time.time()))
q.save(state / 'drain_result.json', dict(completed_unix=time.time(), reports=reports))
print(json.dumps(reports), flush=True)
