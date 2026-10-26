#!/usr/bin/env python3
"""Mirror the approved L EMA exports from hxw into the selected-29 queue."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path('/mnt/huawei_deepcad/dinov3')
REPORT = ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018'
APPROVAL = REPORT / 'approved_l_eval_exports_20261019.json'
SPECS = {
    'complement': (
        '/home/xzj/storage/merged/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019/eval',
        ROOT / 'outputs/01_training_runs/l20tb_complement_bio5tb_from47823_len19520_qi_20261018/eval',
        REPORT / 'mirror_hxw_complement_state.json'),
    'domain': (
        '/home/xzj/storage/merged/l20tb_domain_recoverability_from47823_len19520_hxw_20261019/eval',
        ROOT / 'outputs/01_training_runs/l20tb_domain_recoverability_from47823_len19520_20261018/eval',
        REPORT / 'mirror_hxw_domain_state.json'),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def remote_meta(path):
    code = ('import hashlib,json,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
            'assert p.is_file() and p.stat().st_size>1000000; '
            'assert time.time()-p.stat().st_mtime>120; '
            'h=hashlib.sha256(); f=p.open("rb"); '
            '[h.update(b) for b in iter(lambda:f.read(16<<20),b"")]; '
            'print(json.dumps({"sha256":h.hexdigest(),"bytes":p.stat().st_size}))')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             '5090-hxw-xzj', shlex.join(['python3', '-c', code, path])],
                            capture_output=True, text=True, check=True, timeout=300)
    return json.loads(result.stdout)


def save(state, path):
    state.update(pid=os.getpid(), updated_unix=time.time())
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=sorted(SPECS), required=True)
    args = parser.parse_args()
    source_root, dest_root, state_path = SPECS[args.arm]
    approval = json.loads(APPROVAL.read_text())
    if not approval.get('approved') or approval.get('scope') != 'EMA teacher exports only; no optimizer checkpoints':
        raise RuntimeError('L evaluation export transfer lacks matching approval')
    points = tuple(approval['arms'][args.arm])
    if args.arm == 'complement' and points != (50263, 52703, 57583, 67343):
        raise RuntimeError('Unexpected complement transfer points')
    if args.arm == 'domain' and points != (57583, 67343):
        raise RuntimeError('Unexpected domain transfer points')
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {'points': {}}
        while True:
            for ck in points:
                point = state['points'].setdefault(str(ck), {'status': 'WAITING', 'failures': 0})
                if point['status'] == 'TRANSFER_PAUSED_BY_USER':
                    point['status'] = 'WAITING'
                if point['status'] in ('MIRRORED', 'CONFLICT', 'FAILED_THREE_ATTEMPTS'):
                    continue
                source = f'{source_root}/training_{ck}/teacher_checkpoint.pth'
                try:
                    meta = remote_meta(source)
                except (subprocess.SubprocessError, AssertionError):
                    continue
                target = dest_root / f'training_{ck}/teacher_checkpoint.pth'
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if target.stat().st_size != meta['bytes'] or digest(target) != meta['sha256']:
                            point.update(status='CONFLICT', source=source, destination=str(target))
                            save(state, state_path)
                            continue
                    else:
                        partial = target.with_suffix('.partial')
                        point.update(status='TRANSFERRING', source=source, destination=str(target))
                        save(state, state_path)
                        subprocess.run(['rsync', '-a', '--partial', '--append-verify',
                                        '5090-hxw-xzj:' + source, str(partial)],
                                       check=True, timeout=21600)
                        if partial.stat().st_size != meta['bytes'] or digest(partial) != meta['sha256']:
                            raise RuntimeError(f'Hash mismatch for {source}')
                        partial.replace(target)
                    record = dict(host='5090-hxw-xzj', source=source, checkpoint=ck,
                                  sha256=meta['sha256'], bytes=meta['bytes'],
                                  verified_unix=time.time())
                    (target.parent / 'hxw_mirror_provenance.json').write_text(
                        json.dumps(record, indent=2) + '\n')
                    point.update(status='MIRRORED', **record)
                except Exception as exc:
                    point['failures'] += 1
                    point.update(status=('FAILED_THREE_ATTEMPTS' if point['failures'] >= 3 else 'RETRY'),
                                 error=repr(exc))
                save(state, state_path)
            save(state, state_path)
            if all(state['points'][str(ck)]['status'] in
                   ('MIRRORED', 'CONFLICT', 'FAILED_THREE_ATTEMPTS') for ck in points):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
