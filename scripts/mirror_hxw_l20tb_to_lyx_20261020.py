#!/usr/bin/env python3
"""Mirror only approved L evaluation exports directly from hxw to lyx."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


REPORT = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/l20tb_domain_7b_baseline_20261018')
APPROVAL = REPORT / 'approved_l_eval_exports_20261019.json'
STATE = REPORT / 'mirror_hxw_to_lyx_state_20261020.json'
LYX_ROOT = '/data/xuzijing/cooldown_20261013_eval'
SPECS = {
    'complement': (
        '/home/xzj/storage/merged/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019/eval',
        f'{LYX_ROOT}/20tb_complement_bio5tb_hxw_20261020/eval',
        (50263, 52703, 57583, 67343)),
    'domain': (
        '/home/xzj/storage/merged/l20tb_domain_recoverability_from47823_len19520_hxw_20261019/eval',
        f'{LYX_ROOT}/20tb_domain_recoverability_hxw_20261020/eval',
        (57583, 67343)),
}


def ssh(host, args, timeout=300):
    return subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, shlex.join(args)],
        check=True, capture_output=True, text=True, timeout=timeout)


def metadata(host, path, require_stable=False):
    code = (
        'import hashlib,json,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
        'assert p.is_file() and p.stat().st_size>1000000; '
        'assert not int(sys.argv[2]) or time.time()-p.stat().st_mtime>120; '
        'h=hashlib.sha256(); f=p.open("rb"); '
        '[h.update(b) for b in iter(lambda:f.read(16<<20),b"")]; '
        'print(json.dumps({"sha256":h.hexdigest(),"bytes":p.stat().st_size}))')
    return json.loads(ssh(host, ['python3', '-c', code, path, str(int(require_stable))]).stdout)


def save(state):
    state.update(pid=os.getpid(), updated_unix=time.time())
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def main():
    approval = json.loads(APPROVAL.read_text())
    if not approval.get('approved') or approval.get('scope') != 'EMA teacher exports only; no optimizer checkpoints':
        raise RuntimeError('EMA-only transfer approval is missing')
    for arm, (_, _, points) in SPECS.items():
        if tuple(approval['arms'][arm]) != points:
            raise RuntimeError(f'Unexpected approved points for {arm}')
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {'points': {}}
        while True:
            for arm, (source_root, dest_root, points) in SPECS.items():
                for ck in points:
                    key = f'{arm}/{ck}'
                    entry = state['points'].setdefault(key, {'status': 'WAITING', 'failures': 0})
                    if entry['status'] in ('MIRRORED', 'CONFLICT', 'FAILED_THREE_ATTEMPTS'):
                        continue
                    source = f'{source_root}/training_{ck}/teacher_checkpoint.pth'
                    try:
                        expected = metadata('5090-hxw-xzj', source, require_stable=True)
                    except (subprocess.SubprocessError, AssertionError):
                        continue
                    target = f'{dest_root}/training_{ck}/teacher_checkpoint.pth'
                    try:
                        ssh('5090-lyx-xr', ['mkdir', '-p', str(Path(target).parent)])
                        try:
                            actual = metadata('5090-lyx-xr', target)
                        except subprocess.SubprocessError:
                            actual = None
                        if actual is not None and actual != expected:
                            entry.update(status='CONFLICT', source=source, destination=target)
                            save(state)
                            continue
                        if actual is None:
                            entry.update(status='TRANSFERRING', source=source, destination=target)
                            save(state)
                            ssh('5090-lyx-xr', [
                                'rsync', '-a', '--partial', '--append-verify',
                                '5090-hxw-xzj:' + source, target + '.partial'], timeout=21600)
                            if metadata('5090-lyx-xr', target + '.partial') != expected:
                                raise RuntimeError(f'Checksum mismatch: {key}')
                            ssh('5090-lyx-xr', ['mv', target + '.partial', target])
                        entry.update(status='MIRRORED', source=source, destination=target, **expected)
                    except (subprocess.SubprocessError, RuntimeError) as exc:
                        entry['failures'] += 1
                        entry.update(status=('FAILED_THREE_ATTEMPTS' if entry['failures'] >= 3 else 'RETRY'),
                                     error=repr(exc))
                    save(state)
            save(state)
            if all(state['points'][f'{arm}/{ck}']['status'] in
                   ('MIRRORED', 'CONFLICT', 'FAILED_THREE_ATTEMPTS')
                   for arm, (_, _, points) in SPECS.items() for ck in points):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
