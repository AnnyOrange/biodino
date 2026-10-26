#!/usr/bin/env python3
"""Publish the resumed L checkpoints from hxw to its lyx evaluation workers."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

CONTROL = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/l20tb_domain_7b_baseline_20261018')
SOURCE = '/home/xzj/storage/merged/cooldown_20261013/20tb_distill_cooldown_len4880_lw2_rebal/eval'
DEST = '/data/xuzijing/cooldown_20261013_eval/l20tb_rebal_resumed20261018/eval'
POINTS = (77591, 78567, 79543)


def ssh(host, argv, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           host, shlex.join(argv)], check=True, **kwargs)


def main():
    lock = (CONTROL/'mirror_l_hxw.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = CONTROL/'mirror_l_hxw_state.json'
    state = json.loads(record.read_text()) if record.exists() else dict(points={})
    def save():
        state.update(updated=time.time(), pid=os.getpid())
        tmp = record.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(record)
    probe = ("import pathlib,sys,time,hashlib,json; p=pathlib.Path(sys.argv[1]); "
             "assert p.is_file() and p.stat().st_size>1000000 and time.time()-p.stat().st_mtime>120; "
             "h=hashlib.sha256(); f=p.open('rb'); "
             "[h.update(b) for b in iter(lambda:f.read(16*1024**2),b'')]; "
             "print(json.dumps(dict(sha256=h.hexdigest(),bytes=p.stat().st_size)))")
    receive = "import shutil,sys; f=open(sys.argv[1],'wb'); shutil.copyfileobj(sys.stdin.buffer,f,16*1024**2); f.close()"
    verify = ("import pathlib,hashlib,sys; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(); f=p.open('rb'); "
              "[h.update(b) for b in iter(lambda:f.read(16*1024**2),b'')]; "
              "assert h.hexdigest()==sys.argv[3]; f.close(); p.replace(sys.argv[2])")
    while True:
        for ck in POINTS:
            entry = state['points'].setdefault(str(ck), dict(status='WAITING_FOR_TRAINING', failures=0))
            if entry['status'] in ('MIRRORED', 'FAILED_THREE_ATTEMPTS'):
                continue
            source = f'{SOURCE}/training_{ck}/teacher_checkpoint.pth'
            try:
                metadata = json.loads(ssh('5090-hxw-xzj', ['python3','-c',probe,source],
                    capture_output=True,text=True,timeout=300).stdout)
            except subprocess.SubprocessError:
                continue
            target = f'{DEST}/training_{ck}/teacher_checkpoint.pth'
            try:
                entry.update(status='TRANSFERRING', started=time.time(), **metadata)
                save()
                ssh('5090-lyx-xr',['mkdir','-p',str(Path(target).parent)],timeout=30)
                sender = subprocess.Popen(['ssh','5090-hxw-xzj',shlex.join(['cat',source])],stdout=subprocess.PIPE)
                try:
                    ssh('5090-lyx-xr',['python3','-c',receive,target+'.partial'],stdin=sender.stdout,timeout=3600)
                    sender.stdout.close()
                    if sender.wait(timeout=60):
                        raise RuntimeError('Source transfer failed')
                finally:
                    if sender.poll() is None:
                        sender.terminate()
                ssh('5090-lyx-xr',['python3','-c',verify,target+'.partial',target,metadata['sha256']],timeout=300)
                entry.update(status='MIRRORED', finished=time.time(), destination=target)
            except Exception as exc:
                entry['failures'] += 1
                entry.update(status='FAILED_THREE_ATTEMPTS' if entry['failures'] >= 3 else 'RETRY',error=repr(exc))
            save()
        save()
        if all(state['points'][str(ck)]['status'] in ('MIRRORED','FAILED_THREE_ATTEMPTS') for ck in POINTS):
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
