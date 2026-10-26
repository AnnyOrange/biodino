#!/usr/bin/env python3
"""Export selected committed 7B teachers and publish verified copies to lyx.

Runs on zxr, using its existing SSH configuration. Evaluation workers watch the
destination; temporary copies cannot be mistaken for complete teacher exports.
"""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/l20tb_domain_7b_baseline_20261018')
TRAIN = '/home/xzj/storage/merged/vit7b_20tb_recovered249_20261018'
DEST = '/data/xuzijing/cooldown_20261013_eval/7b_20tb_nogram_recovered249_20261018/eval'
PY = '/home/xzj/miniconda3/envs/dinov3/bin/python'
EXPORT = '/home/xzj/biodino_7b_20261015/scripts/export_7b_dcp_teacher_20261016.py'
POINTS = (499, 999, 1999, 2999)


def ssh(host, argv, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                           shlex.join([str(a) for a in argv])], check=True, **kwargs)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'transfer_paused_by_user_20261019').exists():
        print('Checkpoint transfer paused by user; obtain confirmation before restarting.', flush=True)
        return
    lock = (ROOT/'mirror.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = ROOT/'mirror_state.json'
    state = json.loads(record.read_text()) if record.exists() else dict(points={}, started=time.time())

    def save():
        state.update(updated=time.time(), pid=os.getpid())
        tmp=record.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(record)

    while True:
        if (ROOT / 'transfer_paused_by_user_20261019').exists():
            return
        for ck in POINTS:
            entry = state['points'].setdefault(str(ck), dict(status='WAITING_FOR_TRAINING', attempts=0))
            if entry['status'] in ('MIRRORED', 'FAILED_THREE_ATTEMPTS'):
                continue
            source = f'{TRAIN}/eval/training_{ck}'
            try:
                ssh('5090-hxw-xzj', ['test', '-f', source+'/sharded_teacher_checkpoint/.metadata'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except subprocess.SubprocessError:
                continue
            try:
                entry.update(status='EXPORTING', started=time.time())
                save()
                # Use the already validated converter; it preserves every tensor
                # and dtype, and checks finite values plus exact round-trip.
                exported = subprocess.run(['ssh','5090-hxw-xzj',shlex.join(['test','-f',source+'/teacher_checkpoint.export.json'])]).returncode == 0
                if not exported:
                    ssh('5090-hxw-xzj', [PY, EXPORT, source+'/sharded_teacher_checkpoint', source+'/teacher_checkpoint.pth'], timeout=6*3600)
                metadata = json.loads(ssh('5090-hxw-xzj', ['cat',source+'/teacher_checkpoint.export.json'], capture_output=True,text=True,timeout=30).stdout)
                target = f'{DEST}/training_{ck}/teacher_checkpoint.pth'
                ssh('5090-lyx-xr',['mkdir','-p',str(Path(target).parent)],timeout=30)
                entry['status']='TRANSFERRING'
                save()
                receive_code = "import sys,shutil; f=open(sys.argv[1],'wb'); shutil.copyfileobj(sys.stdin.buffer,f,16*1024**2); f.close()"
                sender = subprocess.Popen(['ssh','5090-hxw-xzj',shlex.join(['cat', source+'/teacher_checkpoint.pth'])],stdout=subprocess.PIPE)
                try:
                    receiver = subprocess.run(['ssh','5090-lyx-xr',shlex.join(['python3','-c',receive_code,target+'.partial'])],
                                              stdin=sender.stdout,timeout=6*3600)
                    sender.stdout.close()
                    if sender.wait(timeout=60) or receiver.returncode:
                        raise RuntimeError('Teacher transfer failed')
                finally:
                    if sender.poll() is None:
                        sender.terminate()
                verify_code = "import hashlib,os,sys; p=sys.argv[1]; h=hashlib.sha256(); f=open(p,'rb'); [h.update(b) for b in iter(lambda:f.read(16*1024**2),b'')]; assert h.hexdigest()==sys.argv[3]; f.close(); os.replace(p,sys.argv[2])"
                ssh('5090-lyx-xr',['python3','-c',verify_code,target+'.partial',target,metadata['sha256']],timeout=6*3600)
                entry.update(status='MIRRORED', sha256=metadata['sha256'], bytes=metadata['bytes'], finished=time.time(), destination=target)
            except Exception as exc:
                entry['attempts']+=1
                entry.update(status='FAILED_THREE_ATTEMPTS' if entry['attempts']>=3 else 'RETRY',error=repr(exc))
            save()
        save()
        if all(state['points'].get(str(ck),{}).get('status') in ('MIRRORED','FAILED_THREE_ATTEMPTS') for ck in POINTS):
            return
        time.sleep(60)


if __name__=='__main__':
    main()
