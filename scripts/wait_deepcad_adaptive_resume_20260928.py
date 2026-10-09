"""Resume saved Adaptive state only after GPU2 is genuinely free."""
import fcntl
import subprocess
import time
from pathlib import Path
import run_deepcad_method_20260927 as train

REPORT=train.REPO/'outputs/00_reports/deepcad_method_20260927'
PY='/home/deepcad/anaconda3/envs/dinov3/bin/python'

def main():
    lock=(REPORT/'adaptive_resume.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ck=train.ROOT/'adaptive_continue_v2_gpu2/ckpt/16103/checkpoint.pth'
    assert ck.is_file()
    cmd=[PY,str(train.REPO/'scripts/run_deepcad_method_20260927.py'),
         '--arm','adaptive_continue_resume16103','--tag','formal','--gpus','2',
         '--batch','128','--end','20007','--port','32759','--resume-from',str(ck)]
    deadline=time.monotonic()+60*3600
    while time.monotonic()<deadline:
        gs=train.gpu_info();occupied={p['uuid'] for p in train.gpu_processes()}
        free=gs[2]['used']<1024 and gs[2]['uuid'] not in occupied
        train.atomic(REPORT/'ADAPTIVE_RESUME_STATUS.json',dict(time=train.now(),
                     state='LAUNCHING' if free else 'WAITING_FREE_GPU2',gpu=gs[2],checkpoint=str(ck),command=cmd))
        if free:
            rc=subprocess.call(cmd)
            train.atomic(REPORT/'ADAPTIVE_RESUME_STATUS.json',dict(time=train.now(),state='EXITED',returncode=rc,command=cmd))
            return rc
        time.sleep(30)
    train.atomic(REPORT/'ADAPTIVE_RESUME_STATUS.json',dict(time=train.now(),state='WAIT_DEADLINE_EXCEEDED'))
    return 1

if __name__=='__main__':
    raise SystemExit(main())
