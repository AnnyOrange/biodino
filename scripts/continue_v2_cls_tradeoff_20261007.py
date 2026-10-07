#!/usr/bin/env python3
"""Run the predeclared fixed-anchor CLS weight-.3 control after slow-anchor .3.
No test score enters the readiness or stopping rules. All full states retained.
"""
import fcntl,json,os,subprocess,time
from pathlib import Path
REPO=Path('/data/xuzijing/biodino_v2_20260930')
ROOT=REPO/'outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930'
PREVIOUS=ROOT/'global_cls_slow2_w03'
TARGET=ROOT/'global_cls_w03'
STATE=ROOT/'cls_tradeoff_20261007.json'

def write(status,**extra):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(dict(time=time.time(),status=status,target='global_cls_w03',weight=.3,anchor_momentum=0.,fork=29279,end=35136,**extra),indent=2));p.replace(STATE)

def main():
 with (ROOT/'cls_tradeoff_20261007.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   if (TARGET/'launch_manifest.json').exists():write('ALREADY_LAUNCHED');return
   complete=PREVIOUS/'ckpt/35135/checkpoint.pth'
   commands=subprocess.check_output(['ps','-eo','args'],text=True)
   alive=any('dinov3/train/train.py' in line and str(PREVIOUS) in line for line in commands.splitlines())
   if not complete.exists() or alive:
    write('WAITING_PREVIOUS_TRAINING',previous_process_alive=alive);time.sleep(60);continue
   env=dict(os.environ,REPO=str(REPO),ROOT=str(ROOT),PY='/home/server/miniconda3/envs/dinov3/bin/python',TARS_1TB='/data/xuzijing/microscopy-100k-patched',TARS_5TB='/data/xuzijing/5TB')
   (TARGET/'eval').mkdir(parents=True,exist_ok=True)
   r=subprocess.run(['bash',str(REPO/'scripts/launch_v2_recovery_fork29279_hxw.sh'),'global_cls_w03','0,1,2,3','29526','35136'],env=env,capture_output=True,text=True)
   if r.returncode==0:
    ev=Path('/data/xuzijing/hs6_l5_v2_recovery_eval_20260930')
    source=ev/'global_cls_w03/source';source.mkdir(parents=True,exist_ok=True)
    config=source/'config.yaml'
    if not config.exists():config.symlink_to(ev/'bin/config.yaml')
    for gpu in (4,5):
     log=(ev/'logs'/f'global_cls_w03_gpu{gpu}_20261007.log').open('a')
     subprocess.Popen(['bash',str(ev/'bin/keep_v2_eval_slot_busy_lyx.sh'),'global_cls_w03',str(gpu),f'fixed03_{gpu}','classification_a,classification_b,classification_c,classification_d,regression,retrieval','29767'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write('LAUNCHED' if r.returncode==0 else 'FAILED',stdout=r.stdout,stderr=r.stderr);return
if __name__=='__main__':main()
