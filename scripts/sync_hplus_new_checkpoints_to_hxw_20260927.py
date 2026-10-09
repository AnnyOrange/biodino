#!/usr/bin/env python3
"""Continuously stage SHA-256 verified new H+ teacher checkpoints for ID tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/data/hs6_hplus_5tb_eval_20260921')
ADMIN = ROOT / 'new_checkpoint_queue_20260927'
INVENTORY = r'''
from pathlib import Path
import json,time
base=Path('/data/xuzijing/biodino/outputs/01_training_runs')
patterns=['HS6_Hplus_5tb_no_fsdp_fromck13175_bs64_4x5090lyxxr_20260924',
 'HS6_Hplus_5tb_no_fsdp_fromck13663_bs64_8x5090lyxxr_20260924',
 'HS6_Hplus_5tb_no_fsdp_fromck*_bs64_8x5090lyxxr_20260927_r*']
items={}
for pattern in patterns:
 for run in base.glob(pattern):
  if (run/'INVALID_RESUME.json').exists():continue
  for p in (run/'eval').glob('training_*/teacher_checkpoint.pth'):
   step=int(p.parent.name.split('_')[-1]);s=p.stat()
   if step<13663 or s.st_size!=1778210727 or time.time()-s.st_mtime<120:continue
   items[step]={'step':step,'path':str(p),'size':s.st_size,'mtime':s.st_mtime}
print(json.dumps(sorted(items.values(),key=lambda x:x['step'],reverse=True)))
'''


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def main():
 import fcntl
 ADMIN.mkdir(parents=True,exist_ok=True)
 with (ADMIN/'sync.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:
    raw=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090-lyx-xr','python3 -'],
                       input=INVENTORY,text=True,capture_output=True,check=True)
    items=json.loads(raw.stdout)
    (ADMIN/'source_checkpoint_inventory.json').write_text(json.dumps(items,indent=2)+'\n')
    for item in items:
     step=item['step'];adapter=ROOT/'adapters'/str(step)
     marker=adapter/'verified_sha256.json'
     destination=ROOT/'source/eval'/f'training_{step}'/'teacher_checkpoint.pth'
     if marker.exists() and destination.exists():continue
     if shutil.disk_usage('/data').free<80*2**30:break
     destination.parent.mkdir(parents=True,exist_ok=True)
     source_sha=subprocess.check_output(['ssh','5090-lyx-xr','sha256sum',item['path']],text=True).split()[0]
     if not destination.exists() or destination.stat().st_size!=item['size'] or sha(destination)!=source_sha:
      print('COPY',step,flush=True)
      subprocess.run(['rsync','-a','--partial','--timeout=120','5090-lyx-xr:'+item['path'],str(destination)],check=True)
     if sha(destination)!=source_sha:raise RuntimeError(f'Checksum mismatch for ck{step}')
     adapter.mkdir(parents=True,exist_ok=True)
     temporary=adapter/'checkpoint.pth.tmp';temporary.unlink(missing_ok=True)
     temporary.symlink_to(destination);temporary.replace(adapter/'checkpoint.pth')
     temporary=marker.with_suffix('.tmp')
     temporary.write_text(json.dumps({'sha256':source_sha,'source_host':'5090-lyx-xr',**item},indent=2)+'\n')
     temporary.replace(marker)
     print('VERIFIED_READY',step,source_sha,flush=True)
   except Exception as exc:print('SYNC_RETRY',repr(exc),flush=True)
   time.sleep(30)


if __name__=='__main__':main()
