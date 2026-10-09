"""Persistent manifest for the user-requested late no-Gram baseline transfers."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/huawei_deepcad/dinov3/outputs/06_data_prep_transfer/lyxxr_5tb_nogram_20260928')
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 import fcntl
 lock=(ROOT/'sync_monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(ROOT/'sync_monitor_process.json',dict(pid=os.getpid(),started=time.time()))
 inv=json.loads((ROOT/'REMOTE_METRIC_RECORDS.json').read_text())
 remote="import hashlib,json;from pathlib import Path;p=Path('/data/hs6_l_5tb_nogram_eval_20260921/source/eval/training_41479/teacher_checkpoint.pth');h=hashlib.sha256();f=p.open('rb');[h.update(b) for b in iter(lambda:f.read(8388608),b'')];print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())))"
 import shlex
 check=subprocess.run(['ssh','5090-hxw-xzj','python3 -c '+shlex.quote(remote)],text=True,capture_output=True,check=True)
 latest=json.loads(check.stdout);atomic(ROOT/'REMOTE_LATEST_SHA256.json',latest)
 verified={}
 while True:
  copied=[];pending=[]
  for entry in inv['checkpoints']:
   rel=Path(entry['path']).relative_to('/data/hs6_l_5tb_nogram_eval_20260921/source')
   target=ROOT/'hxw_eval/source'/rel
   (copied if target.is_file() and target.stat().st_size==entry['bytes'] else pending).append(str(rel))
  for key,target in [('priority_latest',ROOT/'priority_latest/eval/training_41479/teacher_checkpoint.pth'),('bulk_latest',ROOT/'hxw_eval/source/eval/training_41479/teacher_checkpoint.pth')]:
   if key not in verified and target.is_file() and target.stat().st_size==latest['bytes']:
    digest=sha(target);verified[key]=dict(path=str(target),sha256=digest,match=digest==latest['sha256']);assert digest==latest['sha256']
  status=dict(time=time.time(),training_logs_synced=True,metric_records=len(inv['records']),expected_teacher_checkpoints=len(inv['checkpoints']),copied_count=len(copied),copied=copied,pending=pending,latest_verified=verified,complete=len(copied)==len(inv['checkpoints']) and bool(verified))
  atomic(ROOT/'SYNC_STATUS.json',status)
  if status['complete']:return
  time.sleep(60)
if __name__=='__main__':main()
