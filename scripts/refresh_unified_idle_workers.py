"""Roll out a queue fix only to verified idle parents, never live evaluation children."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(path,value):
 tmp=path.with_suffix('.rollout.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def resident(host):
 directory=OUT/'states/queue_worker_rollout';directory.mkdir(exist_ok=True);target=hashlib.sha256((ROOT/'scripts/unified_paper_queue.py').read_bytes()).hexdigest();state_path=directory/f'{host}.json'
 with(OUT/'locks'/f'idle_worker_rollout_{host}.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return dict(host=host,status='rollout_already_live')
  previous=json.loads(state_path.read_text())if state_path.exists()else{};records=previous.get('workers',{})if previous.get('queue_sha256')==target else{};deferred=0
  for path in sorted((OUT/'states').glob('worker_*.json')):
   try:state=json.loads(path.read_text())
   except (OSError,json.JSONDecodeError):continue
   if state.get('host')!=host:continue
   pid=state['pid'];proc=Path(f'/proc/{pid}');name=path.stem
   if records.get(name,{}).get('new_pid')==pid:continue
   try:
    argv=[x.decode()for x in(proc/'cmdline').read_bytes().split(b'\0')if x]
    if not any(x.endswith('/unified_paper_queue.py')for x in argv):continue
    if state.get('status')!='waiting'or(proc/'wchan').read_text()!='hrtimer_nanosleep':deferred+=1;continue
    # Freeze only the idle queue parent for the final race-free inspection.
    # Its child processes, if any, continue uninterrupted and cause a deferral.
    started=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    os.kill(pid,signal.SIGSTOP)
    try:
     for _ in range(50):
      if(proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='T':break
      time.sleep(.01)
     stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
     if stat[0]!='T'or stat[19]!=started:deferred+=1;continue
     children=(proc/'task'/str(pid)/'children').read_text().split()
     if children:deferred+=1;continue
     owned=False
     for claim in(OUT/'locks').glob('*.claimdir'):
      if claim.is_symlink():continue
      try:r=json.loads((claim/'owner.json').read_text())
      except (OSError,json.JSONDecodeError):continue
      if r.get('host')==host and r.get('pid')==pid:owned=True;break
     if owned:deferred+=1;continue
     env={k.decode():v.decode()for part in(proc/'environ').read_bytes().split(b'\0')if part for k,v in[part.split(b'=',1)]}
     if(proc/'cmdline').read_bytes().split(b'\0')[:-1]!=[x.encode()for x in argv]:raise RuntimeError('PID identity changed')
     os.kill(pid,signal.SIGTERM)
    finally:os.kill(pid,signal.SIGCONT)
    for _ in range(50):
     if not(proc/'cmdline').exists()or not(proc/'cmdline').read_bytes():break
     time.sleep(.1)
    if(proc/'cmdline').exists()and(proc/'cmdline').read_bytes():deferred+=1;continue
    log=OUT/'logs'/f'{name}_prefilter_refresh.log'
    with log.open('a')as f:new=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    records[name]=dict(old_pid=pid,new_pid=new.pid,updated=time.time(),log=str(log),reason='Idle parent, no child and no claim at atomic rollout check')
    dump(state_path,dict(host=host,updated=time.time(),queue_sha256=target,workers=records,deferred=deferred))
   except (FileNotFoundError,ProcessLookupError):continue
  payload=dict(host=host,updated=time.time(),queue_sha256=target,workers=records,deferred=deferred);dump(state_path,payload);return dict(host=host,updated=payload['updated'],workers_refreshed=len(records),deferred=deferred)
def controller():
 from manage_unified_ai_branches import HOST,REMOTE_ROOT
 hosts=set()
 for path in(OUT/'states').glob('worker_*.json'):
  try:r=json.loads(path.read_text())
  except (OSError,json.JSONDecodeError):continue
  if r.get('gpu',-1)>=0 and r.get('host')!='server':hosts.add(r['host'])
 hosts.add(HOST)
 def job(host):
  from unified_paper_transport import ssh
  base=REMOTE_ROOT if host==HOST else str(ROOT);cmd=ssh(host)+['python3',base+'/scripts/refresh_unified_idle_workers.py','--resident',host]
  try:
   p=subprocess.run(cmd,text=True,capture_output=True,timeout=50);return dict(host=host,returncode=p.returncode,output=p.stdout[-3000:],error=p.stderr[-1000:])
  except subprocess.TimeoutExpired:return dict(host=host,status='observation_timeout; resident lock prevents duplicate rollout; no evaluation terminated')
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:results=list(pool.map(job,sorted(hosts)))
 dump(OUT/'states/queue_worker_rollout.json',dict(pid=os.getpid(),updated=time.time(),hosts=results))
def main():
 p=argparse.ArgumentParser();p.add_argument('--resident');p.add_argument('--watch',action='store_true');a=p.parse_args()
 if a.resident:print(json.dumps(resident(a.resident)));return
 with(OUT/'locks/idle_worker_rollout_controller.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  from manage_unified_ai_branches import HOST,REMOTE_ROOT
  subprocess.run(['rsync','-azR','scripts/unified_paper_queue.py','scripts/unified_paper_gpu_policy.py','scripts/refresh_unified_idle_workers.py',HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  while True:
   controller()
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
