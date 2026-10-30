"""Keep authorized seven-family workers alive, publish progress and clean owned caches."""
import argparse,concurrent.futures,ctypes,fcntl,json,os,platform,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/seven_family_100tb_1pb_20261029';U=ROOT/'outputs/02_eval_runs/unified_paper'
PY='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python';WORKER=ROOT/'scripts/seven_family_queue_20261029.py'
PLAN={'5090-zxr':[0,1,3],'3090-qi':[0,5,7],'cpu1':[0],'cpu2':[0],'cpu5':[0],'cpu9':[0],'cpu10':[0],'cpu11':[0],'cpu12':[0],'cpu19':[0]}
def slots(host):return 8 if host.startswith('cpu') else 5
def pidfd_open(pid):
 if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
 assert platform.machine()=='x86_64'
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
 if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
 return fd
def pidfd_term(fd):
 if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(fd,signal.SIGTERM)
 assert platform.machine()=='x86_64'
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.syscall(424,fd,signal.SIGTERM,0,0)<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
def read(p):return json.loads(p.read_text())
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def local(host,launch):
 assert host in PLAN
 reaped=[]
 for job in read(OUT/'queue.json')['tasks']:
  guard=U/'locks'/f"execution_{job['key']}.claimdir"
  try:raw=os.readlink(guard)
  except (FileNotFoundError,OSError):continue
  record=json.loads(raw)
  if record.get('host')!=host or record.get('task')!=job['key']:continue
  pid=record['pid']
  try:
   stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
   alive=stat[19]==record['start_ticks'] and stat[0]!='Z'
  except (FileNotFoundError,ProcessLookupError):alive=False
  if not alive and os.readlink(guard)==raw:
   guard.unlink();reaped.append(record)
 # Real /proc scan includes stopped/not-yet-heartbeating workers; no duplicate slot resurrection.
 found={}
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   a=p.read_bytes().split(b'\0')
   if str(WORKER).encode() not in a or b'--slot' not in a:continue
   found[a[a.index(b'--slot')+1].decode()]=dict(pid=int(p.parent.name),state=(p.parent/'stat').read_text().rsplit(')',1)[1].split()[0])
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 started=[];retired=[]
 if (OUT/'COMPLETE.json').exists():
  launch=False
  for slot,identity in found.items():
   pid=identity['pid']
   try:
    if Path(f'/proc/{pid}/task/{pid}/children').read_text().strip():continue
    fd=pidfd_open(pid)
    try:
     argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
     if str(WORKER).encode() in argv and slot.encode() in argv:
      pidfd_term(fd);retired.append(pid)
    finally:os.close(fd)
   except (FileNotFoundError,ProcessLookupError):pass
 if launch:
  for gpu in PLAN[host]:
   for i in range(slots(host)):
    slot=f'seven29_g{gpu}_{i}'
    if slot in found:continue
    command=[PY,'-u',str(WORKER),'--host-label',host,'--gpu',str(gpu),'--slot',slot,'--queue-file',str(OUT/'queue.json'),'--python',PY]
    logfile=OUT/'logs'/f'worker_{host}_{slot}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a') as f:
     proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    started.append(dict(slot=slot,pid=proc.pid));proc.poll()
 mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
 return dict(host=host,existing=found,started=started,retired=retired,reaped_execution_guards=reaped,memory_kib=mem,gpu=gpu,applications=apps,time=time.time())
def remote(host,launch):
 if host=='5090-zxr':return local(host,launch)
 cmd=[PY,str(Path(__file__).resolve()),'--local',host]+(['--launch'] if launch else [])
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(cmd)],text=True,capture_output=True,timeout=75)
 if p.returncode:return dict(host=host,error=p.stderr[-2000:],stdout=p.stdout[-2000:])
 return json.loads(p.stdout)
def valid(j):
 try:
  r=read(Path(j['result']))
  if r.get('model')!=j['model']:return False
  if j['task']=='segmentation':return r.get('sample_manifest_status')=='verified' and Path(r['prediction_path']).is_file() and 0<=r['test']['mDice']<=1
  if j['task']=='box_detection':return Path(r['prediction_path']).is_file() and 'bbox_ap_50_95' in r['metrics']
  return True
 except (FileNotFoundError,ValueError,KeyError,TypeError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--local',choices=PLAN);p.add_argument('--launch',action='store_true');a=p.parse_args()
 if a.local:print(json.dumps(local(a.local,a.launch)));return
 with (OUT/'watch.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  print('WATCH START',time.time(),flush=True)
  tasks=read(OUT/'queue.json')['tasks'];done={j['key']:valid(j) for j in tasks};pending=[j for j in tasks if not done[j['key']]]
  with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
   futures={pool.submit(remote,h,bool(pending)):h for h in PLAN};hosts=[]
   for f,h in futures.items():
    try:hosts.append(f.result())
    except Exception as error:hosts.append(dict(host=h,error=repr(error)))
  print('HOST SCAN COMPLETE',time.time(),flush=True)
  states=[]
  for h, gpus in PLAN.items():
   for gpu in gpus:
    for i in range(slots(h)):
     f=U/'states'/f'worker_{h}_gpu{gpu}_slotseven29_g{gpu}_{i}.json'
     if f.exists():
      d=read(f)
      host_time=next((v.get('time',time.time()) for v in hosts if v['host']==h),time.time())
      if host_time-d.get('heartbeat',0)<90:states.append(d)
  running={d.get('task') for d in states if d.get('status')=='running'}
  print('STATE SCAN COMPLETE',time.time(),flush=True)
  failures=[]
  for j in pending:
   log=Path(j['log'])
   if not log.exists():continue
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-2500));tail=stream.read()
   if b'Traceback' not in tail:continue
   f=U/'states'/f"{j['key']}_attempts.json"
   if f.exists():
    d=read(f)
    if d.get('attempts',0)>=2 and d.get('exit_code')!=0:failures.append(dict(task=j['key'],attempt=d,log=j['log']))
  print('FAILURE SCAN COMPLETE',time.time(),flush=True)
  # Never remove an active reader's cache. Remove only campaign-owned feature files.
  cleaned=[]
  for j in tasks:
   if not done[j['key']] or j['key'] in running:continue
   base=Path(j['result']).parent
   paths=[]
   if j['runner']=='run_unified_paper_global':paths=list(base.glob('*_features.npz'))
   if j['task']=='retrieval_clustering':
    dep=[x for x in tasks if x['model']==j['model'] and x['dataset']=='lc25000' and x['task']=='classification']
    if j['dataset']!='lc25000' or all(done[x['key']] and x['key'] not in running for x in dep):paths=list(base.glob('*features*.npz'))+list(base.glob('*features*.npy'))
   if j['task']=='segmentation_cache_recovery':
    children=[x for x in tasks if j['result'] in x.get('requires',[])]
    if children and all(done[x['key']] and x['key'] not in running for x in children):paths=[base/'cache']
   for path in paths:
    if path.exists() and path.resolve().is_relative_to(OUT) and not path.is_symlink():
     if path.is_dir():shutil.rmtree(path)
     else:path.unlink()
     cleaned.append(str(path))
  if cleaned:
   with (OUT/'cleanup.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),paths=cleaned))+'\n')
  summary={}
  for model in read(OUT/'config.json')['models']:
   m=model['model'];jobs=[j for j in tasks if j['model']==m];summary[m]=dict(total=len(jobs),validated_complete=sum(done[j['key']] for j in jobs),remaining=sum(not done[j['key']] for j in jobs),by_task={t:dict(total=sum(j['task']==t for j in jobs),complete=sum(j['task']==t and done[j['key']] for j in jobs)) for t in sorted({j['task'] for j in jobs})})
  report=dict(time=time.time(),models=summary,workers=states,hosts=hosts,failed=failures,finished=not pending,warning='Queue workers/loading are not completed tests. Five slots on main hosts and eight on selected cpu nodes are bounded by live shared GPU/RAM admission. Existing historical results are not silently counted as current seven-family results.')
  dump(OUT/'status.json',report);print(json.dumps(dict(models=summary,failed=len(failures),workers=len(states))))
  from report_seven_family_20261029 import main as report_scores
  report_scores()
  if not pending and all(r['complete'] for r in read(OUT/'scores.json')['scores']):
   dump(OUT/'COMPLETE.json',dict(time=time.time(),models=summary,report=str(OUT/'scores.json')))
if __name__=='__main__':main()
