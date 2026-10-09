#!/usr/bin/env python3
"""Persistent v4 queue: register each new EMA checkpoint, run bounded GPU slots."""
import argparse, csv, fcntl, hashlib, json, os, signal, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hs6_l5_weightspace_campaign_20260922 as v4

REPO=Path('/mnt/huawei_deepcad/dinov3')
ROOT=REPO/'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923'
TRAIN=REPO/'outputs/01_training_runs/hs6_l5_selective_retention_20260923'
GRAM=REPO/'outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915'
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=1));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def register(arm,checkpoint,config,ckid,priority=0,primary_only=False):
 marker=ROOT/'registered'/f'{arm}.json'
 if marker.exists():return
 v4.ROOT=ROOT;v4.checkpoint_for=lambda _:checkpoint
 v4.SEG_CKPT_ID[arm]=ckid;v4.SEG_RUN_LABEL[arm]=arm
 tasks=[]
 for ds in v4.CLS_SMALL+v4.CLS_LARGE:tasks.append(v4.cls_task(ds,arm,len(tasks)))
 for ds in v4.RET_WITHIN+v4.RET_HPA:tasks.append(v4.ret_task(ds,arm,len(tasks)))
 for ds in v4.DET:tasks.append(v4.det_task(ds,arm,len(tasks)))
 dense=['cellpose','tissuenet','multimodal_cellseg'] if primary_only else ['cellpose','tissuenet','pannuke','conic','livecell','multimodal_cellseg']
 for ds in dense:tasks.append(v4.seg_task(ds,arm,len(tasks)))
 ch=sha(checkpoint)
 for task in tasks:
  cmd=task['cmd'];cmd[cmd.index('--train-config')+1]=str(config)
  task.update(checkpoint_sha256=ch,train_config_sha256=sha(config),created=now(),priority=priority,
              protocol='bio-eval-union-v4',expected_memory_mib=6500 if task.get('heavy') else 3500)
  task['source_entry_sha256']=sha(Path(task['cwd'])/('dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py' if task.get('heavy') else 'dinov3/eval/bio_frozen_eval/encoder.py'))
  atomic(ROOT/'tasks'/f'{task["id"]}.json',task)
 atomic(marker,{'arm':arm,'checkpoint':str(checkpoint),'sha256':ch,'n_tasks':len(tasks),'created':now(),'primary_only':primary_only,
                'inventory_gaps':{'segmentation:monuseg':'Needs matched official-identity source snapshot',
                  'retrieval:rxrx1-cross':'Separate full-manifest runner pending','retrieval:rxrx3-core':'Separate full-manifest runner pending',
                  'tracking:ctc':'Native evaluator admission pending','ood:xray':'Separate OOD evaluator pending','ood:cryo':'Separate OOD evaluator pending'}})
 print('REGISTER',arm,len(tasks),flush=True)

def discover():
 for ck in [20007,29279]:
  p=GRAM/f'eval/training_{ck}/teacher_checkpoint.pth'
  if p.is_file():register(f'G{ck}',p,GRAM/'config.yaml',ck,10)
 for run in sorted(TRAIN.glob('*_formal*')):
  for ck in sorted((run/'eval').glob('training_*/teacher_checkpoint.pth')):
   step=int(ck.parent.name.split('_')[-1])
   if time.time()-ck.stat().st_mtime<45:continue
   # All scheduled snapshots are evaluated; no test-based checkpoint choice.
   register(f'{run.name}_ck{step}',ck,run/'config.yaml',step,0)

def query():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(p[0]):list(map(int,p[1:])) for line in s.splitlines() if (p:=line.split(','))}

def validated_done(done):
 if done['type']!='summary_csv':return v4.task_done(done)
 path=Path(done['path'])
 if not path.exists():return False,'missing'
 with path.open() as f:rows=list(csv.DictReader(f))
 if any(r.get('dataset')!=done['dataset'] for r in rows):return False,'dataset mismatch'
 valid=[r for r in rows if not r.get('error')]
 if len(valid)!=done['rows']:return False,f'valid rows={len(valid)} expected {done["rows"]}'
 return True,f'{len(valid)} successful rows; {len(rows)-len(valid)} preserved failed-attempt rows'

def host_memory_cost(task):
 if not task.get('heavy'):return 5*1024
 # Reserve for future dense cache loading, not just current extraction RSS.
 ds=task['dataset'];primary=task['id'].startswith('seg_primary')
 gib={'monuseg':3,'conic':8 if primary else 22,'livecell':32 if primary else 100,
      'pannuke':12 if primary else 40,'tissuenet':20,'cellpose':10,'multimodal_cellseg':25}.get(ds,40)
 return gib*1024

def gpu_memory_cost(task):
 # High-resolution extraction peaks exceed their initial model-only allocation.
 floor=6500 if task.get('heavy') or task['dataset'] in ('bbbc048-cellcycle','cyclops-protein-loc') else 3500
 return max(floor,task.get('expected_memory_mib',0))

class AdoptedProcess:
 """Observe a still-running task across a supervisor restart without relaunching."""
 def __init__(self,pid,exit_file):
  self.pid=pid;self.exit_file=exit_file
  self.start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21]
 def poll(self):
  try:
   fields=Path(f'/proc/{self.pid}/stat').read_text().split()
   if fields[2]!='Z' and fields[21]==self.start_ticks:return None
  except FileNotFoundError:pass
  # Legacy tasks predate the exit wrapper: completion must validate outputs.
  return json.loads(self.exit_file.read_text())['returncode'] if self.exit_file.exists() else 0

 def wait(self,timeout=None):
  """Wait for an adopted PID using the same timeout contract as Popen."""
  deadline=None if timeout is None else time.monotonic()+timeout
  while True:
   result=self.poll()
   if result is not None:return result
   if deadline is not None and time.monotonic()>=deadline:
    raise subprocess.TimeoutExpired(str(self.pid),timeout)
   time.sleep(.1)

def execute_task(task_id,gpu):
 task=json.loads((ROOT/'tasks'/f'{task_id}.json').read_text())
 cmd=[str(x).replace('{GPU}',str(gpu)) for x in task['cmd']]
 rc=subprocess.run(cmd,cwd=task['cwd']).returncode
 atomic(ROOT/'claims'/task_id/'exit.json',{'returncode':rc,'time':now()})
 raise SystemExit(rc)

def worker(host,hours,min_slots,max_slots):
 ROOT.mkdir(parents=True,exist_ok=True);active={};deadline=time.time()+hours*3600;laststart={g:0 for g in range(8)}
 lock=(ROOT/'supervisors'/f'{host}.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (ROOT/'supervisors'/f'{host}.pid').write_text(str(os.getpid()))
 env0=os.environ.copy();env0.update(v4.THREADS);env0['PYTHONUNBUFFERED']='1'
 failures=0
 for status_file in (ROOT/'claims').glob('*/status.json'):
  st=json.loads(status_file.read_text())
  if st.get('host')!=host or st['state']!='RUNNING':continue
  tid=status_file.parent.name;task=json.loads((ROOT/'tasks'/f'{tid}.json').read_text());pid=st['pid']
  proc_file=Path(f'/proc/{pid}/cmdline')
  argv=proc_file.read_bytes().rstrip(b'\0').decode().split('\0') if proc_file.exists() else []
  ours=argv==st.get('command') or ('--task-id' in argv and tid in argv and 'execute' in argv)
  if not ours:
   ok,reason=validated_done(task['done']);st.update(state='DONE' if ok else 'FAILED',validation=reason,reconciled=now())
   atomic(status_file,st);continue
  proc=AdoptedProcess(pid,status_file.parent/'exit.json');log=Path(st['log']).open('a')
  start=datetime.fromisoformat(st['start']).timestamp()
  active[tid]=(proc,task,st['gpu'],log,start)
  print(now(),'ADOPT',st['gpu'],tid,pid,flush=True)
 while time.time()<deadline or active:
  for tid,item in list(active.items()):
   proc,task,gpu,log,start=item;rc=proc.poll()
   if rc is None:
    if time.time()-start>12*3600:os.killpg(proc.pid,signal.SIGTERM)
    continue
   log.close()
   try:ok,reason=validated_done(task['done'])
   except Exception as e:ok,reason=False,f'{type(e).__name__}: {e}'
   atomic(ROOT/'claims'/tid/'status.json',{'state':'DONE' if rc==0 and ok else 'FAILED','returncode':rc,'validation':reason,'host':host,'gpu':gpu,'end':now(),'pid':proc.pid})
   if rc or not ok:
    failures+=1
    logpath=ROOT/'logs'/f'{tid}.log'
    with logpath.open('rb') as f:
     f.seek(max(0,logpath.stat().st_size-32768));tail=f.read().decode(errors='replace').lower()
    if any(s in tail for s in ('out of memory','cudaerroroutofmemory','cublas_status_alloc_failed')) and task.get('oom_retries',0)<1:
     task['oom_retries']=1;task['expected_memory_mib']=gpu_memory_cost(task)+2500
     atomic(ROOT/'tasks'/f'{tid}.json',task)
     archive=ROOT/'attempts'/f'{tid}__oom1';archive.parent.mkdir(exist_ok=True)
     (ROOT/'claims'/tid).rename(archive)
     print(now(),'REQUEUE_OOM_WITH_MORE_HEADROOM',tid,flush=True)
   del active[tid]
  stats=query();running=Counter(item[2] for item in active.values())
  atomic(ROOT/'telemetry'/f'{host}.json',{'time':now(),'gpus':stats,'running':dict(running),'active_tasks':list(active),'failed_this_worker':failures})
  if time.time()>=deadline:
   if not active:break
   time.sleep(10);continue
  # Individual task failure does not corrupt other arms; stop new work at 12 failures.
  if failures>=12:
   atomic(ROOT/'STOP_EVAL_FAILURES.json',{'host':host,'failures':failures,'time':now()});time.sleep(30);continue
  pending=[]
  for p in (ROOT/'tasks').glob('*.json'):
   if not (ROOT/'claims'/p.stem).exists():pending.append(json.loads(p.read_text()))
  # Dense pipelines include long, low-memory probe phases; a one-pipeline cap
  # leaves GPUs idle. Admit additional useful pipelines with GPU AND host RAM
  # reservations, favoring primary dense readouts over secondary observations.
  pending.sort(key=lambda t:(t['priority'],t['order'],t['id']))
  meminfo={line.split(':')[0]:int(line.split()[1])//1024 for line in Path('/proc/meminfo').read_text().splitlines()}
  future_ram=sum(host_memory_cost(it[1]) for it in active.values())
  for gpu in sorted(stats,key=lambda g:running[g]):
   used,total,util=stats[gpu]
   if running[gpu]>=max_slots or time.time()-laststart[gpu]<25:continue
   if running[gpu]>=min_slots and used/total>=.74:continue
   if (Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0]) and int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])<40*1024**2:continue
   for task in pending:
    heavy=task.get('heavy',False)
    if heavy and sum(bool(it[2]==gpu and it[1].get('heavy',False)) for it in active.values())>=6:continue
    if heavy and (future_ram+host_memory_cost(task)>meminfo['MemTotal']*.80 or meminfo['MemAvailable']<host_memory_cost(task)+48*1024):continue
    # Accommodate measured high-res frozen peaks, leave room for running jobs.
    # A just-started process may not have created its CUDA context yet. Reserve
    # its future peak as well; otherwise the 25-second fill loop over-admits.
    startup_reserve=sum(gpu_memory_cost(it[1]) for it in active.values()
                        if it[2]==gpu and time.time()-it[4]<120)
    reserve=gpu_memory_cost(task)+2500+startup_reserve
    if total-used<reserve:continue
    claim=ROOT/'claims'/task['id']
    try:claim.mkdir(parents=True)
    except FileExistsError:continue
    env=env0.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=task['pythonpath'])
    cmd=[str(x).replace('{GPU}',str(gpu)) for x in task['cmd']]
    logpath=ROOT/'logs'/f'{task["id"]}.log';logpath.parent.mkdir(exist_ok=True)
    wrapper=[sys.executable,str(REPO/'scripts/selective_retention_eval_queue_20260923.py'),'execute','--task-id',task['id'],'--gpu',str(gpu)]
    log=logpath.open('a');proc=subprocess.Popen(wrapper,cwd=task['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(claim/'status.json',{'state':'RUNNING','host':host,'gpu':gpu,'pid':proc.pid,'start':now(),'command':cmd,'log':str(logpath)})
    active[task['id']]=(proc,task,gpu,log,time.time());laststart[gpu]=time.time();running[gpu]+=1;pending.remove(task)
    future_ram+=host_memory_cost(task)
    print(now(),'START',gpu,task['id'],proc.pid,flush=True);break
  time.sleep(10)

def summary():
 counts=Counter();states=[]
 for p in (ROOT/'claims').glob('*/status.json'):
  st=json.loads(p.read_text());counts[st['state']]+=1
  if st['state']=='FAILED':states.append({'task':p.parent.name,**st})
 atomic(ROOT/'STATUS.json',{'time':now(),'counts':dict(counts),'registered':len(list((ROOT/'registered').glob('*.json'))),'task_count':len(list((ROOT/'tasks').glob('*.json'))),'failures':states})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','watch','worker','summary','execute']);ap.add_argument('--host',default='qi3090');ap.add_argument('--hours',type=float,default=18);ap.add_argument('--min-slots',type=int,default=5);ap.add_argument('--max-slots',type=int,default=9);ap.add_argument('--task-id');ap.add_argument('--gpu',type=int);a=ap.parse_args()
 if a.mode=='execute':execute_task(a.task_id,a.gpu)
 elif a.mode=='worker':worker(a.host,a.hours,a.min_slots,a.max_slots)
 elif a.mode=='summary':summary()
 elif a.mode=='register':discover()
 else:
  end=time.time()+a.hours*3600
  while time.time()<end:
   discover();summary();time.sleep(30)
