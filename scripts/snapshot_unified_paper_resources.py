"""Controller-clock process/GPU snapshot; remote clocks are not liveness tests."""
import concurrent.futures,json,subprocess,time,socket
from pathlib import Path
from unified_paper_transport import local_host,ssh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def read_json(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return None

def main():
 workers=[d for p in (OUT/'states').glob('worker_*.json') if (d:=read_json(p)) is not None]
 ctc=OUT/'states/ctc_pilot.json'
 if ctc.exists():
  d=read_json(ctc)
  if d:workers.append({**d,'child_pid':d['pid'],'task':'dinov2__ctc__fold0_seed0'})
 hosts=sorted(set(w['host'] for w in workers)|{'deepcad','3090-qi'}|{f'cpu{i}' for i in range(1,22)})
 def query(h):
  command='nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits\nnvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits\nps -eo pid,args | grep -E "(unified_paper_queue|run_unified_paper_|run_new_fm_v4_segmentation_20261016)"'
  try:
   argv=['bash','-c',command] if local_host(h) else ssh(h,5)+[command]
   r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
   lines=r.stdout.splitlines();gpus=[];pids=[];memory={}
   for line in lines:
    parts=line.split(',')
    if len(parts)==4 and all(p.strip().isdigit() for p in parts):
     i,total,used,util=map(int,parts);gpus.append({'gpu':i,'total_mib':total,'used_mib':used,'vram_fraction':used/total,'gpu_utilization':util,'target_met':used/total>=.6})
    elif len(parts)==2 and all(p.strip().isdigit() for p in parts):memory[parts[0].strip()]=int(parts[1].strip())
    elif line.strip().split() and line.strip().split()[0].isdigit():pids.append(int(line.strip().split()[0]))
   return h,{'gpus':gpus,'live_pids':pids,'gpu_process_memory_mib':memory,'error':r.stderr,'exit_code':r.returncode}
  except Exception as e:return h,{'error':repr(e),'live_pids':[]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:nodes=dict(pool.map(query,hosts))
 live={w['task']:w for w in workers if w.get('task') and w.get('child_pid') in nodes[w['host']]['live_pids']}
 jobs={}
 for name in ['queue.json','global_queue.json','native_detection_queue.json','nucls_queue.json','segmentation_queue.json','segmentation_recovery_queue.json','segmentation_cache_queue.json','detection_queue.json','rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','additional_retrieval_pilot_queue.json','bbbc038_official_queue.json','ctc_queue.json','retrieval_queue.json']:
  p=OUT/name
  if (queue:=read_json(p)) is not None:jobs.update({j['key']:j for j in queue.get('tasks',[])})
 reservations={}
 for p in (OUT/'gpu_reservations').glob('process_*.json'):
  r=read_json(p)
  if r:reservations[r['task'],r['pid']]=r
 for key,w in live.items():w['execution_phase']=reservations.get((key,w.get('child_pid')),{}).get('status','active_or_loading')
 import fcntl,os
 with (OUT/'locks/gpu_observations.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=OUT/'gpu_memory_observations.json';observed=json.loads(p.read_text()) if p.exists() else {'by_model_task':{}}
  for key,w in live.items():
   memory=nodes[w['host']].get('gpu_process_memory_mib',{}).get(str(w.get('child_pid')));job=jobs.get(key)
   if not memory or not job:continue
   k=f"{job['model']}::{job.get('task','ood')}";item=observed['by_model_task'].setdefault(k,{'peak_observed_mib':0,'observations':0})
   item.update(peak_observed_mib=max(item['peak_observed_mib'],memory),observations=item['observations']+1,last_host=w['host'],last_pid=w['child_pid'],updated=time.time())
  observed['note']='Observed NVML process memory, not claimed maximum allocated tensor memory; future admissions reserve15pct above observed maximum and at least original task-specific estimate.'
  t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(observed,indent=2));t.replace(p)
 d={'observed_at_controller':time.time(),'nodes':nodes,'live_tasks':live}
 p=OUT/'resource_snapshot.json';t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def watch():
 import fcntl,os,traceback
 from watch_unified_paper import publish_gpu_snapshot,save
 with (OUT/'locks/resource_observer.local.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  while True:
   started=time.monotonic()
   try:
    main();publish_gpu_snapshot()
    state=dict(pid=os.getpid(),updated=time.time(),status='live',interval_seconds=60,cycle_seconds=time.monotonic()-started)
   except Exception as error:
    traceback.print_exc();state=dict(pid=os.getpid(),updated=time.time(),status='error',error=repr(error))
   save(OUT/'states/resource_observer.json',state)
   time.sleep(max(1,60-(time.monotonic()-started)))

if __name__=='__main__':
 import sys
 if '--watch'in sys.argv:watch()
 else:main()
