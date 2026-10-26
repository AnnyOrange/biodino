"""Restart dead registered native workers; never disturb a live evaluation child."""
import concurrent.futures,fcntl,json,os,shlex,socket,subprocess,time
from pathlib import Path
from unified_paper_transport import ssh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';PY='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
QUEUES=['center_probe_pilot_queue.json','rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','additional_retrieval_pilot_queue.json','bbbc038_official_queue.json','nucls_large_pilot_queue.json','global_queue.json','detection_queue.json','global_recovery_queue.json','ctc_queue.json','bone_age_queue.json','ood_repair_queue.json','rsna_queue.json','film_queue.json','native_detection_queue.json','nucls_queue.json','retrieval_queue.json','segmentation_recovery_queue.json','segmentation_cache_queue.json','segmentation_queue.json']
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def remote(host,code,args=()):
 cmd=['python3','-c',code,*args]
 if host not in {'server',socket.gethostname()}:cmd=ssh(host,5)+[shlex.join(cmd)]
 return subprocess.run(cmd,text=True,capture_output=True,timeout=20)
def check(host,pids):
 code='import json,pathlib,sys; out={}\nfor pid in json.loads(sys.argv[1]):\n try:\n  cmd=pathlib.Path(f"/proc/{pid}/cmdline").read_bytes(); out[str(pid)]={"queue":b"unified_paper_queue.py" in cmd,"evaluation":b"run_unified_paper_" in cmd or b"run_new_fm_v4_segmentation_20261016.py" in cmd}\n except FileNotFoundError: out[str(pid)]={"queue":False,"evaluation":False}\nprint(json.dumps(out))'
 try:
  # The host can be reachable after reboot while its shared project is absent.
  # Probe that prerequisite in the same process check, before launching workers.
  code=code.replace('print(json.dumps(out))',f'print(json.dumps(dict(processes=out,workspace_ready=pathlib.Path({str(ROOT / "scripts/unified_paper_queue.py")!r}).is_file() and pathlib.Path({PY!r}).is_file())))')
  r=remote(host,code,[json.dumps(sorted(pids))]);return host,json.loads(r.stdout) if r.returncode==0 else {'error':r.stderr}
 except Exception as e:return host,{'error':repr(e)}
def launch(item):
 from unified_paper_gpu_policy import excluded
 if excluded(item['host'],item['gpu']):raise ValueError('GPU reserved by user; evaluation launch prohibited')
 argv=[PY,'-u',str(ROOT/'scripts/unified_paper_queue.py'),'--host-label',item['host'],'--slot',item['slot'],'--queue-file',item['queue']]
 argv+=['--cpu'] if item['gpu']<0 else ['--gpu',str(item['gpu'])]
 log=OUT/'logs'/f"worker_{item['ident']}_recovery.log"
 code='import json,pathlib,subprocess,sys; args=json.loads(sys.argv[1]); log=pathlib.Path(sys.argv[2]); f=log.open("a"); p=subprocess.Popen(args,cwd=sys.argv[3],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)'
 r=remote(item['host'],code,[json.dumps(argv),str(log),str(ROOT)])
 if r.returncode:raise RuntimeError(r.stderr)
 return int(r.stdout.strip()),str(log)
def main():
 with (OUT/'locks/native_fleet_guard.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  registry=read(OUT/'native_worker_registry.json');services={}
  while True:
   now=time.time();task_queue={};need={}
   for name in QUEUES:
    tasks=read(OUT/name).get('tasks',[]);task_queue.update({j['key']:str(OUT/name) for j in tasks})
    need[str(OUT/name)]=any(not Path(j['result']).exists() and read(OUT/'states'/f"{j['key']}_attempts.json").get('attempts',0)<2 for j in tasks)
   states={}
   for p in (OUT/'states').glob('worker_*.json'):
    s=read(p);q=s.get('queue') or task_queue.get(s.get('task'))
    if q not in need or '_slot' not in p.stem:continue
    ident=p.stem.removeprefix('worker_');item=registry.setdefault(ident,{})
    item.update(ident=ident,host=s['host'],gpu=s['gpu'],slot=p.stem.split('_slot',1)[1],queue=q,state_path=str(p));states[ident]=s
   hosts={}
   for ident,item in registry.items():
    s=states.get(ident,read(Path(item['state_path'])));pid=s.get('pid')
    if now-item.get('last_launch',0)<120 and pid!=item.get('launched_pid'):pid=item['launched_pid']
    item['checked_pid']=pid;item['checked_child']=s.get('child_pid');hosts.setdefault(item['host'],set()).update(x for x in [pid,s.get('child_pid')] if x)
   with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:checks=dict(pool.map(lambda x:check(*x),hosts.items()))
   for host,node in checks.items():
    if 'error'not in node and not node.get('workspace_ready',True):
     from restore_unified_paper_mounts import restore
     recovery=restore(host)
     if recovery.get('status')!='ready':node['error']='Shared project unavailable; mount recovery: '+json.dumps(recovery)
   for ident,item in registry.items():
    from unified_paper_gpu_policy import excluded
    if excluded(item['host'],item['gpu']):item['status']='excluded_by_user';continue
    node=checks[item['host']]
    if 'error' in node:item['status']='host_liveness_unknown';item['error']=node['error'];continue
    processes=node['processes'];parent=processes.get(str(item['checked_pid']),{}).get('queue',False);child=processes.get(str(item['checked_child']),{}).get('evaluation',False)
    if parent:item['status']='worker_live';continue
    if child:item['status']='orphan_evaluation_live; wait for natural completion';continue
    if not need.get(item['queue'],False):item['status']='queue_complete_or_all_remaining_tasks_failed_twice';continue
    try:
     pid,log=launch(item);item.update(status='restarted',launched_pid=pid,last_launch=now,recovery_log=log);print('RESTARTED WORKER',ident,pid,flush=True)
    except Exception as e:item.update(status='restart_failed',error=repr(e));print('RESTART ERROR',ident,repr(e),flush=True)
   # Local analysis services use their own nonblocking singleton locks.
   commands=['refresh_unified_idle_workers.py','unified_paper_dense_statistics.py','prepare_unified_ai_center_probes.py','prepare_unified_center_probes.py','prepare_unified_ai_rxrx1.py','finalize_unified_ai_local_retrieval.py','prepare_unified_rxrx1.py','admit_unified_rxrx3.py','prepare_unified_ai_retrieval.py','admit_unified_hpa_retrieval.py','unified_paper_bbbc038_statistics.py','prepare_unified_ai_bbbc038_official.py','admit_unified_additional_retrieval.py','admit_unified_bbbc038_official.py','repair_bone_age_precision.py','repair_ood_group_split.py','unified_paper_statistics.py','unified_paper_global_statistics.py','unified_paper_multilabel_statistics.py','unified_paper_bone_age_statistics.py','unified_paper_rsna_statistics.py','unified_paper_film_statistics.py','unified_paper_native_statistics.py','unified_paper_retrieval_statistics.py','unified_paper_clustering_statistics.py','fix_unified_clustering_metric_sampling.py','unified_paper_nucls_statistics.py','unified_paper_segmentation_statistics.py','unified_paper_ctc_statistics.py','route_unified_prediction_recovery_retries.py','manage_unified_ai_branches.py','prepare_unified_ai_bone_age.py','prepare_unified_ai_native.py','prepare_unified_ai_global.py','prepare_unified_ai_film.py','prepare_unified_ai_rsna.py','prepare_unified_ai_ctc.py','prepare_unified_ai_film_pack.py','accelerate_unified_ai_native_transfer.py','finalize_unified_ai_native_transfer.py']
   commands.append('repair_unified_duplicate_executions.py')
   cmdlines=[]
   for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:cmdlines.append(p.read_bytes())
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
   for name in commands:
    if any(str(ROOT/'scripts'/name).encode() in c and b'--watch' in c for c in cmdlines):services[name]={'status':'live'};continue
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');log=OUT/'logs'/f'{name.removesuffix(".py")}_recovery.log'
    with log.open('a') as f:proc=subprocess.Popen([PY,'-u',str(ROOT/'scripts'/name),'--watch'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    services[name]={'status':'restarted','pid':proc.pid,'log':str(log)};print('RESTARTED SERVICE',name,proc.pid,flush=True)
   dump(OUT/'native_worker_registry.json',registry);dump(OUT/'states/native_fleet_guard.json',{'pid':os.getpid(),'updated':time.time(),'workers':len(registry),'status_counts':{s:sum(v.get('status')==s for v in registry.values()) for s in {v.get('status') for v in registry.values()}},'services':services})
   time.sleep(45)
if __name__=='__main__':main()
