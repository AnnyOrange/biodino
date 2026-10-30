#!/usr/bin/env python3
"""Persistent report/statistics maintenance and safe dead-owner queue recovery."""
import json,os,socket,subprocess,time,shutil,fcntl,csv,collections,concurrent.futures,shlex,traceback
from pathlib import Path
from unified_paper_transport import local_host,ssh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
PY='/home/lxy/miniconda3/envs/dinov3/bin/python'
def alive(host,pid):
 cmd=['python3','-c',f'import pathlib; p=pathlib.Path("/proc/{int(pid)}/cmdline"); print(int(p.exists() and b"unified_paper" in p.read_bytes()))']
 if not local_host(host):
  import shlex
  cmd=ssh(host,5)+[shlex.join(cmd)]
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
  return r.returncode!=0 or r.stdout.strip()!='0'
 except subprocess.TimeoutExpired:return True

def recover():
 claims=[];hosts={};children_by_owner={}
 for p in (OUT/'states').glob('worker_*.json'):
  state=load(p)
  if state.get('child_pid'):children_by_owner.setdefault((state.get('host'),state.get('pid')),[]).append(state['child_pid'])
 for claim in (OUT/'locks').glob('*.claimdir'):
  owner=claim/'owner.json'
  try:raw=os.readlink(claim)if claim.is_symlink()else owner.read_text();d=json.loads(raw)
  except (FileNotFoundError,json.JSONDecodeError):continue
  pids={p for p in [d['pid'],d.get('child_pid'),*children_by_owner.get((d['host'],d['pid']),[])] if p}
  hosts.setdefault(d['host'],set()).update(pids);claims.append((claim,raw,d,pids))
 def inspect(item):
  host,pids=item
  code='import pathlib,json,sys; result={}\nfor pid in json.loads(sys.argv[1]):\n try:\n  cmd=pathlib.Path(f"/proc/{pid}/cmdline").read_bytes();result[str(pid)]=b"unified_paper"in cmd or b"repair_unified_duplicate_executions.py"in cmd\n except FileNotFoundError:result[str(pid)]=False\nprint(json.dumps(result))'
  cmd=['python3','-c',code,json.dumps(sorted(pids))]
  if not local_host(host):cmd=ssh(host,5)+[shlex.join(cmd)]
  try:
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
   return host,json.loads(r.stdout) if r.returncode==0 else {}
  except (subprocess.TimeoutExpired,json.JSONDecodeError):return host,{}
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:status=dict(pool.map(inspect,hosts.items()))
 for claim,raw,d,pids in claims:
  if any(status[d['host']].get(str(pid),True) for pid in pids):continue
  owner=claim/'owner.json'
  try:current=os.readlink(claim)if claim.is_symlink()else owner.read_text()
  except FileNotFoundError:continue
  if current!=raw:continue
  archive=OUT/'recovered_locks';archive.mkdir(exist_ok=True)
  claim.rename(archive/f'{claim.name}.{time.time_ns()}')
  print('recovered dead owner',d,flush=True)

def load(path):
 for attempt in range(3):
  try:return json.loads(path.read_text())
  except (FileNotFoundError,json.JSONDecodeError):return {}
  except OSError as error:
   # Atomic NFS replacement can invalidate the just-opened handle. Reopen;
   # never treat an unreadable memory reservation as zero occupied memory.
   if error.errno!=116 or attempt==2:raise
   time.sleep(.05)

def save(path,data):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def full_audit():
 from unified_paper_delivery import queue_frame,write_note
 from reconcile_unified_model_queues import run as reconcile
 reconcile()
 from admit_unified_ctc_after_pilots import run as expand_ctc
 expand_ctc()
 from admit_unified_spatial_after_pilots import run as expand_spatial
 expand_spatial()
 report=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924';cfg=load(OUT/'config.json')
 frame=queue_frame();tmp=report/'execution_jobs.monitor.tmp.csv';frame.to_csv(tmp,index=False);tmp.replace(report/'execution_jobs.csv');write_note(frame)
 table=report/'paper_results.csv';rows=list(csv.DictReader(table.open())) if table.exists() else []
 coverage=[]
 for model in cfg['models']:
  for domain in ['ID','OOD']:
   subset=[r for r in rows if r['model']==model['model'] and r.get('domain')==domain]
   status=collections.Counter(r['status'] for r in subset)
   coverage.append(dict(model=model['model'],checkpoint=model['checkpoint'],kind=model['kind'],domain=domain,table_rows=len(subset),metric_complete=sum(r.get('metric_status')=='complete' for r in subset),**{s:status[s] for s in ['已完成','运行中','已排队','待执行','阻塞']}))
 with (report/'monitor_model_coverage.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
 deadline=load(OUT/'delivery_deadline.json');health=load(OUT/'resource_snapshot.json')
 missing=[dict(model=m['model'],dataset=t['dataset'],task=t['task']) for m in cfg['models'] for t in cfg['tasks'] if not any(r['model']==m['model'] and r['dataset']==t['dataset'] and r['task']==t['task'] for r in rows)]
 current_frame=frame[frame.required_for_current_protocol]
 fm_rows=[]
 for model in cfg['models']:
  if model['kind']!='fm':continue
  subset=current_frame[current_frame.model==model['model']];counts=subset.execution_status.value_counts().to_dict()
  fm_rows.append(dict(model=model['model'],completed=counts.get('已完成',0),running=counts.get('运行中',0),queued=counts.get('已排队',0),blocked=counts.get('阻塞',0),unknown=counts.get('状态未知',0),total=len(subset)))
 tmp=report/'fm_execution_progress.tmp.csv'
 with tmp.open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(fm_rows[0]));writer.writeheader();writer.writerows(fm_rows)
 tmp.replace(report/'fm_execution_progress.csv')
 save(report/'fm_execution_progress.json',dict(updated=time.time(),models=fm_rows,counts={key:sum(row[key]for row in fm_rows)for key in ['completed','running','queued','blocked','unknown','total']},note='Execution ledger units include retained-artifact recovery and repeats; not independent experiments or completed paper task cells. Unknown claimed children are not counted as running or queued. Task statistical blockers remain in execution_status.csv.'))
 audit=dict(updated=time.time(),models=len(cfg['models']),fm_models=sum(m['kind']=='fm' for m in cfg['models']),model_domain_coverage=coverage,missing_table_cells=missing,queue_status_counts=current_frame.execution_status.value_counts().to_dict(),withdrawn_historical_jobs=int((~frame.required_for_current_protocol).sum()),all_ledger_job_counts=frame.execution_status.value_counts().to_dict(),failed_jobs=current_frame[current_frame.execution_status=='阻塞'][['task_key','log','reason']].to_dict('records'),report_age_seconds=time.time()-table.stat().st_mtime if table.exists() else None,resource_snapshot_age=time.time()-health.get('observed_at_controller',0),deadline=deadline,seconds_to_deadline=deadline.get('deadline_epoch',time.time())-time.time(),note='Metric-table rows include folds/budgets/secondary metrics; queue rows are jobs, not independent experiments. All models and both ID/OOD scopes audited every300s.')
 save(OUT/'states/full_scope_monitor.json',audit);save(report/'full_scope_monitor.json',audit)
 publish_gpu_snapshot()
 delivery_path=report/'current_delivery.json';delivery=load(delivery_path)
 if delivery:
  rxrx=[]
  for model in cfg['models']:
   base=OUT/'results'/model['model']/'retrieval_clustering/rxrx1-cross'
   component=load(base/'component_result.json')
   if len(component.get('clustering_results',[]))!=5 or not all(Path(p).is_file() for p in component['clustering_results']):continue
   retrieval=load(base/'retrieval_result.json')
   clustering=[load(Path(p)) for p in component['clustering_results']]
   if not retrieval.get('metrics') or not all(r.get('metrics',{}).get('nmi') is not None for r in clustering):continue
   import statistics
   values=[r['metrics']['nmi'] for r in clustering]
   rxrx.append(dict(model=model['model'],checkpoint=model['checkpoint'],recall_at_1=retrieval['metrics']['recall_at_1'],nmi_mean=statistics.mean(values),nmi_sd=statistics.stdev(values),completed_seeds=5,result=str(base/'component_result.json')))
  native={}
  for name in ['ctc_queue.json','nucls_queue.json','rsna_queue.json']:
   jobs=[job for job in load(OUT/name).get('tasks',[]) if any(m['model']==job['model']and m['kind']=='fm'for m in cfg['models'])]
   native[name]=dict(completed=sum(Path(job['result']).is_file()for job in jobs),required=len(jobs))
  live=health.get('live_tasks',{})
  delivery.update(updated=time.time(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),all_complete=bool(rows)and all(r['status']=='已完成'for r in rows),deadline_met=False,
   rxrx1_models_complete=len(rxrx),rxrx1_fm_complete=sum(any(m['model']==r['model']and m['kind']=='fm'for m in cfg['models'])for r in rxrx),rxrx1_completed_results=rxrx,
   fm_native_task_counts=native,fm_execution_snapshot=load(report/'fm_execution_progress.json'),
   queue_status_snapshot=dict(controller_epoch=time.time(),queues={Path(queue).name:{'total':len(group),**group.execution_status.value_counts().to_dict()}for queue,group in current_frame.groupby('queue')}),
   running_selected=[{'task':key,**value}for key,value in live.items() if value.get('execution_phase') not in {'waiting_for_capacity','exited'}][:24],
   waiting_for_gpu_admission=[{'task':key,'host':value['host'],'gpu':value.get('gpu'),'pid':value.get('child_pid'),'log':value.get('log')}for key,value in live.items() if value.get('execution_phase')=='waiting_for_capacity'],
   gpu_snapshot=load(report/'gpu_memory_monitor.json').get('summary',{}),
   clustering_ci_v2_publication=load(OUT/'validation/clustering_ci_v2_publication.json'),
   gpu_reserved_cards={'deepcad':[0,2,3,5]})
  save(delivery_path,delivery)
 with (OUT/'logs/full_scope_monitor_history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in audit.items() if k not in ['model_domain_coverage','failed_jobs','missing_table_cells']})+'\n')

def publish_gpu_snapshot():
 report=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
 health=load(OUT/'resource_snapshot.json');remote=load(OUT/'remote_resource_snapshot.json');nodes={**health.get('nodes',{}),**remote.get('nodes',{})}
 workers=[load(p)for p in (OUT/'states').glob('worker_*.json')]+remote.get('states',[])
 allocated={(w.get('host'),w.get('gpu'))for w in workers if w.get('gpu',-1)>=0}
 from unified_paper_gpu_policy import excluded
 allocated={item for item in allocated if not excluded(*item)}
 live={**health.get('live_tasks',{}),**remote.get('live_tasks',{})};machines={};flat=[]
 for host,node in nodes.items():
  machines[host]=[]
  for card in node.get('gpus',[]):
   gpu=card['gpu'];jobs=[w for w in live.values()if w.get('host')==host and w.get('gpu')==gpu]
   phases=collections.Counter(w.get('execution_phase','unknown')for w in jobs)
   observed=health.get('observed_at_controller')if host in health.get('nodes',{})else remote.get('observed_at_controller')
   row={**card,'host':host,'observed_at_controller':observed,'registered_test_gpu':(host,gpu)in allocated,'verified_evaluation_jobs':len(jobs),'waiting_for_capacity_jobs':phases['waiting_for_capacity'],'cpu_only_jobs':phases['cpu_only'],'admitted_or_loading_jobs':sum(v for k,v in phases.items()if k not in ['waiting_for_capacity','cpu_only','exited']),'gpu_allocated_job_owner_pids':[w['child_pid']for w in jobs if node.get('gpu_process_memory_mib',{}).get(str(w.get('child_pid')),0)>0],'live_job_pids':[w['child_pid']for w in jobs],'live_tasks':[w['task']for w in jobs]}
   if host=='deepcad':
    row.update(policy_target_fraction=.35,policy_admission_fraction=.35,policy_max_active=2,above_user_memory_cap=card['vram_fraction']>.40)
   machines[host].append(row);flat.append(row)
 registered=[r for r in flat if r['registered_test_gpu']]
 observed_cards={(r['host'],r['gpu'])for r in flat}
 unobserved=[dict(host=host,gpu=gpu,error=nodes.get(host,{}).get('error')or 'No GPU observation returned')for host,gpu in sorted(allocated-observed_cards)]
 host_errors={host:node['error']for host,node in nodes.items()if node.get('error')}
 save(report/'gpu_memory_monitor.json',{'updated':time.time(),'main_observed_at':health.get('observed_at_controller'),'remote_observed_at':remote.get('observed_at_controller'),'machines':machines,'unobserved_registered_test_cards':unobserved,'host_errors':host_errors,'summary':{'monitored_cards':len(flat),'configured_test_cards':len(allocated),'registered_test_cards':len(registered),'unobserved_registered_test_cards':len(unobserved),'registered_test_cards_at_60pct':sum(r['target_met']for r in registered)},'target_fraction':.60,'admission_limit_fraction':.80,'deepcad_gpu_policy':{'allowed_gpus':[1,4,6,7],'excluded_gpus':[0,2,3,5],'target_fraction':.35,'admission_limit_fraction':.35,'hard_memory_limit_fraction':.40,'max_active_tests':2},'loading_reservations':str(OUT/'gpu_reservations'),'topup_actions':load(OUT/'states/automatic_gpu_topup.json'),'counting_note':'GPU memory includes existing other workloads. Deepcad GPUs 1, 4, 6 and 7 admit at most two tests each, with a 35% admission limit and 40% user cap. Configured test cards include currently unobservable hosts; registered_test_cards counts observed registered cards only. SSH failures are unknown states, not zero memory or exited jobs. Evaluation jobs require verified live child processes; capacity waits and CPU stages are separate. Owner PID allocation is not a count of CUDA subprocesses.'})
 if flat:
  tmp=report/f'gpu_memory_audit.{os.getpid()}.tmp.csv'
  with tmp.open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in flat for k in row)));writer.writeheader();writer.writerows({k:json.dumps(v)if isinstance(v,list)else v for k,v in row.items()}for row in flat)
  tmp.replace(report/'gpu_memory_audit.csv')

def top_up():
 from watch_unified_native_fleet import launch
 health=load(OUT/'resource_snapshot.json');live=health.get('live_tasks',{});allowed=set()
 reservations={}
 for path in (OUT/'gpu_reservations').glob('process_*.json'):
  record=load(path)
  if record.get('status')=='admitted':reservations[record.get('task'),record.get('pid')]=record
 observed=load(OUT/'gpu_memory_observations.json').get('by_model_task',{})
 readiness={};queue_tasks={}
 def ready_on_disk(job):
  key=job['key']
  if key not in readiness:
   readiness[key]=not job.get('blocked_reason') and not Path(job['result']).exists() and all(Path(p).exists() for p in job.get('requires',[])) and load(OUT/'states'/f'{key}_attempts.json').get('attempts',0)<2 and not (OUT/'locks'/f'{job.get("claim_key",job["model"])}.claimdir').exists()
  return readiness[key]
 if time.time()-health.get('observed_at_controller',0)>180:return
 for p in (OUT/'states').glob('worker_*.json'):
  w=load(p)
  if w.get('gpu',-1)>=0 and w.get('host') not in {'server',socket.gethostname()}:allowed.add((w['host'],w['gpu']))
 actions=[]
 for host,gpu in sorted(allowed):
  from unified_paper_gpu_policy import excluded
  if excluded(host,gpu):continue
  card=next((g for g in health.get('nodes',{}).get(host,{}).get('gpus',[]) if g['gpu']==gpu),None)
  shared_deepcad_gpu=host=='deepcad'
  limit_fraction=.35 if shared_deepcad_gpu else .80
  target_fraction=.35 if shared_deepcad_gpu else .60
  max_active=2 if shared_deepcad_gpu else 8
  if not card or card['vram_fraction']>=target_fraction:continue
  own=[w for w in live.values() if w['host']==host and w['gpu']==gpu and w.get('execution_phase') not in {'cpu_only','waiting_for_capacity'}]
  if len(own)>=max_active:
   actions.append(dict(host=host,gpu=gpu,status='defer',reason=f'At {max_active} live children; preserve concurrency limit',vram_fraction=card['vram_fraction']));continue
  memory=health['nodes'][host].get('gpu_process_memory_mib',{})
  loading_reserve=sum(max(0,reservations.get((w.get('task'),w.get('child_pid')),{}).get('expected_peak_mib',0)-memory.get(str(w.get('child_pid')),0)) for w in own)
  def peak(job):
   return max(2048.,float(job.get('min_free_mib',8192))-.20*card['total_mib'],float(job.get('expected_peak_mib',0)),observed.get(f"{job['model']}::{job.get('task','ood')}",{}).get('peak_observed_mib',0)*1.15)
  added=False
  for name in ['rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','retrieval_queue.json','segmentation_queue.json','segmentation_cache_queue.json','native_detection_queue.json','detection_queue.json','global_queue.json','film_queue.json','bone_age_queue.json','nucls_queue.json','rsna_queue.json','bbbc038_official_queue.json','ctc_queue.json']:
   if name not in queue_tasks:queue_tasks[name]=load(OUT/name).get('tasks',[])
   ready=next((j for j in queue_tasks[name] if (not j.get('allowed_hosts') or host in j['allowed_hosts']) and card['total_mib']-card['used_mib']>=j['min_free_mib'] and card['used_mib']+loading_reserve+peak(j)<=limit_fraction*card['total_mib'] and ready_on_disk(j)),None)
   if ready is None:continue
   for index in range(4):
    slot='auto_'+name.removesuffix('_queue.json')+'_'+str(index);ident=f'{host}_gpu{gpu}_slot{slot}'
    if (OUT/'states'/f'worker_{ident}.json').exists():continue
    pid,log=launch(dict(host=host,gpu=gpu,slot=slot,queue=str(OUT/name),ident=ident));actions.append(dict(host=host,gpu=gpu,status='worker_started',pid=pid,queue=name,log=log,vram_fraction=card['vram_fraction'],reason=f'Below {target_fraction:.0%}; child admission enforces {limit_fraction:.0%} cap and {max_active}-process limit'));added=True;break
   if added:break
  if not added:actions.append(dict(host=host,gpu=gpu,status='defer',reason='No unclaimed ready job fits current memory plus live loading reservations, or existing slots already registered',vram_fraction=card['vram_fraction'],loading_reserve_mib=loading_reserve))
  if sum(a['status']=='worker_started' for a in actions)>=3:break
 save(OUT/'states/automatic_gpu_topup.json',dict(updated=time.time(),actions=actions))

def main():
 with (OUT/'locks/maintenance.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  report_proc=None;report_log=None;last_report=0;last_audit=0;last_topup=0
  while True:
   cycle_started=time.monotonic()
   phase_errors={}
   def phase(name,action):
    try:action();return True
    except Exception as error:
     phase_errors[name]=repr(error);traceback.print_exc();return False
   try:
    phase('dead_owner_recovery',recover)
    observer=load(OUT/'states/resource_observer.json')
    if observer.get('status')!='live'or time.time()-observer.get('updated',0)>120:
     phase('resource_fallback',lambda:subprocess.run([PY,str(ROOT/'scripts/snapshot_unified_paper_resources.py')],cwd=ROOT,check=True,timeout=45))
     phase('resource_publication',publish_gpu_snapshot)
    if time.monotonic()-last_topup>=120 and phase('gpu_topup',top_up):last_topup=time.monotonic()
    if time.monotonic()-last_audit>=300 and phase('full_scope_audit',full_audit):last_audit=time.monotonic()
    # Only the full validated table builder publishes canonical paper files.
    builder=ROOT/'plot/build_paper_figure_tables.py'
    if builder.stat().st_size and (report_proc is None or report_proc.poll() is not None) and time.monotonic()-last_report>=600:
     existing=False
     for p in Path('/proc').glob('[0-9]*/cmdline'):
      try:parts=p.read_bytes().split(b'\0')
      except OSError:continue
      if any(v.endswith(b'build_paper_figure_tables.py') for v in parts):existing=True;break
     if not existing:
      if report_log:report_log.close()
      report_log=(OUT/'logs/periodic_paper_report.log').open('a');env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
      report_proc=subprocess.Popen([PY,str(builder)],cwd=ROOT,env=env,stdout=report_log,stderr=subprocess.STDOUT,start_new_session=True);last_report=time.monotonic()
    save(OUT/'states/maintenance.json',{'pid':os.getpid(),'host':socket.gethostname(),'report_pid':report_proc.pid if report_proc else None,'updated':time.time(),'status':'degraded'if phase_errors else'live','phase_errors':phase_errors,'intervals':{'GPU_seconds':60,'all_model_ID_OOD_seconds':300,'paper_tables_seconds':600}})
   except Exception as e:
    traceback.print_exc()
    save(OUT/'states/maintenance.json',{'pid':os.getpid(),'updated':time.time(),'status':'error','error':repr(e)});print('MAINTENANCE ERROR',repr(e),flush=True)
   time.sleep(max(1,60-(time.monotonic()-cycle_started)))
if __name__=='__main__':main()
