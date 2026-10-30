"""Exact run-level queue ledger for the stable paper entry, including corrections."""
import json,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';REPORT=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def completed_segmentation_recovery(job):
 if job.get('task')!='segmentation_cache_recovery' or not job['key'].endswith('__recipe16'):return []
 split=Path(job['result']).parents[3].name
 for family in ('segmentation','segmentation_refit'):
  base=OUT/'results'/job['model']/family/job['dataset']/split
  outputs=[base/f'E{budget}'/f'seed{seed}'/'component_result.json'
           for budget in (20,50) for seed in range(5)]
  if all(path.is_file() and read(path).get('split')==split and
         Path(read(path).get('prediction_path','')).is_file() for path in outputs):
   return outputs
 return []
def queue_frame():
 health=read(OUT/'resource_snapshot.json');live=health.get('live_tasks',{}) if time.time()-health.get('observed_at_controller',0)<180 else {};models={x['model']:x for x in read(OUT/'config.json')['models']};rows=[]
 remote=read(OUT/'remote_resource_snapshot.json')
 if time.time()-remote.get('observed_at_controller',0)<180:live.update(remote.get('live_tasks',{}))
 last_workers={}
 for worker in [read(p)for p in (OUT/'states').glob('worker_*.json')]+remote.get('states',[]):
  if worker.get('task')and worker.get('child_pid'):last_workers[worker['task']]=worker
 resident_keys={j['key'] for j in read(OUT/'remote_queue.json').get('tasks',[])}
 for name in sorted(p.name for p in OUT.glob('*queue.json')):
  for j in read(OUT/name).get('tasks',[]):
   if j['model'] not in models:continue
   if name!='remote_queue.json' and j['key'] in resident_keys:continue
   if name=='queue.json' and '__inputfix_v2' not in j['key']:continue
   w=live.get(j['key'],{});attempt=read(OUT/('remote_mirror/5090-lyx-xr/states' if name=='remote_queue.json' else 'states')/f"{j['key']}_attempts.json");complete=Path(j['result']).exists()
   waiting=w.get('execution_phase')=='waiting_for_capacity'
   status='已完成' if complete else '已排队' if waiting else '运行中' if w else '阻塞' if attempt.get('attempts',0)>=2 else '已排队'
   row={'dataset':j['dataset'],'task':j.get('task','ood'),'model':j['model'],'checkpoint':models[j['model']]['checkpoint'],'fold':j.get('fold',j.get('split','fixed')),'seed':j.get('seed','not_applicable'),'budget':j.get('budget',''),'execution_status':status,'metric_run_complete':complete,'task_key':j['key'],'queue':str(OUT/name),'job_id':f"{w['host']}:{w.get('child_pid',w.get('pid'))}" if w else '', 'log':j['log'],'result':j['result'],'attempts':attempt.get('attempts',0),'reason':'Metric run complete; required seed-set/grouped statistics tracked in execution_status.csv' if complete else 'Child process verified on controller clock' if w else f"Two failed attempts; exit={attempt.get('exit_code')}" if status=='阻塞' else 'Registered with live persistent worker fleet; waits for task claim/resource'}
   if waiting and not complete:row['reason']='Evaluation process live, waiting for GPU admission with loading reservations and20pct margin; model computation not started'
   previous=last_workers.get(j['key'])
   if not complete and not w and previous:
    snapshot=remote if previous['host']==remote.get('host')else health
    node=snapshot.get('nodes',{}).get(previous['host'],{})
    stale=time.time()-snapshot.get('observed_at_controller',0)>=180
    if stale or (node.get('error')and not node.get('gpus')):
     row.update(execution_status='状态未知',job_id=f"{previous['host']}:{previous['child_pid']}",reason='Previous claimed child retained; current process/GPU state cannot be verified. '+('Resource snapshot stale'if stale else str(node['error']).strip())+'. Not counted as running or queued; do not reclaim on network failure.')
   row['artifact_sync_complete']=complete
   recovered=completed_segmentation_recovery(j) if not complete else []
   if recovered:
    row.update(execution_status='已完成',metric_run_complete=True,artifact_sync_complete=True,
               result=str(recovered[0]),result_set=json.dumps([str(path) for path in recovered]),
               reason='All 10 final E20/E50 x five-seed results and predictions verified; temporary feature cache was pruned')
   if name=='remote_queue.json' and j.get('remote_result_verified') and not complete:
    row.update(execution_status='已完成',metric_run_complete=True,reason='Resident evaluation and prediction files verified complete; transfer to unified report mirror still running; paper statistics not complete',remote_result=j.get('remote_result_path',''),remote_result_sha256=j.get('remote_result_sha256',''))
   if j.get('blocked_reason'):
    row.update(execution_status='阻塞',metric_run_complete=False,reason=j['blocked_reason'])
   superseded=j['dataset']=='multimodal_cellseg' and 'incorrect cross-set basename' in j.get('blocked_reason','')
   row['required_for_current_protocol']=not superseded
   row['protocol_scope']='withdrawn pairing; preserved correction history, replaced by author-public52 jobs' if superseded else 'current evaluation or required artifact recovery'
   row['execution_mode']=j.get('execution_mode','frozen_test_feature_recovery' if j.get('task')=='segmentation_cache_recovery' else 'evaluation')
   if row['execution_mode'] in ['retained_head_prediction_recovery','frozen_test_feature_recovery']:
    row['reason']+='; inference/artifact recovery only, not an additional downstream training seed'
   rows.append(row)
 # A pilot/priority queue may alias a job in its full matrix. It remains one
 # fit and one execution record, with every dispatch queue retained as an index.
 unique={};aliases={}
 for row in rows:
  key=row['task_key'];aliases.setdefault(key,[]).append(row['queue'])
  if key in unique:
   for field in ['model','task','dataset','seed','result']:
    if unique[key][field]!=row[field]:raise ValueError(f'Conflicting queue identity: {key}/{field}')
   if 'pilot' not in Path(unique[key]['queue']).name:continue
  unique[key]=row
 for key,row in unique.items():row['queue_aliases']=json.dumps(sorted(set(aliases[key])))
 return pd.DataFrame(unique.values())
def write_note(frame):
 counts={}
 for queue,g in frame.groupby('queue'):counts[Path(queue).name]={'total':len(g),**g.execution_status.value_counts().to_dict()}
 current=frame[frame.required_for_current_protocol]
 snapshot={'controller_epoch':time.time(),'queues':counts,'current_protocol_job_counts':current.execution_status.value_counts().to_dict(),'withdrawn_historical_jobs':int((~frame.required_for_current_protocol).sum()),'nucls_cohort':read(OUT/'nucls_qc_cohort_validation.json'),'checkpoint_configuration':str(OUT/'config.json'),'resource_snapshot':str(OUT/'resource_snapshot.json'),'note':'Counts refer to jobs, including artifact recovery, not independent experiments. Withdrawn image/label-pairing jobs remain in the correction ledger but are replaced by author-public52 jobs. Full paper completion additionally requires matched protocols, all required seeds, retained grouped predictions and corrected statistics.'}
 p=REPORT/'delivery_snapshot.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
if __name__=='__main__':
 frame=queue_frame();frame.to_csv(REPORT/'execution_jobs.csv',index=False);write_note(frame);print(frame.execution_status.value_counts().to_dict())
