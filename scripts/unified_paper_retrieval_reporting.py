"""Index existing v4 metrics with retained query and cluster assignments."""
import json,hashlib
from pathlib import Path
import numpy as np
def apply_retrieval_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';queue=out/'retrieval_queue.json'
 if not queue.exists():return []
 all_metrics=[]
 jobs={j['key']:j for j in json.loads(queue.read_text())['tasks']}
 remote=out/'remote_queue.json'
 if remote.exists():
  jobs.update({j['key']:j for j in json.loads(remote.read_text())['tasks'] if j.get('task')=='retrieval_clustering'})
 for name in ['rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','additional_retrieval_pilot_queue.json']:
  path=out/name
  if path.exists():jobs.update({j['key']:j for j in json.loads(path.read_text())['tasks']})
 for j in jobs.values():
  base=Path(j['result']).parent
  for task,metric,paths in [('retrieval','recall_at_1',[base/'retrieval_result.json']),('clustering','nmi',[base/f'seed{s}'/'clustering_result.json' for s in range(5)])]:
   available=[]
   for p in paths:
    if not p.exists():continue
    raw=json.loads(p.read_text());prediction=Path(raw['prediction_path'])
    if not prediction.exists() or hashlib.sha256(prediction.read_bytes()).hexdigest()!=raw['prediction_sha256']:raise ValueError('Retrieval/cluster output missing or hash mismatch')
    available.append((p,raw))
   if not available:continue
   if len({r['manifest_sha256'] for _,r in available})!=1:raise ValueError('Clustering seed membership changed')
   key=(j['model'],task,j['dataset'],'','fixed');row=rows[key];vals=[r['metrics'][metric] for _,r in available];required=5 if task=='clustering' else 0
   row.update(metric=metric,value=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,
    completed_seeds=len(vals) if required else 0,required_seeds=required,seed_values=json.dumps(vals) if required else '[]',seed_ids=json.dumps([r['seed'] for _,r in available]) if required else '[]',
    protocol_verified=True,split_id=available[0][1].get('split','within-set-leave-one-out'),split_sha256=available[0][1]['manifest_sha256'],data_complete='是',
    raw_result_path=json.dumps([str(p) for p,_ in available]) if required else str(available[0][0]),prediction_path=json.dumps([r['prediction_path'] for _,r in available]),
    seed_semantics=available[0][1].get('seed_semantics','not_applicable'),validation_basis='Original v4 solver/retrieval rules, full locked source cohort, retained per-query scores/cluster assignments',
    patient_group_verified=available[0][1].get('primary_grouping_verified',False),uncertainty_group_scope=available[0][1].get('grouping','Conditional source-image analysis only; patient/slide structure not verified'),
    reason=(f'{len(vals)}/5 clustering fits complete; ' if required else 'Deterministic retrieval complete; ')+available[0][1]['group_limitation'])
   ci=out/'statistics'/task/f'{j["model"]}__{j["dataset"]}__ci.json'
   if ci.exists():
    stat=json.loads(ci.read_text());expected=[r['prediction_sha256'] for _,r in available] if required else available[0][1]['prediction_sha256']
    if stat.get('prediction_sha256')==expected:
     value=stat['metrics'][metric];row.update(ci_low=value['ci_low'],ci_high=value['ci_high'],statistics_path=str(ci),uncertainty_note=stat['uncertainty'])
   runs[:]=[r for r in runs if (r['model'],r['task'],r['dataset'])!=(j['model'],task,j['dataset'])]
   for p,raw in available:
    runs.append({**row,'seed':raw['seed'] if required else '', 'run_value':raw['metrics'][metric],'raw_result_path':str(p),'prediction_path':raw['prediction_path'],'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for metric_name,value in raw['metrics'].items():
     fixed=metric_name!='silhouette_cosine' or raw.get('silhouette_evaluation_seed')==0
     all_metrics.append({'model':j['model'],'checkpoint':raw['checkpoint'],'task':task,'dataset':j['dataset'],'seed':raw['seed'],'metric':metric_name,'value':value if fixed else np.nan,'source':str(p),'prediction_path':raw['prediction_path'],
      'status':'complete' if fixed else 'Fixed silhouette evaluation-subset recovery running; original per-seed sampling score not mixed into table'})
 return all_metrics
