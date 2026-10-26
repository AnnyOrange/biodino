"""Index prediction-preserving dense runs without mixing old and new seed sets."""
import json, hashlib
from pathlib import Path
import numpy as np

def shared_queue_jobs(out,name):
 """Include resident jobs once, using the same entrypoint and result validation."""
 entrypoints={'native_detection_queue.json':'run_unified_paper_native_detection.py',
  'ctc_queue.json':'run_unified_paper_ctc.py','nucls_queue.json':'run_unified_paper_nucls.py',
  'detection_queue.json':'run_unified_paper_detection.py'}
 jobs={}
 for queue in [out/name,out/'remote_queue.json']:
  if not queue.exists():continue
  for job in json.loads(queue.read_text())['tasks']:
   if queue.name=='remote_queue.json' and job.get('entrypoint')!=entrypoints[name]:continue
   jobs[job['key']]=job
 return list(jobs.values())

def apply_native_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';groups={}
 recovery_path=out/'duplicate_execution_recovery.json';recovery=json.loads(recovery_path.read_text()).get('tasks',{})if recovery_path.exists()else{}
 for name,task,metric in [('native_detection_queue.json','box_detection','bbox_ap_50_95'),('ctc_queue.json','cell_tracking','TRA'),('nucls_queue.json','nuclei_detection_segmentation','bbox_ap_50_95')]:
  for job in shared_queue_jobs(out,name):
   repair=recovery.get(job['key'],{})
   if repair.get('replay_required')and repair.get('status')!='complete':continue
   p=Path(job['result'])
   if not p.exists():continue
   raw=json.loads(p.read_text());fold=str(job.get('fold','fixed'));key=(job['model'],task,job['dataset'],'',fold)
   if task=='cell_tracking':
    if raw['n_domains']!=4 or any(x['ctc_metrics']['Valid']!=1 for x in raw['domain_rows']):raise ValueError('Incomplete native CTC fold')
    value=raw['macro'][metric];predictions=str(p.parent/'domains')
   else:
    value=raw['metrics'][metric];predictions=raw['prediction_path']
    if not Path(predictions).exists():raise ValueError('Missing native test predictions')
   groups.setdefault(key,[]).append((job,raw,p,value,predictions,metric))
 for key,found in groups.items():
  m,t,d,_,fold=key;row=rows[key];values=[x[3] for x in found]
  row.update(value=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else np.nan,metric=found[0][5],
   protocol_verified=True,completed_seeds=len(found),required_seeds=5,seed_ids=json.dumps([x[0]['seed'] for x in found]),seed_values=json.dumps(values),
   data_complete='是',raw_result_path=json.dumps([str(x[2]) for x in found]),prediction_path=json.dumps([x[4] for x in found]),
   validation_basis='Native shared evaluator, frozen encoder and fixed complete released test members; retained native outputs',
   reason=f'{len(found)}/5 native downstream fits complete; grouped statistics pending',seed_semantics='training_randomness')
  ci=out/'statistics/native_box'/f'{m}__{d}__ci.json'
  if t=='nuclei_detection_segmentation':ci=out/'statistics/nucls'/f'{m}__fold{fold}__ci.json'
  if t in ['box_detection','nuclei_detection_segmentation'] and ci.exists():
   stats=json.loads(ci.read_text());by_seed={x[0]['seed']:x[1]['prediction_sha256'] for x in found}
   if stats['prediction_sha256']==[by_seed[s] for s in sorted(by_seed)]:
    v=stats['metrics'][row['metric']];row.update(ci_low=v['ci_low'],ci_high=v['ci_high'],statistics_path=str(ci),reason=f'{len(found)}/5 native fits complete; grouped AP CI complete; paired family tracked separately',patient_group_verified=True if t=='nuclei_detection_segmentation' else row.get('patient_group_verified',True))
  if t=='cell_tracking':
   ci=out/'statistics/ctc'/f'{m}__fold{fold}__ci.json'
   if ci.exists():
    stats=json.loads(ci.read_text())
    if sorted(s['sha256'] for s in stats['source'])==sorted(hashlib.sha256(x[2].read_bytes()).hexdigest() for x in found):
     v=stats['metrics']['TRA'];row.update(ci_low=v['ci_low'],ci_high=v['ci_high'],statistics_path=str(ci),patient_group_verified=True,reason='5/5 native downstream fits; whole-domain grouped CI complete; paired family tracked separately',sample_manifest=json.dumps([str(x[2].with_name('test_frame_index.csv')) for x in found]))
  for job,raw,p,value,predictions,metric in found:runs.append({**row,'seed':job['seed'],'run_value':value,'raw_result_path':str(p),'prediction_path':predictions,'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

def apply_dense_results(root, rows, runs, models):
 out=root/'outputs/02_eval_runs/unified_paper'
 queue_path=out/'detection_queue.json'
 queue=shared_queue_jobs(out,queue_path.name); grouped={}
 for job in queue:grouped.setdefault((job['model'],job['dataset']),[]).append(job)
 corrections=[]
 for (m,d),jobs in grouped.items():
  key=(m,'detection_proxy',d,'','fixed');row=rows[key];finished=[]
  for job in jobs:
   p=Path(job['result'])
   if not p.exists():continue
   raw=json.loads(p.read_text())
   if raw['batch_size']!=8 or raw['epochs']!=5 or raw['seed']!=job['seed'] or not Path(raw['prediction_path']).is_file():raise ValueError(f'Invalid dense result: {p}')
   finished.append((job,raw,p))
  row['prediction_completed_seeds']=len(finished)
  row['queue_file']=str(queue_path)
  if not finished:continue
  hashes={raw['test_ids_sha256'] for _,raw,_ in finished}
  if len(hashes)!=1:raise ValueError('Dense seeds changed test membership')
  old=[r for r in runs if (r['model'],r['task'],r['dataset'])==(m,'detection_proxy',d)]
  for job,raw,p in finished:
   for r in old:
    if str(r.get('seed'))==str(job['seed']):corrections.append({'model':m,'dataset':d,'seed':job['seed'],'old_source':r.get('raw_result_path',''),'old_value':r.get('run_value'),'new_source':str(p),'new_value':raw['test_patch_f1'],'reason':'Fresh prescribed downstream fit with retained image-indexed logits; old artifacts preserved; no mixed seed-set mean'})
  runs[:]=[r for r in runs if (r['model'],r['task'],r['dataset'])!=(m,'detection_proxy',d)]
  vals=[raw['test_patch_f1'] for _,raw,_ in finished];ids=[raw['seed'] for _,raw,_ in finished]
  row.update(value=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,ci_low=np.nan,ci_high=np.nan,
    completed_seeds=len(vals),seed_values=json.dumps(vals),seed_ids=json.dumps(ids),metric='test_patch_f1',
    protocol_verified=True,validation_basis='Shared frozen B8/E5 training; image-indexed logits verified against patch F1',
    split_id=finished[0][1].get('split_id','v4 fixed BBBC038 train/val/test'),split_sha256=next(iter(hashes)),result=json.dumps([str(p) for _,_,p in finished]),
    raw_result_path=json.dumps([str(p) for _,_,p in finished]),prediction_path=json.dumps([raw['prediction_path'] for _,raw,_ in finished]),
    reason=f'{len(vals)}/5 seed with retained per-image predictions; grouped statistics tracked separately',data_complete='是',seed_semantics='training_randomness')
  if d in ['conic','livecell']:
   first=finished[0][1];canonical=root/'outputs/02_eval_inputs/formal_v4'/Path(first['canonical_manifest']).name
   assert canonical.is_file()and hashlib.sha256(canonical.read_bytes()).hexdigest()==first['manifest_sha256']
   row.update(patient_group_verified=first['primary_grouping_verified'],uncertainty_group_scope=first['grouping'],sample_manifest=str(canonical),manifest_sha256=first['manifest_sha256'])
  ci=out/'statistics'/f'{m}__{d}__ci.json'
  if ci.exists():
   data=json.loads(ci.read_text());metric=data['metrics']['test_patch_f1']
   ordered=sorted(finished,key=lambda item:item[1]['seed'])
   hashes=[hashlib.sha256(Path(raw['prediction_path']).read_bytes()).hexdigest()for _,raw,_ in ordered]
   if len(ordered)==5 and data['training_seeds']==[raw['seed']for _,raw,_ in ordered]and data['prediction_sha256']==hashes:
    assert abs(metric['value']-row['value'])<1e-6
    row.update(ci_low=metric['ci_low'],ci_high=metric['ci_high'],statistics_path=str(ci),uncertainty_note=data['uncertainty'])
    if d=='livecell' and data.get('grouping_version'):
     from unified_paper_livecell_groups import PROOF,VERSION
     assert data['grouping_version']==VERSION and data['grouping_proof_sha256']==hashlib.sha256(PROOF.read_bytes()).hexdigest()
     row.update(patient_group_verified=data['primary_grouping_verified'],uncertainty_group_scope=data['grouping'],grouping_proof=str(PROOF),uncertainty_note=data['uncertainty']+'; '+data['group_limitation'])
  for job,raw,p in finished:
   runs.append({**row,'seed':raw['seed'],'run_value':raw['test_patch_f1'],'raw_result_path':str(p),'prediction_path':raw['prediction_path'],'sample_manifest':raw['sample_manifest'],'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if corrections:
  target=out/'corrections/detection_prediction_recovery.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(corrections,indent=2)+'\n');tmp.replace(target)

def apply_segmentation_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';inventory=out/'biodino_segmentation_inventory.json'
 if not inventory.exists():return
 cells=json.loads(inventory.read_text())['cells']
 fm_inventory=out/'fm_segmentation_inventory.json'
 if fm_inventory.exists():cells+=json.loads(fm_inventory.read_text())['cells']
 for cell in cells:
  if cell.get('superseded_protocol'):continue
  m,d,split=cell['model'],cell['dataset'],cell['split'];fold=split if d=='pannuke' else 'fixed'
  for budget in [20,50]:
   key=(m,'segmentation',d,str(budget),fold)
   row=rows.setdefault(key,dict(model=m,checkpoint=models[m]['checkpoint'],task='segmentation',dataset=d,budget=str(budget),fold=fold,required_seeds=5,status='待执行',protocol='bio-eval-union-v4',reason='Frozen-head fits or retained predictions pending'))
   if cell.get('requires_fresh_training'):
    runs[:]=[r for r in runs if not (r['model']==m and r['task']=='segmentation' and r['dataset']==d and str(r.get('budget',''))==str(budget) and str(r.get('fold','fixed'))==str(fold))]
   found={} if cell.get('requires_fresh_training') else {r['seed']:dict(r,prediction_path='') for r in cell['existing'] if r['budget']==budget}
   for p in (out/'results'/m/cell.get('result_family','segmentation')/d/split/f'E{budget}').glob('seed*/component_result.json'):
    r=json.loads(p.read_text());found[r['seed']]={'seed':r['seed'],'budget':budget,'value':r['test']['mDice'],'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'prediction_path':r['prediction_path'],'patient_group_verified':r.get('patient_group_verified',d not in ['livecell','pannuke','tissuenet','multimodal_cellseg'])}
   if not found:
    if cell.get('requires_fresh_training'):
     row.update(historical_result_paths=json.dumps([x['source'] for x in cell['existing'] if x['budget']==budget]),raw_result_path='',prediction_path='',result='',value=np.nan,sd=np.nan,ci_low=np.nan,ci_high=np.nan,completed_seeds=0,seed_ids='[]',seed_values='[]',reason='Retained head reproduction failed fixed tolerance; fresh shared-protocol fits queued. Historical scores and correction evidence retained in source inventory.',protocol_verified=False)
    continue
   key=(m,'segmentation',d,str(budget),fold)
   row=rows.setdefault(key,dict(model=m,checkpoint=models[m]['checkpoint'],task='segmentation',dataset=d,budget=str(budget),fold=fold,required_seeds=5,status='待执行'))
   vals=[found[s]['value'] for s in sorted(found)];row.update(value=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,completed_seeds=len(vals),seed_ids=json.dumps(sorted(found)),seed_values=json.dumps(vals),metric='mDice',protocol_verified=True,split_id=split,result=cell['cell'],raw_result_path=json.dumps([found[s]['source'] for s in sorted(found)]),validation_basis='Selected model/checkpoint only; original result SHA256s, B32 and best-val-mIoU selection verified; same caches for retained-head recovery or added seeds',data_complete='是',reason=f'{len(vals)}/5 prescribed frozen-head fits; sample prediction/grouped inference pending',prediction_completed_seeds=sum(bool(r['prediction_path']) for r in found.values()),seed_semantics='training_randomness')
   family=cell.get('result_family','segmentation')
   if d=='multimodal_cellseg':row['protocol']=split
   row['patient_group_verified']=all(r.get('patient_group_verified',d not in ['livecell','pannuke','tissuenet','multimodal_cellseg']) for r in found.values())
   ci=out/'statistics'/f'{m}__{family}__{d}__{split}__E{budget}__ci.json'
   if ci.exists():
    stats=json.loads(ci.read_text())
    matched=len(found)==5 and sorted(x['result_sha256']for x in stats.get('source',[]))==sorted(x['sha256']for x in found.values())
    if not matched:
     row.update(ci_low=np.nan,ci_high=np.nan,statistics_path='',reason=f'{len(vals)}/5 prescribed frozen-head fits; statistics source hashes do not match current results; recomputation pending')
     for seed,r in found.items():runs.append({**row,'seed':seed,'run_value':r['value'],'raw_result_path':r['source'],'result_sha256':r['sha256'],'prediction_path':r['prediction_path']})
     continue
    metric=stats['metrics']['mDice'];assert abs(metric['value']-row['value'])<1e-6
    row.update(ci_low=metric['ci_low'],ci_high=metric['ci_high'],statistics_path=str(ci),uncertainty_note=stats['uncertainty']+'; '+stats.get('group_limitation',''),uncertainty_group_scope=stats['group_unit'],reason=f'{len(vals)}/5 prescribed frozen-head fits; CI complete for documented grouping; matched FM paired comparisons tracked separately')
    if d=='livecell' and stats.get('grouping_version'):
     from unified_paper_livecell_groups import PROOF,VERSION
     assert stats['grouping_version']==VERSION and stats['grouping_proof_sha256']==hashlib.sha256(PROOF.read_bytes()).hexdigest()
     assert len(found)==5 and all(hashlib.sha256(Path(x['result']).read_bytes()).hexdigest()==x['result_sha256']for x in stats['source'])
     row.update(patient_group_verified=True,uncertainty_group_scope=stats['group_unit'],grouping_proof=str(PROOF),uncertainty_note=stats['uncertainty']+'; '+stats['group_limitation'])
   for seed,r in found.items():runs.append({**row,'seed':seed,'run_value':r['value'],'raw_result_path':r['source'],'result_sha256':r['sha256'],'prediction_path':r['prediction_path']})
