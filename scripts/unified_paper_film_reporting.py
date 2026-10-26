"""Verified FILM fixed-group fits; preserve each actual training seed and prediction."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def apply_film_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';manifest=root/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json'
 if not manifest.exists():return
 h=hashlib.sha256(manifest.read_bytes()).hexdigest();cfg=json.loads(manifest.read_text());expected=sorted({cfg['records'][i]['group_id'] for i in cfg['splits']['test']})
 queue=out/'film_queue.json';jobs=json.loads(queue.read_text())['tasks'] if queue.exists() else []
 for m in models:
  key=(m,'classification','film_age','','fixed')
  if key not in rows:continue
  r=rows[key];found=[]
  for seed in range(5):
   p=out/'results'/m/'film_age'/f'seed{seed}/component_result.json'
   if not p.exists():continue
   raw=json.loads(p.read_text());pred=json.loads(Path(raw['prediction_path']).read_text())
   assert raw['manifest_sha256']==h and raw['checkpoint']==models[m]['checkpoint'] and sorted(pred['group_id'])==expected and len(pred['group_id'])==len(set(pred['group_id']))
   v=float(balanced_accuracy_score(pred['y_true'],pred['y_pred']));assert abs(v-raw['metrics']['balanced_accuracy'])<1e-12
   found.append((seed,v,p,raw))
  r.update(required_seeds=5,completed_seeds=len(found),protocol=cfg['protocol'],split_sha256=h,split_id='FILM-fixed-train8-val8-test11-biological-groups',data_complete='是',seed_semantics='training_randomness',queue_file=str(queue),sample_manifest=str(manifest),patient_group_verified=True)
  if found:
   vals=[v for _,v,_,_ in found];r.update(value=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,ci_low=np.nan,ci_high=np.nan,seed_values=json.dumps(vals),seed_ids=json.dumps([s for s,_,_,_ in found]),protocol_verified=True,raw_result_path=json.dumps([str(p) for _,_,p,_ in found]),result=str(out/'results'/m/'film_age/component_result.json'),prediction_path=json.dumps([d['prediction_path'] for _,_,_,d in found]),reason='Fixed biological-group age classification, project-defined; five random linear heads, validation-only epoch selection; no changing test cohorts',validation_basis='Recomputed each seed balanced accuracy from saved group predictions; identical manifest/checkpoint validated')
   ci=out/'statistics/film'/f'{m}__film_age__ci.json'
   if ci.exists() and len(found)==5:
    stat=json.loads(ci.read_text());hashes=[hashlib.sha256(Path(raw['prediction_path']).read_bytes()).hexdigest() for _,_,_,raw in found]
    if stat['prediction_sha256']==hashes:
     metric=stat['metrics']['balanced_accuracy'];r.update(ci_low=metric['ci_low'],ci_high=metric['ci_high'],statistics_path=str(ci))
   for seed,v,p,raw in found:runs.append({**r,'seed':seed,'run_value':v,'raw_result_path':str(p),'prediction_path':raw['prediction_path']})
  else:r.update(value=np.nan,reason='Persistent FILM fixed-group queue; waiting for published encoder capacity',status='已排队')
