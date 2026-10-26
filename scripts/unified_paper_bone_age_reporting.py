import json,hashlib
from pathlib import Path
import numpy as np

def apply_bone_age_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';mp=root/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json'
 if not mp.exists():return
 h=hashlib.sha256(mp.read_bytes()).hexdigest();manifest=json.loads(mp.read_text());ids=[r['sample_id'] for r in manifest['records'] if r['split']=='official_validation']
 for m in models:
  key=(m,'bone_age_regression','xray_bone','','fixed')
  if key not in rows:continue
  r=rows[key];r.update(required_seeds=0,completed_seeds=0,seed_semantics='not_applicable',seed_values='[]',seed_ids='[]',data_complete='是',metric='mae',protocol=manifest['protocol'],split_id='RSNA-official-train12611-validation1425',split_sha256=h,queue_file=str(out/'bone_age_queue.json'),sample_manifest=str(mp),status='已排队',reason='Official validation1425images and ages downloaded and joined; fixed deterministic image-only Ridge alpha1 queued; evaluation cohort is validation, not hidden200test')
  p=out/'results'/m/'bone_age/component_result.json'
  if not p.exists():continue
  raw=json.loads(p.read_text())
  if raw.get('solver_precision')!='float64':
   r.update(reason='Legacy float32 Ridge queued for float64 recomputation on identical features');continue
  pred=json.loads(Path(raw['prediction_path']).read_text());assert raw['manifest_sha256']==h and pred['sample_id']==ids and raw['n_test']==1425 and raw['checkpoint']==models[m]['checkpoint'];value=float(np.mean(abs(np.array(pred['y_true'])-pred['y_pred'])));assert abs(value-raw['metrics']['mae'])<1e-8
  r.update(value=value,sd=np.nan,protocol_verified=True,raw_result_path=str(p),result=str(p),prediction_path=raw['prediction_path'],reason='Verified deterministic Ridge on full official validation cohort; grouped statistics pending',validation_basis='All1425saved predictions recomputed MAE in months; official validation membership hash')
  cp=out/'statistics'/f'{m}__xray_bone__ci.json'
  if cp.exists():
   stat=json.loads(cp.read_text());provenance=stat.get('provenance',{})
   if provenance.get('prediction_sha256')==[hashlib.sha256(Path(raw['prediction_path']).read_bytes()).hexdigest()] and provenance.get('component_sha256')==[hashlib.sha256(p.read_bytes()).hexdigest()]:
    v=stat['metrics']['mae'];r.update(ci_low=v['ci_low'],ci_high=v['ci_high'],statistics_path=str(cp),reason='Deterministic float64 Ridge, official1425case validation; recorded-case grouped CI complete; repeated-patient linkage unavailable and not certified; paired family tracked separately')
  runs.append({**r,'seed':'','run_value':value})
