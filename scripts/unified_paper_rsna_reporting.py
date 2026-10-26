import json,hashlib
from pathlib import Path
import numpy as np

def apply_rsna_results(root,rows,runs,models):
 out=root/'outputs/02_eval_runs/unified_paper';source=root/'outputs/02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json';manifest=json.loads(source.read_text());h=hashlib.sha256(source.read_bytes()).hexdigest();queue=out/'rsna_queue.json'
 for m in models:
  key=(m,'box_detection','xray_chest','','fixed')
  if key not in rows:continue
  r=rows[key];found=[]
  for seed in range(5):
   p=out/'results'/m/'rsna_pneumonia'/f'seed{seed}/component_result.json'
   if not p.exists():continue
   d=json.loads(p.read_text());assert d['manifest_sha256']==h and d['metrics']['evaluated_images']==3000 and Path(d['prediction_path']).exists();found.append((seed,p,d))
  r.update(metric='rsna_mean_precision',required_seeds=5,completed_seeds=len(found),protocol='rsna-native-fixed-patient-purged-v1',sample_manifest=str(source),split_sha256=h,split_id='official-test3000/patient-purged-train15663-val1740',seed_semantics='training_randomness',data_complete='是',queue_file=str(queue),reason='Official3000test annotations mapped; NIH patient-independent15663train/1740val; native RCNN10epochs/LRvalidation search/5seeds queued',status='已排队')
  if found:
   vals=[d['metrics']['rsna_mean_precision'] for _,_,d in found];r.update(value=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,seed_values=json.dumps(vals),seed_ids=json.dumps([s for s,_,_ in found]),raw_result_path=json.dumps([str(p) for _,p,_ in found]),prediction_path=json.dumps([d['prediction_path'] for _,_,d in found]),protocol_verified=True)
   cp=out/'statistics'/f'{m}__xray_chest__ci.json'
   if len(found)==5 and cp.exists():
    stat=json.loads(cp.read_text());hashes=[d['prediction_sha256']for _,_,d in found]
    if stat.get('prediction_sha256')==hashes and stat.get('provenance',{}).get('manifest_sha256')==h:
     v=stat['metrics']['rsna_mean_precision'];r.update(ci_low=v['ci_low'],ci_high=v['ci_high'],statistics_path=str(cp),reason='Five independent native RSNA fits; NIH-patient grouped CI complete, including per-fit ignored-negative rule; paired family tracked separately')
   for seed,p,d in found:runs.append({**r,'seed':seed,'run_value':d['metrics']['rsna_mean_precision'],'raw_result_path':str(p),'prediction_path':d['prediction_path']})
