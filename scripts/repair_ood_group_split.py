#!/usr/bin/env python3
"""Frozen source-stratified FOV-disjoint OOD scoring, reusing verified v2 features."""
import argparse,fcntl,hashlib,json,os,socket,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(8)
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_unified_paper_ood import inputs,sha,write_json
from dinov3.eval.eval_ood.metrics import id_vs_ood_knn
OUT=ROOT/'outputs/02_eval_runs/unified_paper';REPORT=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
SELECTION=ROOT/'outputs/02_eval_inputs/formal_v4/fm18_ood_group_selection_v3.json'
REVISION='cyclops_raw_float_percentile_0p5_99p5_v2';SPLIT='source_stratified_fov_disjoint_v3'

def manifest():
 old,_,_,identity,_=inputs('xray');bank=[];test=[]
 for source in sorted({r['source'] for r in identity}):
  groups=sorted({r['group_id'] for r in identity if r['source']==source})
  order=np.random.default_rng(0).permutation(groups);n=int(round(.7*len(order)));chosen=set(order[:n])
  for i,r in enumerate(identity):
   if r['source']==source:(bank if r['group_id'] in chosen else test).append(i)
 bank.sort();test.sort();assert not {identity[i]['group_id'] for i in bank}&{identity[i]['group_id'] for i in test}
 payload={'protocol':SPLIT,'previous_selection':str(ROOT/'outputs/02_eval_inputs/formal_v4/fm18_ood_selection_20261021.json'),
  'source_records':old['source_records'],'source_records_sha256':old['source_records_sha256'],
  'algorithm':'Within each ID source, sorted unique source FOV groups permuted with numpy.default_rng(0); round(0.7*N_groups) reference groups, remainder test. All selected images retained; no score-informed selection.',
  'group_relation':'CyclOps prefix before _gfp/_rfp is paired same numbered field. BBBC048 field prefix; BloodMNIST image ID (donor ID unavailable).',
  'id_bank':[dict(position=i,**identity[i]) for i in bank],'id_test':[dict(position=i,**identity[i]) for i in test],
  'id_bank_sha256':sha([identity[i] for i in bank]),'id_test_sha256':sha([identity[i] for i in test]),'id_group_overlap':0,
  'xray_records':old['xray_records'],'cryo_records':old['cryo_records']}
 if SELECTION.exists():assert json.loads(SELECTION.read_text())==payload
 else:write_json(SELECTION,payload)
 oldb={x['position'] for x in old['id_bank']};oldt={x['position'] for x in old['id_test']}
 overlap={identity[i]['group_id'] for i in oldb}&{identity[i]['group_id'] for i in oldt}
 pd.DataFrame([dict(position=i,**r,old_split='bank' if i in oldb else 'test',split='bank' if i in bank else 'test',previous_crossing_group=r['group_id'] in overlap) for i,r in enumerate(identity)]).to_csv(REPORT/'ood_group_split_v3_manifest.csv',index=False)
 return payload,identity

def run(model_filter=None):
 selected,identity=manifest();bank=np.array([r['position'] for r in selected['id_bank']]);test=np.array([r['position'] for r in selected['id_test']]);completed=[];waiting=[]
 models=json.loads((OUT/'config.json').read_text())['models']
 if model_filter:models=[m for m in models if m['model']==model_filter]
 for model in models:
  name=model['model'];old=OUT/'results'/name/'ood_inputfix_v2';target=OUT/'results'/name/'ood_groupfix_v3'
  for task in ['xray','cryo']:
   rawpath=old/f'{task}_result.json';dst=target/f'{task}_result.json'
   if dst.exists():completed.append(str(dst));continue
   if not rawpath.exists():waiting.append(f'{name}/{task}: corrected features pending');continue
   raw=json.loads(rawpath.read_text());meta=json.loads((old/'id_features.json').read_text())
   if raw.get('input_revision')!=REVISION or meta.get('input_revision')!=REVISION:
    waiting.append(f'{name}/{task}: unverified v2 feature cache metadata; other models continue');continue
   _,_,_,ids,other=inputs(task);assert ids==identity
   idf=np.load(old/'id_features.npy').astype(np.float32);of=np.load(old/f'{task}_features.npy').astype(np.float32)
   assert np.isfinite(idf).all() and np.isfinite(of).all()
   write_json(OUT/'states'/f'{name}__ood__{task}__groupfix_v3.json',{'state':'running','host':socket.gethostname(),'pid':os.getpid(),'started':time.time()})
   metrics,pred=id_vs_ood_knn(idf,of,k=10,return_predictions=True,bank_indices=bank,test_indices=test)
   rows=[identity[i] for i in test]+other;target.mkdir(parents=True,exist_ok=True);prediction=target/f'{task}_predictions.csv'
   frame=pd.DataFrame(rows);frame['model']=name;frame['checkpoint']=raw['checkpoint'];frame['dataset']=task;frame['task']='ood';frame['fold']='group_fixed';frame['seed']='not_applicable';frame['true_label']=pred['labels'];frame['score']=pred['scores'];frame.to_csv(prediction,index=False)
   y=frame.true_label;s=frame.score
   assert np.isclose(metrics['auroc'],roc_auc_score(y,s),atol=1e-12) and np.isclose(metrics['average_precision'],average_precision_score(y,s),atol=1e-12)
   for ext in ['.npy','.json']:
    for prefix in ['id_features',f'{task}_features']:
     link=target/(prefix+ext)
     if not link.exists():os.link(old/link.name,link)
   write_json(dst,{**raw,'metrics':metrics,'prediction_path':str(prediction),'selection':str(SELECTION),'split_revision':SPLIT,
    'test_ids_sha256':sha(rows),'previous_result':str(rawpath),'previous_result_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),
    'feature_reuse':'Exact v2 corrected cached embeddings; only FOV-group reference/test assignment and kNN scores recomputed. No encoder or OOD population changes.',
    'statistics_status':'pending_group_bootstrap','host':socket.gethostname(),'pid':os.getpid()})
   write_json(OUT/'states'/f'{name}__ood__{task}__groupfix_v3.json',{'state':'completed_metrics','host':socket.gethostname(),'pid':os.getpid(),'ended':time.time(),'result':str(dst)})
   print(name,task,metrics,flush=True);completed.append(str(dst))
 write_json(OUT/'states'/('ood_groupfix_v3_'+model_filter+'.json' if model_filter else 'ood_groupfix_v3.json'),{'host':socket.gethostname(),'pid':os.getpid(),'updated':time.time(),'completed':len(completed),'expected':len(models)*2,'waiting':waiting,'bank':len(bank),'test':len(test),'selection':str(SELECTION)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--model');a=p.parse_args()
 with (OUT/'locks'/('ood_groupfix_v3_'+a.model+'.lock' if a.model else 'ood_groupfix_v3.lock')).open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   run(a.model)
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
