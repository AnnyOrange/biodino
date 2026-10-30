#!/usr/bin/env python3
"""Grouped paired inference on fixed OOD predictions; training seeds are N/A."""
import argparse, json, hashlib, time, os, multiprocessing, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def dump(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def groups(frame):
 codes,unique=pd.factorize(frame.group_id,sort=True)
 strata=[]
 for source in sorted(frame.source.unique()):
  strata.append(np.unique(codes[np.asarray(frame.source==source)]))
 return codes,strata

def metrics(y,s,w=None):
 return np.asarray([roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)])

def bootstrap(a,b,config):
 codes,strata=groups(a);rng=np.random.default_rng(config['random_seed']);y=a.true_label.to_numpy();x=a.score.to_numpy()
 z=None if b is None else b.score.to_numpy();draws=[]
 # Perfect ranking remains perfect under every nonnegative source-stratified group weight.
 perfect=np.all(metrics(y,x)==1.)
 if perfect and z is None:return np.ones((2,2))
 if perfect and z is not None and np.all(metrics(y,z)==1.):return np.zeros((2,2))
 for _ in range(config['bootstrap_replicates']):
  counts=np.zeros(codes.max()+1)
  for ids in strata:counts+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=len(counts))
  w=counts[codes];v=metrics(y,x,w)
  if z is not None:v-=metrics(y,z,w)
  draws.append(v)
 return np.quantile(draws,[.025,.975],axis=0)

FRAMES={}
META={}
SETTINGS={}
VERSION='ood-groupfix-v3-parallel-prediction-hash'
SELECTION=ROOT/'outputs/02_eval_inputs/formal_v4/fm18_ood_group_selection_v3.json'
OOD_IDENTITY_FIELDS=('protocol','selection','split_revision','input_revision','test_ids_sha256')

def current_ood_identity(raw, dataset, expected=None):
 identity=tuple(raw.get(field) for field in OOD_IDENTITY_FIELDS)
 if (raw.get('dataset')!=dataset or raw.get('protocol')!='bio-eval-union-v4/formal-ood'
     or raw.get('selection')!=str(SELECTION)
     or raw.get('split_revision')!='source_stratified_fov_disjoint_v3'
     or raw.get('input_revision')!='cyclops_raw_float_percentile_0p5_99p5_v2'
     or not raw.get('test_ids_sha256')):
  raise ValueError(f'Non-current OOD protocol: {raw.get("model")}/{dataset}')
 if expected is not None and identity!=expected:
  raise ValueError(f'FM/BioDINO OOD identity mismatch: {raw.get("model")}/{dataset}')
 return identity

def provenance(keys):
 return dict(implementation=VERSION,prediction_sha256=[META[k]['prediction_sha256'] for k in keys],
             component_sha256=[META[k]['component_sha256'] for k in keys],bootstrap_replicates=SETTINGS['bootstrap_replicates'],
             permutation_replicates=SETTINGS['permutation_replicates'],random_seed=SETTINGS['random_seed'])

def valid(path,signature):
 try:return json.loads(path.read_text()).get('provenance')==signature
 except (FileNotFoundError,json.JSONDecodeError):return False

def permutation(a,b,s):
 y=a.true_label.to_numpy();x=a.score.to_numpy();z=b.score.to_numpy();delta=metrics(y,x)-metrics(y,z)
 codes,_=groups(a);rng=np.random.default_rng(s['random_seed']);extreme=np.zeros(2,dtype=int)
 for _ in range(0 if np.all(np.abs(delta)<1e-12) else s['permutation_replicates']):
  swap=rng.integers(0,2,codes.max()+1).astype(bool)[codes]
  diff=metrics(y,np.where(swap,z,x))-metrics(y,np.where(swap,x,z))
  extreme+=np.abs(diff)>=np.abs(delta)-1e-12
 p=np.ones(2) if np.all(np.abs(delta)<1e-12) else (extreme+1)/(s['permutation_replicates']+1)
 return delta,p

def job(task):
 candidate,dataset,reference=task;key=(candidate,dataset);a=FRAMES[key];r=META[key]['raw']
 target=OUT/'statistics/ood_groupfix_v3'/(f'{candidate}__vs__{reference}__{dataset}.json' if reference else f'{candidate}__{dataset}__ci.json')
 try:
  keys=[key]+([(reference,dataset)] if reference else []);sig=provenance(keys)
  if valid(target,sig):return dict(task=task,status='cached')
  payload=dict(input_revision='cyclops_raw_float_percentile_0p5_99p5_v2',split_revision='source_stratified_fov_disjoint_v3',dataset=dataset,
   n_test=len(a),n_groups=int(a.group_id.nunique()),bootstrap_replicates=SETTINGS['bootstrap_replicates'],
   permutation_replicates=SETTINGS['permutation_replicates'],training_seed='not_applicable',provenance=sig,
   uncertainty='Source-stratified independent-group bootstrap conditional on fixed ID reference bank; limited-source cryo inference retains whole project groups',
   group_note='Recorded source groups; BloodMNIST donor mapping unavailable; cryo four project groups',primary_metric='auroc')
  b=FRAMES[reference,dataset] if reference else None
  if b is not None:
   if current_ood_identity(r,dataset)!=current_ood_identity(META[reference,dataset]['raw'],dataset):
    raise ValueError(f'OOD protocol mismatch: {candidate}/{reference}/{dataset}')
   for col in ['sample_id','group_id','true_label','source']:
    if not np.array_equal(a[col],b[col]):raise ValueError(f'Unpaired {candidate}/{reference}/{dataset}/{col}')
   delta,p=permutation(a,b,SETTINGS);ci=bootstrap(a,b,SETTINGS)
   payload.update(candidate=candidate,reference=reference,
     metrics={m:{'difference':float(delta[i]),'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i]),'p_two_sided':float(p[i])} for i,m in enumerate(['auroc','average_precision'])})
  else:
   ci=bootstrap(a,None,SETTINGS)
   payload.update(model=candidate,prediction_path=r['prediction_path'],prediction_sha256=META[key]['prediction_sha256'],
     metrics={m:{'value':float(r['metrics'][m]),'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i])} for i,m in enumerate(['auroc','average_precision'])})
  if target.exists():
   old=target.read_bytes();archive=OUT/'statistics/history/ood_groupfix_v3';archive.mkdir(parents=True,exist_ok=True)
   history=archive/(target.stem+'__'+hashlib.sha256(old).hexdigest()+'.json')
   if not history.exists():history.write_bytes(old)
  dump(target,payload);return dict(task=task,status='complete')
 except Exception as e:return dict(task=task,status='error',error=repr(e),traceback=traceback.format_exc())

def run(workers=4):
 workers=min(workers,4)
 global FRAMES,META,SETTINGS
 cfg=json.loads((OUT/'config.json').read_text());SETTINGS=cfg['statistics'];FRAMES={};META={};errors=[]
 models={m['model']:m for m in cfg['models']};store=OUT/'statistics/ood_groupfix_v3';store.mkdir(exist_ok=True,parents=True);identities={}
 for path in sorted((OUT/'results').glob('*/ood_groupfix_v3/*_result.json')):
  try:
   r=json.loads(path.read_text());key=r['model'],r['dataset']
   if key[0] not in models or key[1] not in ['xray','cryo']:continue
   identity=current_ood_identity(r,key[1],identities.get(key[1]))
   p=Path(r['prediction_path']);f=pd.read_csv(p)
   if not np.isfinite(f.score).all() or f.sample_id.duplicated().any():raise ValueError('Invalid predictions')
   if not np.allclose(metrics(f.true_label,f.score),[r['metrics'][m] for m in ['auroc','average_precision']],atol=1e-12,rtol=0):raise ValueError('Point metric mismatch')
   identities[key[1]]=identity
   FRAMES[key]=f;META[key]={'raw':r,'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'component_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  except Exception as e:errors.append(dict(path=str(path),error=repr(e)))
 jobs=[]
 for (m,d),f in FRAMES.items():
  if not valid(store/f'{m}__{d}__ci.json',provenance([(m,d)])):jobs.append((m,d,None))
  if models[m]['kind']!='biodino':continue
  for fm,model in models.items():
   if model['kind']=='fm' and (fm,d) in FRAMES and not valid(store/f'{m}__vs__{fm}__{d}.json',provenance([(m,d),(fm,d)])):jobs.append((m,d,fm))
 # Publish individual CIs before potentially expensive paired AP permutations.
 jobs.sort(key=lambda x:x[2] is not None)
 state=dict(pid=os.getpid(),updated=time.time(),workers=workers,available_predictions=len(FRAMES),scheduled=len(jobs),completed_this_pass=0,errors=errors,status='running')
 statepath=OUT/'states/ood_statistics.json';dump(statepath,state);print('OOD STATISTICS',len(FRAMES),'results',len(jobs),'jobs',flush=True)
 if jobs:
  from unified_paper_statistics_resources import pool as bounded_pool
  with bounded_pool(workers) as pool:
   for result in pool.imap_unordered(job,jobs,chunksize=1):
    if result['status']=='error':state['errors'].append(result)
    else:state['completed_this_pass']+=1
    print('OOD',result,flush=True);state['updated']=time.time();dump(statepath,state)
 state.update(status='watching',updated=time.time());dump(statepath,state)

if __name__=='__main__':
 import fcntl
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
 with (OUT/'locks/ood_inputfix_statistics.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(args.workers)
   if not args.watch:break
   time.sleep(60)
