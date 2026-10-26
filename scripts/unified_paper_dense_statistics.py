#!/usr/bin/env python3
"""Image-cluster inference on mean single-seed F1, never an ensemble or patch bootstrap."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def dump(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def counts(result):
 d=json.loads(result.read_text());p=Path(d['prediction_path']);a=np.load(p)
 y=a['labels']>=.5;pred=a['predicted']if 'predicted'in a.files else a['logits']>=0
 y=y.reshape(len(y),-1);pred=pred.reshape(len(pred),-1)
 c=np.stack([(y&pred).sum(1),(~y&pred).sum(1),(y&~pred).sum(1)],axis=-1).astype(float)
 manifest=json.loads(Path(d['sample_manifest']).read_text())['test']
 if d.get('dataset')=='livecell':
  from unified_paper_livecell_groups import regroup
  manifest,metadata=regroup(manifest);d.update(metadata)
 ids=[r['sample_id'] for r in manifest];groups=[r['group_id'] for r in manifest]
 if len(set(ids))!=len(ids):raise ValueError('Repeated sample IDs')
 # Keep first-occurrence order so existing unique-image BBBC038 bootstrap
 # draws remain bit-for-bit identical, while correlated records stay together.
 unique=list(dict.fromkeys(groups));positions={g:i for i,g in enumerate(unique)};inverse=np.array([positions[g]for g in groups]);grouped=np.zeros((len(unique),3),dtype=float);np.add.at(grouped,inverse,c);c=grouped;groups=unique
 return d,c,ids,groups,hashlib.sha256(p.read_bytes()).hexdigest()

def metric(counts):
 # Last two axes are seed and TP/FP/FN; average metrics across independent fits.
 den=2*counts[...,0]+counts[...,1]+counts[...,2]
 f=np.divide(200*counts[...,0],den,out=np.zeros_like(den),where=den!=0)
 return f.mean(axis=-1)

def bootstrap(a,b,replicates,seed):
 rng=np.random.default_rng(seed);draw=[];n=a.shape[1]
 for start in range(0,replicates,128):
  size=min(128,replicates-start);weights=rng.multinomial(n,np.full(n,1/n),size=size)
  v=metric(np.einsum('bn,snc->bsc',weights,a))
  if b is not None:v-=metric(np.einsum('bn,snc->bsc',weights,b))
  draw.extend(v.tolist())
 return np.quantile(draw,[.025,.975]).tolist()

def permutation(a,b,replicates,seed):
 rng=np.random.default_rng(seed);delta=float(metric(a.sum(1))-metric(b.sum(1)));extreme=0
 total_a=a.sum(1);total_b=b.sum(1)
 for start in range(0,replicates,128):
  swaps=rng.integers(0,2,size=(min(128,replicates-start),a.shape[1]))
  change=np.einsum('bn,snc->bsc',swaps,b-a)
  dif=metric(total_a[None]+change)-metric(total_b[None]-change)
  extreme+=int(np.count_nonzero(np.abs(dif)>=abs(delta)-1e-12))
 return delta,(extreme+1)/(replicates+1)

def run(dataset='bbbc038'):
 cfg=json.loads((OUT/'config.json').read_text());s=cfg['statistics'];store=OUT/'statistics';data={}
 for model in cfg['models']:
  name=model['model'];paths=[OUT/'results'/name/'detection_proxy'/dataset/f'seed{k}'/'component_result.json' for k in range(5)]
  if not all(p.exists() for p in paths):continue
  arr=[counts(p) for p in paths];ids=arr[0][2];groups=arr[0][3]
  if any(x[2]!=ids or x[3]!=groups for x in arr):raise ValueError('Unpaired seed memberships')
  a=np.stack([x[1] for x in arr]);data[name]=(a,ids,groups,arr[0][0])
  actual=[float(metric(x[1].sum(0)[None])) for x in arr]
  if not np.allclose(actual,[x[0]['test_patch_f1'] for x in arr],atol=1e-6):raise ValueError('Saved logits do not reproduce F1')
  target=store/f'{name}__{dataset}__ci.json'
  if target.exists():
   old=json.loads(target.read_text())
   if old.get('prediction_sha256')==[x[4]for x in arr] and old.get('grouping_version')==arr[0][0].get('grouping_version'):continue
   from unified_paper_livecell_groups import archive_previous
   archive_previous(target)
  low,high=bootstrap(a,None,s['bootstrap_replicates'],s['random_seed'])
  dump(target,{'model':name,'task':'detection_proxy','dataset':dataset,'n_test':len(ids),'n_groups':len(set(groups)),
    'training_seeds':list(range(5)),'per_seed_values':actual,'seed_sd':float(np.std(actual,ddof=1)),
    'uncertainty':('Whole-culture-well'if dataset=='livecell'else'Original-source-group')+' cluster bootstrap of mean of five single-seed patch F1 scores; same group draw for every seed; conditional on these five fits',
    **{k:arr[0][0][k]for k in ['grouping_version','grouping_proof','grouping_proof_sha256','group_limitation']if k in arr[0][0]},
    'prediction_sha256':[x[4] for x in arr],'bootstrap_replicates':s['bootstrap_replicates'],'primary_grouping_verified':arr[0][0].get('primary_grouping_verified',False),'grouping':arr[0][0].get('grouping','Original image ID; parent sample hierarchy unavailable'),
    'metrics':{'test_patch_f1':{'value':float(np.mean(actual)),'ci_low':low,'ci_high':high}}})
  print('DENSE CI',name,flush=True)
 for candidate in [m['model'] for m in cfg['models'] if m['kind']=='biodino']:
  if candidate not in data:continue
  a,ids,groups,meta=data[candidate]
  for reference in [m['model'] for m in cfg['models'] if m['kind']=='fm']:
   if reference not in data:continue
   target=store/f'{candidate}__vs__{reference}__{dataset}.json'
   b,bids,bgroups,bmeta=data[reference]
   source_ci_sha256=[hashlib.sha256((store/f'{name}__{dataset}__ci.json').read_bytes()).hexdigest()for name in [candidate,reference]]
   if target.exists():
    old=json.loads(target.read_text())
    if old.get('source_ci_sha256')==source_ci_sha256:continue
    from unified_paper_livecell_groups import archive_previous
    archive_previous(target)
   if ids!=bids or groups!=bgroups:raise ValueError('Model test membership mismatch')
   low,high=bootstrap(a,b,s['bootstrap_replicates'],s['random_seed']);delta,p=permutation(a,b,s['permutation_replicates'],s['random_seed'])
   dump(target,{'candidate':candidate,'reference':reference,'task':'detection_proxy','dataset':dataset,'n_test':len(ids),'n_groups':len(set(groups)),
     'training_seeds':list(range(5)),'bootstrap_replicates':s['bootstrap_replicates'],'permutation_replicates':s['permutation_replicates'],
     'primary_grouping_verified':meta.get('primary_grouping_verified',False)and bmeta.get('primary_grouping_verified',False),'source_ci_sha256':source_ci_sha256,
     **{k:meta[k]for k in ['grouping_version','grouping_proof','grouping_proof_sha256','group_limitation']if k in meta},
     'test_definition':'Two-sided paired model swap by '+('culture well'if dataset=='livecell'else'original source group')+'; all patches, records and seed predictions in a group swap together; mean single-seed F1 difference',
     'metrics':{'test_patch_f1':{'difference':delta,'ci_low':low,'ci_high':high,'p_two_sided':p}}})
   print('DENSE PAIR',candidate,reference,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with (OUT/'locks/dense_statistics.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   for dataset in ['bbbc038','conic','livecell']:run(dataset)
   if not a.watch:break
   time.sleep(60)
