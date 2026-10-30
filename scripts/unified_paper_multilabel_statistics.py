"""Whole-group multilabel AUC inference; synchronized labels, fixed fitted probes.

Paired AUC difference under complementary swaps is linear in sample swap signs:
quadratic terms cancel because both models share labels and class denominators.
The coefficient implementation is checked against direct recomputation, including ties.
"""
import argparse,fcntl,hashlib,json,multiprocessing,os,time,traceback
from pathlib import Path
import numpy as np
import unified_paper_global_statistics as common
ROOT=common.ROOT;OUT=common.OUT
VERSION='multilabel-grouped-auc-v1'
DATA={};SETTINGS={}

def weighted_auc(y,score,weights):
 """Weights are [bootstrap, sample], shared across all labels."""
 result=np.zeros(len(weights),dtype=np.float64)
 for col in range(y.shape[1]):
  order=np.argsort(score[:,col],kind='stable');ss=score[order,col];yy=y[order,col]
  starts=np.r_[0,np.flatnonzero(np.diff(ss)!=0)+1]
  w=weights[:,order];pos=np.add.reduceat(w*yy,starts,axis=1);neg=np.add.reduceat(w*(1-yy),starts,axis=1)
  numerator=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)
  den=pos.sum(axis=1)*neg.sum(axis=1)
  result+=np.divide(numerator,den,out=np.full_like(numerator,np.nan),where=den>0)/y.shape[1]
 return result

def permutation_coefficients(y,a,b,code):
 """AUC_A(swap)-AUC_B(swap) = group_coeff @ signs, signs=+1 for no swap."""
 coeff=np.zeros(len(y),dtype=np.float64)
 for col in range(y.shape[1]):
  pos=y[:,col]==1;neg=~pos;den=int(pos.sum())*int(neg.sum())
  if not den:raise ValueError('Undefined AUC: label has only one class')
  negatives=np.sort(np.r_[a[neg,col],b[neg,col]])
  positives=np.sort(np.r_[a[pos,col],b[pos,col]])
  def lower_mid(values,x):return .5*(np.searchsorted(values,x,side='left')+np.searchsorted(values,x,side='right'))
  coeff[pos]+=.5*(lower_mid(negatives,a[pos,col])-lower_mid(negatives,b[pos,col]))/den/y.shape[1]
  # Greater-positive counts: n_positive_union - lower_mid(positive_union,score).
  coeff[neg]+=.5*(lower_mid(positives,b[neg,col])-lower_mid(positives,a[neg,col]))/den/y.shape[1]
 return np.bincount(code,weights=coeff,minlength=int(code.max())+1)

def load(path):
 r=json.loads(path.read_text());p=Path(r['prediction_path'])
 with np.load(p) as z:y=z['true_label'];score=z['score'];ids=z['sample_id'].astype(str);groups=z['group_id'].astype(str)
 if y.ndim!=2 or score.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(score).all():raise ValueError('Invalid multilabel outputs')
 names,code=np.unique(groups,return_inverse=True)
 point=float(weighted_auc(y,score,np.ones((1,len(y))))[0])
 if not np.isclose(point,r['macro_auc'],rtol=0,atol=1e-12):raise ValueError(f'Metric mismatch: {point} vs {r["macro_auc"]}')
 return dict(raw=r,y=y,score=score,ids=ids,groups=groups,code=code,n_groups=len(names),point=point,
             prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),component_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def signature(a,b=None):
 items=[a] if b is None else [a,b]
 return dict(implementation=VERSION,prediction_sha256=[x['prediction_sha256'] for x in items],
             component_sha256=[x['component_sha256'] for x in items],bootstrap_replicates=SETTINGS['bootstrap_replicates'],
             permutation_replicates=SETTINGS['permutation_replicates'],random_seed=SETTINGS['random_seed'])

def base(a):
 r=a['raw']
 return dict(dataset=r['dataset'],task=r['task'],checkpoint=r['checkpoint'],n_test=len(a['y']),n_groups=a['n_groups'],
  training_seed='not_applicable',primary_metric='macro_auc',group_note=r.get('group_note'),
  uncertainty='Whole recorded source-group percentile bootstrap; same draws for all labels and models; conditional on fixed probe and available group IDs',
  test_definition='Two-sided whole-group paired prediction swaps, synchronized across labels; exact AUC difference algebra with 10000 Monte Carlo swaps',
  bootstrap_replicates=SETTINGS['bootstrap_replicates'],permutation_replicates=SETTINGS['permutation_replicates'])

def ci_job(model):
 a=DATA[model];r=a['raw'];target=OUT/'statistics'/f'{model}__chestmnist__ci.json';cache=OUT/'statistics/multilabel_draws'/f'{model}.npz';sig=signature(a)
 try:
  if common.valid_existing(target,sig) and cache.exists():
   with np.load(cache) as z:
    if str(z['signature'])==json.dumps(sig,sort_keys=True):return dict(model=model,status='cached',cache=str(cache))
  rng=np.random.default_rng(SETTINGS['random_seed']);draws=[];g=a['n_groups']
  # Each group draw is shared by all labels; do not independently stratify labels.
  for start in range(0,SETTINGS['bootstrap_replicates'],32):
   size=min(32,SETTINGS['bootstrap_replicates']-start);w=rng.multinomial(g,np.full(g,1/g),size=size)
   draws.extend(weighted_auc(a['y'],a['score'],w[:,a['code']]).tolist())
  draws=np.asarray(draws);valid=np.isfinite(draws)
  if valid.sum()<.95*len(draws):raise ValueError('Too many class-degenerate grouped AUC draws')
  lo,hi=np.quantile(draws[valid],[.025,.975]);result=base(a)
  result.update(model=model,prediction_path=r['prediction_path'],provenance=sig,valid_bootstrap_replicates=int(valid.sum()),
                metrics={'macro_auc':dict(value=a['point'],ci_low=float(lo),ci_high=float(hi))})
  cache.parent.mkdir(exist_ok=True);tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,draws=draws,signature=json.dumps(sig,sort_keys=True));tmp.replace(cache)
  common.save_stat(target,result);return dict(model=model,status='complete',cache=str(cache))
 except Exception as e:return dict(model=model,status='error',error=repr(e),traceback=traceback.format_exc())

def pair_job(job):
 m,fm=job;a=DATA[m];b=DATA[fm];sig=signature(a,b);target=OUT/'statistics'/f'{m}__vs__{fm}__chestmnist.json'
 try:
  if common.valid_existing(target,sig):return dict(job=job,status='cached')
  for key in ['y','ids','groups']:
   if not np.array_equal(a[key],b[key]):raise ValueError('Unpaired '+key)
  for key in ['protocol','split','split_sha256','probe']:
   if a['raw'].get(key)!=b['raw'].get(key):raise ValueError('Protocol mismatch '+key)
  draws=[]
  for model,item in [(m,a),(fm,b)]:
   with np.load(OUT/'statistics/multilabel_draws'/f'{model}.npz') as z:
    if str(z['signature'])!=json.dumps(signature(item),sort_keys=True):raise ValueError('Stale bootstrap cache')
    draws.append(z['draws'])
  differences=draws[0]-draws[1];valid=np.isfinite(differences)
  if valid.sum()<.95*len(differences):raise ValueError('Too many degenerate paired draws')
  lo,hi=np.quantile(differences[valid],[.025,.975]);delta=a['point']-b['point']
  coeff=permutation_coefficients(a['y'],a['score'],b['score'],a['code'])
  if not np.isclose(coeff.sum(),delta,rtol=0,atol=1e-12):raise ValueError('Permutation coefficient identity failed')
  rng=np.random.default_rng(SETTINGS['random_seed']);reps=SETTINGS['permutation_replicates'];count=0
  for start in range(0,reps,128):
   signs=1-2*rng.integers(0,2,(min(128,reps-start),len(coeff)));null=signs@coeff
   count+=int((np.abs(null)>=abs(delta)-1e-12).sum())
  result=base(a);result.update(candidate=m,reference=fm,reference_checkpoint=b['raw']['checkpoint'],provenance=sig,
   valid_bootstrap_replicates=int(valid.sum()),metrics={'macro_auc':dict(difference=delta,ci_low=float(lo),ci_high=float(hi),p_two_sided=(count+1)/(reps+1))})
  common.save_stat(target,result);return dict(job=job,status='complete')
 except Exception as e:return dict(job=job,status='error',error=repr(e),traceback=traceback.format_exc())

def run(workers):
 workers=min(workers,4)
 global DATA,SETTINGS
 cfg=json.loads((OUT/'config.json').read_text());SETTINGS=cfg['statistics'];DATA={};models={m['model']:m for m in cfg['models']};errors=[]
 for path in (OUT/'results').glob('*/global/chestmnist/component_result.json'):
  try:
   a=load(path)
   if a['raw']['model'] in models:DATA[a['raw']['model']]=a
  except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 state=dict(pid=os.getpid(),updated=time.time(),models=len(DATA),workers=workers,errors=errors,status='running',ci_completed=0,pairs_completed=0)
 statepath=OUT/'states/multilabel_statistics.json';common.dump(statepath,state)
 from unified_paper_statistics_resources import pool as bounded_pool
 with bounded_pool(workers) as pool:
  good=set()
  for r in pool.imap_unordered(ci_job,list(DATA)):
   print('MULTILABEL CI',r,flush=True)
   if r['status']=='error':state['errors'].append(r)
   else:good.add(r['model']);state['ci_completed']+=1
   state['updated']=time.time();common.dump(statepath,state)
  jobs=[(m,fm) for m in good if models[m]['kind']=='biodino' for fm in good if models[fm]['kind']=='fm']
  state['pairs_scheduled']=len(jobs)
  for r in pool.imap_unordered(pair_job,jobs):
   print('MULTILABEL PAIR',r,flush=True)
   if r['status']=='error':state['errors'].append(r)
   else:state['pairs_completed']+=1
   state['updated']=time.time();common.dump(statepath,state)
 state.update(status='watching',updated=time.time());common.dump(statepath,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 with (OUT/'locks/multilabel_statistics.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(args.workers)
   if not args.watch:break
   time.sleep(60)
