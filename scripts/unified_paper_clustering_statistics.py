"""Label-invariant paired grouped NMI inference for five fixed fitted partitions.

Use shared whole-group bootstrap draws. A centered paired-bootstrap null test
avoids exchanging non-identifiable cluster IDs across models. This is an
approximate bootstrap test, explicitly not an exact prediction-swap test.
"""
import argparse,fcntl,hashlib,json,multiprocessing,os,time,traceback
from pathlib import Path
import numpy as np
from unified_paper_retrieval_statistics import nmi
from unified_paper_sparse_nmi import prepare as sparse_prepare,score as sparse_score,group_count
import unified_paper_global_statistics as common
ROOT=common.ROOT;OUT=common.OUT;STORE=OUT/'statistics/clustering';DATA={};SETTINGS={}
VERSION='clustering-paired-source-group-centered-normal-bootstrap-v2'
PREVIOUS_VERSION='clustering-paired-source-group-bootstrap-v1'
METHOD_CONFIG=OUT/'clustering_statistics_config.json'

def centered_interval(observed,draws,bounds):
 # Resampling finite NMI partitions changes the plug-in estimator's mean.
 # Estimate its sampling SE from whole-group draws and center on the v4 score.
 radius=1.959963984540054*np.std(draws,ddof=1)
 return np.clip([observed-radius,observed+radius],*bounds)

def load(model,dataset,spec):
 base=OUT/'results'/model/'retrieval_clustering'/dataset;paths=[base/f'seed{s}'/'clustering_result.json'for s in range(5)]
 if not all(p.exists() for p in paths):return None
 assignments=[];hashes=[];metrics=[];raws=[];prediction_paths=[]
 for seed,path in enumerate(paths):
  r=json.loads(path.read_text());pred=Path(r['prediction_path']);h=hashlib.sha256(pred.read_bytes()).hexdigest();assert h==r['prediction_sha256'];assert r['seed']==seed and r['checkpoint']==spec['checkpoint']
  with np.load(pred) as z:
   ids=z['sample_id'].astype(str);groups=z['group_id'].astype(str);labels=z['labels'];cluster=z['cluster']
  if not seed:
   ids0=ids;groups0=groups;labels0=labels;manifest=r['manifest_sha256'];names,code=np.unique(groups,return_inverse=True);classes,yi=np.unique(labels,return_inverse=True)
  else:assert np.array_equal(ids,ids0) and np.array_equal(groups,groups0) and np.array_equal(labels,labels0) and r['manifest_sha256']==manifest
  assignments.append(np.unique(cluster,return_inverse=True)[1]);hashes.append(h);metrics.append(r['metrics']['nmi']);raws.append(r);prediction_paths.append(str(pred))
 assert len(set(ids))==len(ids)
 k=max(int(x.max())+1 for x in assignments)
 if len(classes)>128:
  tables=sparse_prepare(code,yi,assignments,len(names));observed=sparse_score(tables,np.ones((1,len(names))))[0]
 else:
  tables=np.zeros((len(names),5,len(classes),k))
  for seed,cluster in enumerate(assignments):np.add.at(tables[:,seed],(code,yi,cluster),1)
  observed=nmi(tables.sum(0))
 assert np.allclose(observed,metrics,atol=1e-12,rtol=0)
 return dict(model=model,dataset=dataset,counts=tables,ids=ids0,groups=groups0,labels=labels0,manifest_sha256=manifest,prediction_sha256=hashes,prediction_paths=prediction_paths,point=float(observed.mean()),seed_values=metrics,checkpoint=spec['checkpoint'],grouping=raws[0].get('grouping','Source image groups'),group_limitation=raws[0]['group_limitation'],primary_grouping_verified=raws[0].get('primary_grouping_verified',False))

def signature(a,b=None):
 items=[a] if b is None else [a,b]
 return dict(implementation=VERSION,ci_method=SETTINGS['clustering_ci_method'],test_method=SETTINGS['clustering_test_method'],prediction_sha256=[x['prediction_sha256']for x in items],manifest_sha256=a['manifest_sha256'],bootstrap_replicates=SETTINGS['bootstrap_replicates'],test_replicates=SETTINGS.get('clustering_test_replicates',10000),random_seed=SETTINGS['random_seed'])

def cache_path(a):return STORE/'bootstrap_draws'/f"{a['model']}__{a['dataset']}.npz"

def compute_draws(tables,repeats,seed):
 rng=np.random.default_rng(seed);g=group_count(tables);draws=[]
 for start in range(0,repeats,64):
  weights=rng.multinomial(g,np.full(g,1/g),size=min(64,repeats-start))
  values=sparse_score(tables,weights)if isinstance(tables,dict)else nmi(np.tensordot(weights,tables,axes=(1,0)))
  draws.extend(values.mean(-1).tolist())
 return np.asarray(draws)

def model_job(key):
 a=DATA[key];sig=signature(a);cache=cache_path(a)
 try:
  draws=None
  if cache.exists():
   with np.load(cache) as z:
    prior=json.loads(str(z['signature']))
    comparable={k:v for k,v in prior.items()if k not in ['implementation','ci_method','test_method']}
    expected={k:v for k,v in sig.items()if k not in ['implementation','ci_method','test_method']}
    if comparable==expected and prior['implementation']in [VERSION,PREVIOUS_VERSION]:draws=z['draws'].copy()
  if draws is None:draws=compute_draws(a['counts'],max(sig['bootstrap_replicates'],sig['test_replicates']),sig['random_seed'])
  assert len(draws)>=max(sig['bootstrap_replicates'],sig['test_replicates']) and np.isfinite(draws).all()
  ci=centered_interval(a['point'],draws[:sig['bootstrap_replicates']],(0,1));existing=STORE/f"{a['model']}__{a['dataset']}__ci.json"
  if existing.exists():
   old=json.loads(existing.read_text())
   if old.get('provenance')==sig and old.get('prediction_sha256')==a['prediction_sha256']:
    v=old['metrics']['nmi'];assert np.allclose(ci,[v['ci_low'],v['ci_high']],atol=1e-12,rtol=0)
    return dict(key=key,status='cached')
  cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,draws=draws,signature=json.dumps(sig,sort_keys=True));tmp.replace(cache)
  result=dict(model=a['model'],dataset=a['dataset'],task='clustering',n_test=len(a['ids']),n_groups=group_count(a['counts']),training_seeds=list(range(5)),per_seed_values=a['seed_values'],seed_sd=float(np.std(a['seed_values'],ddof=1)),prediction_sha256=a['prediction_sha256'],bootstrap_replicates=sig['bootstrap_replicates'],provenance=sig,method_config=str(METHOD_CONFIG),primary_grouping_verified=a['primary_grouping_verified'],uncertainty='Normal-approximation 95% interval around the unaltered empirical NMI using the whole-group bootstrap standard error; the resampling mean is not substituted for the reported score. Same group draws across five separate fitted partitions. '+a['grouping']+'. '+a['group_limitation'],paired_test_status='Paired source-group difference and centered bootstrap null test; explicit method in paired files',metrics={'nmi':dict(value=a['point'],ci_low=float(ci[0]),ci_high=float(ci[1]))})
  common.save_stat(existing,result);return dict(key=key,status='complete',n_groups=group_count(a['counts']))
 except Exception as e:return dict(key=key,status='error',error=repr(e),traceback=traceback.format_exc())

def pair_job(job):
 candidate,reference,dataset=job;a=DATA[candidate,dataset];b=DATA[reference,dataset];sig=signature(a,b);target=STORE/f'{candidate}__vs__{reference}__{dataset}.json'
 try:
  if common.valid_existing(target,sig):return dict(job=job,status='cached')
  for name in ['ids','groups','labels']:
   assert np.array_equal(a[name],b[name]),'Unpaired '+name
  assert a['manifest_sha256']==b['manifest_sha256']
  draws=[]
  for item in [a,b]:
   with np.load(cache_path(item)) as z:
    assert str(z['signature'])==json.dumps(signature(item),sort_keys=True);draws.append(z['draws'])
  differences=draws[0]-draws[1];delta=a['point']-b['point'];ci=centered_interval(delta,differences[:sig['bootstrap_replicates']],(-1,1));null=differences[:sig['test_replicates']]-differences[:sig['test_replicates']].mean()
  p=(int((np.abs(null)>=abs(delta)-1e-12).sum())+1)/(len(null)+1)
  result=dict(candidate=candidate,reference=reference,dataset=dataset,task='clustering',n_test=len(a['ids']),n_groups=group_count(a['counts']),training_seeds=list(range(5)),primary_metric='nmi',primary_grouping_verified=a['primary_grouping_verified']and b['primary_grouping_verified'],statistical_scope='Exploratory conditional inference. '+a['grouping']+'. '+a['group_limitation'],provenance=sig,method_config=str(METHOD_CONFIG),prediction_sha256=[a['prediction_sha256'],b['prediction_sha256']],bootstrap_replicates=sig['bootstrap_replicates'],test_replicates=sig['test_replicates'],permutation_replicates=0,paired_test_method='two-sided centered paired whole-group bootstrap test',test_definition='Resample identical whole recorded groups in both models and all five fitted partitions; recompute each seed NMI and average five scores before subtraction. Normal-approximation95percent CI uses paired whole-group bootstrap standard error centered on the unchanged empirical difference. The two-sided approximate null test centers the paired bootstrap differences at their bootstrap mean and uses plus-one correction; it is not an exact cluster-ID permutation. No seed/sample inflation, retraining or ensemble.',metrics={'nmi':dict(difference=delta,ci_low=float(ci[0]),ci_high=float(ci[1]),p_two_sided=float(p))})
  common.save_stat(target,result);return dict(job=job,status='complete')
 except Exception as e:return dict(job=job,status='error',error=repr(e),traceback=traceback.format_exc())

def run(workers):
 workers=min(workers,4)
 global DATA,SETTINGS
 cfg=json.loads((OUT/'config.json').read_text());SETTINGS={**cfg['statistics'],**json.loads(METHOD_CONFIG.read_text())};models={m['model']:m for m in cfg['models']};DATA={};errors=[]
 for dataset in ['nct-crc-he-1k','crc-val-he-7k','nct-crc-he-100','lc25000','hpa-subcellular','rxrx3-core','rxrx1-cross']:
  for m,spec in models.items():
   try:
    a=load(m,dataset,spec)
    if a is not None:DATA[m,dataset]=a
   except Exception as e:errors.append(dict(model=m,dataset=dataset,error=repr(e)))
 state=dict(pid=os.getpid(),updated=time.time(),models_datasets=len(DATA),workers=workers,errors=errors,status='running',draw_sets_completed=0,pairs_completed=0);statepath=OUT/'states/clustering_paired_statistics.json';common.dump(statepath,state)
 from unified_paper_statistics_resources import pool as bounded_pool
 with bounded_pool(workers) as pool:
  good=set()
  for result in pool.imap_unordered(model_job,list(DATA)):
   print('CLUSTER BOOTSTRAP',result,flush=True)
   if result['status']=='error':state['errors'].append(result)
   else:good.add(result['key']);state['draw_sets_completed']+=1
   state['updated']=time.time();common.dump(statepath,state)
  jobs=[(m,fm,d)for m,d in good if models[m]['kind']=='biodino' for fm,spec in models.items() if spec['kind']=='fm' and (fm,d) in good]
  state['pairs_scheduled']=len(jobs)
  for result in pool.imap_unordered(pair_job,jobs):
   print('CLUSTER PAIR',result,flush=True)
   if result['status']=='error':state['errors'].append(result)
   else:state['pairs_completed']+=1
   state['updated']=time.time();common.dump(statepath,state)
 state.update(status='watching',updated=time.time());common.dump(statepath,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 STORE.mkdir(exist_ok=True)
 with (OUT/'locks/clustering_paired_statistics.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(args.workers)
   if not args.watch:break
   time.sleep(60)
