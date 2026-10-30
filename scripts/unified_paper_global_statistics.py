#!/usr/bin/env python3
"""Grouped deterministic-probe inference from sufficient statistics; no training repeats."""
import argparse,fcntl,json,time,hashlib,os,multiprocessing,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def load(p):
 r=json.loads(p.read_text());z=np.load(r['prediction_path']);y=z['true_label'];pred=z['prediction']
 if y.ndim!=1:raise ValueError('Multilabel AUC requires score-based grouped evaluator')
 ids=z['sample_id'].astype(str);groups=z['group_id'].astype(str);names,code=np.unique(groups,return_inverse=True);g=len(names)
 strata=[np.arange(g)];classes=None
 if r['task']=='classification':
  classes,yi=np.unique(y,return_inverse=True);c=np.zeros((g,len(classes),2))
  np.add.at(c,(code,yi,0),1);np.add.at(c,(code,yi,1),(y==pred).astype(float));metric_name='balanced_accuracy'
  # Preserve the class-conditional estimand of balanced accuracy when each source group has one label.
  if np.all((c[:,:,0]>0).sum(1)==1):strata=[np.flatnonzero(c[:,i,0]>0) for i in range(len(classes))]
 else:
  # Columns count, sum(y), sum(y^2), squared error, absolute error.
  c=np.zeros((g,5));sample=np.stack([np.ones(len(y)),y,y*y,(y-pred)**2,np.abs(y-pred)],axis=-1)
  np.add.at(c,code,sample);metric_name='r2'
  if r['dataset']=='bbbc013':strata=[np.flatnonzero(np.isin(names,list('ABCD'))),np.flatnonzero(np.isin(names,list('EFGH')))]
 return {'raw':r,'data':c,'ids':ids,'groups':groups,'y':y,'metric':metric_name,'strata':strata,'classes':classes,'prediction_sha256':hashlib.sha256(Path(r['prediction_path']).read_bytes()).hexdigest(),'component_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def metric(totals,kind):
 # Leading axes are replicate and stratum; last axes describe metric sufficient statistics.
 if kind=='balanced_accuracy':
  totals=totals.sum(axis=1);den=totals[...,0];v=np.divide(totals[...,1],den,out=np.full_like(den,np.nan),where=den>0)
  return v.mean(axis=-1)
 if kind=='mae':return (totals[...,4]/np.maximum(totals[...,0],1)).mean(axis=-1)
 n=totals[...,0];ss=totals[...,2]-totals[...,1]**2/np.maximum(n,1)
 v=np.divide(totals[...,3],ss,out=np.full_like(ss,np.nan),where=ss>1e-12)
 return (1-v).mean(axis=-1)

def sums(c,weights,strata):
 return np.stack([np.einsum('bg,g...->b...',weights[:,ix],c[ix]) for ix in strata],axis=1)

def point(a):return float(metric(sums(a['data'],np.ones((1,len(a['data']))),a['strata']),a['metric'])[0])

def bootstrap(a,b,reps,seed):
 rng=np.random.default_rng(seed);draw=[];invalid=0;g=len(a['data'])
 for start in range(0,reps,128):
  n=min(128,reps-start);w=np.zeros((n,g))
  for ix in a['strata']:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
  v=metric(sums(a['data'],w,a['strata']),a['metric'])
  if b is not None:v-=metric(sums(b['data'],w,a['strata']),a['metric'])
  invalid+=int((~np.isfinite(v)).sum());draw.extend(v[np.isfinite(v)].tolist())
 if len(draw)<reps*.95:raise ValueError(f'Too many class/variance-degenerate group bootstrap samples: {invalid}/{reps}')
 return np.quantile(draw,[.025,.975]).tolist(),len(draw)

def permutation(a,b,reps,seed):
 rng=np.random.default_rng(seed);delta=point(a)-point(b);count=0;g=len(a['data']);ones=np.ones((1,g))
 ta=sums(a['data'],ones,a['strata']);tb=sums(b['data'],ones,a['strata'])
 for start in range(0,reps,128):
  swap=rng.integers(0,2,size=(min(128,reps-start),g));change=sums(b['data']-a['data'],swap,a['strata'])
  d=metric(ta+change,a['metric'])-metric(tb-change,a['metric']);count+=int((np.abs(d)>=abs(delta)-1e-12).sum())
 return delta,(count+1)/(reps+1)

DATA={}
SETTINGS={}
VERSION='grouped-global-v2-prediction-hash'

def provenance(a,b=None):
 items=[a] if b is None else [a,b]
 return {'implementation':VERSION,'prediction_sha256':[x['prediction_sha256'] for x in items],
         'component_sha256':[x['component_sha256'] for x in items],
         'bootstrap_replicates':SETTINGS['bootstrap_replicates'],
         'permutation_replicates':SETTINGS['permutation_replicates'],
         'random_seed':SETTINGS['random_seed']}

def valid_existing(path,signature):
 try:return json.loads(path.read_text()).get('provenance')==signature
 except (FileNotFoundError,json.JSONDecodeError):return False

def save_stat(path,result):
 # Preserve previous outputs and their exact bytes before any correction/metadata refresh.
 if path.exists():
  old=path.read_bytes();archive=OUT/'statistics/history/global';archive.mkdir(parents=True,exist_ok=True)
  dest=archive/(path.stem+'__'+hashlib.sha256(old).hexdigest()+'.json')
  if not dest.exists():dest.write_bytes(old)
 dump(path,result)

def compute_job(job):
 kind,candidate,d,reference=job;a=DATA[candidate,d];b=DATA[reference,d] if reference else None
 path=OUT/'statistics'/(f'{candidate}__vs__{reference}__{d}.json' if b else f'{candidate}__{d}__ci.json')
 try:
  signature=provenance(a,b)
  if valid_existing(path,signature):return {'job':job,'status':'cached'}
  if b is not None:
   for name in ['ids','groups','y','classes']:
    same=np.allclose(a[name],b[name],rtol=0,atol=1e-12) if name=='y' and a['raw']['task']=='regression' else np.array_equal(a[name],b[name])
    if not same:raise ValueError(f'Unpaired {candidate}/{reference}/{d}/{name}')
   for name in ['protocol','split','split_sha256','probe']:
    if a['raw'].get(name)!=b['raw'].get(name):raise ValueError(f'Protocol mismatch: {name}')
  metrics={};valid=None
  for metric_name in (['r2','mae'] if a['raw']['task']=='regression' else [a['metric']]):
   am={**a,'metric':metric_name};bm={**b,'metric':metric_name} if b else None
   (lo,hi),valid=bootstrap(am,bm,SETTINGS['bootstrap_replicates'],SETTINGS['random_seed'])
   if b:
    delta,p=permutation(am,bm,SETTINGS['permutation_replicates'],SETTINGS['random_seed'])
    metrics[metric_name]={'difference':delta,'ci_low':lo,'ci_high':hi,'p_two_sided':p}
   else:metrics[metric_name]={'value':point(am),'ci_low':lo,'ci_high':hi}
  r=a['raw'];result={'task':r['task'],'dataset':d,'n_test':len(a['ids']),'n_groups':len(a['data']),
    'checkpoint':r['checkpoint'],'training_seed':'not_applicable','group_note':r.get('group_note','See source manifest'),
    'group_stratum_sizes':[len(ix) for ix in a['strata']],
    'stratification':'Class when groups have homogeneous labels; compound for BBBC013; otherwise none',
    'uncertainty':'Grouped source-unit percentile bootstrap, conditional on fixed fitted estimator and documented available group IDs; BBBC013 compound-stratified replicate rows',
    'test_definition':'Two-sided paired model-prediction swap by whole recorded source group; no sample/seed inflation',
    'bootstrap_replicates':SETTINGS['bootstrap_replicates'],'valid_bootstrap_replicates':valid,
    'permutation_replicates':SETTINGS['permutation_replicates'],'provenance':signature,
    'primary_metric':'mae' if r['task']=='regression' else a['metric'],'metrics':metrics}
  if d=='lc25000':
   result['primary_grouping_verified']=r.get('protocol')=='lc25000-clean-original-tile-80-20-seed0-v1' and r.get('annotation_sha256')=='2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23' and (b is None or b['raw'].get('annotation_sha256')==r['annotation_sha256'])
   result['statistical_scope']='Published original-tile groups; independent of augmentation descendants, conditional on unknown patient/slide identities'
  if b:result.update(candidate=candidate,reference=reference,reference_checkpoint=b['raw']['checkpoint'])
  else:result.update(model=candidate,prediction_path=r['prediction_path'])
  save_stat(path,result)
  return {'job':job,'status':'complete','result':str(path)}
 except Exception as e:return {'job':job,'status':'error','error':repr(e),'traceback':traceback.format_exc()}

def run(workers=4):
 workers=min(workers,4)
 global DATA,SETTINGS
 cfg=json.loads((OUT/'config.json').read_text());SETTINGS=cfg['statistics'];DATA={};errors=[];deferred=[]
 models={x['model']:x for x in cfg['models']}
 approved={(t['dataset'],t['task']) for t in cfg['tasks'] if t['task'] in {'classification','regression'}}
 for p in sorted((OUT/'results').glob('*/global/*/component_result.json')):
  try:
   raw=json.loads(p.read_text());m=raw['model'];d=raw['dataset']
   if m not in models or (d,raw['task']) not in approved:continue
   if d=='chestmnist':deferred.append({'model':m,'dataset':d,'reason':'Multilabel score-based AUC statistics handled separately'});continue
   a=load(p);v=point(a)
   if not np.isclose(v,raw[a['metric']],atol=1e-7):raise ValueError(f'Statistic mismatch {m}/{d}: {v} vs {raw[a["metric"]]}')
   DATA[m,d]=a
  except Exception as e:errors.append({'path':str(p),'error':repr(e)})
 jobs=[]
 for (m,d),a in DATA.items():
  target=OUT/'statistics'/f'{m}__{d}__ci.json'
  if not valid_existing(target,provenance(a)):jobs.append(('ci',m,d,None))
 for (m,d),a in DATA.items():
  if models[m]['kind']!='biodino':continue
  for fm,model in models.items():
   if model['kind']!='fm' or (fm,d) not in DATA:continue
   target=OUT/'statistics'/f'{m}__vs__{fm}__{d}.json'
   if not valid_existing(target,provenance(a,DATA[fm,d])):jobs.append(('pair',m,d,fm))
 state={'pid':os.getpid(),'updated':time.time(),'workers':workers,'available_results':len(DATA),'datasets':sorted({d for m,d in DATA}),
        'scheduled':len(jobs),'completed_this_pass':0,'errors':errors,'deferred':deferred,'status':'running'}
 state_path=OUT/'states/global_statistics.json';dump(state_path,state)
 print('GLOBAL STATISTICS',len(DATA),'results',len(jobs),'jobs',workers,'workers',flush=True)
 if jobs:
  from unified_paper_statistics_resources import pool as bounded_pool
  with bounded_pool(workers) as pool:
   for result in pool.imap_unordered(compute_job,jobs,chunksize=1):
    if result['status']=='error':state['errors'].append(result);print('GLOBAL ERROR',result,flush=True)
    else:state['completed_this_pass']+=1;print('GLOBAL',result['status'],result['job'],flush=True)
    state['updated']=time.time();dump(state_path,state)
 state.update(status='watching',updated=time.time());dump(state_path,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 with (OUT/'locks/global_statistics.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(a.workers)
   if not a.watch:break
   time.sleep(60)
