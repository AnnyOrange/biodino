"""Patient-block inference for mean of five individual native RSNA precisions."""
import argparse,fcntl,hashlib,json,multiprocessing,os,time,traceback,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dinov3.eval.bio_detection.rsna import image_precision
import unified_paper_global_statistics as common
ROOT=common.ROOT;OUT=common.OUT;DATA={};SETTINGS={}
VERSION='rsna-native-patient-five-fit-v1'

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def signature(p):
 s=p.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)

def index_seed(path,manifest,manifest_hash):
 r=json.loads(path.read_text());pred=Path(r['prediction_path']);assert r['manifest_sha256']==manifest_hash
 records=[row for row in manifest['records'] if row['split']=='test'];ids=[x['sample_id']for x in records];groups=[x['group']for x in records]
 cache=OUT/'statistics/rsna_index'/r['model']/f"seed{r['seed']}.npz";meta=cache.with_suffix('.json');proof=dict(component_sha256=digest(path),prediction_sha256=r['prediction_sha256'],prediction_stat=signature(pred),manifest_sha256=manifest_hash)
 if cache.exists() and meta.exists() and json.loads(meta.read_text()).get('provenance')==proof:
  with np.load(cache) as z:return dict(raw=r,values=z['values'],ids=z['ids'],groups=z['groups'],prediction_sha256=r['prediction_sha256'])
 assert digest(pred)==r['prediction_sha256'];payload=torch.load(pred,map_location='cpu',weights_only=False)
 assert payload['records']==records and len(payload['predictions'])==3000
 values=[]
 for row,p,saved in zip(records,payload['predictions'],payload['per_image_precision']):
  value=image_precision(row['boxes'],p['boxes'].numpy(),p['scores'].numpy())
  assert (value is None and saved is None) or (value is not None and saved is not None and abs(value-saved)<1e-12)
  values.append(np.nan if value is None else value)
 assert len(values)==3000 and abs(np.nanmean(values)-r['metrics']['rsna_mean_precision'])<1e-12
 cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,values=values,ids=ids,groups=groups);tmp.replace(cache)
 common.dump(meta,dict(provenance=proof,model=r['model'],seed=r['seed'],n_test=3000,n_groups=len(set(groups)),metric_reproduced=r['metrics']['rsna_mean_precision'],prediction_path=str(pred)))
 return dict(raw=r,values=np.array(values),ids=np.array(ids),groups=np.array(groups),prediction_sha256=r['prediction_sha256'])

def assemble(seeds):
 assert [s['raw']['seed']for s in seeds]==list(range(5))
 for s in seeds[1:]:
  assert np.array_equal(s['ids'],seeds[0]['ids']) and np.array_equal(s['groups'],seeds[0]['groups'])
 values=np.stack([s['values']for s in seeds],axis=1);names,code=np.unique(seeds[0]['groups'],return_inverse=True)
 c=np.zeros((len(names),5,2));np.add.at(c[:,:,0],code,np.nan_to_num(values,nan=0.));np.add.at(c[:,:,1],code,np.isfinite(values))
 point=float((c[:,:,0].sum(0)/c[:,:,1].sum(0)).mean());assert abs(point-np.mean([s['raw']['metrics']['rsna_mean_precision']for s in seeds]))<1e-12
 return dict(data=c,ids=seeds[0]['ids'],groups=seeds[0]['groups'],point=point,seeds=seeds)

def metric(total):return np.mean(total[...,0]/total[...,1],axis=-1)

def inference(a,b=None):
 g=len(a['data']);rng=np.random.default_rng(SETTINGS['random_seed']);draws=[]
 for start in range(0,SETTINGS['bootstrap_replicates'],128):
  w=rng.multinomial(g,np.full(g,1/g),size=min(128,SETTINGS['bootstrap_replicates']-start));v=metric(np.einsum('bg,gsc->bsc',w,a['data']))
  if b is not None:v-=metric(np.einsum('bg,gsc->bsc',w,b['data']))
  draws.extend(v.tolist())
 draws=np.array(draws);valid=np.isfinite(draws)
 if valid.sum()<.95*len(draws):raise ValueError('Too many zero-scored bootstrap draws')
 lo,hi=np.quantile(draws[valid],[.025,.975])
 if b is None:return dict(value=a['point'],ci_low=float(lo),ci_high=float(hi)),int(valid.sum())
 observed=a['point']-b['point'];rng=np.random.default_rng(SETTINGS['random_seed']);count=0;ta=a['data'].sum(0);tb=b['data'].sum(0)
 for start in range(0,SETTINGS['permutation_replicates'],128):
  swap=rng.integers(0,2,(min(128,SETTINGS['permutation_replicates']-start),g));change=np.einsum('bg,gsc->bsc',swap,b['data']-a['data']);delta=metric(ta+change)-metric(tb-change);count+=int((np.abs(delta)>=abs(observed)-1e-12).sum())
 return dict(difference=observed,ci_low=float(lo),ci_high=float(hi),p_two_sided=(count+1)/(SETTINGS['permutation_replicates']+1)),int(valid.sum())

def provenance(a,b=None):
 items=[a] if b is None else [a,b]
 return dict(implementation=VERSION,prediction_sha256=[[s['prediction_sha256']for s in x['seeds']]for x in items],manifest_sha256=a['seeds'][0]['raw']['manifest_sha256'],bootstrap_replicates=SETTINGS['bootstrap_replicates'],permutation_replicates=SETTINGS['permutation_replicates'],random_seed=SETTINGS['random_seed'])

def job(task):
 m,fm=task;a=DATA[m];b=DATA[fm] if fm else None;sig=provenance(a,b);target=OUT/'statistics'/(f'{m}__vs__{fm}__xray_chest.json' if fm else f'{m}__xray_chest__ci.json')
 try:
  if common.valid_existing(target,sig):return dict(task=task,status='cached')
  if b:
   assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['groups'],b['groups'])
  metrics,valid=inference(a,b);result=dict(task='box_detection',dataset='xray_chest',fold='fixed',training_seeds=list(range(5)),n_test=len(a['ids']),n_groups=len(a['data']),primary_metric='rsna_mean_precision',provenance=sig,bootstrap_replicates=SETTINGS['bootstrap_replicates'],permutation_replicates=SETTINGS['permutation_replicates'],valid_bootstrap_replicates=valid,uncertainty='NIH patient-group bootstrap; all exams and all five separately fitted heads retained per patient; mean of five per-fit scored-image precisions; ignored negatives handled separately within each fit',test_definition='Two-sided paired patient swaps synchronizing all five fitted prediction sets, including model-dependent scored/ignored counts; no prediction ensemble',metrics={'rsna_mean_precision':metrics})
  if fm:result.update(candidate=m,reference=fm)
  else:result.update(model=m,prediction_sha256=sig['prediction_sha256'][0],seed_values=[s['raw']['metrics']['rsna_mean_precision']for s in a['seeds']])
  common.save_stat(target,result);return dict(task=task,status='complete')
 except Exception as e:return dict(task=task,status='error',error=repr(e),traceback=traceback.format_exc())

def run(workers):
 workers=min(workers,4)
 global DATA,SETTINGS
 cfg=json.loads((OUT/'config.json').read_text());SETTINGS=cfg['statistics'];models={m['model']:m for m in cfg['models']};DATA={};errors=[];available={}
 mp=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json';manifest=json.loads(mp.read_text());mh=digest(mp)
 for m in models:
  seeds=[]
  for seed in range(5):
   p=OUT/'results'/m/'rsna_pneumonia'/f'seed{seed}/component_result.json'
   if not p.exists():continue
   try:
    s=index_seed(p,manifest,mh);assert s['raw']['checkpoint']==models[m]['checkpoint'];seeds.append(s)
   except Exception as e:errors.append(dict(path=str(p),error=repr(e)))
  available[m]=len(seeds)
  if len(seeds)==5:DATA[m]=assemble(seeds)
 jobs=[(m,None)for m in DATA]+[(m,fm)for m in DATA if models[m]['kind']=='biodino' for fm in DATA if models[fm]['kind']=='fm']
 state=dict(pid=os.getpid(),updated=time.time(),indexed_seed_counts=available,complete_models=len(DATA),scheduled=len(jobs),completed_this_pass=0,errors=errors,status='running');statepath=OUT/'states/rsna_statistics.json';common.dump(statepath,state)
 if jobs:
  from unified_paper_statistics_resources import pool as bounded_pool
  with bounded_pool(workers) as pool:
   for r in pool.imap_unordered(job,jobs):
    print('RSNA STATISTICS',r,flush=True)
    if r['status']=='error':state['errors'].append(r)
    else:state['completed_this_pass']+=1
    state['updated']=time.time();common.dump(statepath,state)
 state.update(status='watching',updated=time.time());common.dump(statepath,state)

if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 with (OUT/'locks/rsna_statistics.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(a.workers)
   if not a.watch:break
   time.sleep(60)
