"""Native box AP: exact per-image COCO matches, grouped resampling of five fits."""
import argparse,contextlib,fcntl,hashlib,io,json,sys,time,os,multiprocessing,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
sys.path[:0]=[str(ROOT),str(ROOT/'outputs/02_eval_runtime/pycocotools_2_0_10')]
import numpy as np,torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRICS=['bbox_ap_50_95','bbox_ap_50','bbox_ap_75']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
 tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(payload):
 annotations=[];detections=[];images=[]
 for i,(row,pred) in enumerate(zip(payload['records'],payload['predictions'])):
  images.append({'id':i,'width':row['width'],'height':row['height']})
  for box,label in zip(row['boxes'],row['labels']):
   x,y,xx,yy=box;annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':label,'bbox':[x,y,xx-x,yy-y],'area':(xx-x)*(yy-y),'iscrowd':0})
  for box,label,score in zip(pred['boxes'].tolist(),pred['labels'].tolist(),pred['scores'].tolist()):
   x,y,xx,yy=box;detections.append({'image_id':i,'category_id':label,'bbox':[x,y,xx-x,yy-y],'score':score})
 with contextlib.redirect_stdout(io.StringIO()):
  truth=COCO();truth.dataset={'info':{},'licenses':[],'images':images,'annotations':annotations,'categories':[{'id':i+1,'name':x} for i,x in enumerate(payload['classes'])]};truth.createIndex()
  if detections:result=truth.loadRes(detections)
  else:
   result=COCO();result.dataset={**truth.dataset,'annotations':[]};result.createIndex()
  ev=COCOeval(truth,result,'bbox');ev.params.maxDets=[1,10,300];ev.params.areaRng=[[0,1e10]];ev.params.areaRngLbl=['all'];ev.evaluate()
 categories=[];n=len(images)
 for cat in range(len(payload['classes'])):
  rows=[]
  for image in range(n):
   item=ev.evalImgs[cat*n+image]
   rows.append({'scores':np.asarray(item['dtScores']),'tp':np.logical_and(item['dtMatches']>0,~item['dtIgnore']),
    'fp':np.logical_and(item['dtMatches']==0,~item['dtIgnore']),'n_gt':int(np.count_nonzero(item['gtIgnore']==0))} if item is not None else {'scores':np.zeros(0),'tp':np.zeros((10,0),bool),'fp':np.zeros((10,0),bool),'n_gt':0})
  categories.append(rows)
 return categories

def ap(cache,indices,other=None,swaps=None):
 values=[]
 for cat,images in enumerate(cache):
  rows=[other[cat][i] if other is not None and swaps[i] else images[i] for i in indices]
  n_gt=sum(r['n_gt'] for r in rows)
  if not n_gt:continue
  scores=np.concatenate([r['scores'] for r in rows]);order=np.argsort(-scores,kind='mergesort')
  tp=np.concatenate([r['tp'] for r in rows],axis=1)[:,order].cumsum(1,dtype=np.float64)
  fp=np.concatenate([r['fp'] for r in rows],axis=1)[:,order].cumsum(1,dtype=np.float64)
  precision=tp/(tp+fp+np.spacing(1));recall=tp/n_gt
  if len(scores):precision=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
  sampled=np.zeros((10,101),float)
  for threshold in range(10):
   inds=np.searchsorted(recall[threshold],np.linspace(0,1,101),side='left');valid=inds<len(scores)
   sampled[threshold,valid]=precision[threshold,inds[valid]]
  values.append(sampled.mean(1))
 if not values:raise ValueError('No true annotated category in resample')
 v=np.mean(values,axis=0);return np.array([v.mean(),v[0],v[5]])

def metric(caches,indices,other=None,swaps=None):
 return np.mean([ap(c,indices,None if other is None else other[s],swaps) for s,c in enumerate(caches)],axis=0)

def bootstrap(a,b,groups,repeats,seed):
 rng=np.random.default_rng(seed);members=[np.flatnonzero(np.asarray(groups)==g) for g in sorted(set(groups))];samples=[]
 for _ in range(repeats):
  indices=np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))]);v=metric(a,indices)
  if b is not None:v-=metric(b,indices)
  samples.append(v)
 return np.quantile(samples,[.025,.975],axis=0)

def permutation(a,b,groups,repeats,seed):
 rng=np.random.default_rng(seed);unique=sorted(set(groups));mapping={g:i for i,g in enumerate(unique)};group_index=np.array([mapping[g] for g in groups]);indices=np.arange(len(groups))
 delta=metric(a,indices)-metric(b,indices);extreme=np.zeros(len(METRICS),int)
 for _ in range(repeats):
  swaps=rng.integers(2,size=len(unique))[group_index].astype(bool)
  difference=metric(a,indices,b,swaps)-metric(b,indices,a,swaps)
  extreme+=(np.abs(difference)>=np.abs(delta)-1e-12)
 return delta,(extreme+1)/(repeats+1)

def load(paths):
 caches=[];hashes=[];ids=groups=records=None;values=[]
 for path in paths:
  raw=json.loads(path.read_text());pred=Path(raw['prediction_path']);digest=sha(pred);assert digest==raw['prediction_sha256']
  payload=torch.load(pred,map_location='cpu',weights_only=False);this_ids=[r['sample_id'] for r in payload['records']];this_groups=[r.get('group_id',r.get('group',r['sample_id'])) for r in payload['records']]
  signature=hashlib.sha256(json.dumps(payload['records'],sort_keys=True).encode()).hexdigest()
  if ids is not None and (ids!=this_ids or groups!=this_groups or records!=signature):raise ValueError('Unpaired native seed targets or members')
  ids,groups,records=this_ids,this_groups,signature;cache_path=pred.with_name('coco_per_image_matches.pth')
  saved=torch.load(cache_path,map_location='cpu',weights_only=False) if cache_path.exists() else {}
  if saved.get('prediction_sha256')!=digest or saved.get('version')!=1:
   saved={'version':1,'prediction_sha256':digest,'matches':prepare(payload)};temp=cache_path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(cache_path)
  c=saved['matches'];score=ap(c,np.arange(len(ids)));expected=[raw['metrics'][k] for k in METRICS]
  if not np.allclose(score,expected,atol=1e-12,rtol=0):raise ValueError(f'COCO reconstruction mismatch {score} != {expected}')
  caches.append(c);hashes.append(digest);values.append(score)
 return {'caches':caches,'ids':ids,'groups':groups,'target_sha256':records,'prediction_sha256':hashes,'values':np.array(values)}

PAIR_BANK={}

def compute_job(job):
 kind,name,reference,dataset,target,s=job;a=PAIR_BANK[name,dataset]
 if kind=='ci':
  ci=bootstrap(a['caches'],None,a['groups'],s['bootstrap_replicates'],s['random_seed'])
  result={'model':name,'task':'box_detection','dataset':dataset,'n_test':len(a['ids']),'n_groups':len(set(a['groups'])),'training_seeds':list(range(5)),
   'per_seed_values':a['values'].tolist(),'metric_order':METRICS,'seed_sd':np.std(a['values'],axis=0,ddof=1).tolist(),'prediction_sha256':a['prediction_sha256'],
   'bootstrap_replicates':s['bootstrap_replicates'],'uncertainty':'Source-image cluster bootstrap synchronized across five fitted seeds; mean of individual AP values, not ensemble; conditional on these fits. Patient mapping unavailable.',
   'metrics':{k:{'value':float(a['values'][:,i].mean()),'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i])} for i,k in enumerate(METRICS)}}
 else:
  b=PAIR_BANK[reference,dataset]
  if a['ids']!=b['ids'] or a['groups']!=b['groups'] or a['target_sha256']!=b['target_sha256']:raise ValueError('Native comparison targets differ')
  ci=bootstrap(a['caches'],b['caches'],a['groups'],s['bootstrap_replicates'],s['random_seed']);delta,p=permutation(a['caches'],b['caches'],a['groups'],s['permutation_replicates'],s['random_seed'])
  result={'candidate':name,'reference':reference,'task':'box_detection','dataset':dataset,'n_test':len(a['ids']),'n_groups':len(set(a['groups'])),
   'training_seeds':list(range(5)),'prediction_sha256':a['prediction_sha256']+b['prediction_sha256'],'bootstrap_replicates':s['bootstrap_replicates'],'permutation_replicates':s['permutation_replicates'],
   'test_definition':'Two-sided paired swap of every seed prediction for each independent source image; identical group draws for difference CI; primary AP50:95 with Holm family of all configured BioDINO anchors x18 FM per dataset; AP50/AP75 secondary separate families',
   'metrics':{k:{'difference':float(delta[i]),'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i]),'p_two_sided':float(p[i])} for i,k in enumerate(METRICS)}}
 dump(Path(target),result)
 return dict(kind=kind,model=name,reference=reference,dataset=dataset,target=str(target),pid=os.getpid())

def run():
 global PAIR_BANK
 cfg=json.loads((OUT/'config.json').read_text());s=cfg['statistics'];store=OUT/'statistics/native_box';store.mkdir(exist_ok=True);available={}
 jobs=[]
 for dataset in ['bccd','bbbc041']:
  for model in cfg['models']:
   name=model['model'];paths=[OUT/'results'/name/'native_box'/dataset/f'seed{i}'/'component_result.json' for i in range(5)]
   if not all(p.exists() for p in paths):continue
   try:
    a=load(paths);available[name,dataset]=a;target=store/f'{name}__{dataset}__ci.json'
    if target.exists() and json.loads(target.read_text()).get('prediction_sha256')==a['prediction_sha256']:continue
    jobs.append(('ci',name,None,dataset,str(target),s))
   except Exception as error:
    dump(store/f'{name}__{dataset}__error.json',{'error':repr(error),'updated':time.time()});print('NATIVE STATS ERROR',name,dataset,repr(error),flush=True)
  for candidate in [m['model'] for m in cfg['models'] if m['kind']=='biodino']:
   for reference in [m['model'] for m in cfg['models'] if m['kind']=='fm']:
    if (candidate,dataset) not in available or (reference,dataset) not in available:continue
    a,b=available[candidate,dataset],available[reference,dataset];target=store/f'{candidate}__vs__{reference}__{dataset}.json'
    hashes=a['prediction_sha256']+b['prediction_sha256']
    if target.exists() and json.loads(target.read_text()).get('prediction_sha256')==hashes:continue
    if a['ids']!=b['ids'] or a['groups']!=b['groups'] or a['target_sha256']!=b['target_sha256']:raise ValueError('Native comparison targets differ')
    jobs.append(('pair',candidate,reference,dataset,str(target),s))
 PAIR_BANK=available;workers=min(4,max(1,(os.cpu_count() or 2)//2));state=OUT/'states/native_statistics_parallel.json';done=[];errors=[]
 dump(state,dict(pid=os.getpid(),updated=time.time(),workers=workers,scheduled=len(jobs),completed=0,errors=[]))
 if not jobs:return
 # Fork only CPU state. Each process reads immutable validated COCO matches;
 # every comparison retains its original RNG seed and complete resampling budget.
 from unified_paper_statistics_resources import initialize_worker
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
  futures={pool.submit(compute_job,job):job for job in jobs}
  for future in concurrent.futures.as_completed(futures):
   job=futures[future]
   try:result=future.result();done.append(result);print('NATIVE STAT COMPLETE',json.dumps(result),flush=True)
   except Exception as error:
    item=dict(job=list(job[:5]),error=repr(error));errors.append(item);print('NATIVE STAT ERROR',item,flush=True)
   dump(state,dict(pid=os.getpid(),updated=time.time(),workers=workers,scheduled=len(jobs),completed=len(done),errors=errors))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 with (OUT/'locks/native_statistics.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run()
   if not a.watch:break
   time.sleep(60)
