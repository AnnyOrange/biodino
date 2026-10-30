"""Released-case grouped MAE inference for the fixed official bone-age validation."""
import argparse,fcntl,hashlib,json,multiprocessing,os,time
from pathlib import Path
import numpy as np
import unified_paper_global_statistics as common
OUT=common.OUT;ROOT=common.ROOT
common.VERSION='bone-age-official-validation-grouped-mae-v1'

def load(path,manifest,manifest_hash):
 raw=json.loads(path.read_text())
 if raw.get('solver_precision')!='float64':raise ValueError('Awaiting prescribed float64 solve')
 if raw['manifest_sha256']!=manifest_hash:raise ValueError('Different official validation membership')
 pp=Path(raw['prediction_path']);p=json.loads(pp.read_text());rows=[r for r in manifest['records'] if r['split']=='official_validation']
 ids=np.array(p['sample_id'],dtype=str);groups=np.array(p['group_id'],dtype=str);y=np.array(p['y_true'],dtype=np.float64);pred=np.array(p['y_pred'],dtype=np.float64)
 assert ids.tolist()==[r['sample_id'] for r in rows] and groups.tolist()==[r['group_id'] for r in rows]
 assert np.array_equal(y,np.array([r['age_months'] for r in rows])) and len(ids)==1425 and np.isfinite(pred).all()
 names,code=np.unique(groups,return_inverse=True);c=np.zeros((len(names),5));values=np.stack([np.ones(len(y)),y,y*y,(y-pred)**2,np.abs(y-pred)],axis=-1);np.add.at(c,code,values)
 meta={**raw,'group_note':manifest['grouping'],'split':'official_validation','split_sha256':manifest_hash}
 a=dict(raw=meta,data=c,ids=ids,groups=groups,y=y,metric='mae',strata=[np.arange(len(names))],classes=None,
        prediction_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),component_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 assert abs(common.point(a)-raw['metrics']['mae'])<1e-10
 return a

def run(workers):
 workers=min(workers,4)
 cfg=json.loads((OUT/'config.json').read_text());common.SETTINGS=cfg['statistics'];models={m['model']:m for m in cfg['models']}
 mp=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json';raw=mp.read_bytes();manifest=json.loads(raw);h=hashlib.sha256(raw).hexdigest();common.DATA={};errors=[]
 for path in (OUT/'results').glob('*/bone_age/component_result.json'):
  try:
   a=load(path,manifest,h);m=a['raw']['model']
   if m in models:
    assert a['raw']['checkpoint']==models[m]['checkpoint'];common.DATA[m,'xray_bone']=a
  except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 jobs=[]
 for (m,d),a in common.DATA.items():
  if not common.valid_existing(OUT/'statistics'/f'{m}__{d}__ci.json',common.provenance(a)):jobs.append(('ci',m,d,None))
  if models[m]['kind']=='biodino':
   for fm,spec in models.items():
    if spec['kind']=='fm' and (fm,d) in common.DATA:
     b=common.DATA[fm,d]
     if not common.valid_existing(OUT/'statistics'/f'{m}__vs__{fm}__{d}.json',common.provenance(a,b)):jobs.append(('pair',m,d,fm))
 state=dict(pid=os.getpid(),updated=time.time(),available_results=len(common.DATA),scheduled=len(jobs),completed_this_pass=0,errors=errors,status='running',workers=workers)
 statepath=OUT/'states/bone_age_statistics.json';common.dump(statepath,state)
 if jobs:
  from unified_paper_statistics_resources import pool as bounded_pool
  with bounded_pool(workers) as pool:
   for r in pool.imap_unordered(common.compute_job,jobs,chunksize=1):
    print('BONE AGE',r,flush=True)
    if r['status']=='error':state['errors'].append(r)
    else:state['completed_this_pass']+=1
    state['updated']=time.time();common.dump(statepath,state)
 state.update(status='watching',updated=time.time());common.dump(statepath,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 with (OUT/'locks/bone_age_statistics.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run(a.workers)
   if not a.watch:break
   time.sleep(60)
