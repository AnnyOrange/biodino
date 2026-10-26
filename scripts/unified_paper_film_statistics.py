"""FILM paired biological-sample inference, preserving all five fits together."""
import argparse,json,hashlib,time,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';STORE=OUT/'statistics/film';STORE.mkdir(exist_ok=True)
def dump(p,x):
 if p.exists() and 'task' not in json.loads(p.read_text()):
  old=p.read_bytes();history=OUT/'statistics/history/film';history.mkdir(parents=True,exist_ok=True);dest=history/(p.stem+'__'+hashlib.sha256(old).hexdigest()+'.json')
  if not dest.exists():dest.write_bytes(old)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run():
 cfg=json.loads((OUT/'config.json').read_text());frames={};rng=np.random.default_rng(cfg['statistics']['random_seed']);draws=None
 for p in sorted((OUT/'results').glob('*/film_age/component_result.json')):
  d=json.loads(p.read_text())
  if len(d['seeds'])!=5:continue
  seed_paths=[p.parent/f'seed{seed}'/'component_result.json' for seed in range(5)]
  if not all(path.is_file() for path in seed_paths):continue
  seeds=[json.loads(path.read_text()) for path in seed_paths]
  assert all(s['seed']==i and s['model']==d['model'] and s['checkpoint']==d['checkpoint'] and s['manifest_sha256']==d['manifest_sha256'] for i,s in enumerate(seeds))
  ps=[json.loads(Path(s['prediction_path']).read_text()) for s in seeds];ids=ps[0]['group_id'];y=np.array(ps[0]['y_true'])
  assert all(x['group_id']==ids and x['y_true']==y.tolist() for x in ps)
  acc=np.mean([np.array(x['y_pred'])==y for x in ps],axis=0);weights=np.array([1/(4*(y==v).sum()) for v in y]);value=float(acc@weights);assert abs(value-d['metrics']['balanced_accuracy'])<1e-12
  if draws is None:
   counts=[]
   for _ in range(2000):
    sampled=np.zeros(len(y))
    for c in np.unique(y):
     ix=np.flatnonzero(y==c);sampled+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(y))*weights
    counts.append(sampled)
   draws=np.array(counts);baseids=ids;basey=y
  assert ids==baseids and np.array_equal(y,basey)
  hashes=[hashlib.sha256(Path(s['prediction_path']).read_bytes()).hexdigest() for s in seeds];frames[d['model']]=(acc,hashes)
  ci=np.quantile(draws@acc,[.025,.975]);dump(STORE/f"{d['model']}__film_age__ci.json",dict(model=d['model'],task='classification',dataset='film_age',n_test=len(y),n_groups=len(y),prediction_sha256=hashes,uncertainty='Class-stratified biological-group bootstrap; all5training fits retained together;2000draws',metrics={'balanced_accuracy':dict(value=value,ci_low=float(ci[0]),ci_high=float(ci[1]))}))
 for m in [x['model'] for x in cfg['models'] if x['kind']=='biodino']:
  if m not in frames:continue
  for fm in [x['model'] for x in cfg['models'] if x['kind']=='fm']:
   if fm not in frames:continue
   path=STORE/f'{m}__vs__{fm}__film_age.json';a,ah=frames[m];b,bh=frames[fm]
   if path.exists() and json.loads(path.read_text()).get('prediction_sha256')==[ah,bh] and json.loads(path.read_text()).get('task')=='classification':continue
   diff=a-b;observed=float(diff@weights);ci=np.quantile(draws@diff,[.025,.975]);perms=np.random.default_rng(cfg['statistics']['random_seed']).integers(0,2,(10000,len(y)))*2-1;null=(perms*diff)@weights;p=(int((np.abs(null)>=abs(observed)-1e-12).sum())+1)/10001
   dump(path,dict(candidate=m,reference=fm,task='classification',dataset='film_age',n_test=len(y),n_groups=len(y),prediction_sha256=[ah,bh],bootstrap_replicates=2000,permutation_replicates=10000,uncertainty='Class-stratified group bootstrap and paired group swaps of all5fits; multiplicity handled by common report Holm family',metrics={'balanced_accuracy':dict(difference=observed,ci_low=float(ci[0]),ci_high=float(ci[1]),p_two_sided=float(p))}))
 print('FILM STATISTICS',len(frames),'models',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with (OUT/'locks/film_statistics.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   run()
   if not a.watch:break
   time.sleep(60)
