"""Grouped query inference and conditional fitted-partition NMI bootstrap."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def nmi(counts):
 total=counts.sum(axis=(-2,-1),keepdims=True);p=counts/np.maximum(total,1);py=p.sum(-1,keepdims=True);pc=p.sum(-2,keepdims=True)
 mi=np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)/np.maximum(py*pc,1e-300)),0),axis=(-2,-1))
 hy=-np.sum(np.where(py>0,py*np.log(np.maximum(py,1e-300)),0),axis=(-2,-1));hc=-np.sum(np.where(pc>0,pc*np.log(np.maximum(pc,1e-300)),0),axis=(-2,-1))
 return np.divide(2*mi,hy+hc,out=np.ones_like(mi),where=(hy+hc)>1e-15)
def grouped_ci(counts,metric,repeats,seed):
 rng=np.random.default_rng(seed);n=len(counts);draw=[]
 for start in range(0,repeats,64):
  weights=rng.multinomial(n,np.full(n,1/n),size=min(64,repeats-start));sample=np.tensordot(weights,counts,axes=(1,0));draw.extend(metric(sample).tolist())
 return np.quantile(draw,[.025,.975],axis=0)
def query_metric(counts):return counts[...,0]/counts[...,1]

def compute_draws(counts,settings):
 """Shared whole-group draws; columns are models, never independent samples."""
 successes=np.stack([a[:,0]for a in counts],axis=1);sizes=counts[0][:,1];n=len(sizes)
 assert all(np.array_equal(a[:,1],sizes)for a in counts)
 rng=np.random.default_rng(settings['random_seed']);bootstrap=[]
 for start in range(0,settings['bootstrap_replicates'],64):
  weights=rng.multinomial(n,np.full(n,1/n),size=min(64,settings['bootstrap_replicates']-start));bootstrap.append((weights@successes)/(weights@sizes)[:,None])
 rng=np.random.default_rng(settings['random_seed']);swaps=[]
 for start in range(0,settings['permutation_replicates'],128):
  signs=2*rng.integers(2,size=(min(128,settings['permutation_replicates']-start),n))-1;swaps.append((signs@successes)/sizes.sum())
 return np.concatenate(bootstrap),np.concatenate(swaps)

def run():
 import os,traceback
 import unified_paper_global_statistics as common
 cfg=json.loads((OUT/'config.json').read_text());settings=cfg['statistics'];store=OUT/'statistics/retrieval';store.mkdir(exist_ok=True);cache=store/'draws';cache.mkdir(exist_ok=True)
 state=dict(pid=os.getpid(),updated=time.time(),models_datasets=0,ci_completed=0,pairs_completed=0,errors=[],status='running');statepath=OUT/'states/retrieval_statistics.json';dump(statepath,state)
 for dataset in ['nct-crc-he-1k','crc-val-he-7k','nct-crc-he-100','lc25000','hpa-subcellular','rxrx3-core','rxrx1-cross']:
  available={};pending=[]
  for spec in cfg['models']:
   model=spec['model'];path=OUT/'results'/model/'retrieval_clustering'/dataset/'retrieval_result.json'
   if not path.exists():continue
   try:
    raw=json.loads(path.read_text());pred=Path(raw['prediction_path']);digest=hashlib.sha256(pred.read_bytes()).hexdigest();assert digest==raw['prediction_sha256']and raw['checkpoint']==spec['checkpoint']
    with np.load(pred)as z:ids=z['sample_id'].copy();groups=z['group_id'].copy();labels=z['labels'].copy();scores=z['recall_at_1'].copy()
    names,code=np.unique(groups,return_inverse=True);counts=np.zeros((len(names),2));np.add.at(counts[:,0],code,scores);np.add.at(counts[:,1],code,1)
    point=float(query_metric(counts.sum(0)));assert abs(point-raw['metrics']['recall_at_1'])<1e-12
    if available:
     reference=next(iter(available.values()))
     assert np.array_equal(ids,reference['ids'])and np.array_equal(groups,reference['groups'])and np.array_equal(labels,reference['labels'])and raw['manifest_sha256']==reference['raw']['manifest_sha256']
    sig=dict(implementation='shared-query-group-draws-v1',prediction_sha256=digest,manifest_sha256=raw['manifest_sha256'],bootstrap_replicates=settings['bootstrap_replicates'],permutation_replicates=settings['permutation_replicates'],random_seed=settings['random_seed'])
    item=dict(raw=raw,ids=ids,groups=groups,labels=labels,counts=counts,point=point,digest=digest,signature=sig,cache=cache/f'{model}__{dataset}.npz')
    if item['cache'].exists():
     with np.load(item['cache'])as z:
      if str(z['signature'])==json.dumps(sig,sort_keys=True):item.update(bootstrap=z['bootstrap'].copy(),swaps=z['swaps'].copy())
    available[model]=item
    if 'bootstrap'not in item:pending.append(model)
   except Exception as e:state['errors'].append(dict(model=model,dataset=dataset,error=repr(e)))
  if pending:
   bootstrap,swaps=compute_draws([available[m]['counts']for m in pending],settings)
   for col,model in enumerate(pending):
    item=available[model];item.update(bootstrap=bootstrap[:,col],swaps=swaps[:,col]);tmp=item['cache'].with_suffix('.tmp.npz');np.savez_compressed(tmp,bootstrap=item['bootstrap'],swaps=item['swaps'],signature=json.dumps(item['signature'],sort_keys=True));tmp.replace(item['cache'])
  state['models_datasets']+=len(available)
  for model,a in available.items():
   try:
    ci=np.quantile(a['bootstrap'],[.025,.975]);target=store/f'{model}__{dataset}__ci.json';old=json.loads(target.read_text())if target.exists()else{}
    if old.get('prediction_sha256')==a['digest']:
     assert np.allclose(ci,[old['metrics']['recall_at_1']['ci_low'],old['metrics']['recall_at_1']['ci_high']],atol=1e-12,rtol=0)
    else:
     common.save_stat(target,dict(model=model,dataset=dataset,task='retrieval',n_test=len(a['ids']),n_groups=len(a['counts']),training_seed='not_applicable',prediction_sha256=a['digest'],bootstrap_replicates=settings['bootstrap_replicates'],provenance=a['signature'],primary_grouping_verified=a['raw'].get('primary_grouping_verified',False),uncertainty='Whole recorded query-group bootstrap, conditional on identical fixed gallery and encoder. '+a['raw'].get('grouping','Source image groups')+'. '+a['raw']['group_limitation'],metrics={'recall_at_1':dict(value=a['point'],ci_low=float(ci[0]),ci_high=float(ci[1]))}))
    state['ci_completed']+=1
   except Exception as e:state['errors'].append(dict(model=model,dataset=dataset,error=repr(e)))
  for candidate in [m['model']for m in cfg['models']if m['kind']=='biodino'and m['model']in available]:
   for reference in [m['model']for m in cfg['models']if m['kind']=='fm'and m['model']in available]:
    a=available[candidate];b=available[reference];target=store/f'{candidate}__vs__{reference}__{dataset}.json'
    try:
     delta=a['point']-b['point'];ci=np.quantile(a['bootstrap']-b['bootstrap'],[.025,.975]);null=a['swaps']-b['swaps'];p=float((1+int((np.abs(null)>=abs(delta)-1e-12).sum()))/(len(null)+1));old=json.loads(target.read_text())if target.exists()else{}
     if old.get('prediction_sha256')==[a['digest'],b['digest']]:
      v=old['metrics']['recall_at_1'];assert np.allclose([delta,*ci,p],[v['difference'],v['ci_low'],v['ci_high'],v['p_two_sided']],atol=1e-12,rtol=0)
     else:
      common.save_stat(target,dict(candidate=candidate,reference=reference,dataset=dataset,task='retrieval',n_test=len(a['ids']),n_groups=len(a['counts']),prediction_sha256=[a['digest'],b['digest']],bootstrap_replicates=settings['bootstrap_replicates'],permutation_replicates=settings['permutation_replicates'],provenance={'candidate':a['signature'],'reference':b['signature']},primary_grouping_verified=a['raw'].get('primary_grouping_verified',False)and b['raw'].get('primary_grouping_verified',False),test_definition='Two-sided paired whole recorded query-group swaps of Recall@1 outcomes, conditional on identical fixed gallery. Shared cached draw projections are algebraically identical to direct paired differences. Configured BioDINO x18 FM family; no training seed and no ensemble.',statistical_scope=a['raw']['group_limitation'],metrics={'recall_at_1':dict(difference=delta,ci_low=float(ci[0]),ci_high=float(ci[1]),p_two_sided=p)}))
     state['pairs_completed']+=1
    except Exception as e:state['errors'].append(dict(candidate=candidate,reference=reference,dataset=dataset,error=repr(e)))
  state['updated']=time.time();dump(statepath,state)
 state.update(status='watching',updated=time.time());dump(statepath,state)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
 with (OUT/'locks/retrieval_statistics.local.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run()
   if not args.watch:break
   time.sleep(60)
