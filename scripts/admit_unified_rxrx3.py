"""Lock the existing complete RxRx3 v4 inputs and gate bulk execution on a real pilot."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from unified_paper_retrieval_scheduling import requirements
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';DATA=ROOT/'outputs/02_eval_inputs/formal_v3/rxrx3-core';MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/rxrx3-core_retrieval_manifest.json'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,r):
 t=p.with_suffix('.rxrx3.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def run():
 if not MANIFEST.exists():
  meta=json.loads((DATA/'metadata.json').read_text());rules=json.loads((ROOT/'Evaluation Rules/protocol_v4.json').read_text())['retrieval_splits']['rxrx3-core'];rows=[json.loads(line)for line in(DATA/'split_manifest.jsonl').read_text().splitlines()]
  assert meta['rows']==rows and sha(DATA/'split_manifest.jsonl')==meta['manifest_sha256']==rules['manifest_sha256']and meta['input_mapping']==rules['input_mapping']
  g={r['gene']:r for r in rows if r['split']=='gallery'};q={r['gene']:r for r in rows if r['split']=='query'};assert len(rows)==1468 and len(g)==734 and g.keys()==q.keys()
  assert len({r['key']for r in rows})==1468 and len({r['well_id']for r in rows})==1468
  for gene in g:assert g[gene]['well_id']!=q[gene]['well_id']and g[gene]['plate']!=q[gene]['plate']
  dump(MANIFEST,dict(dataset='rxrx3-core',protocol=meta['protocol_id'],source_hashes={n:sha(DATA/n)for n in ['images.npy','labels.npy','metadata.json','split_manifest.jsonl']},records=rows,grouping='Released well_id; one recorded field per biological well; gallery and query wells and plates disjoint within each gene',group_limitation='Conditional on the locked fixed gallery, selected eligible genes and sampled acquisition batches; well-level uncertainty does not estimate generalization to unseen experiments.',primary_grouping_verified=True))
 cfg=json.loads((OUT/'config.json').read_text());pilot=next(m for m in cfg['models']if m['model']=='biodino_1tb_splus');p=OUT/'results'/pilot['model']/'retrieval_clustering/rxrx3-core/component_result.json';passed=False
 if p.exists():
  r=json.loads(p.read_text());assert r['checkpoint']==pilot['checkpoint']and r['completed_clustering_seeds']==5 and r['n_query']==734 and r['n_gallery']==734 and r['manifest_sha256']==sha(MANIFEST)
  for path in[r['retrieval_result'],*r['clustering_results']]:
   x=json.loads(Path(path).read_text());assert sha(Path(x['prediction_path']))==x['prediction_sha256']
  passed=True
 jobs=[]
 for spec in cfg['models']if passed else[pilot]:
  m=spec['model'];key=m+'__retrieval_clustering__rxrx3-core';j=dict(key=key,model=m,task='retrieval_clustering',dataset='rxrx3-core',fold='fixed',seed='bundled0to4',entrypoint='run_unified_paper_rxrx3.py',args=['--model',m],claim_key=key,min_free_mib=16000 if spec['min_free_mib']>=16000 else 6000,result=str(OUT/'results'/m/'retrieval_clustering/rxrx3-core/component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
  j.update(requirements(spec,'rxrx3-core'))
  if spec.get('execution_host'):j['blocked_reason']='Complete locked RxRx3 cache requires resident-host preparation; no cross-building checkpoint transfer'
  jobs.append(j)
 p=OUT/'rxrx3_retrieval_queue.json';old=json.loads(p.read_text())if p.exists()else{}
 if old.get('tasks')!=jobs:dump(p,dict(tasks=jobs))
 dump(OUT/'states/rxrx3_retrieval_admission.json',dict(pid=os.getpid(),updated=time.time(),status='all_models_admitted_after_real_pilot'if passed else'one_real_full_cohort_pilot',jobs=len(jobs)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');a=parser.parse_args()
 with(OUT/'locks/rxrx3_retrieval_admission.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run()
   if not a.watch:break
   time.sleep(45)
