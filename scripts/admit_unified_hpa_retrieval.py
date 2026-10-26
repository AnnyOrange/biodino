"""Real-pilot-gated execution of the existing HPA v4 cohorts."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from unified_paper_retrieval_scheduling import requirements
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(p,v):
 t=p.with_suffix('.hpa.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def run():
 cfg=json.loads((OUT/'config.json').read_text());pilot=next(m for m in cfg['models']if m['model']=='biodino_1tb_splus')
 manifest=ROOT/'outputs/02_eval_inputs/formal_v4/hpa-subcellular_retrieval_manifest.json'
 result=OUT/'results'/pilot['model']/'retrieval_clustering/hpa-subcellular/component_result.json';passed=False
 if result.exists():
  r=json.loads(result.read_text());assert r['checkpoint']==pilot['checkpoint']and r['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
  assert r['completed_clustering_seeds']==5 and r['n_query']==1786 and r['n_gallery']==1786 and r['n_samples']==1458
  for p in [r['retrieval_result'],*r['clustering_results']]:
   a=json.loads(Path(p).read_text());assert hashlib.sha256(Path(a['prediction_path']).read_bytes()).hexdigest()==a['prediction_sha256']
  passed=True
 jobs=[]
 for spec in cfg['models']if passed else[pilot]:
  m=spec['model'];key=f'{m}__retrieval_clustering__hpa-subcellular'
  j=dict(key=key,model=m,task='retrieval_clustering',dataset='hpa-subcellular',fold='fixed',seed='bundled0to4',entrypoint='run_unified_paper_hpa_retrieval.py',args=['--model',m],claim_key='hpa_'+m,min_free_mib=16000 if spec['min_free_mib']>=16000 else 6000,result=str(OUT/'results'/m/'retrieval_clustering/hpa-subcellular/component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
  j.update(requirements(spec,'hpa-subcellular'))
  if spec.get('execution_host'):j.update(blocked_reason='Complete HPA cohort requires resident-host preparation; no cross-building checkpoint transfer',allowed_hosts=[spec['execution_host']])
  jobs.append(j)
 path=OUT/'hpa_retrieval_queue.json';old=json.loads(path.read_text())if path.exists()else{}
 if old.get('tasks')!=jobs:dump(path,dict(tasks=jobs))
 dump(OUT/'states/hpa_retrieval_admission.json',dict(pid=os.getpid(),updated=time.time(),status='all_models_admitted_after_real_pilot'if passed else'one_real_full_cohort_pilot',jobs=len(jobs)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/hpa_retrieval_admission.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run()
   if not a.watch:break
   time.sleep(45)
