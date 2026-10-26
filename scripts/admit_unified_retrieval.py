"""Expand the two existing retrieval cohorts after a complete20TB-best pilot."""
import fcntl,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
 with (OUT/'locks/retrieval_admission.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  pilot=OUT/'results/biodino_20tb_l/retrieval_clustering/nct-crc-he-1k/component_result.json'
  while True:
   ok=False
   if pilot.exists():
    x=json.loads(pilot.read_text());ok=x['n_samples']==999 and x['completed_clustering_seeds']==5 and all(Path(p).exists() for p in [x['retrieval_result'],*x['clustering_results']])
   state={'pid':os.getpid(),'status':'waiting_for_real_pilot','planned_jobs':48,'planned_clustering_fits':240,'updated':time.time()}
   (OUT/'states/retrieval_admission.json').write_text(json.dumps(state,indent=2)+'\n')
   if ok:break
   time.sleep(15)
  jobs=[]
  for spec in json.loads((OUT/'config.json').read_text())['models']:
   m=spec['model']
   for d in ['nct-crc-he-1k','crc-val-he-7k']:
    key=f'{m}__retrieval_clustering__{d}'
    jobs.append({'key':key,'model':m,'task':'retrieval_clustering','dataset':d,'fold':'fixed','seed':'bundled0to4',
     'entrypoint':'run_unified_paper_retrieval.py','args':['--model',m,'--dataset',d],'claim_key':'retrieval_'+m,
     'min_free_mib':16000 if spec['min_free_mib']>=16000 else 6000,
     'result':str(OUT/'results'/m/'retrieval_clustering'/d/'component_result.json'),'log':str(OUT/'logs'/f'{key}.log')})
  p=OUT/'retrieval_queue.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({'tasks':jobs,'stage':'Full selected20TB-best pilot passed;24models x2existing cohorts, each deterministic retrieval plus5clustering fits'},indent=2)+'\n');temp.replace(p)
  state.update(status='admitted',jobs=len(jobs),updated=time.time());(OUT/'states/retrieval_admission.json').write_text(json.dumps(state,indent=2)+'\n');print('ADMITTED RETRIEVAL',len(jobs),flush=True)
if __name__=='__main__':main()
