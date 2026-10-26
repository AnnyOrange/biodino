"""Admit all approved NuCLS fits only after the complete real pilot result."""
import fcntl,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
 with (OUT/'locks/nucls_admission.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  pilot=OUT/'results/biodino_1tb_b/nucls_qc/fold2/seed0/component_result.json'
  while True:
   valid=False
   if pilot.exists():
    p=json.loads(pilot.read_text());valid=p['metrics']['evaluated_images']==505 and p['epochs']==10 and p['frozen_encoder'] and Path(p['prediction_path']).exists()
   state={'pid':os.getpid(),'status':'waiting_for_real_pilot','pilot':str(pilot),'planned_jobs':480,'updated':time.time()}
   (OUT/'states/nucls_admission.json').write_text(json.dumps(state,indent=2)+'\n')
   if valid:break
   time.sleep(30)
  jobs=[]
  for spec in json.loads((OUT/'config.json').read_text())['models']:
   model=spec['model'];base=OUT/'results'/model/'nucls_qc'
   for fold in [2,3,4,5]:
    for seed in range(5):
     key=f'{model}__nucls_qc__fold{fold}_seed{seed}'
     jobs.append({'key':key,'model':model,'task':'nuclei_detection_segmentation','dataset':'nucls_qc','fold':fold,'seed':seed,
      'entrypoint':'run_unified_paper_nucls.py','args':['--model',model,'--fold',str(fold),'--seed',str(seed)],'claim_key':'nucls_'+model,
      'requires':[str(base/'selected_recipe.json')] if (fold,seed)!=(2,0) else [],
      'min_free_mib':22000 if model in ['genbio_pathfm','hoptimus0','gigapath','virchow2','biodino_1tb_hplus'] else 12000,
      'result':str(base/f'fold{fold}'/f'seed{seed}'/'component_result.json'),'log':str(OUT/'logs'/f'{key}.log')})
  p=OUT/'nucls_queue.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'tasks':jobs,'stage':'Full real pilot passed; all24 models x4 official evaluation folds x5 downstream seeds'},indent=2)+'\n');tmp.replace(p)
  state.update(status='admitted',jobs=len(jobs),updated=time.time());(OUT/'states/nucls_admission.json').write_text(json.dumps(state,indent=2)+'\n');print('ADMITTED NUCLS',len(jobs),flush=True)
if __name__=='__main__':main()
