#!/usr/bin/env python3
"""Gate the complete native CTC matrix on three actual full-fold adapter pilots."""
import fcntl,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def main():
 with (OUT/'locks/ctc_admission.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  pilots=['dinov2','biodino_1tb_splus','distillpath']
  while True:
   passed=[]
   for model in pilots:
    p=OUT/'results'/model/'ctc/fold0/seed0/component_result.json'
    if not p.exists():continue
    result=json.loads(p.read_text())
    if result['n_domains']==4 and all(int(x['ctc_metrics']['Valid'])==1 for x in result['domain_rows']):passed.append(model)
   (OUT/'states/ctc_admission.json').write_text(json.dumps({'status':'waiting_for_real_pilots','passed':passed,'required':pilots,'planned_jobs':600,'updated':time.time()},indent=2))
   if len(passed)==len(pilots):break
   time.sleep(30)
  jobs=[]
  for model in json.loads((OUT/'config.json').read_text())['models']:
   for fold in range(5):
    for seed in range(5):
     name=model['model'];key=f'{name}__ctc__fold{fold}_seed{seed}'
     jobs.append(dict(key=key,model=name,task='cell_tracking',dataset='ctc',fold=fold,seed=seed,
       entrypoint='run_unified_paper_ctc.py',args=['--model',name,'--fold',str(fold),'--seed',str(seed)],
       min_free_mib=16000 if name in {'genbio_pathfm','biodino_1tb_hplus','gigapath','hoptimus0','virchow2'} else 8000,
       result=str(OUT/'results'/name/'ctc'/f'fold{fold}'/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log')))
  path=OUT/'ctc_queue.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'tasks':jobs,'stage':'Validated established FM/BioDINO/new FM full-fold pilots; 24 fixed anchors x5folds x5downstream seeds'},indent=2));temp.replace(path)
  (OUT/'states/ctc_admission.json').write_text(json.dumps({'status':'admitted','jobs':len(jobs),'passed':passed,'updated':time.time()},indent=2));print('ADMITTED',len(jobs),flush=True)

if __name__=='__main__':main()
