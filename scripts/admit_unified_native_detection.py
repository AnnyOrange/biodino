"""Admit the full shared native box matrix after own-model and FM real pilots."""
import fcntl,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def main():
 with (OUT/'locks/native_detection_admission.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  models=['biodino_1tb_b','cytoself']
  while True:
   done=[]
   for name in models:
    p=OUT/'results'/name/'native_box/bccd/seed0/component_result.json'
    if p.exists():
     x=json.loads(p.read_text())
     if x['metrics']['evaluated_images']==72 and x['epochs']==10 and Path(x['prediction_path']).exists():done.append(name)
   (OUT/'states/native_detection_admission.json').write_text(json.dumps({'status':'waiting_for_real_pilots','completed':done,'required':models,'planned_jobs':240,'updated':time.time()},indent=2))
   if len(done)==2:break
   time.sleep(30)
  jobs=[]
  for model in json.loads((OUT/'config.json').read_text())['models']:
   name=model['model']
   for dataset in ['bccd','bbbc041']:
    for seed in range(5):
     key=f'{name}__native_box__{dataset}__seed{seed}';root=OUT/'results'/name/'native_box'/dataset
     jobs.append(dict(key=key,model=name,task='box_detection',dataset=dataset,fold='fixed',seed=seed,
       entrypoint='run_unified_paper_native_detection.py',args=['--model',name,'--dataset',dataset,'--seed',str(seed)],
       requires=[str(root/'selected_recipe.json')] if seed else [],
       min_free_mib=18000 if name in ['genbio_pathfm','hoptimus0','gigapath','virchow2','biodino_1tb_hplus'] else 4000 if name in ['cytoself','distillpath','morphem'] else 8000,
       result=str(root/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log')))
  path=OUT/'native_detection_queue.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'tasks':jobs,'stage':'Own-model and FM pilots passed; full shared native box comparison'},indent=2));temp.replace(path)
  (OUT/'states/native_detection_admission.json').write_text(json.dumps({'status':'admitted','jobs':len(jobs),'updated':time.time()},indent=2));print('ADMITTED',len(jobs),flush=True)

if __name__=='__main__':main()
