#!/usr/bin/env python3
"""Admit missing cached head seeds only after a real shared segmentation pilot completes."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
q=OUT/'segmentation_queue.json';pilot=json.loads(q.read_text())['tasks'][0]
while not Path(pilot['result']).exists():time.sleep(20)
r=json.loads(Path(pilot['result']).read_text());assert r['_meta']['seed']==3 and r['_meta']['probe_batch_size']==32 and Path(r['prediction_path']).exists()
cells=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells'];jobs=[]
for index,c in enumerate(cells):
 if not c['cache_ready']:continue
 for budget in [20,50]:
  existing={x['seed'] for x in c['existing'] if x['budget']==budget}
  for seed in range(5):
   if seed in existing:continue
   key=f"{c['model']}__segmentation__{c['dataset']}__{c['split']}__E{budget}_seed{seed}"
   jobs.append({'key':key,'model':c['model'],'task':'segmentation','dataset':c['dataset'],'split':c['split'],'budget':budget,'seed':seed,'cell_index':index,'entrypoint':'run_unified_paper_segmentation.py','min_free_mib':2500,'args':['--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],'result':str(OUT/'results'/c['model']/'segmentation'/c['dataset']/c['split']/f'E{budget}'/f'seed{seed}'/'component_result.json'),'log':str(OUT/'logs'/f'{key}.log')})
tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps({'tasks':jobs,'stage':'validated cached head extension; preserve existing seeds0-2'},indent=2));tmp.replace(q)
workers=[]
for h,g,py in [('deepcad',1,'/home/deepcad/anaconda3/envs/dinov3/bin/python'),('cpu11',0,'/home/inspur/anaconda3/envs/dinov3/bin/python'),('cpu12',0,'/home/inspur/anaconda3/envs/dinov3/bin/python')]:
 log=OUT/'logs'/f'worker_{h}_gpu{g}_segmentation.log';cmd=f'cd {ROOT} && (nohup {py} -u scripts/unified_paper_queue.py --gpu {g} --slot segmentation --host-label {h} --queue-file {q} > {log} 2>&1 < /dev/null & echo $!)'
 x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',h,cmd],capture_output=True,text=True,timeout=15);workers.append({'host':h,'gpu':g,'pid':x.stdout.strip(),'error':x.stderr,'log':str(log)})
(OUT/'states/segmentation_dispatch.json').write_text(json.dumps({'tasks':len(jobs),'workers':workers,'pilot':pilot['result']},indent=2));print('ADMITTED',len(jobs),workers,flush=True)
