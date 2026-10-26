#!/usr/bin/env python3
"""Validate the real BBBC038 pilot, then launch frozen-head prediction recovery."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
pilot=OUT/'results/biodino_20tb_l/detection_proxy/bbbc038/seed3/component_result.json'
while not pilot.exists():time.sleep(20)
d=json.loads(pilot.read_text())
assert d['seed']==3 and d['batch_size']==8 and d['epochs']==5 and Path(d['prediction_path']).exists()
models=json.loads((OUT/'config.json').read_text())['models'];tasks=[]
for model in models:
 for seed in range(5):
  m=model['model'];key=f'{m}__detection_proxy__bbbc038__seed{seed}'
  tasks.append({'key':key,'model':m,'task':'detection_proxy','dataset':'bbbc038','seed':seed,'entrypoint':'run_unified_paper_detection.py','min_free_mib':model['min_free_mib'],
   'args':['--model',m,'--dataset','bbbc038','--seed',str(seed)],'log':str(OUT/'logs'/f'{key}.log'),
   'result':str(OUT/'results'/m/'detection_proxy/bbbc038'/f'seed{seed}'/'component_result.json')})
q=OUT/'detection_queue.json';temp=q.with_suffix('.tmp');temp.write_text(json.dumps({'task':'fixed-v4-BBBC038-proxy-prediction-recovery','models':models,'tasks':tasks,'note':'Existing metrics preserved. Rerun fixed frozen heads to recover missing per-image test logits; no protocol/budget change.'},indent=2)+'\n');temp.replace(q)
workers=[]
for host,python,gpu in [('deepcad','/home/deepcad/anaconda3/envs/dinov3/bin/python',1),('deepcad','/home/deepcad/anaconda3/envs/dinov3/bin/python',5)]+[(h,'/home/inspur/anaconda3/envs/dinov3/bin/python',0) for h in ['cpu2','cpu5','cpu11','cpu12','cpu19','cpu20']]:
 log=OUT/'logs'/f'worker_{host}_gpu{gpu}_detection.log'
 cmd=f'cd {ROOT} && (nohup {python} -u scripts/unified_paper_queue.py --gpu {gpu} --slot detection --host-label {host} --queue-file {q} > {log} 2>&1 < /dev/null & echo $!)'
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6',host,cmd],capture_output=True,text=True,timeout=15)
 workers.append({'host':host,'gpu':gpu,'pid':r.stdout.strip(),'exit_code':r.returncode,'error':r.stderr,'log':str(log)})
 print('WORKER',workers[-1],flush=True)
(OUT/'states/detection_dispatch.json').write_text(json.dumps({'tasks':len(tasks),'workers':workers,'pilot':str(pilot)},indent=2))
print('ADMITTED',len(tasks),'seed jobs',flush=True)
