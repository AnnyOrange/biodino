"""Wait for atomic teacher transfer completion, validate it, append only two OOD jobs."""
import json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
cfg=json.loads((OUT/'config.json').read_text());model=next(m for m in cfg['models'] if m['model']=='biodino_1tb_b');path=Path(model['checkpoint'])
while not path.exists():time.sleep(30)
d=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
assert len(d['teacher'])==188
queue=json.loads((OUT/'queue.json').read_text())
if not any(x['model']==model['model'] for x in queue['models']):queue['models'].append(model)
for task in ['xray','cryo']:
 key=f"{model['model']}__ood__{task}"
 if any(t['key']==key for t in queue['tasks']):continue
 queue['tasks'].append({'key':key,'model':model['model'],'dataset':task,'task':'ood','args':['--model',model['model'],'--task',task,'--batch-size','32','--checkpoint',str(path),'--train-config',model['train_config']],'min_free_mib':8000,'log':str(OUT/'logs'/f'{key}.log'),'result':str(OUT/'results'/model['model']/'ood'/f'{task}_result.json')})
t=OUT/'queue.tmp';t.write_text(json.dumps(queue,indent=2)+'\n');t.replace(OUT/'queue.json');print('Validated teacher; two tasks admitted to persistent queue',flush=True)
