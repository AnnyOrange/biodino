#!/usr/bin/env python3
"""Add native-final dense, admitted MoNuSeg, and full RxRx1/RxRx3 cells."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import selective_retention_eval_queue_20260923 as q
import hs6_l5_weightspace_campaign_20260922 as v4
SRC=Path('/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923')

def add(arm,checkpoint,config,ckid,baselines_only=False):
 marker=q.ROOT/'extensions'/f'{arm}.json'
 if marker.exists():
  previous=json.loads(marker.read_text()).get('added',[])
  if baselines_only or f'ret__rxrx3-core__{arm}' in previous:return
 v4.ROOT=q.ROOT;v4.checkpoint_for=lambda _:checkpoint;v4.SEG_CKPT_ID[arm]=ckid;v4.SEG_RUN_LABEL[arm]=arm
 tasks=[]
 for ds in ['conic','livecell','pannuke','monuseg']:
  t=v4.seg_task(ds,arm,100);c=t['cmd']
  c[c.index('--protocol')+1]='manual'
  c+=['--layer-preset','last1','--feature-img-size',str(768 if ds=='monuseg' else 512 if ds=='livecell' else 256),
      '--resize-mode','pad' if ds in ('livecell','monuseg') else 'stretch',
      '--probe-class-weight-mode','sqrt_inverse' if ds=='conic' else 'none']
  name=f'hs6_l5_{arm}_primary';c[c.index('--run-name')+1]=name
  t['id']=f'seg_primary__{ds}__{arm}';t['done']['run_prefix']=name+'_'
  t['readout']='v4 primary native-final last1'
  if ds=='monuseg':t['cwd']=t['pythonpath']=str(SRC)
  tasks.append(t)
 if not baselines_only:
  t=v4.ret_task('rxrx1-cross',arm,90);t['done']['rows']=7
  # Standard v4 RxRx1 cross-experiment core (four cell types), no sample cap.
  t['protocol_detail']='official-cross-experiment-core';tasks.append(t)
  out=q.ROOT/'retrieval/rxrx3-core'/arm
  t={'id':f'ret__rxrx3-core__{arm}','family':'retrieval','dataset':'rxrx3-core','arm':arm,'order':91,
     'cwd':str(SRC),'pythonpath':str(SRC),
     'cmd':[str(v4.PYTHON),str(q.REPO/'scripts/run_selective_rxrx3_20260923.py'),'--arm',arm,'--checkpoint',str(checkpoint),'--config',str(config)],
     'done':{'type':'summary_csv','path':str(out/'summary.csv'),'dataset':'rxrx3-core','rows':1}}
  tasks.append(t)
 checkpoint_sha=q.sha(checkpoint)
 for t in tasks:
  c=t['cmd']
  if '--train-config' in c:c[c.index('--train-config')+1]=str(config)
  t.update(protocol='bio-eval-union-v4',created=q.now(),priority=5,checkpoint_sha256=checkpoint_sha,
           expected_memory_mib=9000 if t['dataset']=='monuseg' else 6500 if t.get('heavy') else 3500)
  task_file=q.ROOT/'tasks'/f'{t["id"]}.json'
  if not task_file.exists():q.atomic(task_file,t)
 q.atomic(marker,{'arm':arm,'added':[t['id'] for t in tasks],'remaining':['ctc native new-arm admission','OOD separate evaluation']})
 print('EXTEND',arm,len(tasks),flush=True)

def once():
 for p in (q.ROOT/'registered').glob('*.json'):
  j=json.loads(p.read_text());ck=Path(j['checkpoint']);add(j['arm'],ck,ck.parents[2]/'config.yaml',int(ck.parent.name.split('_')[-1]))
 for role,ckid in [('E',12687),('M',20007),('L',29279)]:
  ck=v4.RUN/f'eval/training_{ckid}/teacher_checkpoint.pth';add(f'baseline_{role}',ck,v4.RUN/'config.yaml',ckid,False)

if __name__=='__main__':
 if '--watch' in sys.argv:
  end=time.time()+20*3600
  while time.time()<end:once();time.sleep(60)
 else:once()
