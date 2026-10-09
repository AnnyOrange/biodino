#!/usr/bin/env python3
"""Twenty-hour supervisor: liveness, matched-stream audit, reports, local eval handoff.

Never select a method/checkpoint using test scores. Retry an interrupted trainer
once using its existing full checkpoint; retain all attempt logs and statuses.
"""
import json,math,os,subprocess,time
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
REPO=Path('/mnt/huawei_deepcad/dinov3')
TRAIN=REPO/'outputs/01_training_runs/hs6_l5_selective_retention_20260923'
EVAL=REPO/'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923'

def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default
def atomic(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def alive(pid):
 try:os.kill(pid,0);return True
 except (ProcessLookupError,ValueError,TypeError):return False

def main():
 end=time.time()+20*3600;retries={};local_worker=None
 while time.time()<end:
  trains={};digests={}
  for i,arm in enumerate(['vanilla','gram','fixed','adaptive']):
   run=TRAIN/f'{arm}_formal';rows=[]
   p=run/'raw_loss_metrics.jsonl'
   if p.exists():
    for line in p.read_text().splitlines():
     try:rows.append(json.loads(line))
     except ValueError:pass
   # A resumed attempt can append repeated updates after its last checkpoint.
   # Audit the most recent observation at each actual optimizer update.
   rows=list({r['optimizer_update']:r for r in rows}.values())
   rows.sort(key=lambda r:r['optimizer_update'])
   last=rows[-1] if rows else {};exit_state=read(run/'exit.json')
   pid=int((run/'trainer.pid').read_text()) if (run/'trainer.pid').exists() else 0
   live=alive(pid) if pid else False
   trains[arm]={'pid':pid,'alive':live,'exit':exit_state,'updates':len(rows),'last':last}
   digests[arm]={r['optimizer_update']:r.get('batch_sample_key_digest') for r in rows}
   if exit_state and exit_state['returncode']!=0 and not live and retries.get(arm,0)<1 and not (run/'STOP_REQUESTED_NEXT_METHOD').exists():
    full=list((run/'ckpt').glob('*/checkpoint.pth'))
    if full:
     archive=run/f'exit_attempt_{retries.get(arm,0)+1}.json';(run/'exit.json').rename(archive)
     cmd=['python',str(REPO/'scripts/launch_selective_retention_20260923.py'),'--arm',arm,'--gpus',f'{2*i},{2*i+1}',
          '--batch','128','--checkpoint-blocks','24','--steps','2440','--period','488','--tag','formal','--port',str(32640+2*i),'--resume']
     with (TRAIN/f'{arm}_retry_supervisor.log').open('a') as log:
      subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     retries[arm]=retries.get(arm,0)+1
  shared=set.intersection(*(set(v) for v in digests.values()))
  common=len(shared)
  matched=all(all(v[k]==digests['vanilla'][k] for k in shared) for v in digests.values())
  states=Counter();failures=[]
  for p in (EVAL/'claims').glob('*/status.json'):
   st=read(p,{})
   states[st.get('state','UNKNOWN')]+=1
   if st.get('state')=='FAILED':failures.append(p.parent.name)
  snapshot={'utc':datetime.now(timezone.utc).isoformat(),'training':trains,'common_updates':common,'sample_digests_matched':matched,
            'evaluation':dict(states),'failed_tests':failures,'retry_attempts':retries}
  atomic(EVAL/'OVERNIGHT_STATUS.json',snapshot)
  lines=['# Overnight selective retention status','',snapshot['utc'],'',f'Common audited updates: {common}; data digests matched: {matched}.',
         '', '| Arm | Updates | Alive | Exit |','|---|---:|---|---|']
  for arm,s in trains.items():lines.append(f'| {arm} | {s["updates"]} | {s["alive"]} | {s["exit"]} |')
  lines+=['',f'Evaluation queue: {dict(states)}. Failed tasks: {failures}.','',
          'The queue evaluates predeclared checkpoints; no test-side model selection. Full v4 coverage requires the explicit inventory/admission ledger.',
          '', '## Regression: canonical fixed alpha1 vs train-only selected alpha', '', '| Dataset / arm | Fixed R² | Tuned R² |','|---|---:|---:|']
  for p in sorted((EVAL/'regression_tuning').glob('*.json')):
   if p.name.endswith('_selection.json'):continue
   data=read(p,{})
   for arm,result in data.get('results',{}).items():
    lines.append(f'| {data["dataset"]} / {arm} | {result["fixed"]["r2"]:.5f} | {result["tuned"]["r2"]:.5f} |')
  (EVAL/'OVERNIGHT_STATUS.md').write_text('\n'.join(lines)+'\n')
  if not matched:atomic(EVAL/'DATA_STREAM_MISMATCH.json',snapshot)
  if local_worker is None and not (EVAL/'LOCAL_TRAINING_RESERVED').exists() and all(s['exit']=={'returncode':0} for s in trains.values()):
   with (EVAL/'supervisors/local5090.log').open('a') as log:
    local_worker=subprocess.Popen(['python',str(REPO/'scripts/selective_retention_eval_queue_20260923.py'),'worker','--host','local5090','--hours','6','--min-slots','5','--max-slots','12'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(45)

if __name__=='__main__':main()
