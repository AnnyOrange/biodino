#!/usr/bin/env python3
"""Write a live, evidence-based status for the two authorized training tracks."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3')
OUT = ROOT/'outputs/00_reports/l20tb_domain_7b_baseline_20261018'
L = ROOT/'outputs/01_training_runs/l20tb_domain_recoverability_from47823_len19520_20261018'
QI = ROOT/'outputs/01_training_runs/l20tb_complement_bio5tb_from47823_len19520_qi_20261018'
REMOTE = r'''
import json,sys
from pathlib import Path
kind=sys.argv[1]
if kind in ('train','l_train'):
 p=Path('/home/xzj/storage/merged/vit7b_20tb_recovered249_20261018' if kind=='train' else
  '/home/xzj/storage/merged/cooldown_20261013/20tb_distill_cooldown_len4880_lw2_rebal')
 state=json.loads((p/('recovery_state.json' if kind=='train' else 'supervisor_state.json')).read_text())
 metric=p/'raw_loss_metrics.jsonl'
 rows=metric.read_text().splitlines() if metric.exists() else []
 records=[json.loads(line) for line in rows]
 started=state.get('attempts',[{}])[-1].get('started',0)
 active=[row for row in records if row.get('time_unix',0)>=started]
 phase=('TRAINING' if active else 'RESTORING_CHECKPOINT') if state.get('status')=='TRAINING' else state.get('status')
 print(json.dumps(dict(state=state,latest=json.loads(rows[-1]) if rows else None,
  phase=phase,active_attempt_updates=len(active),active_attempt_latest=active[-1] if active else None,
  train_process_exists=Path('/proc/'+str(state.get('train_pid',-1))).exists(),
  checkpoints=[x.name for x in (p/'ckpt').glob('*') if x.name.isdigit() and
   ((x/'.metadata').exists() or (x/'checkpoint.pth').exists())])))
else:
 p=Path('/data/xuzijing/cooldown_20261013_eval/7b_20tb_nogram_recovered249_20261018/results/_state')
 print(json.dumps({s:len(list((p/s).glob('*'))) for s in ('done','claims','terminal','failures')}))
'''


def remote(host, kind):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-c',
                      __import__('shlex').quote(REMOTE),kind],text=True,capture_output=True,timeout=30)
    if p.returncode:
        return dict(error=p.stderr[-1000:])
    return json.loads(p.stdout)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'status.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            status=dict(updated=time.time(),pid=os.getpid(),promotion='WAITING_FOR_VALIDATED_L_WINNER_AND_PLAIN_7B_BASELINE')
            for key, path in [('L', L), ('L_QI', QI)]:
                rows=[json.loads(s) for s in (path/'raw_loss_metrics.jsonl').read_text().splitlines()]
                status[key]=dict(state=json.loads((path/'supervisor_state.json').read_text()),
                    latest=rows[-1] if rows else None,recorded_updates=len(rows),
                    finite_losses=all(math.isfinite(r['total_loss']) for r in rows),
                    evaluation_checkpoints=[50263,52703,57583,67343],
                    primary_evaluation_protocol='SELECTED29_LOCKED_E20_3SEEDS_20261019',
                    historical_screening_checkpoints=[48311,50263] if key == 'L' else [48311])
            status['resource_plan'] = {'deepcad': 'L domain method: GPUs3/4/5/7',
                '3090-qi': 'L complementary biology method: GPUs2/5/6/7, legacy evaluations retained',
                '5090-hxw-xzj': '7B GPUs0/1/2/3; previous L distillation completed79543',
                'online_evaluation': 'selected29: deepcad GPUs1/2/6; historical45 in-flight: cpu2/local; hxw L and7B: lyx'}
            for name in ('L_dispatch',):
                status[name]=[json.loads(p.read_text()) for p in (OUT/name/'workers').glob('*.json')]
            for key,host,kind in [('7B','5090-hxw-xzj','train'),('L_HXW','5090-hxw-xzj','l_train'),
                                  ('7B_eval','5090-lyx-xr','eval')]:
                try:
                    status[key]=remote(host,kind)
                except Exception as exc:
                    status[key]=dict(error=repr(exc))
            try:
                from summarize_l20tb_innovation_20261018 import summarize
                summary = summarize()
                status['L_screening'] = dict(file=str(OUT/'L_screening_summary.json'),
                    complete_points=sum(p['complete'] for p in summary['points'].values()))
            except Exception as exc:
                status['L_screening'] = dict(error=repr(exc))
            primary = ROOT/'outputs/02_eval_runs/l20tb_selected29_20261019/manager_status.json'
            if primary.exists():
                status['selected29_evaluation'] = json.loads(primary.read_text())
            tmp=OUT/'live_status.tmp'
            tmp.write_text(json.dumps(status,indent=2)+'\n')
            tmp.replace(OUT/'live_status.json')
        except Exception as exc:
            print(repr(exc),flush=True)
        time.sleep(60)


if __name__=='__main__':
    main()
