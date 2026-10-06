#!/usr/bin/env python3
"""Admit resident recovery exports to the existing verified MoNuSeg 30/7/14 campaign."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

SITE = Path('/data/xuzijing/monuseg_t30v7_remote_20260930')
EVAL = Path('/data/xuzijing/hs6_l5_v2_recovery_eval_20260930')
RUNS = Path('/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930')
OUT = SITE / 'campaign_v3'
sys.path.insert(0, str(SITE / 'snapshot/scripts'))
import run_retest_fleet_20260918 as fleet

with (OUT / '_state/round5_online.lock').open('a') as singleton:
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        with (OUT / '_state/manifest.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = OUT / 'campaign_manifest.json'
            manifest = json.loads(path.read_text())
            known = {a['path'] for a in manifest['checkpoint_assets']}
            new = []
            for arm in ('global_cls_w3','global_cls_slow2_w3','global_cls_slow2_w03','gram_ext'):
                for p in (RUNS / arm / 'eval').glob('training_*/teacher_checkpoint.pth'):
                    ck = int(p.parent.name.split('_')[-1])
                    if ck not in range(29767,35136,488) or str(p) in known or time.time()-p.stat().st_mtime < 180:
                        continue
                    new.append(dict(arm='v2_'+arm,checkpoint_id=str(ck),path=str(p),
                        config=str(EVAL/'bin/config.yaml'),kind='dinov3',model_id='',reserve_mib=4096))
            manifest['online_checkpoints'] = True
            if new:
                manifest['checkpoint_assets'] += new
                manifest['tasks'] += fleet.tasks_for(new, manifest['datasets'])
                manifest.setdefault('admission_history',[]).append(dict(time=time.time(),arms=[a['arm'] for a in new],
                    authorization='2026-10-06 continue existing method development and evaluation',
                    watcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
                print('ADMITTED',len(new),'resident exports to official 30/7/14 evaluation',flush=True)
            fleet.queue.save(path,manifest)
        time.sleep(120)
