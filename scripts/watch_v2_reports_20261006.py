#!/usr/bin/env python3
"""Refresh ongoing recovery reports without launching duplicate scorers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', choices=('shared','lyx'), required=True)
p.add_argument('--wait-pid', type=int)
p.add_argument('--interval', type=int, default=1800)
a=p.parse_args()
if a.site == 'lyx':
    root=Path('/data/xuzijing/hs6_l5_v2_recovery_eval_20260930')
    output=root/'scores/latest_20261006'
    command=['/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python',str(root/'bin/score_v2_arms_20261002.py')]
    extra=dict(V2_MIRROR=str(root),V2_UNION=str(root/'bin/SELECTED_CELLS.csv'),
        V2_CR_MODULE=str(root/'bin/capability_regret_20260923.py'),V2_OUT=str(output),
        V2_MONUSEG_ROOT='/data/xuzijing/monuseg_t30v7_remote_20260930/campaign_v3',
        V2_ARMS='global,global_local,global_cls,global_slow,global_w03,global_cls_slow,global_cls_slow2,global_cls_w3,global_cls_slow2_w3,global_cls_early,global_cls_slow2_w03')
else:
    root=Path('/mnt/huawei_deepcad/dinov3')
    output=root/'outputs/00_reports/deepcad_method_20260927/progress_20261006/20tb'
    command=['python3',str(root/'scripts/summarize_v2_paired_20261006.py')]
    extra={}
output.mkdir(parents=True,exist_ok=True)
with (output/'watch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while a.wait_pid and Path(f'/proc/{a.wait_pid}/cmdline').exists():
        cmd=Path(f'/proc/{a.wait_pid}/cmdline').read_bytes()
        if b'score_v2_arms_20261002.py' not in cmd:
            break
        time.sleep(30)
    while True:
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',**extra)
        with (output/'refresh.log').open('a') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        status=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),returncode=result.returncode,command=command)
        temp=output/'refresh_status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(output/'refresh_status.json')
        time.sleep(a.interval)
