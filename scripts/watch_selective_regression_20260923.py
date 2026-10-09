#!/usr/bin/env python3
"""Apply the predeclared train-only readout sweep to every completed model bank."""
import json, os, subprocess, time
from pathlib import Path
import selective_retention_eval_queue_20260923 as q

DATASETS=('bbbc005','bbbc013','conic-cell-count','livecell-cell-count')
PYTHON='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'

def main():
    end=time.time()+24*3600;active={}
    env=os.environ.copy();env.update(PYTHONPATH=str(q.REPO),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    while time.time()<end or active:
        for key,(proc,log) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                log.close();ds,arm=key
                path=q.ROOT/'regression_tuning/arms'/arm/f'{ds}.json'
                ok=False;parity=None
                if rc==0 and path.exists():
                    data=json.loads(path.read_text())['results'][arm]
                    official=json.loads((q.ROOT/'frozen'/ds/arm/'last_result.json').read_text())
                    parity=abs(float(official['r2'])-data['fixed']['r2'])
                    # Same float32 sklearn estimator differs slightly across
                    # CPU BLAS architectures for weakly regularized matrices.
                    # Verified G29279/LIVECell: exact match on extraction host;
                    # float64 predictions match across hosts to <1e-13 R².
                    ok=parity<1e-3
                q.atomic(q.ROOT/'regression_tuning/claims'/f'{arm}__{ds}'/'status.json',
                         {'state':'DONE' if ok else 'FAILED','returncode':rc,'fixed_r2_difference':parity,
                          'numeric_tolerance':1e-3,'cross_host_float32_drift':parity is not None and parity>1e-4,'time':q.now()})
                print(q.now(),'END',key,rc,'fixed parity',parity,flush=True);del active[key]
        if time.time()<end:
            for p in sorted((q.ROOT/'registered').glob('*.json')):
                arm=p.stem
                for ds in DATASETS:
                    if len(active)>=2:break
                    claim=q.ROOT/'claims'/f'cls__{ds}__{arm}'/'status.json'
                    if not claim.exists() or json.loads(claim.read_text())['state']!='DONE':continue
                    dest=q.ROOT/'regression_tuning/claims'/f'{arm}__{ds}'
                    try:dest.mkdir(parents=True)
                    except FileExistsError:continue
                    log=(dest/'console.log').open('a')
                    cmd=[PYTHON,str(q.REPO/'scripts/tune_5tb_regression_v4_20260923.py'),
                         '--dataset',ds,'--arm',arm,'--campaign',str(q.ROOT)]
                    proc=subprocess.Popen(cmd,cwd=q.REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    q.atomic(dest/'status.json',{'state':'RUNNING','pid':proc.pid,'command':cmd,'start':q.now()})
                    active[(ds,arm)]=(proc,log);print(q.now(),'START',ds,arm,proc.pid,flush=True)
        time.sleep(15)

if __name__=='__main__':main()
