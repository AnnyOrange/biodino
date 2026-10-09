"""Admit preregistered C/K arms only after frozen calibration is verified."""
import fcntl,json,math,os,subprocess,time
from pathlib import Path
import run_deepcad_method_20260927 as r
ROOT=r.REPO/'outputs/00_reports/deepcad_method_20260927/ck_recovery'
CFG=r.REPO/'outputs/00_reports/deepcad_method_20260927/coexist/config.json'
CAL=r.ROOT/'ck_calibration_e12687_smoke'

def admit():
    import torch
    torch.set_num_threads(2)
    assert json.loads((ROOT/'CPU_CHECKS.json').read_text())['status']=='PASS_ENGINEERING_ONLY'
    assert json.loads((CAL/'exit.json').read_text())['returncode']==0
    rows=[json.loads(s) for s in (CAL/'raw_loss_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==16 and rows[-1]['optimizer_update']==12703,(len(rows),rows[-1].get('optimizer_update'))
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        assert row['lr']==0 and row['backbone_lr']==0
    reports={}
    for stream in ['global','local']:
        p=ROOT/'calibration_E12687'/f'{stream}.json';d=json.loads(p.read_text())
        assert d['status']=='READY' and d['steps']==128 and d['observations']==64 and d['dim']==1024
        assert all(math.isfinite(v) and v>=0 for v in d['reference'])
        assert all(math.isfinite(v) and v>=.01999 for v in d['noise_band'])
        assert math.isfinite(d['gain_cap']) and d['gain_cap']>=1
        reports[stream]=dict(sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),gain_cap=d['gain_cap'],mean_reference=sum(d['reference'])/1024,mean_band=sum(d['noise_band'])/1024)
    manifest=json.loads((CAL/'launch_manifest.json').read_text())
    assert r.hash_files(Path(manifest['source']))==manifest['source_hashes']
    anchor=next(s.split('=',1)[1] for s in manifest['command'] if s.startswith('student.resume_from_teacher_chkpt='))
    a=torch.load(anchor,map_location='cpu',mmap=True,weights_only=False)['teacher']
    s=torch.load(CAL/'ckpt/12703/checkpoint.pth',map_location='cpu',mmap=True,weights_only=False)['model']
    compared=0
    for key,value in s.items():
        if not key.startswith('student.backbone.'):continue
        other=a[key.removeprefix('student.')].to(dtype=value.dtype)
        assert torch.equal(value,other),f'Calibration changed backbone: {key}'
        compared+=1
    assert compared>300,compared
    admission=dict(time=r.now(),status='ADMITTED_EXPLORATORY_2440_UPDATES',frozen_backbone_tensors_equal=compared,zero_lr_steps=16,calibration=reports,biological_improvement_established=False,comparison_caveat='Historical Adaptive was two-rank; C/K are paired one-rank and cannot establish matched-control superiority alone')
    r.atomic(ROOT/'ADMISSION.json',admission)
    config=json.loads(CFG.read_text())
    for item in config['training_jobs']:
        if Path(item['output']).name in ['ck_c_e12687_formal','ck_k_e12687_formal']:item['admitted']=True
    r.atomic(CFG,config)
    print(json.dumps(admission),flush=True)

def main():
    lock=(ROOT/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.atomic(ROOT/'manager_process.json',dict(pid=os.getpid(),time=r.now()))
    while not (ROOT/'ADMISSION.json').exists():
        if (CAL/'exit.json').exists():
            try:admit()
            except Exception as exc:
                r.atomic(ROOT/'ADMISSION_FAILED.json',dict(time=r.now(),error=repr(exc)));raise
            break
        r.atomic(ROOT/'status.json',dict(time=r.now(),stage='FROZEN_CALIBRATION_RUNNING_OR_QUEUED'))
        time.sleep(15)
    # Keep provenance of the live queue and stop only our new arms on nonfinite training.
    while True:
        states={}
        for name in ['ck_c_e12687_formal','ck_k_e12687_formal']:
            out=r.ROOT/name;p=out/'raw_loss_metrics.jsonl';last=None
            if p.exists():
                lines=p.read_text().splitlines()
                try:last=json.loads(lines[-1]) if lines else None
                except json.JSONDecodeError:pass
            st='QUEUED'
            if (out/'process.json').exists():st='STARTED'
            if (out/'exit.json').exists():st='COMPLETE' if json.loads((out/'exit.json').read_text())['returncode']==0 else 'INTERRUPTED_OR_FAILED'
            if last and any(not math.isfinite(v) for v in last.values() if isinstance(v,float)) and not (out/'NONFINITE_STOP.json').exists():
                proc=json.loads((out/'process.json').read_text())['torchrun']
                code="import sys;sys.path.insert(0,'/mnt/huawei_deepcad/dinov3/scripts');import run_deepcad_method_20260927 as r;import selective_retention_eval_queue_20260923 as q;from pathlib import Path;import json;"+f"p={proc};out=Path({str(out)!r});expected=json.loads((out/'launch_manifest.json').read_text())['command'];actual=Path(f'/proc/{{p}}/cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0');assert actual==expected;r.stop(q.AdoptedProcess(p,out/'child_exit.json'))"
                import shlex
                subprocess.run(['ssh','deepcad',r.PY+' -c '+shlex.quote(code)],check=True)
                r.atomic(out/'NONFINITE_STOP.json',dict(time=r.now(),iteration=last.get('optimizer_update')))
                st='STOPPED_NONFINITE'
            states[name]=dict(state=st,last_iteration=last.get('optimizer_update') if last else None)
        r.atomic(ROOT/'status.json',dict(time=r.now(),stage='FORMAL_ARMS',arms=states))
        if all(v['state'] in ['COMPLETE','INTERRUPTED_OR_FAILED','STOPPED_NONFINITE'] for v in states.values()):return
        time.sleep(30)

if __name__=='__main__':main()
