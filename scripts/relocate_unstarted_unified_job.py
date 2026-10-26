"""Move an admission-waiting child with zero CUDA allocation; never stop computation."""
import argparse,fcntl,json,shlex,subprocess,time
from pathlib import Path
from watch_unified_native_fleet import launch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--host',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--peak-mib',type=int);a=p.parse_args()
    matches=[(q,j) for q in OUT.glob('*queue.json') if q.name!='remote_queue.json' for j in json.loads(q.read_text()).get('tasks',[]) if j['key']==a.key];assert len(matches)==1;q,job=matches[0]
    states=[json.loads(f.read_text()) for f in (OUT/'states').glob('worker*.json')];s=next(x for x in states if x.get('task')==a.key)
    code='''import json,pathlib,subprocess,sys,signal,os
pid=int(sys.argv[1]);job=json.loads(sys.argv[2]);out=pathlib.Path(sys.argv[3]);cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\\0')[:-1]
assert any(x.endswith('/'+job['entrypoint']) for x in cmd) and cmd[-len(job['args']):]==job['args']
records=[json.loads(p.read_text()) for p in (out/'gpu_reservations').glob(f'process_*_{pid}.json')];assert len(records)==1 and records[0]['status']=='waiting_for_capacity'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
assert not any(line.split(',')[0].strip()==str(pid) for line in apps.splitlines()),'Child has CUDA allocation; migration forbidden'
os.kill(pid,signal.SIGTERM);print(json.dumps(records[0]))
'''
    cmd=['python3','-c',code,str(s['child_pid']),json.dumps(job),str(OUT)]
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',s['host'],shlex.join(cmd)],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;proof=json.loads(r.stdout)
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(q.read_text())
        for j in d['tasks']:
            if j['key']==a.key:
                j['allowed_hosts']=[a.host]
                if a.peak_mib:j.update(expected_peak_mib=a.peak_mib,min_free_mib=a.peak_mib+5000)
        d['tasks'].sort(key=lambda j:j['key']!=a.key);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(q)
    claim=OUT/'locks'/f'{job.get("claim_key",job["model"])}.claimdir'
    deadline=time.monotonic()+45
    while claim.exists() and time.monotonic()<deadline:time.sleep(1)
    assert not claim.exists(),'Original parent has not released claim; no duplicate launched'
    ident=f'{a.host}_gpu{a.gpu}_slotrelocated_spatial';pid,log=launch(dict(host=a.host,gpu=a.gpu,slot='relocated_spatial',queue=str(q),ident=ident))
    record=dict(updated=time.time(),key=a.key,original=s,zero_cuda_wait_proof=proof,target_host=a.host,target_gpu=a.gpu,worker_pid=pid,log=log,reason='Relocated before encoder loading or training; actual admission still applies; original attempt/log retained')
    (OUT/'corrections'/f'{a.key}__capacity_relocation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
