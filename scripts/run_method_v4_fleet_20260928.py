"""Shared atomic task claims; one formal test per selected GPU per host."""
import argparse, collections, fcntl, hashlib, importlib.metadata, json, os
import signal, socket, subprocess, sys, time
from pathlib import Path
import run_deepcad_method_20260927 as resource
import selective_retention_eval_queue_20260923 as queue

REPO=Path('/mnt/huawei_deepcad/dinov3')
ROOT=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'

def validate(task):
    done=task['done']
    if done['type']=='ctc_native':
        p=Path(done['path'])
        if not p.exists():return False,'missing'
        d=json.loads(p.read_text());rows=d.get('domain_rows',[])
        ok=d.get('status')=='VALID_COMPLETE' and len(d.get('folds',[]))==5 and len(rows)==20
        ok=ok and len({r['domain'] for r in rows})==20 and all(int(r['ctc_metrics']['Valid'])==1 for r in rows)
        return ok,'native five-fold / 20-domain validation'
    if done['type']=='ood_json':
        p=Path(done['path'])
        if not p.exists():return False,'missing'
        d=json.loads(p.read_text());key=done['dataset']+'_ood_'
        ok=all(key+x in d for x in ['auroc','average_precision','id_bank','id_test','ood_test'])
        return ok and d.get(key+'ood_test')==done['expected_ood'],'OOD metrics and fixed population'
    return queue.validated_done(done)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--gpus',required=True)
    ap.add_argument('--allow-existing',action='store_true');ap.add_argument('--hours',type=float,default=60)
    a=ap.parse_args();gpus=[int(g) for g in a.gpus.split(',')]
    dest=ROOT/'fleet'/a.host;dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks=[]
    for g in gpus:
        f=open(f'/tmp/dinov3_method_fleet_gpu_{g}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    initial=resource.gpu_info();baseline={g:set() for g in gpus}
    if a.allow_existing:
        for p in resource.gpu_processes():
            if Path(f'/proc/{p["pid"]}').exists() and Path(f'/proc/{p["pid"]}').stat().st_uid==os.getuid():
                for g in gpus:
                    if p['uuid']==initial[g]['uuid']:baseline[g].add(p['pid'])
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn']}
    queue.atomic(dest/'manifest.json',dict(time=queue.now(),host=socket.gethostname(),worker_host=a.host,pid=os.getpid(),
        gpus=gpus,max_tasks_per_gpu=1,initial_gpu=initial,permitted_existing={g:sorted(v) for g,v in baseline.items()},
        numerical_environment=versions,python=sys.version,protocol='bio-eval-union-v4',command=sys.argv))
    active={};deadline=time.monotonic()+a.hours*3600;failures=0;recent={};laststart=0
    # Recover only this host's own in-flight claims; never interpret another host's PID.
    for p in (ROOT/'claims').glob('*/status.json'):
        st=json.loads(p.read_text())
        if st.get('host')!=a.host or st['state']!='RUNNING':continue
        task=json.loads((ROOT/'tasks'/f'{p.parent.name}.json').read_text())
        cmdline=Path(f'/proc/{st["pid"]}/cmdline')
        argv=cmdline.read_bytes().rstrip(b'\0').decode().split('\0') if cmdline.exists() else []
        if argv!=st['command']:
            ok,why=validate(task);queue.atomic(p,dict(st,state='DONE' if ok else 'FAILED',validation=why));continue
        proc=queue.AdoptedProcess(st['pid'],p.parent/'exit.json')
        active[st['gpu']]=(proc,task,st,(ROOT/'logs'/f'{task["id"]}.log').open('a'))
    while time.monotonic()<deadline or active:
        for g,(proc,t,st,log) in list(active.items()):
            for pid in resource.descendants(proc.pid):recent[pid]=time.monotonic()+60
            rc=proc.poll()
            if rc is None:continue
            log.close()
            try:ok,why=validate(t)
            except Exception as exc:ok,why=False,repr(exc)
            queue.atomic(ROOT/'claims'/t['id']/'status.json',dict(st,state='DONE' if rc==0 and ok else 'FAILED',
                         returncode=rc,validation=why,end=queue.now()))
            if rc or not ok:failures+=1
            del active[g]
        recent={p:expiry for p,expiry in recent.items() if expiry>time.monotonic()}
        gs=resource.gpu_info();processes=resource.gpu_processes();foreign={}
        for g in gpus:
            own=resource.descendants(active[g][0].pid) if g in active else set()
            foreign[g]=[p for p in processes if p['uuid']==gs[g]['uuid'] and p['pid'] not in baseline[g]|own|set(recent)
                        and Path(f'/proc/{p["pid"]}').exists()]
            if foreign[g] and g in active:
                proc,t,st,_=active[g];queue.atomic(ROOT/'claims'/t['id']/'resource_interruption.json',dict(time=queue.now(),foreign=foreign[g]));resource.stop(proc)
        queue.atomic(dest/'status.json',dict(time=queue.now(),pid=os.getpid(),gpus={g:gs[g] for g in gpus},
                     active={g:v[1]['id'] for g,v in active.items()},foreign=foreign,failures=failures))
        if failures>=3 or time.monotonic()>=deadline:
            if not active:break
            time.sleep(10);continue
        pending=[json.loads(p.read_text()) for p in (ROOT/'tasks').glob('*.json') if not (ROOT/'claims'/p.stem).exists()]
        pending.sort(key=lambda t:(0 if 'adaptive_mid_retry2' in t['arm'] else 1,t.get('priority',1),t['order'],t['id']))
        for g in gpus:
            if g in active or foreign[g] or time.monotonic()-laststart<8:continue
            avail=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])//1024
            for t in pending:
                peak=max(queue.gpu_memory_cost(t),9000 if t.get('heavy') else 4200)
                if gs[g]['total']-gs[g]['used']<peak+2500:continue
                if avail<queue.host_memory_cost(t)+20000:continue
                sid=hashlib.sha256(t['cwd'].encode()).hexdigest()[:16]
                manifest=json.loads((ROOT/'provenance'/f'source_{sid}.json').read_text())
                if resource.hash_files(Path(t['cwd']))!=manifest['files']:raise RuntimeError('Source hash changed: '+t['cwd'])
                if 'runtime_adapter_sha256' in t:assert queue.sha(Path(t['cmd'][2]))==t['runtime_adapter_sha256']
                claim=ROOT/'claims'/t['id']
                try:claim.mkdir()
                except FileExistsError:continue
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=t['pythonpath'],DINOV3_ROOT=t['cwd'],
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                    NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_NET='Socket',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0',PYTHONFAULTHANDLER='1')
                cmd=[str(x).replace('{GPU}',str(g)) for x in t['cmd']]
                log=(ROOT/'logs'/f'{t["id"]}.log').open('a')
                proc=subprocess.Popen(cmd,cwd=t['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                st=dict(state='RUNNING',host=a.host,gpu=g,pid=proc.pid,command=cmd,start=queue.now(),
                        source_manifest=str(ROOT/'provenance'/f'source_{sid}.json'),environment=versions)
                queue.atomic(claim/'status.json',st);active[g]=(proc,t,st,log);pending.remove(t);laststart=time.monotonic();break
        time.sleep(5)
    queue.atomic(dest/'exit.json',dict(time=queue.now(),failures=failures))

if __name__=='__main__':main()
