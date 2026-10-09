"""Own training and v4 evaluations share GPUs with explicit peak reservations."""
import collections, fcntl, hashlib, json, os, signal, subprocess, time
from pathlib import Path
import run_deepcad_method_20260927 as resource
import selective_retention_eval_queue_20260923 as q
from run_method_v4_fleet_20260928 import validate

REPO=Path('/mnt/huawei_deepcad/dinov3')
EVAL=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'
ROOT=REPO/'outputs/00_reports/deepcad_method_20260927/coexist'
CONFIG=ROOT/'config.json'

def argv(pid):
    p=Path(f'/proc/{pid}/cmdline')
    try:return p.read_bytes().rstrip(b'\0').decode().split('\0')
    except FileNotFoundError:return []

def cost(t):
    return max(q.gpu_memory_cost(t),9000 if t.get('heavy') else 3500)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    locks=[]
    for path in [EVAL/'eval.lock',ROOT/'manager.lock']:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    config=json.loads(CONFIG.read_text());gpus=config['gpus']
    for g in gpus:
        f=open(f'/tmp/dinov3_deepcad_gpu_{g}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    q.atomic(EVAL/'WORKER.json',dict(time=q.now(),pid=os.getpid(),gpus=gpus,mode='train_test_coexist'))
    trains={};tests={};recent={};failures=0
    for item in config.get('adopt_training',[]):
        assert argv(item['pid'])==item['command'],'Training PID identity changed'
        p=resource.Path(item['output'])
        trains[item['gpu']]=(q.AdoptedProcess(item['pid'],p/'child_exit.json'),item,None)
        q.atomic(p/'coexist_adoption.json',dict(time=q.now(),supervisor=os.getpid(),**item))
    for f in (EVAL/'claims').glob('*/status.json'):
        st=json.loads(f.read_text())
        if st['state']!='RUNNING' or st.get('host') not in (None,'deepcad','inspur'):continue
        t=json.loads((EVAL/'tasks'/f'{f.parent.name}.json').read_text())
        if argv(st['pid'])!=st.get('command'):
            ok,why=validate(t)
            if not ok and (f.parent/'yield_reason.json').exists():
                archive=EVAL/'attempts'/f'{t["id"]}__coexist_reconcile_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True);f.parent.rename(archive)
            else:q.atomic(f,dict(st,state='DONE' if ok else 'FAILED',validation=why))
            continue
        assert st['gpu'] in gpus
        tests[t['id']]=(q.AdoptedProcess(st['pid'],f.parent/'exit.json'),t,st,None)
    def release(tid,reason):
        proc,t,st,log=tests[tid]
        for pid in resource.descendants(proc.pid):recent[pid]=time.monotonic()+90
        q.atomic(EVAL/'claims'/tid/'yield_reason.json',dict(time=q.now(),reason=reason))
        resource.stop(proc)
        if log:log.close()
        ok,why=validate(t)
        if ok:q.atomic(EVAL/'claims'/tid/'status.json',dict(st,state='DONE',validation=why,end=q.now()))
        else:
            dest=EVAL/'attempts'/f'{tid}__coexist_{time.time_ns()}'
            dest.parent.mkdir(exist_ok=True);(EVAL/'claims'/tid).rename(dest)
        del tests[tid]
    while True:
        config=json.loads(CONFIG.read_text())
        for proc,*_ in list(trains.values())+list(tests.values()):
            for pid in resource.descendants(proc.pid):recent[pid]=time.monotonic()+90
        for tid,(proc,t,st,log) in list(tests.items()):
            rc=proc.poll()
            if rc is None:continue
            if log:log.close()
            try:ok,why=validate(t)
            except Exception as exc:ok,why=False,repr(exc)
            q.atomic(EVAL/'claims'/tid/'status.json',dict(st,state='DONE' if rc==0 and ok else 'FAILED',returncode=rc,validation=why,end=q.now()))
            if rc or not ok:failures+=1
            del tests[tid]
        for g,(proc,item,log) in list(trains.items()):
            rc=proc.poll()
            if rc is None:continue
            out=Path(item['output']);expected=out/f'ckpt/{item["end"]}/checkpoint.pth'
            ok=rc==0 and expected.exists()
            q.atomic(out/'exit.json',dict(time=q.now(),returncode=0 if ok else rc or 1,
                    reason=None if ok else 'training exited without expected complete checkpoint'))
            if log:log.close()
            del trains[g]
        gs=resource.gpu_info();recent={p:v for p,v in recent.items() if v>time.monotonic()}
        own=set()
        for proc,*_ in list(trains.values())+list(tests.values()):own|=resource.descendants(proc.pid)
        foreign={g:[p for p in resource.gpu_processes() if p['uuid']==gs[g]['uuid'] and p['pid'] not in own|set(recent) and Path(f'/proc/{p["pid"]}').exists()] for g in gpus}
        for g in gpus:
            if foreign[g]:
                for tid,v in list(tests.items()):
                    if v[2]['gpu']==g:release(tid,'foreign process entered shared GPU')
                if g in trains:
                    proc,item,log=trains[g]
                    q.atomic(Path(item['output'])/'resource_interruption.json',dict(time=q.now(),foreign=foreign[g]))
                    resource.stop(proc)
                continue
            reserve=config['train_reserve_mib'].get(str(g),30000)
            budget=gs[g]['total']-reserve-config.get('headroom_mib',1500)
            # Preserve the expensive native tracking job when rebalancing.
            selected=sorted([tid for tid,v in tests.items() if v[2]['gpu']==g],key=lambda tid:(0 if tests[tid][1].get('heavy') else 1,tests[tid][2].get('start','')))
            committed=0;slots=0
            for tid in selected:
                peak=cost(tests[tid][1])
                if committed+peak<=budget and slots<config.get('tests_per_gpu',2):committed+=peak;slots+=1
                else:release(tid,'user requested training reservation and test redistribution')
        # Launch only explicitly prepared training manifests, one per GPU.
        for item in config.get('training_jobs',[]):
            g=item['gpu'];out=Path(item['output'])
            if g in trains or foreign[g] or (out/'process.json').exists():continue
            used=resource.gpu_info()[g]['used']
            if gs[g]['total']-used<config['train_reserve_mib'][str(g)]+config.get('headroom_mib',1500):continue
            if not item.get('admitted',False):continue
            source=Path(item['source']);manifest=json.loads((out/'launch_manifest.json').read_text())
            assert resource.hash_files(source)==manifest['source_hashes'],'Training source changed'
            log=(out/'console.log').open('a',buffering=1)
            env=os.environ.copy();env.update(item['environment'])
            proc=subprocess.Popen(item['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            q.atomic(out/'process.json',dict(time=q.now(),supervisor=os.getpid(),torchrun=proc.pid))
            trains[g]=(proc,item,log)
        pending=[json.loads(p.read_text()) for p in (EVAL/'tasks').glob('*.json') if not (EVAL/'claims'/p.stem).exists()]
        pending.sort(key=lambda t:(0 if 'adaptive_mid_retry2' in t['arm'] else 1,t['priority'],t['order'],t['id']))
        for g in gpus:
            if foreign[g] or failures>=6:continue
            selected=[v for v in tests.values() if v[2]['gpu']==g]
            if len(selected)>=config.get('tests_per_gpu',2):continue
            committed=sum(cost(v[1]) for v in selected)
            budget=gs[g]['total']-config['train_reserve_mib'][str(g)]-config.get('headroom_mib',1500)
            current=resource.gpu_info()[g]
            for t in pending:
                peak=cost(t)
                if committed+peak>budget or current['total']-current['used']<peak+config.get('headroom_mib',1500):continue
                avail=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])//1024
                if avail<q.host_memory_cost(t)+20000:continue
                sid=hashlib.sha256(t['cwd'].encode()).hexdigest()[:16]
                assert resource.hash_files(Path(t['cwd']))==json.loads((EVAL/'provenance'/f'source_{sid}.json').read_text())['files']
                if 'runtime_adapter_sha256' in t:assert q.sha(Path(t['cmd'][2]))==t['runtime_adapter_sha256']
                claim=EVAL/'claims'/t['id']
                try:claim.mkdir()
                except FileExistsError:continue
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=t['pythonpath'],DINOV3_ROOT=t['cwd'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_NET='Socket',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0')
                cmd=[str(x).replace('{GPU}',str(g)) for x in t['cmd']]
                log=(EVAL/'logs'/f'{t["id"]}.log').open('a')
                proc=subprocess.Popen(cmd,cwd=t['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                st=dict(state='RUNNING',host='deepcad',gpu=g,pid=proc.pid,command=cmd,start=q.now(),coexist=True)
                q.atomic(claim/'status.json',st);tests[t['id']]=(proc,t,st,log);pending.remove(t);break
        status=dict(time=q.now(),supervisor=os.getpid(),gpus=resource.gpu_info(),
                    training={g:dict(pid=v[0].pid,output=v[1]['output']) for g,v in trains.items()},
                    tests={tid:dict(gpu=v[2]['gpu'],pid=v[0].pid) for tid,v in tests.items()},foreign=foreign,failures=failures)
        q.atomic(ROOT/'status.json',status)
        q.atomic(EVAL/'RESOURCE_STATUS.json',dict(time=status['time'],supervisor=os.getpid(),gpus={g:status['gpus'][g] for g in gpus},running=dict(collections.Counter(v[2]['gpu'] for v in tests.values())),foreign=foreign,failures=failures,coexist=True))
        time.sleep(10)

if __name__=='__main__':main()
