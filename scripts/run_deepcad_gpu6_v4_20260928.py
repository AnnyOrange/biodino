"""Persistent full-v4 consumer for new GPU3/5/7 experiments, on a selected free GPU."""
import argparse,fcntl,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import run_deepcad_method_20260927 as resource
import selective_retention_eval_queue_20260923 as q
from run_method_v4_fleet_20260928 import validate

REPO=Path('/mnt/huawei_deepcad/dinov3')
ROOT=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'
DEST=ROOT/'fleet/deepcad_gpu6'
HOST='deepcad_gpu6';GPU=6
DEFAULT=dict(target_fraction=.55,min_tasks=5,max_tasks=12,headroom_mib=2500,launch_interval_seconds=20,
             preferred_arms=['ck_c_e12687_formal','ck_k_e12687_formal','adaptive_continue_resume'],
             fallback='unfinished same-campaign comparisons',protocol='full bio-eval-union-v4, 56 cells per checkpoint')

def argv(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    except FileNotFoundError:return []

def priority(t,config):
    preferred=any(t['arm'].startswith(p) for p in config['preferred_arms'])
    continuation=t['arm'].startswith('adaptive_continue')
    return (0 if preferred else 1 if continuation else 2,t.get('priority',1),t['order'],t['id'])

def main():
    global GPU,HOST,DEST
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=range(8),default=6)
    args=parser.parse_args();GPU=args.gpu;HOST=f'deepcad_gpu{GPU}';DEST=ROOT/'fleet'/HOST
    DEST.mkdir(parents=True,exist_ok=True);locks=[]
    for path in [DEST/'worker.lock',Path(f'/tmp/dinov3_deepcad_gpu_{GPU}.lock'),Path(f'/tmp/dinov3_method_fleet_gpu_{GPU}.lock')]:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    config_path=DEST/'config.json'
    if not config_path.exists():q.atomic(config_path,DEFAULT)
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn']}
    q.atomic(DEST/'manifest.json',dict(time=q.now(),pid=os.getpid(),host=HOST,gpus=[GPU],configuration=DEFAULT,
             command=sys.argv,numerical_environment=versions,script_sha256=q.sha(Path(__file__))))
    active={};recent={};failures=0;laststart=0
    for p in (ROOT/'claims').glob('*/status.json'):
        st=json.loads(p.read_text())
        if st.get('host')!=HOST or st['state']!='RUNNING':continue
        t=json.loads((ROOT/'tasks'/f'{p.parent.name}.json').read_text())
        if argv(st['pid'])!=st['command']:
            ok,why=validate(t);q.atomic(p,dict(st,state='DONE' if ok else 'FAILED',validation=why));continue
        active[t['id']]=(q.AdoptedProcess(st['pid'],p.parent/'exit.json'),t,st,None)
    while True:
        cfg=json.loads(config_path.read_text());now=time.monotonic()
        for tid,(proc,t,st,log) in list(active.items()):
            for pid in resource.descendants(proc.pid):recent[pid]=now+90
            rc=proc.poll()
            if rc is None:continue
            if log:log.close()
            try:ok,why=validate(t)
            except Exception as exc:ok,why=False,repr(exc)
            q.atomic(ROOT/'claims'/tid/'status.json',dict(st,state='DONE' if rc==0 and ok else 'FAILED',returncode=rc,validation=why,end=q.now()))
            resource_yield=(ROOT/'claims'/tid/'resource_interruption.json').exists()
            if (rc or not ok) and not (rc==-15 and resource_yield):failures+=1
            del active[tid]
        recent={p:v for p,v in recent.items() if v>now}
        gs=resource.gpu_info()[GPU];own=set()
        for proc,*_ in active.values():own|=resource.descendants(proc.pid)
        foreign=[p for p in resource.gpu_processes() if p['uuid']==gs['uuid'] and p['pid'] not in own|set(recent) and Path(f'/proc/{p["pid"]}').exists()]
        if foreign:
            for proc,t,st,log in active.values():
                q.atomic(ROOT/'claims'/t['id']/'resource_interruption.json',dict(time=q.now(),foreign=foreign))
                resource.stop(proc)
        pending=[json.loads(p.read_text()) for p in (ROOT/'tasks').glob('*.json') if not (ROOT/'claims'/p.stem).exists()]
        pending.sort(key=lambda t:priority(t,cfg))
        committed=sum(q.gpu_memory_cost(v[1]) for v in active.values())
        reason='target reached'
        need=gs['used']/gs['total']<cfg['target_fraction'] or len(active)<cfg['min_tasks']
        if foreign:reason='foreign GPU occupancy; owned tests yield'
        elif failures>=6:reason='failure guard; inspect logs'
        elif not pending:reason='waiting for new teacher checkpoints or other pending tasks'
        elif len(active)>=cfg['max_tasks']:reason='concurrency limit'
        elif need and now-laststart<cfg['launch_interval_seconds']:reason='staggered startup'
        elif need:
            reason='GPU peak reservations or host memory limit'
            if not foreign and failures<6 and len(active)<cfg['max_tasks'] and now-laststart>=cfg['launch_interval_seconds']:
                available=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])//1024
                rss_kib=0
                for pid in own:
                    try:
                        fields=Path(f'/proc/{pid}/status').read_text().split('VmRSS:')
                        if len(fields)>1:rss_kib+=int(fields[1].split()[0])
                    except FileNotFoundError:pass
                host_reserved=sum(q.host_memory_cost(v[1]) for v in active.values())
                future_host=max(0,host_reserved-rss_kib//1024)
                for t in pending:
                    peak=q.gpu_memory_cost(t)
                    if committed+peak>gs['total']-cfg['headroom_mib']:continue
                    if gs['total']-gs['used']<peak+cfg['headroom_mib']:continue
                    if available<future_host+q.host_memory_cost(t)+25000:continue
                    sid=hashlib.sha256(t['cwd'].encode()).hexdigest()[:16]
                    provenance=ROOT/'provenance'/f'source_{sid}.json'
                    assert resource.hash_files(Path(t['cwd']))==json.loads(provenance.read_text())['files'],'Evaluator source changed'
                    if 'runtime_adapter_sha256' in t:assert q.sha(Path(t['cmd'][2]))==t['runtime_adapter_sha256']
                    claim=ROOT/'claims'/t['id']
                    try:claim.mkdir()
                    except FileExistsError:continue
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(GPU),PYTHONPATH=t['pythonpath'],DINOV3_ROOT=t['cwd'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_NET='Socket',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0',PYTHONFAULTHANDLER='1')
                    cmd=[str(x).replace('{GPU}',str(GPU)) for x in t['cmd']]
                    log=(ROOT/'logs'/f'{t["id"]}.log').open('a')
                    proc=subprocess.Popen(cmd,cwd=t['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    st=dict(state='RUNNING',host=HOST,gpu=GPU,pid=proc.pid,command=cmd,start=q.now(),source_manifest=str(provenance),environment=versions)
                    q.atomic(claim/'status.json',st);active[t['id']]=(proc,t,st,log);laststart=time.monotonic();reason='launched '+t['id'];break
        q.atomic(DEST/'status.json',dict(time=q.now(),pid=os.getpid(),gpu=GPU,memory=resource.gpu_info()[GPU],target_fraction=cfg['target_fraction'],
                 active={tid:dict(pid=v[0].pid,arm=v[1]['arm'],family=v[1]['family']) for tid,v in active.items()},
                 pending_count=len(pending),foreign=foreign,failures=failures,scheduling_reason=reason,peak_committed_mib=committed))
        time.sleep(5)

if __name__=='__main__':main()
