"""Owned-process supervisor for the user-authorized September 27 continuation."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone

REPO=Path('/mnt/huawei_deepcad/dinov3')
ROOT=REPO/'outputs/01_training_runs/hs6_l5_deepcad_method_20260927'
OLD=REPO/'outputs/01_training_runs/hs6_l5_selective_retention_20260923/adaptive_formal'
PY='/home/deepcad/anaconda3/envs/dinov3/bin/python'

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(data,indent=2,ensure_ascii=False));p.replace(path)
def gpu_info():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(x[0]):dict(uuid=x[1].strip(),used=int(x[2]),total=int(x[3]),util=int(x[4])) for line in text.splitlines() if (x:=line.split(','))}
def gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [dict(uuid=x[0].strip(),pid=int(x[1]),memory=x[2].strip()) for line in text.splitlines() if (x:=line.split(',')) and len(x)==3]
def descendants(root):
    parents={}
    # Processes can exit during directory traversal, before glob reaches stat.
    # Enumerate names first, and handle disappearance inside the protected read.
    for name in os.listdir('/proc'):
        if not name.isdecimal():continue
        p=Path('/proc')/name/'stat'
        try:parents[int(p.parent.name)]=int(p.read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    found={root}
    while True:
        more={pid for pid,parent in parents.items() if parent in found}
        if more<=found:return found
        found|=more
def stop(child):
    owned=descendants(child.pid)
    for pid in sorted(owned,reverse=True):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        for pid in owned|descendants(child.pid):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        child.wait()
def hash_files(source):
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.suffix in ('.py','.yaml','.json','.md','.toml','.sh')
            and p.name!='SOURCE_FINGERPRINT.json'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',default='adaptive_continue')
    ap.add_argument('--gpus',default='0,2');ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--end',type=int,default=20007);ap.add_argument('--tag',default='formal')
    ap.add_argument('--port',type=int,default=32727);ap.add_argument('--fresh-mid',action='store_true')
    ap.add_argument('--resume-from',help='Existing full checkpoint; resume in a separate output directory')
    ap.add_argument('--checkpoint-period',type=int,default=488)
    ap.add_argument('--source',default='/mnt/huawei_deepcad/dinov3_adaptive_continue_v2_snapshot_20260927')
    args=ap.parse_args();gpus=[int(g) for g in args.gpus.split(',')]
    assert not (args.fresh_mid and args.resume_from)
    assert len(gpus) in (1,2) and set(gpus)<=set([0,2,3,4,5])
    assert 1024%(len(gpus)*args.batch)==0
    out=ROOT/(args.arm+'_'+args.tag);out.mkdir(parents=True,exist_ok=True)
    # Independent launch/adoption may reach the same paired arm. Serialize by
    # output, then reuse only a successful checkpoint at the requested end.
    run_lock=(out/'supervisor.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX)
    if (out/'exit.json').exists():
        previous=json.loads((out/'exit.json').read_text())
        completed=out/f'eval/training_{args.end}/teacher_checkpoint.pth'
        if previous['returncode']==0 and completed.exists():
            print('ALREADY_COMPLETE',str(out),flush=True);return 0
    locks=[]
    for gpu in gpus:
        f=open(f'/tmp/dinov3_deepcad_gpu_{gpu}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    running=gpu_info();occupied={p['uuid'] for p in gpu_processes()}
    busy=[g for g in gpus if running[g]['used']>1024 or running[g]['uuid'] in occupied]
    if busy:raise RuntimeError(f'Refusing occupied GPUs: {busy}')
    source=Path(args.source);fingerprint=hash_files(source)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH=str(source),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_CACHE_DIR=str(out/'compiler_cache'))
    base=json.loads((OLD/'launch_manifest.json').read_text())['command']
    overrides={s.split('=',1)[0]:s.split('=',1)[1] for s in base[base.index('compute_precision.distributed_mode=ddp'):]}
    overrides.update({'train.batch_size_per_gpu':str(args.batch),'train.num_workers':'4',
        'optim.gradient_accumulation_steps':str(1024//(len(gpus)*args.batch)),
        'train.max_updates':str(args.end+1),'train.start_iteration_override':'null',
        'checkpointing.max_to_keep':'100','train.checkpointing_blocks':'24'})
    assert args.checkpoint_period>0
    overrides['checkpointing.period']=str(args.checkpoint_period)
    start=15128
    if args.fresh_mid:
        start=20008
        anchor=overrides['student.resume_from_teacher_chkpt'].replace('training_12687','training_20007')
        overrides.update({'student.resume_from_teacher_chkpt':anchor,'train.start_iteration_override':str(start)})
    else:
        resume=Path(args.resume_from).resolve() if args.resume_from else OLD/'ckpt/15127/checkpoint.pth'
        assert resume.is_file(),resume
        resume_step=int(resume.parent.name);start=resume_step+1
        initial=out/'ckpt'/str(resume_step);initial.mkdir(parents=True,exist_ok=True)
        link=initial/'checkpoint.pth'
        if not link.exists():link.symlink_to(resume)
    cmd=[PY,'-m','torch.distributed.run',f'--nproc_per_node={len(gpus)}',f'--master_port={args.port}',
         'dinov3/train/train.py']
    if args.fresh_mid:cmd+=['--no-resume']
    cmd+=['--config-file','dinov3/configs/train/microscopy_continual_vitl16.yaml','--output-dir',str(out),'--seed','0']
    cmd += [k+'='+v for k,v in overrides.items()]
    import torch, sklearn, numpy
    versions=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,sklearn=sklearn.__version__)
    manifest=dict(time=now(),host=socket.gethostname(),gpus=gpus,batch=args.batch,effective_batch=1024,
                  start=start,end=args.end,source=str(source),command=cmd,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF']},
                  source_hashes=fingerprint,versions=versions,initial_gpu=running,
                  initialization='fresh from M teacher' if args.fresh_mid else f'full model+optimizer+EMA+recovery resume from {resume}',
                  seed=0,full_v4_aggregate_allowed=False)
    atomic(out/'launch_manifest.json',manifest)
    log=(out/'console.log').open('a',buffering=1)
    child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(out/'process.json',dict(supervisor=os.getpid(),torchrun=child.pid,time=now()))
    def interrupted(sig,frame):stop(child);raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    reason=None;start_time=time.monotonic()
    while child.poll() is None:
        gs=gpu_info();owned=descendants(child.pid);uuids={gs[g]['uuid'] for g in gpus}
        foreign=[p for p in gpu_processes() if p['uuid'] in uuids and p['pid'] not in owned]
        stat=dict(time=now(),elapsed_seconds=time.monotonic()-start_time,gpus={g:gs[g] for g in gpus},owned_pids=sorted(owned),foreign=foreign)
        atomic(out/'telemetry.json',stat)
        with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(stat)+'\n')
        if foreign:reason='foreign process entered selected GPU';stop(child);break
        if (out/'STOP_REQUESTED').exists():reason='STOP_REQUESTED';stop(child);break
        # User-wide RSS limit avoids affecting other experiments on the borrowed host.
        cg=Path(f'/sys/fs/cgroup/memory/user.slice/user-{os.getuid()}.slice/memory.stat')
        if cg.exists():
            usage=dict(x.split() for x in cg.read_text().splitlines())
            if int(usage.get('total_rss',0))>340_000_000_000:reason='user RSS guard';stop(child);break
        time.sleep(15)
    rc=child.wait();log.close()
    atomic(out/'exit.json',dict(returncode=rc,time=now(),reason=reason))
    print(json.dumps(dict(output=str(out),returncode=rc,reason=reason)),flush=True)
    return rc

if __name__=='__main__':raise SystemExit(main())
