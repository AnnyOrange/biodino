"""Cross-process GPU reservations for queued evaluation children before loading.

Reservations end only after verified process exit, never due to elapsed time.
Existing unrelated allocations remain included in NVML memory.used.
"""
import atexit,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
QUEUES=['bone_age_queue.json','ood_repair_queue.json','rsna_queue.json','film_queue.json','queue.json','global_queue.json','lc25000_grouped_queue.json','native_detection_queue.json','nucls_queue.json','segmentation_queue.json','segmentation_recovery_queue.json','segmentation_cache_queue.json','detection_queue.json','rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','additional_retrieval_pilot_queue.json','bbbc038_official_queue.json','ctc_queue.json','retrieval_queue.json']
_release_callback=None
_execution_guard=None

def release_current_reservation():
    """Release the loading/compute peak after a job has entered CPU-only work.

    Residual CUDA-context memory remains in NVML memory.used and is still counted.
    Callers must make no subsequent CUDA operation requiring model allocations.
    """
    if _release_callback is not None:_release_callback('cpu_only')

def matching_job():
    entry=Path(sys.argv[0]).name
    for name in QUEUES:
        p=OUT/name
        if not p.exists():continue
        for job in json.loads(p.read_text()).get('tasks',[]):
            if job.get('entrypoint','run_unified_paper_ood.py')==entry and job.get('args',[])==sys.argv[1:]:return job
    return None

def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return {}

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]

def alive(pid,started):
    try:return bool(Path(f'/proc/{pid}/cmdline').read_bytes()) and start_ticks(pid)==started
    except (FileNotFoundError,ProcessLookupError):return False

def admission_budget(used,expected,reservations,memory,pid,total,limit_fraction=.80):
    reserve=sum(max(0,v['expected_peak_mib']-memory.get(int(k),0)) for k,v in reservations.items() if int(k)!=pid)
    additional=max(0,expected-memory.get(pid,0))
    return used+reserve+additional<=limit_fraction*total,reserve

def owns_current_fit_claim(job):
    if job.get('claim_schema')!='per_fit_after_locked_selection_v1':return True
    owner_path=OUT/'locks'/f"{job['claim_key']}.claimdir/owner.json"
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    host=parent[parent.index(b'--host-label')+1].decode()if b'--host-label'in parent else None
    if host is None:return False
    for _ in range(20):
        owner=read(owner_path)
        if owner.get('child_pid')==os.getpid()and owner.get('pid')==os.getppid()and owner.get('task')==job['key']and owner.get('host')==host:return True
        if owner.get('child_pid'):return False
        time.sleep(.1)
    return False

def release_execution_guard():
    global _execution_guard
    if _execution_guard is not None:
        path,raw=_execution_guard
        try:
            if os.readlink(path)==raw:path.unlink()
        except FileNotFoundError:pass
        _execution_guard=None

def acquire_execution_guard(job):
    """One process per task across historical model-wide and per-task claims."""
    global _execution_guard
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    host=parent[parent.index(b'--host-label')+1].decode()if b'--host-label'in parent else socket.gethostname()
    path=OUT/'locks'/f"execution_{job['key']}.claimdir"
    raw=json.dumps(dict(host=host,pid=os.getpid(),child_pid=os.getpid(),task=job['key'],start_ticks=start_ticks(os.getpid()),token=time.time_ns(),kind='atomic_execution_guard'),sort_keys=True)
    # This NFS mount does not propagate flock between hosts. Atomic symlink
    # creation publishes ownership with no empty-directory/owner-write gap.
    try:os.symlink(raw,path)
    except FileExistsError:return False
    _execution_guard=(path,raw)
    aliases={job['key'],job.get('claim_key',job['model']),job['model'],'nucls_'+job['model'],job['key'].rsplit('__seed',1)[0]}
    # A process started before this guard existed may own a different legacy
    # claim. Its verified-dead recovery remains the controller's responsibility.
    # New contenders exit on the atomic marker; allow parents to clear claims.
    for attempt in range(16):
        others=[]
        for alias in aliases:
            owner=read(OUT/'locks'/f'{alias}.claimdir/owner.json')
            own=owner.get('child_pid')==os.getpid()and owner.get('pid')==os.getppid()and owner.get('host')==host
            if owner.get('task')==job['key']and owner.get('child_pid')and not own:others.append(owner)
        if not others:atexit.register(release_execution_guard);return True
        if attempt<15:time.sleep(1)
    print('DEFER EXISTING TASK CLAIM',job['key'],[(v.get('host'),v.get('child_pid'))for v in others],flush=True)
    release_execution_guard();return False

def reserve_for_current_command():
    global _release_callback
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    if device is not None and device.isdigit():
        from unified_paper_gpu_policy import excluded
        parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
        host=parent[parent.index(b'--host-label')+1].decode()if b'--host-label'in parent else socket.gethostname()
        if excluded(host,int(device)):
            print('GPU reserved by user; evaluation admission prohibited',host,device,flush=True)
            raise SystemExit(75)
    job=matching_job()
    if job is None:return  # Direct validation commands are not queue submissions.
    if not acquire_execution_guard(job):
        print('DEFER DUPLICATE TASK; no model loaded',job['key'],flush=True)
        raise SystemExit(75)
    if not owns_current_fit_claim(job):
        print('DEFER STALE CLAIM; no model loaded',job['key'],flush=True)
        raise SystemExit(75)
    feature_only=device in (None,'','-1') and job.get('allow_feature_only_remote_host') and job.get('execution_mode')=='deterministic_frozen_feature_probe' and job.get('entrypoint')=='run_unified_paper_lc25000_grouped.py'
    if job.get('blocked_reason') and not (feature_only and job['blocked_reason'].startswith('Main-cluster placeholder;')):raise ValueError('Queue task is blocked: '+job['blocked_reason'])
    # A persistent worker can have read its queue before a dependency repair.
    # Recheck current dependencies before reserving GPU memory or loading weights.
    dependency_notice=0
    while any(not Path(p).exists() for p in job.get('requires',[])):
        if time.monotonic()-dependency_notice>60:
            print('WAIT CURRENT TASK DEPENDENCIES',job['key'],flush=True)
            dependency_notice=time.monotonic()
        time.sleep(10)
        job=matching_job() or job
        if job.get('blocked_reason'):raise ValueError('Queue task is blocked: '+job['blocked_reason'])
    if job.get('selection_locked_sha256'):
        if hashlib.sha256(Path(job['selection_locked']).read_bytes()).hexdigest()!=job['selection_locked_sha256']:
            raise ValueError('Locked validation selection changed before independent fit')
    from unified_paper_host_admission import reserve_for_current_command as reserve_host, launch_ready as host_ready
    reserve_host(job)
    if device is None or device.strip() in ['', '-1']:return
    if ',' in device:raise ValueError('Unified single-GPU queue expected one visible device')
    paired_deepcad_gpu=host=='deepcad'
    limit_fraction=.35 if paired_deepcad_gpu else .80
    max_active=2 if paired_deepcad_gpu else 8
    values=subprocess.check_output(['nvidia-smi',f'--id={device}','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,total=values[0].strip(),int(values[1]);directory=OUT/'gpu_reservations';directory.mkdir(exist_ok=True)
    path=directory/f'{uuid}.json';lock_path=directory/f'{uuid}.lock';pid=os.getpid();last_notice=0
    # Existing task-specific free-memory estimates include a safety margin.
    # Retain at least20% of the card and account for every not-yet-loaded peer.
    expected=max(2048.,float(job.get('min_free_mib',8192))-.20*total)
    expected=max(expected,float(job.get('expected_peak_mib',0)))
    observations=read(OUT/'gpu_memory_observations.json').get('by_model_task',{})
    observed=observations.get(f"{job['model']}::{job.get('task','ood')}",{}).get('peak_observed_mib',0)
    expected=max(expected,observed*1.15)
    while True:
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=read(path);reservations={k:v for k,v in state.get('reservations',{}).items() if alive(int(k),v.get('start_ticks'))}
            used=float(subprocess.check_output(['nvidia-smi',f'--id={device}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            apps=subprocess.check_output(['nvidia-smi',f'--id={device}','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            memory={}
            for line in apps.splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==2 and all(x.isdigit() for x in fields):memory[int(fields[0])]=float(fields[1])
            fits,reserve=admission_budget(used,expected,reservations,memory,pid,total,limit_fraction)
            active=set(map(int,reservations))
            for other_pid in memory:
                try:cmd=Path(f'/proc/{other_pid}/cmdline').read_bytes()
                except (FileNotFoundError,ProcessLookupError):continue
                if b'run_unified_paper_' in cmd or b'run_new_fm_v4_segmentation_' in cmd:active.add(other_pid)
            fits=fits and len(active|{pid})<=max_active
            # A job may have waited for its GPU for hours after reserving RAM.
            # Recheck system headroom before it actually starts loading.
            host_fits,host_details=host_ready(job,pid=pid)
            fits=fits and host_fits
            record={'pid':pid,'start_ticks':start_ticks(pid),'host':socket.gethostname(),'gpu_uuid':uuid,'task':job['key'],'expected_peak_mib':expected,'current_used_mib':used,'other_loading_reserve_mib':reserve,'total_mib':total,'updated':time.time()}
            write(directory/f'process_{socket.gethostname()}_{pid}.json',{**record,'status':'admitted' if fits else 'waiting_for_capacity'})
            if fits:
                reservations[str(pid)]=record;write(path,{'reservations':reservations,'updated':time.time()})
            else:write(path,{'reservations':reservations,'updated':time.time()})
        if fits:break
        if time.monotonic()-last_notice>60:
            print(f"GPU ADMISSION WAIT {job['key']} used={used:.0f} pending_reserve={reserve:.0f} requested_peak={expected:.0f} limit={limit_fraction*total:.0f} MiB host_ram={host_details}",flush=True);last_notice=time.monotonic()
        time.sleep(10)
    print(f"GPU ADMITTED {job['key']} reserved_peak={expected:.0f} MiB; limit={limit_fraction*total:.0f} MiB; live loading peers counted",flush=True)
    def release(status='exited'):
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=read(path);state.get('reservations',{}).pop(str(pid),None);write(path,state)
        write(directory/f'process_{socket.gethostname()}_{pid}.json',{**record,'status':status,'updated':time.time()})
    _release_callback=release
    atexit.register(release)
