#!/usr/bin/env python3
"""Persistent flock queue. Safe restart, live heartbeat, explicit per-task logs."""
import argparse, json, os, socket, subprocess, sys, time, signal, shutil, fcntl, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
SOURCE=Path(__file__).resolve()
LOADED_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
RELOAD_CHECKED_AT=0
LOCK_FD_ENV='UNIFIED_PAPER_QUEUE_LOCK_FD'

def open_slot_lock(path):
 """Keep the same open file description (and flock) across an idle exec."""
 inherited=os.environ.pop(LOCK_FD_ENV,None)
 if inherited is None:return path.open('a')
 fd=int(inherited);actual=os.fstat(fd);expected=path.stat()
 if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
  os.close(fd)
  raise RuntimeError('Inherited slot lock identity mismatch')
 os.set_inheritable(fd,False)
 return os.fdopen(fd,'a')

def reload_if_changed(lock):
 """Called only with no task claim or child; never stop another process."""
 global RELOAD_CHECKED_AT
 if time.monotonic()-RELOAD_CHECKED_AT<5:return
 RELOAD_CHECKED_AT=time.monotonic()
 try:
  source=SOURCE.read_bytes()
  if hashlib.sha256(source).hexdigest()==LOADED_SHA256:return
  compile(source,str(SOURCE),'exec')
 except (OSError,SyntaxError) as error:
  print('QUEUE_RELOAD_DEFERRED',repr(error),flush=True);return
 fd=lock.fileno();env=os.environ.copy();env[LOCK_FD_ENV]=str(fd)
 os.set_inheritable(fd,True)
 try:
  print('QUEUE_IDLE_RELOAD',os.getpid(),flush=True)
  os.execve(sys.executable,[sys.executable,'-u',str(SOURCE),*sys.argv[1:]],env)
 except OSError as error:
  print('QUEUE_RELOAD_DEFERRED',repr(error),flush=True)
 finally:os.set_inheritable(fd,False)

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def completed_result(path,cache,now=None):
 """Cache only completed immutable outputs; refresh within five minutes.

 Queue edits clear this cache. Pending outputs and the final check after an
 atomic claim always use the filesystem, so a cache miss cannot duplicate work.
 """
 now=time.monotonic()if now is None else now;key=str(path)
 if key in cache and now-cache[key]<300:return True
 if path.exists():cache[key]=now;return True
 cache.pop(key,None);return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=-1);p.add_argument('--cpu',action='store_true');p.add_argument('--slot',default='0');p.add_argument('--python',default=sys.executable);p.add_argument('--host-label',default=socket.gethostname());p.add_argument('--queue-file',type=Path,default=OUT/'queue.json');p.add_argument('--model-kind',choices=('fm','biodino'));a=p.parse_args()
 if not a.cpu and a.gpu<0:p.error('--gpu is required unless --cpu is specified')
 ident=f'{a.host_label}_gpu{a.gpu}_slot{a.slot}'
 from unified_paper_gpu_policy import excluded,worker_disabled
 if not a.cpu and excluded(a.host_label,a.gpu):return
 worker=OUT/'states'/f'worker_{ident}.json'
 signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
 with open_slot_lock(OUT/'locks'/f'worker_{ident}.lock') as wlock:
  try:fcntl.flock(wlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  free_memory=None;memory_checked_at=0;completed_cache={};cached_queue_version=None;host_wait=None
  while True:
   if not a.cpu and excluded(a.host_label,a.gpu):return
   reload_if_changed(wlock)
   if worker_disabled(ident):
    dump(worker,{'host':a.host_label,'pid':os.getpid(),'gpu':a.gpu,'heartbeat':time.time(),'status':'disabled_idle','queue':str(a.queue_file)})
    return
   queue_version=a.queue_file.stat().st_mtime_ns
   if cached_queue_version!=queue_version:completed_cache.clear();cached_queue_version=queue_version
   model_config={m['model']:m for m in json.loads((OUT/'config.json').read_text())['models']}
   model_hosts={name:m.get('execution_host')for name,m in model_config.items()}
   tasks=json.loads(a.queue_file.read_text())['tasks']; ran=False
   # Finish one downstream training seed for each cell before later repeats.
   tasks.sort(key=lambda task: (task.get('seed') not in (None, 0), task.get('seed') or 0))
   for task in tasks:
    if not a.cpu and excluded(a.host_label,a.gpu):return
    reload_if_changed(wlock)
    if a.model_kind and model_config[task['model']]['kind']!=a.model_kind:continue
    # Inventories and dependency paths may be repaired while an earlier task
    # runs. Reload the queue before using any remaining stale task definition.
    if a.queue_file.stat().st_mtime_ns!=queue_version:ran=True;break
    feature_only=a.cpu and task.get('allow_feature_only_remote_host') and task.get('execution_mode')=='deterministic_frozen_feature_probe' and task.get('entrypoint')=='run_unified_paper_lc25000_grouped.py'
    if task.get('blocked_reason') and not (feature_only and task['blocked_reason'].startswith('Main-cluster placeholder;')):continue
    if model_hosts.get(task['model']) and model_hosts[task['model']]!=a.host_label and not feature_only:continue
    if task.get('allowed_hosts') and a.host_label not in task['allowed_hosts'] and not feature_only:continue
    key=task['key'];result=Path(task['result'])
    fm_claim=Path('/mnt/liyixin22/fm/queue/claims')/key
    if fm_claim.exists() or fm_claim.is_symlink():continue
    if completed_result(result,completed_cache):continue
    if any(not Path(p).exists() for p in task.get('requires',[])):continue
    attempt=OUT/'states'/f'{key}_attempts.json'
    prior=json.loads(attempt.read_text()) if attempt.exists() else {'attempts':0}
    if prior['attempts']>=2:continue
    claim_key = task.get('claim_key', key if task.get('entrypoint') == 'run_unified_paper_global.py' else task['model'])
    claim=OUT/'locks'/f'{claim_key}.claimdir'
    if claim.exists():continue
    # Do not fill RAM with Python/CUDA imports while the host cannot admit a
    # fit. The child's locked check handles races between these prefilters.
    from unified_paper_host_admission import launch_ready
    host_fits,host_details=launch_ready(task)
    if not host_fits:
     host_wait=host_details
     dump(worker,{'host':a.host_label,'pid':os.getpid(),'gpu':a.gpu,'heartbeat':time.time(),'status':'waiting_host_memory','queue':str(a.queue_file),'memory':host_details})
     break
    host_wait=None
    # This is only a launch prefilter. Every child performs fresh, locked GPU
    # admission against live allocations and loading reservations. Cache the
    # prefilter to avoid spawning NVML tools once per pending task.
    if not a.cpu:
     if free_memory is None or time.monotonic()-memory_checked_at>=5:
      mem=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
      free_memory,gpu_total=map(int,mem.strip().split(','));memory_checked_at=time.monotonic()
     if a.host_label=='deepcad' and gpu_total-free_memory>=.35*gpu_total:break
     if free_memory<task['min_free_mib']:continue
    try:claim.mkdir()
    except FileExistsError:continue
    dump(claim/'owner.json',{'host':a.host_label,'pid':os.getpid(),'task':key})
    try:
     if a.queue_file.stat().st_mtime_ns!=queue_version:ran=True;break
     if fm_claim.exists() or fm_claim.is_symlink():continue
     if result.exists():continue
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if a.cpu else str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT))
     cmd=[a.python,'-u',str(ROOT/'scripts'/task.get('entrypoint','run_unified_paper_ood.py')),*task['args']]
     log=Path(task['log']);log.parent.mkdir(parents=True,exist_ok=True)
     with log.open('a') as f:
      f.write(f'\nSTART {time.time()} {ident}\n');f.flush()
      proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      dump(claim/'owner.json',{'host':a.host_label,'pid':os.getpid(),'child_pid':proc.pid,'task':key})
      # A child must not outlive its claim if this worker is terminated.
      try:
       while proc.poll() is None:
        dump(worker,{'host':a.host_label,'pid':os.getpid(),'child_pid':proc.pid,'gpu':a.gpu,'task':key,'heartbeat':time.time(),'status':'running','log':str(log)})
        time.sleep(10)
      finally:
       if proc.poll() is None:proc.terminate();proc.wait()
     if proc.returncode!=75:prior['attempts']+=1
     prior.update(exit_code=proc.returncode,host=a.host_label,pid=proc.pid,ended=time.time());dump(attempt,prior)
     print(key,proc.returncode,flush=True);ran=True;free_memory=None
    finally:shutil.rmtree(claim)
   dump(worker,{'host':a.host_label,'pid':os.getpid(),'gpu':a.gpu,'heartbeat':time.time(),'status':'waiting_host_memory' if host_wait else 'waiting','memory':host_wait,'queue':str(a.queue_file)})
   time.sleep(10 if ran else 30)
if __name__=='__main__':main()
