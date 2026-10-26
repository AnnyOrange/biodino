#!/usr/bin/env python3
"""Persistent flock queue. Safe restart, live heartbeat, explicit per-task logs."""
import argparse, json, os, socket, subprocess, sys, time, signal, shutil, fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
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
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=-1);p.add_argument('--cpu',action='store_true');p.add_argument('--slot',default='0');p.add_argument('--python',default=sys.executable);p.add_argument('--host-label',default=socket.gethostname());p.add_argument('--queue-file',type=Path,default=OUT/'queue.json');a=p.parse_args()
 if not a.cpu and a.gpu<0:p.error('--gpu is required unless --cpu is specified')
 ident=f'{a.host_label}_gpu{a.gpu}_slot{a.slot}'
 from unified_paper_gpu_policy import excluded
 if not a.cpu and excluded(a.host_label,a.gpu):return
 worker=OUT/'states'/f'worker_{ident}.json'
 signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
 with (OUT/'locks'/f'worker_{ident}.lock').open('a') as wlock:
  try:fcntl.flock(wlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  free_memory=None;memory_checked_at=0;completed_cache={};cached_queue_version=None
  while True:
   if not a.cpu and excluded(a.host_label,a.gpu):return
   queue_version=a.queue_file.stat().st_mtime_ns
   if cached_queue_version!=queue_version:completed_cache.clear();cached_queue_version=queue_version
   model_hosts={m['model']:m.get('execution_host')for m in json.loads((OUT/'config.json').read_text())['models']}
   tasks=json.loads(a.queue_file.read_text())['tasks']; ran=False
   for task in tasks:
    if not a.cpu and excluded(a.host_label,a.gpu):return
    # Inventories and dependency paths may be repaired while an earlier task
    # runs. Reload the queue before using any remaining stale task definition.
    if a.queue_file.stat().st_mtime_ns!=queue_version:ran=True;break
    feature_only=a.cpu and task.get('allow_feature_only_remote_host') and task.get('execution_mode')=='deterministic_frozen_feature_probe' and task.get('entrypoint')=='run_unified_paper_lc25000_grouped.py'
    if task.get('blocked_reason') and not (feature_only and task['blocked_reason'].startswith('Main-cluster placeholder;')):continue
    if model_hosts.get(task['model']) and model_hosts[task['model']]!=a.host_label and not feature_only:continue
    if task.get('allowed_hosts') and a.host_label not in task['allowed_hosts'] and not feature_only:continue
    key=task['key'];result=Path(task['result'])
    if completed_result(result,completed_cache):continue
    if any(not Path(p).exists() for p in task.get('requires',[])):continue
    attempt=OUT/'states'/f'{key}_attempts.json'
    prior=json.loads(attempt.read_text()) if attempt.exists() else {'attempts':0}
    if prior['attempts']>=2:continue
    claim=OUT/'locks'/f'{task.get("claim_key",task["model"])}.claimdir'
    if claim.exists():continue
    # This is only a launch prefilter. Every child performs fresh, locked GPU
    # admission against live allocations and loading reservations. Cache the
    # prefilter to avoid spawning NVML tools once per pending task.
    if not a.cpu:
     if free_memory is None or time.monotonic()-memory_checked_at>=5:
      mem=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
      free_memory=int(mem.strip());memory_checked_at=time.monotonic()
     if free_memory<task['min_free_mib']:continue
    try:claim.mkdir()
    except FileExistsError:continue
    dump(claim/'owner.json',{'host':a.host_label,'pid':os.getpid(),'task':key})
    try:
     if a.queue_file.stat().st_mtime_ns!=queue_version:ran=True;break
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
   dump(worker,{'host':a.host_label,'pid':os.getpid(),'gpu':a.gpu,'heartbeat':time.time(),'status':'waiting','queue':str(a.queue_file)})
   time.sleep(10 if ran else 30)
if __name__=='__main__':main()
