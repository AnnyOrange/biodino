"""Retire an identified duplicate and replay affected CTC predictions safely.

Original jobs finish naturally. Outputs are archived only after all task claims
and verified live processes disappear. A verified epoch50 head is reused;
ambiguous training artifacts are retained in the archive and retrained.
"""
import argparse,contextlib,fcntl,hashlib,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from unified_paper_transport import ssh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
PLAN=OUT/'duplicate_execution_recovery.json'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def write(p,d):
 t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def legacy_claim_owners():
 # Execution claims are symlinks containing JSON, not directory targets.
 # Following their JSON payload can raise ENAMETOOLONG during glob traversal.
 with os.scandir(OUT/'locks')as entries:
  for entry in entries:
   if entry.name.endswith('.claimdir')and entry.is_dir(follow_symlinks=False):
    owner=read(Path(entry.path)/'owner.json')
    if owner:yield owner
@contextlib.contextmanager
def execution_lock(job):
 from unified_paper_gpu_admission import acquire_execution_guard,release_execution_guard
 acquired=acquire_execution_guard(job)
 try:yield acquired
 finally:
  if acquired:release_execution_guard()
def retire_target(item):
 target=item.get('stop_target')
 if not target or target.get('status')=='retired':return
 code='''import pathlib,json,os,signal,sys
r=json.loads(sys.argv[1]);p=pathlib.Path('/proc/'+str(r['pid']));out=pathlib.Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/unified_paper')
if not p.exists()or not(p/'cmdline').read_bytes():print(json.dumps(dict(status='retired',already_exited=True)));raise SystemExit(0)
assert(p/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']
args=(p/'cmdline').read_bytes().split(bytes([0]));assert any(v.endswith(b'run_unified_paper_ctc.py')for v in args)
owner=json.loads((out/'locks'/r['claim']/'owner.json').read_text());assert owner['task']==r['task']and owner['child_pid']==r['pid']
had_compute=False
for f in(out/'gpu_reservations').glob('process_*_'+str(r['pid'])+'.json'):
 d=json.loads(f.read_text())
 if d.get('start_ticks')==r['start_ticks']and d.get('task')==r['task']:had_compute|=d.get('status')!='waiting_for_capacity'
os.kill(r['pid'],signal.SIGTERM);print(json.dumps(dict(status='termination_requested',had_compute=had_compute)))
'''
 r=subprocess.run(ssh(target['host'])+[shlex.join(['python3','-c',code,json.dumps(target)])],text=True,capture_output=True,timeout=15)
 if r.returncode:target.update(last_error=r.stderr[-1500:],last_checked=time.time());return
 response=json.loads(r.stdout);target.update(response,last_checked=time.time())
 if response.get('had_compute'):item['replay_required']=True
def run():
 plan=read(PLAN);claims={}
 for d in legacy_claim_owners():
  if d.get('task'):claims.setdefault(d['task'],[]).append(d)
 health=read(OUT/'resource_snapshot.json');queue=read(OUT/'ctc_queue.json');jobs={j['key']:j for j in queue['tasks']}
 for key,item in plan.get('tasks',{}).items():
  if item.get('status')=='complete':continue
  retire_target(item);job=jobs[key];base=Path(job['result']).parent
  ready=OUT/'validation'/f'duplicate_replay_ready_{key}.json'
  if item.get('replay_required')and ready.exists()and item.get('status')!='replay_queued':
   item.update(read(base/'duplicate_execution_archive/correction.json'));item['status']='replay_queued'
  if item.get('status')=='replay_queued':
   if Path(job['result']).exists():item.update(status='complete',completed=time.time(),new_component_sha256=sha(Path(job['result'])))
   continue
  if not item.get('replay_required'):
   if item.get('stop_target',{}).get('status')=='retired':item.update(status='complete',note='Duplicate retired before GPU admission; original fit retained')
   continue
  with(OUT/'locks/model_admission.lock').open('a')as admission:
   fcntl.flock(admission,fcntl.LOCK_EX);path=OUT/'ctc_queue.json';current=read(path);changed=False
   for queued in current['tasks']:
    if queued['key']==key and str(ready)not in queued.get('requires',[]):queued.setdefault('requires',[]).append(str(ready));changed=True
   if changed:write(path,current)
  if claims.get(key)or key in health.get('live_tasks',{})or time.time()-health.get('observed_at_controller',0)>180:
   item.update(status='waiting_for_original_execution_to_finish');continue
  head=base/'head_epoch50.pth';meta=read(base/'head_metadata.json');invocation=base/'invocation_manifest.json'
  valid=False
  try:
   history=[json.loads(line)for line in(base/'train_epoch50.jsonl').read_text().splitlines()]
   valid=meta['epochs']==50 and meta['status']=='VALID_EPOCH_50'and sha(head)==meta['head_sha256']and sha(invocation)==meta['campaign_manifest_sha256']and [r['epoch']for r in history]==list(range(1,51))and abs(history[-1]['loss']-meta['final_loss'])<1e-12
  except (KeyError,OSError,ValueError):pass
  archive=base/'duplicate_execution_archive';archive.mkdir(exist_ok=True)
  with execution_lock(job)as acquired:
   if not acquired:continue
   # Claim refresh under the execution lock protects the mutation boundary.
   if any(owner.get('task')==key for owner in legacy_claim_owners()):continue
   item.update(status='archiving',archive=str(archive));write(PLAN,plan)
   names=['domains','component_result.json','test_frame_index.csv']
   if not valid:names+=['head_epoch50.pth','head_metadata.json','train_epoch50.jsonl']
   moved=[]
   for name in names:
    source=base/name
    if source.exists():
     destination=archive/name
     if destination.exists():raise RuntimeError('Existing recovery archive: '+str(destination))
     source.rename(destination);moved.append(name)
   attempt=OUT/'states'/f'{key}_attempts.json'
   if attempt.exists():write(archive/'original_attempts.json',read(attempt))
   write(attempt,dict(attempts=0,reset_reason='Replay predictions after duplicate execution overlap',archive=str(archive),updated=time.time()))
   item.update(status='replay_queued',queued=time.time(),archive=str(archive),preserved_epoch50_head=valid,moved=moved,head_sha256=meta.get('head_sha256'),queue=str(OUT/'ctc_queue.json'),result=job['result'],log=job['log'])
   write(archive/'correction.json',item)
   write(ready,dict(status='PASS',updated=time.time(),task=key,archive=str(archive),preserved_epoch50_head=valid,training_seed_unchanged=True))
 plan.update(updated=time.time(),pid=os.getpid());write(PLAN,plan)
 write(OUT/'states/duplicate_execution_recovery.json',dict(updated=time.time(),pid=os.getpid(),tasks={k:v.get('status')for k,v in plan.get('tasks',{}).items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/duplicate_execution_recovery.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   try:run()
   except Exception:traceback.print_exc()
   if not a.watch:break
   time.sleep(45)
