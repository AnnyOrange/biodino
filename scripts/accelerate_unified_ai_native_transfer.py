"""Concurrent resumable file batches for complete canonical native image transfers.

Original transfers continue. Independent rsync temporary files atomically publish
identical immutable sources; each batch is SHA256 checked on the resident host.
No weights, split changes, or partial-data evaluation are involved.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,PY,REMOTE_OUT
BENCH=Path('/mnt/huawei_deepcad/benchmark');REMOTE_BENCH='/data/xuzijing/benchmark'

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def alive(pid,needle):
 try:return needle.encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):return False

def transfer(task):
 ds,index,names,expected,proof_sha=task;key=f'{ds}_{index:04d}';folder=OUT/'prepared_inputs/native_transfer_batches'/ds;folder.mkdir(parents=True,exist_ok=True)
 filelist=folder/f'{index:04d}.txt';filelist.write_text(''.join(name+'\n' for name in names));state=OUT/'states'/f'native_transfer_batch_{key}.json';log=OUT/'logs'/f'native_transfer_batch_{key}.log'
 try:
  if state.exists():
   old=json.loads(state.read_text())
   if old.get('status')=='complete_verified' and old.get('canonical_proof_sha256')==proof_sha:return dict(key=key,status='cached',files=len(names))
  active=[]
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:argv=p.read_bytes().split(b'\0')
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if str(filelist).encode() in argv and argv and Path(os.fsdecode(argv[0])).name=='rsync':active.append(int(p.parent.name))
  command=['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','--relative','--partial',f'--partial-dir=.rsync-partial-parallel-{key}','--files-from',str(filelist),str(BENCH)+'/',HOST+':'+REMOTE_BENCH+'/']
  process=None
  if active:pid=min(active)
  else:
   with log.open('a') as f:process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   pid=process.pid
  write(state,dict(status='transfer_running',updated=time.time(),pid=pid,files=len(names),log=str(log),canonical_proof_sha256=proof_sha,adopted=bool(active)))
  while alive(pid,str(filelist)) if process is None else process.poll() is None:time.sleep(5)
  if process is not None and process.returncode:raise RuntimeError(f'rsync exit {process.returncode}; partial files retained')
  write(state,dict(status='resident_sha_verification',updated=time.time(),files=len(names),log=str(log),canonical_proof_sha256=proof_sha))
  code='''from pathlib import Path
import hashlib,json,os,time
out=Path(__OUT__);root=Path(__BENCH__);expected=__EXPECTED__;proof=out/'validation/native_transfer_chunks'/__DS__/__FILE__
old=json.loads(proof.read_text()) if proof.exists() else {};verified={}
def stat(p):
 s=p.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
for name,digest in expected.items():
 p=root/name;before=stat(p);prior=old.get('files',{}).get(name,{})
 if old.get('canonical_proof_sha256')==__HASH__ and prior.get('sha256')==digest and prior.get('stat')==before:verified[name]=prior;continue
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 after=stat(p)
 if before!=after:
  with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  before=after;after=stat(p)
 assert before==after and actual==digest,(name,actual,digest)
 verified[name]=dict(sha256=actual,stat=after)
proof.parent.mkdir(parents=True,exist_ok=True);payload=dict(status='PASS',dataset=__DS__,canonical_proof_sha256=__HASH__,files=verified,updated=time.time());tmp=proof.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\\n');tmp.replace(proof)
print(json.dumps(dict(status='PASS',proof=str(proof),proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),files=len(verified))))
'''.replace('__OUT__',repr(REMOTE_OUT)).replace('__BENCH__',repr(REMOTE_BENCH)).replace('__EXPECTED__',repr({name:expected[name] for name in names})).replace('__DS__',repr(ds)).replace('__FILE__',repr(f'{index:04d}.json')).replace('__HASH__',repr(proof_sha))
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join([PY,'-u','-'])],input=code,text=True,capture_output=True)
  if result.returncode:raise RuntimeError(result.stderr[-4000:])
  verified=json.loads(result.stdout);write(state,dict(status='complete_verified',updated=time.time(),files=len(names),canonical_proof_sha256=proof_sha,verification=verified,log=str(log)))
  return dict(key=key,status='complete_verified',files=len(names))
 except Exception as error:
  write(state,dict(status='retryable_error',updated=time.time(),error=repr(error),canonical_proof_sha256=proof_sha,log=str(log)))
  return dict(key=key,status='error',error=repr(error))

def run(workers):
 jobs=[]
 for ds in ['bbbc041','rsna_pneumonia']:
  prep=OUT/'states'/f'ai_native_{ds}_preparation.json'
  if prep.exists() and json.loads(prep.read_text()).get('status')=='resident_queue_admitted':continue
  proof=OUT/'validation'/f'ai_native_{ds}_canonical.json'
  if not proof.exists():continue
  raw=proof.read_bytes();d=json.loads(raw);names=sorted(d['canonical_files']);h=hashlib.sha256(raw).hexdigest()
  jobs.extend((ds,i//64,names[i:i+64],d['canonical_files'],h)for i in range(0,len(names),64))
 state=dict(pid=os.getpid(),updated=time.time(),status='running',workers=workers,batches=len(jobs),finished=0,errors=[]);path=OUT/'states/parallel_native_transfer.json';write(path,state)
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(transfer,job) for job in jobs]
  for future in concurrent.futures.as_completed(futures):
   result=future.result()
   if result['status']=='error':state['errors'].append(result)
   else:state['finished']+=1
   print('NATIVE INPUT BATCH',result,flush=True);state['updated']=time.time();write(path,state)
 state.update(status='watching',updated=time.time());write(path,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
 with (OUT/'locks/parallel_native_transfer.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   try:run(args.workers)
   except Exception:traceback.print_exc()
   if not args.watch:break
   time.sleep(60)
