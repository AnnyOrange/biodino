"""Admit only SHA-certified resident copies; retire only redundant data transfers."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,PY
DATASETS=['hpa-subcellular','lc25000','rxrx3-core']
def dump(p,r):
 t=p.with_suffix('.finalize_local.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def run():
 code='''import pathlib,json,subprocess,os
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';result={}
for ds in __DATASETS__:
 proof=out/'validation'/f'ai_retrieval_{ds}_resident.json';state=out/'states'/f'ai_local_retrieval_{ds}.json';r=json.loads(state.read_text())if state.exists()else{}
 if proof.exists()and json.loads(proof.read_text()).get('source_host')=='5090-hxw-xzj':result[ds]=dict(proof=json.loads(proof.read_text()),state=r);continue
 pid=r.get('pid');p=pathlib.Path(f'/proc/{pid}/cmdline');live=p.exists()and b'prepare_unified_ai_local_retrieval.py'in p.read_bytes()
 if not live and r.get('attempts',0)<2:
  with(out/'logs'/f'ai_local_retrieval_{ds}.log').open('a')as f:
   child=subprocess.Popen([__PY__,'-u',str(root/'scripts/prepare_unified_ai_local_retrieval.py'),'--dataset',ds],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  r.update(status='launched',pid=child.pid)
 result[ds]=dict(state=r)
print(json.dumps(result))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__PY__',repr(PY)).replace('__DATASETS__',repr(DATASETS))
 response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,capture_output=True,timeout=30);response.check_returncode();states=json.loads(response.stdout)
 for ds,item in states.items():
  proof=item.get('proof')
  if not proof:continue
  canonical=OUT/'validation'/f'ai_retrieval_{ds}_canonical.json';assert proof['status']=='PASS'and proof['canonical_proof_sha256']==hashlib.sha256(canonical.read_bytes()).hexdigest();assert proof['files']==len(json.loads(canonical.read_text())['files'])
  marker=OUT/'validation'/f'ai_retrieval_{ds}_resident_copy_complete.json';dump(marker,proof);retired=[];pack=f'outputs/02_eval_runs/unified_paper/prepared_inputs/retrieval_{ds}.tar'.encode()
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:args=p.read_bytes().split(b'\0')
   except OSError:continue
   if args and Path(args[0].decode()).name=='rsync'and pack in args and (HOST+':'+REMOTE_ROOT+'/').encode()in args:
    try:os.kill(int(p.parent.name),signal.SIGTERM);retired.append(int(p.parent.name))
    except ProcessLookupError:pass
  if retired:dump(OUT/'corrections'/f'ai_retrieval_{ds}_redundant_transfer_retired.json',dict(updated=time.time(),verified_resident_proof=proof,retired_data_transfer_pids=retired,reason='Complete identical inputs already verified on resident host; partial archive is not needed and is retained for provenance'))
 dump(OUT/'states/ai_local_retrieval_finalization.json',dict(pid=os.getpid(),updated=time.time(),datasets=states))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/ai_local_retrieval_finalization.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   run()
   if not a.watch:break
   time.sleep(30)
