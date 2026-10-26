"""Verify complete canonical inputs copied inside the authorized AI building."""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,r):
 t=p.with_suffix('.localcopy.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['hpa-subcellular','lc25000','rxrx3-core']);a=p.parse_args();ds=a.dataset
 with(OUT/'locks'/f'ai_local_retrieval_{ds}.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  proof_path=OUT/'validation'/f'ai_retrieval_{ds}_canonical.json';proof=json.loads(proof_path.read_text());state=OUT/'states'/f'ai_local_retrieval_{ds}.json';previous=json.loads(state.read_text())if state.exists()else{};record=dict(pid=os.getpid(),updated=time.time(),dataset=ds,status='copying_within_AI_building',source_host='5090-hxw-xzj',attempts=previous.get('attempts',0)+1,files=len(proof['files']));dump(state,record)
  try:
   base=ROOT if ds=='rxrx3-core'else Path('/data/xuzijing/benchmark');filelist=OUT/'validation'/f'ai_local_retrieval_{ds}_files.txt'
   if ds=='rxrx3-core':
    source='5090-hxw-xzj:/data/hs6_5tb_v4_rxrx3_20260924/cache/rxrx3-core/';destination=ROOT/'outputs/02_eval_inputs/formal_v3/rxrx3-core';names=[Path(n).name for n in proof['files']]
   else:source='5090-hxw-xzj:/data/benchmark/';destination=base;names=list(proof['files'])
   destination.mkdir(parents=True,exist_ok=True);filelist.write_text(''.join(n+'\n'for n in names));subprocess.run(['rsync','-a','--partial','--partial-dir=.rsync-partial','--files-from',str(filelist),'-e','ssh -o BatchMode=yes -o ConnectTimeout=10',source,str(destination)+'/'],check=True)
   record.update(updated=time.time(),status='verifying_all_canonical_SHA256');dump(state,record)
   for i,(name,expected)in enumerate(proof['files'].items(),1):
    assert sha(base/name)==expected,name
    if i%1000==0:record.update(updated=time.time(),files_verified=i);dump(state,record)
   target=OUT/'validation'/f'ai_retrieval_{ds}_resident.json';dump(target,dict(status='PASS',files=len(proof['files']),manifest_sha256=proof['manifest_sha256'],canonical_proof_sha256=sha(proof_path),updated=time.time(),source_host='5090-hxw-xzj',source_root=source,checkpoint_transfer=False))
   record.update(updated=time.time(),status='complete_verified',files_verified=len(proof['files']),proof=str(target));dump(state,record)
  except Exception as e:record.update(updated=time.time(),status='failed',error=repr(e));dump(state,record);traceback.print_exc();raise
if __name__=='__main__':main()
