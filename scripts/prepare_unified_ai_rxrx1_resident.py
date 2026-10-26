"""Reproduce the byte-identical canonical core from the AI-resident source ZIP."""
import fcntl,hashlib,json,os,subprocess,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def sha(path):
 with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,value):
 tmp=path.with_suffix('.prepare.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
 with(OUT/'locks/ai_rxrx1_resident.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  state=OUT/'states/ai_rxrx1_resident.json';old=json.loads(state.read_text())if state.exists()else{};r=dict(pid=os.getpid(),updated=time.time(),attempts=old.get('attempts',0)+1,status='copying_complete_source_within_AI_building',source_host='5090-hxw-xzj');dump(state,r)
  try:
   manifest=ROOT/'outputs/02_eval_inputs/formal_v4/rxrx1-cross_retrieval_manifest.json';m=json.loads(manifest.read_text());index=OUT/'validation/rxrx1_v4_member_hashes.json';assert sha(index)==m['member_index_sha256'];hashes=json.loads(index.read_text());dates=json.loads((OUT/'validation/rxrx1_canonical_zip_dates.json').read_text());assert set(dates)==set(hashes)
   bench=Path('/data/xuzijing/benchmark');source=bench/'Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip';source.parent.mkdir(parents=True,exist_ok=True)
   subprocess.run(['rsync','-a','--partial','--partial-dir=.rsync-partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=10','5090-hxw-xzj:/data/benchmark/Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip',str(source)],check=True)
   pack=OUT/'prepared_inputs/rxrx1_v4_complete_core.zip'
   if not pack.exists():
    tmp=pack.with_suffix('.resident.pending.zip');r.update(status='rebuilding_and_SHA_verifying_all_106368_members',updated=time.time());dump(state,r)
    with zipfile.ZipFile(source)as src,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True)as dst:
     for number,(name,date)in enumerate(dates.items(),1):
      data=src.read(name);expected=hashes[name];assert len(data)==expected['bytes']and hashlib.sha256(data).hexdigest()==expected['sha256'],name
      dst.writestr(zipfile.ZipInfo(name,tuple(date)),data)
      if number%2000==0:r.update(updated=time.time(),files_verified=number);dump(state,r)
    assert sha(tmp)==m['packed_archive_sha256'],'Canonical archive byte identity failed';tmp.replace(pack)
   assert sha(pack)==m['packed_archive_sha256']
   csv=bench/'Retrieval_Clustering/protocols/v1/rxrx1_official_cross_experiment_core.csv';assert sha(csv)==m['source_csv_sha256']
   proof=OUT/'validation/ai_rxrx1_resident.json';dump(proof,dict(status='PASS',updated=time.time(),files=106368,views=17728,manifest_sha256=sha(manifest),packed_archive_sha256=sha(pack),source_host='5090-hxw-xzj',checkpoint_transfer=False))
   r.update(status='complete_verified',updated=time.time(),files_verified=106368,proof=str(proof));dump(state,r)
  except Exception as exc:r.update(status='failed',updated=time.time(),error=repr(exc));dump(state,r);traceback.print_exc();raise
if __name__=='__main__':main()
