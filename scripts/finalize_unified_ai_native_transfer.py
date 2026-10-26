"""Admit each complete native dataset as soon as all transfer batches are verified."""
import argparse,fcntl,hashlib,json,os,time,traceback
from pathlib import Path
from prepare_unified_ai_native import OUT,admit_prepared_dataset,write

def run():
 rows=[]
 for ds in ['bbbc041','rsna_pneumonia']:
  prep=OUT/'states'/f'ai_native_{ds}_preparation.json'
  if prep.exists() and json.loads(prep.read_text()).get('status')=='resident_queue_admitted':rows.append(dict(dataset=ds,status='resident_queue_admitted'));continue
  proof=OUT/'validation'/f'ai_native_{ds}_canonical.json'
  if not proof.exists():rows.append(dict(dataset=ds,status='canonical_proof_missing'));continue
  raw=proof.read_bytes();h=hashlib.sha256(raw).hexdigest();n=len(json.loads(raw)['canonical_files']);batches=(n+63)//64;complete=0
  for index in range(batches):
   p=OUT/'states'/f'native_transfer_batch_{ds}_{index:04d}.json'
   if not p.exists():continue
   d=json.loads(p.read_text());complete+=d.get('status')=='complete_verified' and d.get('canonical_proof_sha256')==h
  if complete==batches:
   admit_prepared_dataset(ds);rows.append(dict(dataset=ds,status='resident_queue_admitted',verified_batches=complete))
   print('COMPLETE SOURCE ADMITTED',ds,n,flush=True)
  else:rows.append(dict(dataset=ds,status='waiting_for_full_verified_source',verified_batches=complete,required_batches=batches))
 write(OUT/'states/native_parallel_admission.json',dict(pid=os.getpid(),updated=time.time(),datasets=rows))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
 with (OUT/'locks/native_parallel_admission.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   try:run()
   except Exception:traceback.print_exc()
   if not args.watch:break
   time.sleep(30)
