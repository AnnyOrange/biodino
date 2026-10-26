"""Prepare complete existing v4 retrieval cohorts on the checkpoint-resident host."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,tarfile,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
BENCH=Path('/mnt/huawei_deepcad/benchmark')
def dump(p,r):
 t=p.with_suffix('.ai_retrieval.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def prepare(dataset):
 with (OUT/'locks'/f'ai_retrieval_{dataset}.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  return prepare_locked(dataset)
def prepare_locked(dataset):
 state=OUT/'states'/f'ai_retrieval_{dataset}.json';old=json.loads(state.read_text())if state.exists()else{}
 if old.get('status')=='resident_queue_admitted':return
 manifest=ROOT/'outputs/02_eval_inputs/formal_v4'/f'{dataset}_retrieval_manifest.json';raw=json.loads(manifest.read_text())
 pack=OUT/'prepared_inputs'/f'retrieval_{dataset}.tar';proof=OUT/'validation'/f'ai_retrieval_{dataset}_canonical.json'
 source_base=ROOT if dataset=='rxrx3-core'else BENCH
 if not proof.exists():
  if dataset=='rxrx3-core':
   files={str(ROOT/'outputs/02_eval_inputs/formal_v3/rxrx3-core'/n):h for n,h in raw['source_hashes'].items()}
  elif dataset=='hpa-subcellular':
   files={r['sample_id']:r['image_sha256']for rows in raw['records'].values()for r in rows};files.update(raw['source_hashes'])
  elif dataset=='nct-crc-he-100':
   files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (BENCH/'Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data').glob('nct_crc_he_100-*.parquet')};assert files
  else:files={r['sample_id']:r['image_sha256']for r in raw['records']}
  dump(proof,dict(dataset=dataset,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),files={str(Path(p).relative_to(source_base)):h for p,h in files.items()}))
 proof_data=json.loads(proof.read_text());assert proof_data['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
 resident_marker=OUT/'validation'/f'ai_retrieval_{dataset}_resident_copy_complete.json'
 if resident_marker.exists():
  certified=json.loads(resident_marker.read_text());assert certified['canonical_proof_sha256']==hashlib.sha256(proof.read_bytes()).hexdigest()and certified['status']=='PASS';old['status']='complete_inputs_waiting_for_real_pilot'
 if old.get('status')!='complete_inputs_waiting_for_real_pilot':
  dump(state,dict(pid=os.getpid(),updated=time.time(),status='preparing_complete_archive',dataset=dataset,files=len(proof_data['files'])))
  if not pack.exists():
   tmp=pack.with_suffix('.pending.tar')
   with tarfile.open(tmp,'w')as archive:
    for name in proof_data['files']:archive.add(source_base/name,arcname=name,recursive=False)
   tmp.replace(pack)
  dump(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_complete_archive',dataset=dataset,files=len(proof_data['files']),bytes=pack.stat().st_size))
  # Adopt a transfer started by the previous controller; never race its partial file.
  while True:
   if resident_marker.exists():return prepare_locked(dataset)
   active=[]
   for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:argv=proc.read_bytes().split(b'\0')
    except OSError:continue
    if argv and Path(argv[0].decode()).name=='rsync'and str(pack.relative_to(ROOT)).encode()in argv and (HOST+':'+REMOTE_ROOT+'/').encode()in argv:active.append(int(proc.parent.name))
   if not active:break
   dump(state,dict(pid=os.getpid(),updated=time.time(),status='adopting_live_complete_archive_transfer',dataset=dataset,transfer_pids=active,files=len(proof_data['files']),bytes=pack.stat().st_size));time.sleep(10)
  subprocess.run(['rsync','-aR','--partial','--partial-dir=.rsync-partial',*[str(p.relative_to(ROOT))for p in [pack,manifest,proof]],HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  code='''import pathlib,json,tarfile,hashlib,time
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';bench=pathlib.Path('/data/xuzijing/benchmark');ds=__DATASET__
if ds=='rxrx3-core':bench=root
proof=json.loads((out/'validation'/f'ai_retrieval_{ds}_canonical.json').read_text());archive=out/'prepared_inputs'/f'retrieval_{ds}.tar';target=out/'validation'/f'ai_retrieval_{ds}_resident.json'
if not target.exists():
 with tarfile.open(archive)as tar:
  for member in tar:
   assert member.name in proof['files'] and member.isfile()
   tar.extract(member,bench,filter='data')
 for name,digest in proof['files'].items():
  with(bench/name).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='PASS',files=len(proof['files']),manifest_sha256=proof['manifest_sha256'],updated=time.time()),indent=2));tmp.replace(target)
print(target.read_text())
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__DATASET__',repr(dataset))
  subprocess.run(['ssh','-o','BatchMode=yes',HOST,PY+' -u -'],input=code,text=True,check=True)
  dump(state,dict(pid=os.getpid(),updated=time.time(),status='complete_inputs_waiting_for_real_pilot',dataset=dataset,files=len(proof_data['files'])))
 pilot=OUT/'results/biodino_1tb_splus/retrieval_clustering'/dataset/'component_result.json'
 if not pilot.exists():return
 p=json.loads(pilot.read_text());assert p['completed_clustering_seeds']==5 and p['manifest_sha256']==proof_data['manifest_sha256']
 queue_name={'hpa-subcellular':'hpa_retrieval_queue.json','rxrx3-core':'rxrx3_retrieval_queue.json'}.get(dataset,'retrieval_queue.json');queue=json.loads((OUT/queue_name).read_text())
 template=next(j for j in queue['tasks']if j['model']=='biodino_1tb_splus'and j['dataset']==dataset);jobs=[]
 for model in MODELS:
  j=json.loads(json.dumps(template).replace(str(ROOT),REMOTE_ROOT).replace('biodino_1tb_splus',model));j.pop('allowed_hosts',None);j.pop('blocked_reason',None);j['claim_key']=j['key'];j['min_free_mib']=10000;j['expected_peak_mib']=5500;j['requires']=[REMOTE_OUT+'/validation/'+f'ai_retrieval_{dataset}_resident.json'];jobs.append(j)
 code='''import pathlib,json,fcntl
out=pathlib.Path(__OUT__)
with(out/'locks/model_admission.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/__QUEUE__;q=json.loads(path.read_text())if path.exists()else dict(tasks=[]);known={j['key']for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key']not in known);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
'''.replace('__OUT__',repr(REMOTE_OUT)).replace('__QUEUE__',repr(queue_name)).replace('__JOBS__',repr(jobs))
 subprocess.run(['ssh','-o','BatchMode=yes',HOST,PY+' -u -'],input=code,text=True,check=True)
 dump(state,dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',dataset=dataset,jobs=len(jobs),files=len(proof_data['files']),checkpoint_transfer=False))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');a=parser.parse_args()
 with(OUT/'locks/ai_retrieval_preparation.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  subprocess.run(['rsync','-azR','scripts/run_unified_paper_rxrx3.py','scripts/run_external4_fixedbudget_model.py','scripts/run_unified_paper_retrieval.py','scripts/run_unified_paper_hpa_retrieval.py','scripts/prepare_unified_hpa_retrieval.py','scripts/unified_paper_resident_retrieval_inputs.py','scripts/unified_paper_gpu_admission.py','dinov3/eval/bio_frozen_eval/retrieval_clustering.py',HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  while True:
   with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
    futures={pool.submit(prepare,d):d for d in ['nct-crc-he-100','hpa-subcellular','lc25000','rxrx3-core']}
    for future in concurrent.futures.as_completed(futures):
     try:future.result()
     except Exception:traceback.print_exc()
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
