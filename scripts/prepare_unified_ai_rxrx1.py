"""Persistently prepare and admit the two resident BioDINO branches for RxRx1."""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback,zipfile
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
def dump(path,value):
 tmp=path.with_suffix('.ai_rxrx1.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,capture_output=True,timeout=45);p.check_returncode();return json.loads(p.stdout)
def run():
 response=remote('''import pathlib,json,subprocess,time
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';state=out/'states/ai_rxrx1_resident.json';r=json.loads(state.read_text())if state.exists()else{};proof=out/'validation/ai_rxrx1_resident.json'
if proof.exists():r['proof_data']=json.loads(proof.read_text())
else:
 p=pathlib.Path('/proc/'+str(r.get('pid'))+'/cmdline');live=p.exists()and b'prepare_unified_ai_rxrx1_resident.py'in p.read_bytes()
 if not live and r.get('attempts',0)<2:
  with(out/'logs/ai_rxrx1_resident.log').open('a')as f:child=subprocess.Popen([__PY__,'-u',str(root/'scripts/prepare_unified_ai_rxrx1_resident.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  r.update(status='launched',pid=child.pid)
print(json.dumps(r))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__PY__',repr(PY)))
 proof=response.get('proof_data');admitted=False
 if proof:
  manifest=ROOT/'outputs/02_eval_inputs/formal_v4/rxrx1-cross_retrieval_manifest.json';m=json.loads(manifest.read_text());assert proof['status']=='PASS'and proof['files']==106368 and proof['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()and proof['packed_archive_sha256']==m['packed_archive_sha256']
  dump(OUT/'validation/ai_rxrx1_resident_copy_complete.json',proof)
  pilot=OUT/'validation/rxrx1_retrieval_pilot.json'
  if pilot.exists():
   p=json.loads(pilot.read_text());assert p['status']=='PASS'and p['n_query']==8864 and p['n_gallery']==8864 and p['manifest_sha256']==proof['manifest_sha256'];queue=json.loads((OUT/'rxrx1_retrieval_queue.json').read_text());template=next(j for j in queue['tasks']if j['model']=='biodino_1tb_splus');jobs=[]
   for model in MODELS:
    j=json.loads(json.dumps(template).replace(str(ROOT),REMOTE_ROOT).replace('biodino_1tb_splus',model));j.pop('allowed_hosts',None);j.pop('blocked_reason',None);j.update(claim_key=j['key'],min_free_mib=10000,expected_peak_mib=5500,requires=[REMOTE_OUT+'/validation/ai_rxrx1_resident.json']);jobs.append(j)
   remote('''import pathlib,json,fcntl
out=pathlib.Path(__OUT__)
with(out/'locks/model_admission.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'rxrx1_retrieval_queue.json';q=json.loads(path.read_text())if path.exists()else dict(tasks=[]);known={j['key']for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key']not in known);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
print('{}')
'''.replace('__OUT__',repr(REMOTE_OUT)).replace('__JOBS__',repr(jobs)));admitted=True
 dump(OUT/'states/ai_rxrx1_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted'if admitted else'resident_inputs_in_preparation_or_waiting_for_pilot',resident=response,jobs=2 if admitted else 0))
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/ai_rxrx1_controller.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  dates=OUT/'validation/rxrx1_canonical_zip_dates.json'
  if not dates.exists():
   with zipfile.ZipFile(OUT/'prepared_inputs/rxrx1_v4_complete_core.zip')as z:dump(dates,{i.filename:i.date_time for i in z.infolist()})
  files=['scripts/prepare_unified_ai_rxrx1_resident.py','scripts/prepare_unified_rxrx1.py','scripts/run_unified_paper_rxrx1.py','scripts/unified_paper_retrieval_scheduling.py','scripts/unified_paper_resident_retrieval_inputs.py','outputs/02_eval_inputs/formal_v4/rxrx1-cross_retrieval_manifest.json','outputs/02_eval_runs/unified_paper/validation/rxrx1_v4_member_hashes.json',str(dates.relative_to(ROOT))]
  subprocess.run(['rsync','-azR',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  subprocess.run(['rsync','-azR','Retrieval_Clustering/protocols/v1/rxrx1_official_cross_experiment_core.csv',HOST+':/data/xuzijing/benchmark/'],cwd='/mnt/huawei_deepcad/benchmark',check=True)
  while True:
   try:run()
   except Exception:traceback.print_exc()
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
