"""Reuse complete resident inputs for the approved center probes; pilot gated."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
from unified_paper_center_inputs import manifest_path,sha
def dump(path,value):
 tmp=path.with_suffix('.residentcenter.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def run(dataset):
 manifest=manifest_path(dataset)
 if not manifest.exists():return
 state=OUT/'states'/f'ai_{dataset}_center_preparation.json';old=json.loads(state.read_text())if state.exists()else{}
 if old.get('status')=='resident_queue_admitted':return
 if old.get('status')!='complete_inputs_waiting_for_pilot':
  subprocess.run(['rsync','-azR','scripts/run_unified_paper_detection.py','scripts/unified_paper_center_inputs.py',str(manifest.relative_to(ROOT)),HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  dump(state,dict(pid=os.getpid(),updated=time.time(),status='verifying_complete_resident_inputs',dataset=dataset))
  code='''import pathlib,json,os,sys,time
root=pathlib.Path(__ROOT__);sys.path[:0]=[str(root),str(root/'scripts')];out=root/'outputs/02_eval_runs/unified_paper'
from unified_paper_center_inputs import load_verified,manifest_path,sha
ds=__DATASET__;raw=load_verified(ds,'/data/xuzijing/benchmark');proof=out/'validation'/f'ai_{ds}_center_resident.json';tmp=proof.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='PASS',manifest_sha256=sha(manifest_path(ds)),files_verified=len(raw['source_hashes']),updated=time.time()),indent=2));tmp.replace(proof)
print(proof.read_text())
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__DATASET__',repr(dataset))
  # This full input verification may be I/O bound; never kill it on elapsed time.
  p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,capture_output=True);p.check_returncode();proof=json.loads(p.stdout[p.stdout.index('{'):]);assert proof['status']=='PASS'and proof['manifest_sha256']==sha(manifest)
  dump(state,dict(pid=os.getpid(),updated=time.time(),status='complete_inputs_waiting_for_pilot',dataset=dataset,proof=proof))
 pilot=OUT/'results/biodino_1tb_splus/detection_proxy'/dataset/'seed0/component_result.json'
 if not pilot.exists():return
 raw=json.loads(pilot.read_text());assert raw['manifest_sha256']==sha(manifest)and raw['epochs']==5 and raw['batch_size']==8 and Path(raw['prediction_path']).exists()
 queue=json.loads((OUT/'detection_queue.json').read_text());template=next(j for j in queue['tasks']if j['model']=='biodino_1tb_splus'and j['dataset']==dataset and j['seed']==0);jobs=[]
 for model in MODELS:
  for seed in range(5):
   j=json.loads(json.dumps(template).replace(str(ROOT),REMOTE_ROOT).replace('biodino_1tb_splus',model).replace('seed0',f'seed{seed}'));j['seed']=seed;j['args']=['--model',model,'--dataset',dataset,'--seed',str(seed)];j.pop('blocked_reason',None);j.update(claim_key=j['key'],min_free_mib=9000,expected_peak_mib=5500,requires=[REMOTE_OUT+'/validation/'+f'ai_{dataset}_center_resident.json']);jobs.append(j)
 code='''import pathlib,json,fcntl
out=pathlib.Path(__OUT__)
with(out/'locks/model_admission.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'detection_queue.json';q=json.loads(path.read_text());known={j['key']for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key']not in known);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
'''.replace('__OUT__',repr(REMOTE_OUT)).replace('__JOBS__',repr(jobs));subprocess.run(['ssh','-o','BatchMode=yes',HOST,PY+' -u -'],input=code,text=True,check=True)
 dump(state,dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',dataset=dataset,jobs=len(jobs),checkpoint_transfer=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/ai_center_probes_preparation.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  while True:
   for dataset in ['conic','livecell']:
    try:run(dataset)
    except Exception:traceback.print_exc()
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
