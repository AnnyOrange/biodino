"""Publish the same BBBC038 patch-detection probe on resident BioDINO branches."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,PY,MODELS
sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_segmentation.datasets.bbbc038 import get_bbbc038_paths

def main():
 with (OUT/'locks/ai_detection_proxy_preparation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  benchmark=Path('/mnt/huawei_deepcad/benchmark');source=benchmark/'segmentation/bbbc038/extracted';members={};files=set()
  for split in ['train','val','test']:
   images,masks=get_bbbc038_paths(str(source),split=split)
   members[split]=[dict(sample_id=Path(p).stem,group_id=Path(p).stem,image_path=p,mask_path=m) for p,m in zip(images,masks)]
   files.update(Path(p) for p in images)
   for folder in masks:files.update(p for p in Path(folder).glob('*') if p.is_file())
  assert sum(map(len,members.values()))==670
  fingerprints={}
  for path in sorted(files):
   with path.open('rb') as stream:fingerprints[str(path.relative_to(benchmark))]=hashlib.file_digest(stream,'sha256').hexdigest()
  split=source/'bbbc038_splits.npz';owned_split=OUT/'manifests/bbbc038_splits.npz';owned_split.write_bytes(split.read_bytes())
  proof=OUT/'validation/ai_bbbc038_canonical.json';proof.write_text(json.dumps(dict(membership_sha256=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest(),counts={k:len(v) for k,v in members.items()},source_files=fingerprints,split_sha256=hashlib.sha256(owned_split.read_bytes()).hexdigest()),indent=2)+'\n')
  filelist=OUT/'validation/ai_bbbc038_files.txt';filelist.write_text(''.join(p+'\n' for p in fingerprints))
  subprocess.run(['rsync','-ac','--relative','--partial','--files-from',str(filelist),str(benchmark)+'/',HOST+':/data/xuzijing/benchmark/'],check=True)
  code_files=['scripts/run_unified_paper_detection.py','dinov3/eval/bio_detection/center_probe.py','dinov3/eval/bio_segmentation/datasets/bbbc038.py',str(proof.relative_to(ROOT)),str(owned_split.relative_to(ROOT))]
  subprocess.run(['rsync','-azR',*code_files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
  code='''import pathlib,json,hashlib,fcntl
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';proof=json.loads((out/'validation/ai_bbbc038_canonical.json').read_text());fingerprints={};stats={};split=out/'manifests/bbbc038_splits.npz'
assert hashlib.sha256(split.read_bytes()).hexdigest()==proof['split_sha256']
for name,digest in proof['source_files'].items():
 p=pathlib.Path('/data/xuzijing/benchmark')/name
 with p.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
 fingerprints[str(p)]=digest;s=p.stat();stats[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
fingerprints[str(split)]=proof['split_sha256'];s=split.stat();stats[str(split)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
verified=out/'validation/ai_bbbc038_resident.json';verified.write_text(json.dumps(dict(membership_sha256=proof['membership_sha256'],counts=proof['counts'],source_fingerprints=fingerprints,source_stats=stats),indent=2)+'\\n')
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'detection_queue.json';q=json.loads(path.read_text()) if path.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']}
 pilot=out/'results'/__MODELS__[0]/'detection_proxy/bbbc038/seed0/component_result.json'
 for model in __MODELS__:
  for seed in range(5):
   key=model+'__detection_proxy__bbbc038__seed'+str(seed)
   if key in known:continue
   requires=[str(verified)]
   if model!=__MODELS__[0] or seed!=0:requires.append(str(pilot))
   q['tasks'].append(dict(key=key,model=model,dataset='bbbc038',task='detection_proxy',seed=seed,entrypoint='run_unified_paper_detection.py',args=['--model',model,'--dataset','bbbc038','--seed',str(seed)],claim_key=key,min_free_mib=8000,expected_peak_mib=4000,requires=requires,result=str(out/'results'/model/'detection_proxy/bbbc038'/f'seed{seed}'/'component_result.json'),log=str(out/'logs'/f'{key}.log')))
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(path);print('RESIDENT BBBC038 QUEUED',len(q['tasks']),proof['counts'])
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__MODELS__',repr(MODELS))
  subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
  (OUT/'states/ai_detection_proxy_preparation.json').write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',models=MODELS,queue=REMOTE_OUT+'/detection_queue.json',pilot_gated=True,verified_source_files=len(files)),indent=2)+'\n')
if __name__=='__main__':main()
