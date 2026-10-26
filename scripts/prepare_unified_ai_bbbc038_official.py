"""Prepare complete official DSB inputs at resident weights; never copy weights."""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback,zipfile
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,PY,MODELS
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json'
STATE=OUT/'states/ai_bbbc038_official_preparation.json'
def save(value):
    value.update(pid=os.getpid(),updated=time.time());tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)
def remote(code):
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
def prepare():
    pack=OUT/'prepared_inputs/bbbc038_official_gt.zip';manifest=json.loads(MANIFEST.read_text())
    if not pack.exists():
        tmp=pack.with_suffix('.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
            for rows in manifest['splits'].values():
                for row in rows:z.write(row['instances'],Path(row['instances']).name)
        tmp.replace(pack)
    digest=hashlib.sha256(pack.read_bytes()).hexdigest();save(dict(status='transferring_verified_ground_truth_pack',archive_sha256=digest,archive_bytes=pack.stat().st_size))
    files=['scripts/unified_paper_gpu_admission.py','scripts/run_unified_paper_bbbc038_official.py','scripts/run_unified_paper_ctc.py','dinov3/eval/bio_segmentation/instance_seg/train.py','dinov3/eval/bio_segmentation/datasets/bbbc038_official.py',str(MANIFEST.relative_to(ROOT)),str(pack.relative_to(ROOT))]
    subprocess.run(['rsync','-aR','--partial','--partial-dir=.rsync-partial',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
    save(dict(status='downloading_official_images_and_verifying_all841',archive_sha256=digest))
    code='''import concurrent.futures,fcntl,hashlib,json,os,pathlib,subprocess,zipfile,time
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';source=pathlib.Path('/data/xuzijing/benchmark/segmentation/bbbc038');source.mkdir(parents=True,exist_ok=True)
with(out/'locks/bbbc038_resident_preparation.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 archive=out/'prepared_inputs/bbbc038_official_gt.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==__SHA__
 manifest=root/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json';data=json.loads(manifest.read_text());gt=out/'prepared_inputs/bbbc038_official';gt.mkdir(exist_ok=True)
 expected={pathlib.Path(r['instances']).name for rows in data['splits'].values()for r in rows}
 with zipfile.ZipFile(archive)as z:
  assert set(z.namelist())==expected
  z.extractall(gt)
 def fetch(name):
  path=source/(name+'.zip');partial=source/(name+'.zip.part')
  if not path.exists():
   subprocess.run(['curl','-fL','--retry','8','--retry-all-errors','--connect-timeout','20','-C','-','-o',str(partial),'https://data.broadinstitute.org/bbbc/BBBC038/'+name+'.zip'],check=True)
   partial.replace(path)
  with zipfile.ZipFile(path)as z:
   destination=source/'extracted'/name;destination.mkdir(parents=True,exist_ok=True)
   for item in z.infolist():
    assert not pathlib.PurePosixPath(item.filename).is_absolute()and '..'not in pathlib.PurePosixPath(item.filename).parts
   z.extractall(destination)
  return name
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
  for name in pool.map(fetch,['stage1_train','stage1_test','stage2_test_final']):print('OFFICIAL ARCHIVE EXTRACTED',name,flush=True)
 count=0
 for rows in data['splits'].values():
  for r in rows:
   image=source/pathlib.Path(r['image']).relative_to('/mnt/huawei_deepcad/benchmark/segmentation/bbbc038');mask=gt/pathlib.Path(r['instances']).name
   assert hashlib.sha256(image.read_bytes()).hexdigest()==r['image_sha256'],str(image)
   assert hashlib.sha256(mask.read_bytes()).hexdigest()==r['instances_sha256'],str(mask)
   count+=1
 assert count==841
 proof=dict(status='PASS',canonical_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),images=841,train=670,stage1=65,stage2=106,ground_truth_archive_sha256=__SHA__,updated=time.time())
 path=out/'validation/bbbc038_resident_inputs.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2));tmp.replace(path);print(proof,flush=True)
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__SHA__',repr(digest))
    compile(code,'resident_bbbc038_preparation','exec');remote(code)
    save(dict(status='complete_inputs_waiting_for_real_pilot',archive_sha256=digest,images=841,checkpoint_transfer=False))
def admit():
    pilot=OUT/'results/biodino_1tb_splus/bbbc038_official/seed0/component_result.json'
    if not pilot.exists():return
    raw=json.loads(pilot.read_text());assert raw['completed_epochs']==50 and {s:r['n_test']for s,r in raw['stages'].items()}==dict(stage1=65,stage2=106)
    code='''import fcntl,json,pathlib
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper'
proof=json.loads((out/'validation/bbbc038_resident_inputs.json').read_text());assert proof['status']=='PASS'and proof['images']==841
with(out/'locks/model_admission.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'bbbc038_official_queue.json';q=json.loads(path.read_text())if path.exists()else{'tasks':[]};known={j['key']for j in q['tasks']}
 for model in __MODELS__:
  for seed in range(5):
   key=model+'__bbbc038_official__seed'+str(seed)
   if key in known:continue
   q['tasks'].append(dict(key=key,model=model,task='instance_segmentation',dataset='bbbc038-official',fold='stage1+stage2',seed=seed,entrypoint='run_unified_paper_bbbc038_official.py',args=['--model',model,'--seed',str(seed)],claim_key=key,min_free_mib=12000,expected_peak_mib=10000,result=str(out/'results'/model/'bbbc038_official'/('seed'+str(seed))/'component_result.json'),log=str(out/'logs'/(key+'.log'))))
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
print('OFFICIAL DSB RESIDENT QUEUE',len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__MODELS__',repr(MODELS))
    remote(code);save(dict(status='resident_queue_admitted',models=MODELS,pilot=str(pilot),images=841,checkpoint_transfer=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    with(OUT/'locks/ai_bbbc038_official_preparation.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            try:
                state=json.loads(STATE.read_text())if STATE.exists()else{}
                if state.get('status')not in ['complete_inputs_waiting_for_real_pilot','resident_queue_admitted']:prepare()
                if json.loads(STATE.read_text()).get('status')!='resident_queue_admitted':admit()
            except Exception:traceback.print_exc()
            if not args.watch:break
            time.sleep(60)
