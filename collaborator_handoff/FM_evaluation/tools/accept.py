#!/usr/bin/env python3
"""Fail-closed acceptance: input hashes, real evaluations, retained artifacts."""
import gzip
import argparse,hashlib,json,math,os,subprocess,sys,time,fcntl,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1]
CANON=Path('/mnt/huawei_deepcad/dinov3')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def finite_tree(x):
 if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite result')
 if isinstance(x,dict):
  for v in x.values():finite_tree(v)
 if isinstance(x,list):
  for v in x:finite_tree(v)
def verify_file(p,expected):
 if not p.is_file():raise ValueError(f'Missing input: {p}')
 if sha(p)!=expected:raise ValueError(f'SHA256 mismatch: {p}')
def validate_result(repo,job,checkpoint,model="biodino_7b"):
 p=repo/job['result'];d=json.loads(p.read_text());finite_tree(d)
 if job['task']=='prepare_segmentation':
  if not d.get('frozen_encoder'):raise ValueError('Not a frozen cache result')
 else:
  if d.get('model')!=model:raise ValueError('Wrong/missing model identity')
  if isinstance(job.get('seed'),int) and d.get('seed',d.get('_meta',{}).get('seed'))!=job['seed']:raise ValueError('Wrong/missing seed')
  if d.get('checkpoint') and isinstance(d['checkpoint'],str) and d['checkpoint']!=checkpoint:raise ValueError('Wrong checkpoint')
 task=job['task']
 if task=='classification' and job['dataset']!='film_age':
  if not any(k in d for k in ('balanced_accuracy','macro_auc','macro_auroc','auroc')):raise ValueError('Missing classification metric')
 if task=='regression' and not all(k in d for k in ('mae','r2')):raise ValueError('Missing regression metrics')
 if task=='segmentation':
  if not all(k in d.get('test',{}) for k in ('mDice','mIoU')) or d.get('budget')!=job['budget'] or d.get('split')!=job['split']:raise ValueError('Segmentation metric/budget/split mismatch')
 if task=='detection_proxy' and (d.get('epochs')!=5 or 'test_patch_f1' not in d):raise ValueError('Incomplete detection probe')
 if task=='instance_segmentation' and (d.get('completed_epochs')!=50 or set(d.get('stages',{}))!={'stage1','stage2'}):raise ValueError('Incomplete BBBC038 stages/epochs')
 if task=='cell_tracking' and (d.get('fold')!=job['fold'] or not all(k in d.get('macro',{}) for k in ('TRA','SEG')) or not d.get('domain_rows')):raise ValueError('Incomplete CTC fold')
 if task in ('box_detection','nuclei_detection_segmentation','bone_age_regression','ood','centroid_detection_3d') and not d.get('metrics'):raise ValueError('Missing native metrics')
 if task=='ood' and d.get('split_revision')!='source_stratified_fov_disjoint_v3':raise ValueError('Wrong OOD split')
 if job['dataset'] in ('film_age','spheroid_lsfm'):
  if d.get('completed_seeds')!=5 or {v.get('seed') for v in d.get('seeds',[])}!=set(range(5)):raise ValueError('Missing downstream seeds')
 if task=='retrieval_clustering':
  if d.get('completed_clustering_seeds')!=5 or len(d.get('clustering_results',[]))!=5:raise ValueError('Incomplete retrieval/clustering')
  observed=set()
  for ref in [d['retrieval_result'],*d['clustering_results']]:
   child=json.loads(Path(ref).read_text());finite_tree(child)
   if not child.get('metrics'):raise ValueError('Missing referenced retrieval/clustering metrics')
   if ref in d['clustering_results']:observed.add(child.get('seed'))
  if observed!=set(range(5)):raise ValueError('Repeated/missing clustering seed')
 # Validate every explicitly referenced retained prediction, including bundled seeds.
 artifacts=[]
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in ('prediction_path','sample_manifest') and isinstance(v,str):
     q=Path(v)
     if not q.is_absolute():q=repo/q
     if not q.is_file() or q.stat().st_size==0:raise ValueError(f'Missing artifact: {q}')
     expected=x.get('prediction_sha256') if k=='prediction_path' else x.get('sample_manifest_sha256')
     if expected and sha(q)!=expected:raise ValueError(f'Artifact SHA mismatch: {q}')
     artifacts.append(str(q))
    walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(d)
 if task=='retrieval_clustering':
  for ref in [d['retrieval_result'],*d['clustering_results']]:walk(json.loads(Path(ref).read_text()))
 return dict(result_sha256=sha(p),artifacts=artifacts)
def environment(repo):
 env=os.environ.copy();env.update(UNIFIED_PAPER_PYTHON=sys.executable,UNIFIED_PAPER_BENCHMARK_ROOT='/mnt/huawei_deepcad/benchmark',DINOV3_CODE_ROOT=str(repo),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',UNIFIED_PAPER_OOD_SOURCE_RECORDS=str(repo/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/provenance/ood_selected_records.json'))
 env['UNIFIED_PAPER_OOD_INPUT_FINGERPRINTS']=str(repo/'outputs/02_eval_runs/unified_paper/validation/ood_input_fingerprints')
 env['PYTHONPATH']=str(repo)+':'+str(repo/'scripts')+':/mnt/huawei_deepcad/benchmark_model'
 return env
def verify_checkout(repo):
 for name,h in json.loads((P/'PACKAGE_SHA256.json').read_text()).items():
  if name.startswith('checkout/'):verify_file(repo/name[len('checkout/'):],h)
 for name,h in json.loads((P/'RUNTIME_SHA256.json').read_text()).items():verify_file(repo/name[len('checkout/'):],h)
def configure(a):
 repo=a.repo.resolve()
 if repo!=CANON:raise ValueError('Run configuration inside the documented container mounts; canonical paths preserve locked manifest hashes')
 verify_checkout(repo)
 for p in [a.checkpoint,a.train_config]:
  if not p.is_file():raise ValueError(f'Missing: {p}')
 u=repo/'outputs/02_eval_runs/unified_paper';state=u/'acceptance';state.mkdir(parents=True,exist_ok=True)
 identity=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),train_config=str(a.train_config.resolve()),train_config_sha256=sha(a.train_config))
 receipt=state/'identity.json'
 if receipt.exists() and json.loads(receipt.read_text())!=identity:raise ValueError('Another checkpoint owns this checkout; use a fresh workspace')
 cfg=json.loads((P/'eval_config.template.json').read_text());cfg['models']=[dict(model='biodino_7b',kind='biodino',checkpoint=identity['checkpoint'],train_config=identity['train_config'],min_free_mib=60000)]
 cfg['global_input_proofs']={}
 for p in (u/'validation').glob('ai_global_*_canonical.json'):
  d=json.loads(p.read_text());cfg['global_input_proofs'][d['dataset']]=dict(path=str(p),sha256=sha(p))
 dst=u/'config.json'
 if dst.exists() and json.loads(dst.read_text())!=cfg:raise ValueError('Refusing to replace a different evaluation config')
 for name in ['locks','states','logs','results']:(u/name).mkdir(exist_ok=True)
 dump(dst,cfg);dump(receipt,identity);print('CONFIGURED; data and GPU execution are not yet accepted')
def data(a):
 rows=json.loads(gzip.decompress((P/'required_inputs.json.gz').read_bytes()));errors=[];checked=0
 # Stream one file at a time. Quick checks never authorize full execution.
 for row in rows:
  p=Path(row['path']);p=Path(str(p).replace(str(CANON),str(a.repo.resolve()),1)) if str(p).startswith(str(CANON)+'/') else p
  try:
   if not p.is_file():raise ValueError('missing')
   if a.full_hash:verify_file(p,row['sha256'])
   checked+=1
  except (OSError,ValueError) as e:errors.append(dict(path=str(p),error=str(e)))
  if (checked+len(errors))%10000==0:print('INPUTS',checked,'OK',len(errors),'failed',flush=True)
 report=dict(stage='data_sha256' if a.full_hash else 'data_presence_only',status='FAIL' if errors else 'PASS',checked=checked,required=len(rows),errors=errors,input_lock_sha256=sha(P/'required_inputs.json.gz'))
 dump(a.repo/'outputs/02_eval_runs/unified_paper/acceptance/data_report.json',report)
 print(json.dumps({k:v for k,v in report.items() if k!='errors'}));return 1 if errors else 0
def run(a):
 repo=a.repo.resolve();u=repo/'outputs/02_eval_runs/unified_paper';state=u/'acceptance';identity=json.loads((state/'identity.json').read_text())
 if repo!=CANON:raise ValueError('Use the isolated canonical-path container')
 verify_checkout(repo)
 proof=json.loads((state/'data_report.json').read_text())
 if proof['status']!='PASS' or proof['stage']!='data_sha256' or proof['input_lock_sha256']!=sha(P/'required_inputs.json.gz'):raise ValueError('A full input SHA256 pass is required')
 subprocess.run([sys.executable,str(P/'tools/check_env.py')],env=environment(repo),check=True)
 verify_file(Path(identity['checkpoint']),identity['checkpoint_sha256']);verify_file(Path(identity['train_config']),identity['train_config_sha256'])
 jobs=json.loads((P/'jobs.json').read_text())
 if a.mode=='smoke':jobs=[j for j in jobs if (j['dataset']=='bloodmnist' and j['task']=='classification') or (j['dataset']=='xray' and j['task']=='ood')]
 assert jobs
 with (state/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for j in jobs:
   receipt=state/'jobs'/f"{j['key']}.json"
   argv=[sys.executable,str(repo/'scripts'/j['entrypoint']),*[v.replace('@CHECKPOINT@',identity['checkpoint']).replace('@TRAIN_CONFIG@',identity['train_config']) for v in j['args']]]
   signature=hashlib.sha256(json.dumps([j,identity],sort_keys=True).encode()).hexdigest()
   if receipt.exists():
    prior=json.loads(receipt.read_text())
    if prior.get('status')=='PASS' and prior.get('signature')==signature:
     validated=validate_result(repo,j,identity['checkpoint'])
     if validated['result_sha256']!=prior['result_sha256']:raise ValueError('Result changed after acceptance')
     continue
   # Do not accept an arbitrary pre-existing result merely because the runner skips it.
   if (repo/j['result']).exists():raise ValueError(f"Unreceipted result exists: {j['result']}; inspect it or use a clean workspace")
   log=state/'logs'/f"{j['key']}.log";log.parent.mkdir(exist_ok=True)
   started=time.time();print('RUN',j['key'],flush=True)
   with log.open('w') as f:rc=subprocess.run(argv,cwd=repo,env=environment(repo),stdout=f,stderr=subprocess.STDOUT).returncode
   result=dict(status='FAIL',exit_code=rc,signature=signature,command=argv,log=str(log),started=started,ended=time.time())
   try:
    if rc:raise ValueError(f'Process exit={rc}')
    result.update(validate_result(repo,j,identity['checkpoint']));result['status']='PASS'
   except Exception as e:result['error']=repr(e)
   dump(receipt,result)
   if result['status']!='PASS':raise ValueError(f"Failed {j['key']}; inspect {log}")
  dump(state/f'{a.mode}_report.json',dict(status='PASS',scope=a.mode,jobs=len(jobs),identity=identity,full_matrix_complete=a.mode=='full',statistics='Paired significance/CI are not certified by this execution acceptance'))
  print(a.mode.upper(),'PASS',len(jobs),'jobs')
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='stage',required=True)
 c=s.add_parser('configure');c.add_argument('--checkpoint',type=Path,required=True);c.add_argument('--train-config',type=Path,required=True)
 d=s.add_parser('data');d.add_argument('--full-hash',action='store_true')
 r=s.add_parser('run');r.add_argument('--mode',choices=['smoke','full'],required=True)
 for q in [c,d,r]:q.add_argument('--repo',type=Path,default=CANON)
 a=p.parse_args()
 try:return {'configure':configure,'data':data,'run':run}[a.stage](a) or 0
 except Exception as e:print('ACCEPTANCE FAILED:',e,file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
