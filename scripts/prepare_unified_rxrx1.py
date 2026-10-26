"""Preserve every member of the existing v4 balanced cross-experiment core."""
import argparse,csv,fcntl,hashlib,json,os,time,zipfile
from pathlib import Path
from unified_paper_retrieval_scheduling import requirements
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';BENCH=Path('/mnt/huawei_deepcad/benchmark')
CSV=BENCH/'Retrieval_Clustering/protocols/v1/rxrx1_official_cross_experiment_core.csv';SOURCE=BENCH/'Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip'
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/rxrx1-cross_retrieval_manifest.json';PACK=OUT/'prepared_inputs/rxrx1_v4_complete_core.zip'
def dump(p,r):
 t=p.with_suffix('.rxrx1.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def signature(p):
 s=p.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
def prepare():
 if MANIFEST.exists():return
 rows=list(csv.DictReader(CSV.open()));assert len(rows)==17728 and len({r['site_id']for r in rows})==17728
 records={role:[dict(sample_id=r['site_id'],group_id=r['site_id'].rsplit('_',1)[0],label=int(r['sirna_id']),cell_type=r['cell_type'],experiment=r['experiment'],plate=r['plate'])for r in rows if r['role']==role]for role in ['gallery','query']}
 assert all(len(v)==8864 for v in records.values());g=records['gallery'];q=records['query']
 assert not({r['group_id']for r in g}&{r['group_id']for r in q})and not({r['experiment']for r in g}&{r['experiment']for r in q})
 assert {(r['cell_type'],r['label'])for r in g}=={(r['cell_type'],r['label'])for r in q}
 members=sorted({r[f'c{i}']for r in rows for i in range(1,7)});assert len(members)==106368
 hashes={};tmp=PACK.with_suffix('.pending.zip');start=signature(SOURCE)
 with zipfile.ZipFile(SOURCE)as source,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True)as target:
  assert set(members)<=set(source.namelist())
  for index,name in enumerate(members,1):
   data=source.read(name);info=source.getinfo(name);assert len(data)==info.file_size
   hashes[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),crc32=f'{info.CRC:08x}');target.writestr(name,data)
   if index%2000==0:dump(OUT/'states/rxrx1_input_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='copying_and_CRC_SHA_verifying_complete_required_members',files_verified=index,files_required=len(members)))
 assert signature(SOURCE)==start
 tmp.replace(PACK);index=OUT/'validation/rxrx1_v4_member_hashes.json';dump(index,hashes)
 dump(MANIFEST,dict(dataset='rxrx1-cross',protocol='official-cross-experiment-core',records=records,source_csv=str(CSV),source_csv_sha256=sha(CSV),source_archive=str(SOURCE),source_archive_stat=start,packed_archive=str(PACK),packed_archive_sha256=sha(PACK),packed_archive_stat=signature(PACK),member_index=str(index),member_index_sha256=sha(index),grouping='Actual experiment/plate/well from official site_id; two fields in each culture well grouped together',group_limitation='Existing balanced cross-experiment core, not a newly selected subset. Gallery and query experiment IDs and wells disjoint. Conditional on fixed gallery and the specified sampled experiments; well-level inference does not estimate unseen-experiment uncertainty.',primary_grouping_verified=True,primary_retrieval_aggregation='global',secondary_retrieval_aggregations=['HEPG2','HUVEC','RPE','U2OS','macro-cell-type']))
 dump(OUT/'states/rxrx1_input_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='complete_verified',files_verified=len(members),views=17728,manifest=str(MANIFEST)))
def verified_retrieval_pilot(pilot):
 """Gate inference on its completed real retrieval task, not five CPU fits.

 The unchanged shared v4 clustering solver continues in each bundled job;
 component_result still requires all five fits and is the completion target.
 """
 import numpy as np
 path=OUT/'results'/pilot['model']/'retrieval_clustering/rxrx1-cross/retrieval_result.json'
 if not path.exists():return False
 r=json.loads(path.read_text());manifest=json.loads(MANIFEST.read_text());records=manifest['records']['query']
 assert r['checkpoint']==pilot['checkpoint']and r['manifest_sha256']==sha(MANIFEST)and r['n_samples']==8864 and r['n_gallery']==8864
 assert r['task']=='retrieval'and r['required_seeds']==0 and r['aggregation']=='global'
 assert {x['aggregation']for x in r['secondary_cell_types']}=={'HEPG2','HUVEC','RPE','U2OS'}
 for x in [r,*r['secondary_cell_types']]:
  prediction=Path(x['prediction_path']);assert sha(prediction)==x['prediction_sha256']
  expected=records if x is r else [v for v in records if v['cell_type']==x['aggregation']]
  with np.load(prediction,allow_pickle=False)as z:
   assert z['sample_id'].tolist()==[v['sample_id']for v in expected]
   assert z['group_id'].tolist()==[v['group_id']for v in expected]
   assert z['labels'].tolist()==[v['label']for v in expected]
   assert z['neighbor_indices'].shape==z['neighbor_scores'].shape and len(z['neighbor_indices'])==len(expected)
   assert np.all((z['neighbor_indices']>=0)&(z['neighbor_indices']<8864))and np.isfinite(z['neighbor_scores']).all()
   for key,value in x['metrics'].items():assert np.isfinite(z[key]).all()and abs(float(z[key].mean())-value)<1e-12
 for key,value in r['macro_cell_type_metrics'].items():assert abs(value-np.mean([x['metrics'][key]for x in r['secondary_cell_types']]))<1e-12
 proof=dict(status='PASS',model=pilot['model'],checkpoint=pilot['checkpoint'],manifest_sha256=r['manifest_sha256'],retrieval_result=str(path),prediction_sha256=r['prediction_sha256'],n_gallery=8864,n_query=8864,scope='Complete original-v4 retrieval task; clustering completion remains five genuine fits per model; no solver or budget changes')
 dump(OUT/'validation/rxrx1_retrieval_pilot.json',proof)
 return True
def admit():
 cfg=json.loads((OUT/'config.json').read_text());pilot=next(m for m in cfg['models']if m['model']=='biodino_1tb_splus');passed=verified_retrieval_pilot(pilot)
 jobs=[]
 for spec in cfg['models']if passed else[pilot]:
  m=spec['model'];key=m+'__retrieval_clustering__rxrx1-cross';j=dict(key=key,model=m,task='retrieval_clustering',dataset='rxrx1-cross',fold='fixed',seed='bundled0to4',entrypoint='run_unified_paper_rxrx1.py',args=['--model',m],claim_key=key,min_free_mib=16000 if spec['min_free_mib']>=16000 else 6000,result=str(OUT/'results'/m/'retrieval_clustering/rxrx1-cross/component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
  j.update(requirements(spec,'rxrx1-cross'))
  if not passed:j['allowed_hosts']=['cpu19']
  if spec.get('execution_host'):j['blocked_reason']='Complete RxRx1 core requires resident-host preparation; no checkpoint transfer'
  jobs.append(j)
 path=OUT/'rxrx1_retrieval_queue.json';old=json.loads(path.read_text())if path.exists()else{}
 if old.get('tasks')!=jobs:dump(path,dict(tasks=jobs))
 dump(OUT/'states/rxrx1_retrieval_admission.json',dict(pid=os.getpid(),updated=time.time(),status='all_models_admitted_after_real_retrieval_pilot'if passed else'one_real_complete_core_pilot',jobs=len(jobs),completion_requirement='Complete retrieval and all five original-v4 clustering fits; pilot gating does not mark pending fits complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with(OUT/'locks/rxrx1_preparation.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  prepare()
  while True:
   admit()
   if not a.watch:break
   time.sleep(45)
