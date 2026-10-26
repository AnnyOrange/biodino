"""Prepare complete fixed v4 center-probe cohorts, then gate the 37-model matrix."""
import argparse,fcntl,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from unified_paper_center_inputs import membership,manifest_path,sha,signature
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(path,value):
 tmp=path.with_suffix('.center.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def prepare(dataset):
 p=manifest_path(dataset)
 if p.exists():return
 records,sources=membership(dataset);hashes={};stats={}
 for i,source in enumerate(sources,1):
  before=signature(source);hashes[source]=sha(source);assert signature(source)==before;stats[source]=before
  if i%100==0:dump(OUT/'states'/f'{dataset}_center_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='verifying_full_cohort',files_verified=i,files_required=len(sources)))
 overlap=sorted({r['group_id']for r in records['train']}&{r['group_id']for r in records['val']})
 dump(p,dict(dataset=dataset,records=records,source_hashes=hashes,source_stats=stats,split_id='official-baseline-fold0-nested-v1'if dataset=='conic'else'official-train-val-test',grouping='Released original source image prefix in patch_info.csv; not patient identity'if dataset=='conic'else'Original file_name groups; COCO annotation IDs retained separately; culture/well hierarchy unavailable',primary_grouping_verified=dataset=='conic',protocol='bio-eval-union-v4',recipe=dict(batch_size=8,epochs=5,lr=.001,weight_decay=.0001,image_size=224,solver='AdamW',frozen=True,seeds=list(range(5))),validation_role='Diagnostic only; fixed final epoch5, no validation-based hyperparameter/checkpoint selection',train_val_shared_original_images=overlap,test_label_scope='Author baseline development holdout; challenge test labels unavailable'if dataset=='conic'else'Official released test annotations'))
 dump(OUT/'states'/f'{dataset}_center_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='complete_verified',files_verified=len(sources),split_counts={s:len(v)for s,v in records.items()},group_counts={s:len({r['group_id']for r in v})for s,v in records.items()},manifest=str(p)))
def admit(dataset):
 with(OUT/'locks/model_admission.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);cfg=json.loads((OUT/'config.json').read_text());pilot='biodino_1tb_splus';result=OUT/'results'/pilot/'detection_proxy'/dataset/'seed0/component_result.json';passed=False
  if result.exists():
   r=json.loads(result.read_text());assert r['batch_size']==8 and r['epochs']==5 and r['seed']==0 and r['manifest_sha256']==sha(manifest_path(dataset))and Path(r['prediction_path']).exists();passed=True
  path=OUT/'detection_queue.json';q=json.loads(path.read_text());jobs={j['key']:j for j in q['tasks']}
  for spec in cfg['models']if passed else[m for m in cfg['models']if m['model']==pilot]:
   for seed in range(5)if passed else[0]:
    model=spec['model'];key=f'{model}__detection_proxy__{dataset}__seed{seed}';j=dict(key=key,model=model,task='detection_proxy',dataset=dataset,fold='fixed',seed=seed,entrypoint='run_unified_paper_detection.py',args=['--model',model,'--dataset',dataset,'--seed',str(seed)],claim_key=key,min_free_mib=spec['min_free_mib'],result=str(OUT/'results'/model/'detection_proxy'/dataset/f'seed{seed}/component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
    if spec.get('execution_host'):j['blocked_reason']='Complete canonical source images and metadata must be verified on resident host; no checkpoint transfer'
    jobs.setdefault(key,j)
  q['tasks']=list(jobs.values());dump(path,q)
  dump(OUT/'center_probe_pilot_queue.json',dict(tasks=[j for j in jobs.values()if j['model']==pilot and j['dataset']in['conic','livecell']and j['seed']==0]))
  dump(OUT/'states'/f'{dataset}_center_admission.json',dict(pid=os.getpid(),updated=time.time(),status='all_models_admitted_after_real_pilot'if passed else'one_real_pilot_admitted',jobs=sum(j['dataset']==dataset for j in jobs.values())))
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--cache-only',choices=['conic','livecell']);a=p.parse_args()
 if a.cache_only:
  from unified_paper_center_inputs import prepare_tensor_cache
  prepare_tensor_cache(a.cache_only);return
 with(OUT/'locks/center_probes_preparation.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  while True:
   for dataset in ['conic','livecell']:
    try:prepare(dataset);admit(dataset)
    except Exception as exc:dump(OUT/'states'/f'{dataset}_center_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='failed',error=repr(exc)));traceback.print_exc()
   if not a.watch:break
   time.sleep(45)
if __name__=='__main__':main()
