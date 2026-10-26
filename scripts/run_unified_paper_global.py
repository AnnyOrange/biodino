#!/usr/bin/env python3
"""Shared deterministic v4 probe with exact memberships and retained fitted predictions."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
os.environ['DINOV3_CODE_ROOT']=str(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from dinov3.eval.bio_frozen_eval.registry import build_dataset,NATIVE_TEST_SPLIT_DATASETS,UNSUPPORTED_OFFICIAL_SPLIT_DATASETS
from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder,extract_features
from dinov3.eval.bio_frozen_eval.group_keys import GROUP_SPLIT_DATASETS,group_keys
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
from dinov3.eval.bio_frozen_eval.run_classification import split_protocol_for_dataset,resolve_image_size,resolve_dataset_resize_size,_validate_explicit_split_labels
from dinov3.eval.bio_frozen_eval import probes
from run_shared_frozen_stage1_20260917 import identities as source_identities
BENCHMARK_ROOT=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark')

def identities(dataset,split):
 records,labels=source_identities(dataset,split)
 if hasattr(dataset,'coco_image_ids'):
  assert len(records)==len(dataset.coco_image_ids)
  records=[(name+'#image_id='+image_id,label) for (name,label),image_id in zip(records,dataset.coco_image_ids)]
 canonical=os.environ.get('UNIFIED_PAPER_CANONICAL_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark')
 if canonical!=BENCHMARK_ROOT:
  records=[(canonical+name[len(BENCHMARK_ROOT):] if name.startswith(BENCHMARK_ROOT+'/') else name,label) for name,label in records]
 if hasattr(dataset,'coco_image_ids'):
  records=[(name.replace(canonical+'/segmentation/LIVECell/LIVECell_dataset_2021/',canonical+'/Regression/LIVECell_Cell_Count/data/'),label) for name,label in records]
 return records,labels

def make_encoder(spec,dataset,size,resize):
 if spec['kind']=='biodino':
  return Dinov3CkptEncoder(checkpoint=Path(spec['checkpoint']),train_config=Path(spec['train_config']),device='cuda:0',n_last_blocks=1,use_avgpool=True,autocast_dtype=torch.bfloat16,image_size=size,resize_size=resize,channel_policy='auto')
 from run_new_fm_v4_linear_20261016 import MODELS
 if spec['model'] in MODELS:
  from unified_paper_encoders import NewFormalGlobalEncoder
  encoder=NewFormalGlobalEncoder(spec['model'],'cuda:0',64);encoder.set_dataset(dataset);return encoder
 from run_fm_rules import StandardExternalEncoder
 return StandardExternalEncoder(spec['model'],size,resize)

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--cache-only',action='store_true');a=p.parse_args()
 cfg=json.loads((OUT/'config.json').read_text());spec=next(m for m in cfg['models'] if m['model']==a.model)
 fingerprints=dict(cfg.get('source_fingerprints',{}).get(a.dataset,{}));input_proof=cfg.get('global_input_proofs',{}).get(a.dataset);verified_stats={};proof={}
 if input_proof:
  proof_path=Path(input_proof['path']);assert hashlib.sha256(proof_path.read_bytes()).hexdigest()==input_proof['sha256']
  proof=json.loads(proof_path.read_text());fingerprints.update(proof['source_fingerprints']);verified_stats=proof.get('source_stats',{})
 for path,expected in fingerprints.items():
  stat=Path(path).stat();signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
  if verified_stats.get(path)==signature:continue
  with Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
  if actual!=expected:raise ValueError(f'Remote data source differs from shared canonical file: {path}')
 if a.dataset in UNSUPPORTED_OFFICIAL_SPLIT_DATASETS:raise ValueError('Open-set task requires its existing retrieval/classification evaluator')
 if a.dataset not in NATIVE_TEST_SPLIT_DATASETS and a.dataset not in GROUP_SPLIT_DATASETS:raise ValueError('No verified native or grouped split for this entry')
 torch.set_num_threads(2);np.random.seed(0);torch.manual_seed(0)
 out=OUT/'results'/a.model/'global'/a.dataset;out.mkdir(parents=True,exist_ok=True);target=out/'component_result.json'
 if target.exists():return
 size=resolve_image_size(a.dataset,'best',224);resize=resolve_dataset_resize_size(a.dataset,size,0)
 if a.cache_only:
  class CachedOnly:
   def encode_images(self,images):raise ValueError('Cache-only repair forbids encoder extraction')
  for part in ['train','test'] if a.dataset in NATIVE_TEST_SPLIT_DATASETS else ['train']:
   with np.load(out/f'{part}_features.npz') as z:
    assert str(z['model'])==a.model and len(z['features'])==len(z['labels'])
  encoder=CachedOnly()
 else:encoder=make_encoder(spec,a.dataset,size,resize)
 # Frozen inference microbatch does not alter downstream solver/training budget.
 original_encode=encoder.encode_images
 microbatch=2 if spec['min_free_mib']>=16000 or size>224 else 8
 inference_batches=set();oom_backoffs=0;calibrated=False
 def encode_micro(images):
  nonlocal microbatch,oom_backoffs,calibrated
  chunks=[];offset=0
  while offset<len(images):
   n=min(microbatch,len(images)-offset)
   free_before=torch.cuda.mem_get_info()[0];baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
   try:encoded=original_encode(images[offset:offset+n])
   except torch.cuda.OutOfMemoryError:
    if n<=1:raise
    microbatch=max(1,n//2);oom_backoffs+=1;calibrated=True;torch.cuda.empty_cache();continue
   chunks.append(encoded);offset+=n;inference_batches.add(n)
   if not calibrated and n==microbatch:
    transient=max(torch.cuda.max_memory_allocated()-baseline,64*2**20)
    # Reserve half the observed free VRAM for other jobs and fragmentation.
    capacity=int(n*(max(0,free_before-2*2**30)*.5)/transient)
    limit=8 if size>224 else 32
    microbatch=max(1,min(limit,max(n,capacity)));calibrated=True
    print('FROZEN INFERENCE BATCH',a.model,a.dataset,microbatch,'initial_peak_delta_mib',transient/2**20,flush=True)
  return np.concatenate(chunks)
 encoder.encode_images=encode_micro
 ds,task=build_dataset(a.dataset,'train',None,None,benchmark_root=BENCHMARK_ROOT)
 if proof.get('source_order',{}).get('train'):
  from verify_unified_global_inputs import align_source_order
  align_source_order(ds,proof['source_order']['train'],BENCHMARK_ROOT)
 records,_=identities(ds,'train');x,y=extract_features(ds,encoder,out/'train_features.npz',64,2,False,a.model,save_features=True,save_paths=True)
 split=split_protocol_for_dataset(a.dataset);test_records=[];train_records=[]
 if a.dataset in NATIVE_TEST_SPLIT_DATASETS:
  test_ds,_=build_dataset(a.dataset,'test',None,None,benchmark_root=BENCHMARK_ROOT)
  if proof.get('source_order',{}).get('test'):
   from verify_unified_global_inputs import align_source_order
   align_source_order(test_ds,proof['source_order']['test'],BENCHMARK_ROOT)
  test_records,_=identities(test_ds,'test')
  xt,yt=extract_features(test_ds,encoder,out/'test_features.npz',64,2,False,a.model,save_features=True,save_paths=True);train_records=records
  # Public compressed MedMNIST lacks donor mapping: preserve image identifiers and document the limitation.
  test_groups=[r[0] for r in test_records];group_note='Published test image IDs; higher-level donor/source metadata unavailable in this loader; inference conditional on image independence'
  if a.dataset=='livecell-cell-count':
   test_groups=[r[0].split('#image_id=')[0] for r in test_records]
   group_note='Official COCO annotation records retained; repeated records of the same original image share one source-image group. Acquisition/patient grouping unresolved; image-conditional inference only.'
 elif a.dataset=='bbbc013':
  test_records=records;test_groups=[v.split('-')[0] for v in group_keys(a.dataset,ds)];group_note='Replicate row within compound; 4 OOF folds per compound; all 12 wells in a replicate stay together for uncertainty'
 else:
  tr,te=group_split_indices(a.dataset,ds,BENCHMARK_ROOT);groups=group_keys(a.dataset,ds,BENCHMARK_ROOT)
  train_records=[records[i] for i in tr];test_records=[records[i] for i in te];test_groups=[groups[i] for i in te];group_note='Existing committed group_keys source unit'
  if set(groups[i] for i in tr)&set(test_groups):raise ValueError('Group leakage')
  xt,yt=x[te],y[te];x,y=x[tr],y[tr]
 # CPU solvers no longer need the frozen encoder or its bound-method closure.
 del encoder,original_encode
 import gc
 gc.collect();torch.cuda.empty_cache()
 from unified_paper_gpu_admission import release_current_reservation
 release_current_reservation()
 if a.dataset=='bbbc013':result=probes.run_bbbc013_compound_oof_probe(x,y,[r[0] for r in records])
 else:
  _validate_explicit_split_labels(task,y,yt,a.dataset)
  fn={'classification':probes.run_classification_probe_split,'multilabel_classification':probes.run_multilabel_classification_probe_split,'regression':probes.run_regression_probe_split}[task]
  result=fn(x,y,xt,yt)
 if result.predictions is None:raise RuntimeError('Missing fitted predictions')
 ids=[r[0] for r in test_records]
 if len(ids)!=result.n_test or len(set(ids))!=len(ids):raise ValueError('Prediction identity mismatch')
 payload={**result.predictions,'sample_id':np.asarray(ids),'group_id':np.asarray(test_groups)}
 if a.dataset!='bbbc013' and set(r[0] for r in train_records)&set(ids):raise ValueError('Train/test membership overlap')
 pp=out/'test_predictions.npz';tmp=out/'test_predictions.tmp.npz';np.savez_compressed(tmp,**payload);tmp.replace(pp)
 manifest={'protocol':split,'train':train_records,'test':test_records,'test_groups':test_groups,'group_note':group_note}
 mp=out/'sample_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 row={**result.to_dict(),'model':a.model,'dataset':a.dataset,'task':'classification' if task=='multilabel_classification' else task,'checkpoint':spec['checkpoint'],
  'protocol':'bio-eval-union-v4','split':split,'training_seed':'not_applicable','solver_seed':0,'required_seeds':0,
  'probe':'StandardScaler+Ridge(alpha=1)' if task=='regression' else 'StandardScaler+balanced-LBFGS-LogisticRegression',
  'feature_readout':'final CLS + patch mean; documented no-CLS native fallback','prediction_path':str(pp),'sample_manifest':str(mp),
  'split_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'image_size':size,'resize_size':resize,'feature_inference_microbatch':microbatch,'observed_inference_batches':sorted(inference_batches),'inference_oom_backoffs':oom_backoffs,'group_note':group_note,
  'feature_cache':str(out),'checkpoint_selection':'fixed predeclared checkpoint','resident_input_proof':input_proof}
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(target);print('COMPLETE',json.dumps(row),flush=True)
if __name__=='__main__':main()
