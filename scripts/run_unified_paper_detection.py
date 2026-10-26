#!/usr/bin/env python3
"""One real shared B8/E5 detection-proxy fit with image-indexed predictions."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
os.environ['DINOV3_CODE_ROOT']=str(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import torch,numpy as np
from dinov3.eval.bio_detection import center_probe

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['bbbc038','conic','livecell'],default='bbbc038');p.add_argument('--seed',type=int,choices=range(5),required=True);p.add_argument('--recover-existing-predictions',action='store_true');a=p.parse_args()
 if not a.recover_existing_predictions:reserve_for_current_command()
 torch.set_num_threads(2);spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 out=OUT/'results'/a.model/'detection_proxy'/a.dataset/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 result=out/'component_result.json'
 if result.exists():return
 if spec['kind']=='fm':
  from run_fm_companion_rules import FMSpatialBackbone
  from run_new_fm_v4_detection_20261016 import NewSpatialBackbone,MODELS
  from benchmark_eval.encoders import MODEL_REGISTRY
  checkpoint=str(MODELS[a.model]) if a.model in MODELS else str(MODEL_REGISTRY[a.model].path)
  center_probe.load_backbone=lambda **kwargs: NewSpatialBackbone(a.model) if a.model in MODELS else FMSpatialBackbone(a.model,'imagenet')
  config=None
 else:checkpoint=spec['checkpoint'];config=spec['train_config']
 benchmark=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark')
 manifests={};canonical=None;cache_used=False
 if a.dataset=='bbbc038':
  for split in ['train','val','test']:
   images,masks=center_probe.get_bbbc038_paths(benchmark+'/segmentation/bbbc038/extracted',split=split)
   rows=[{'sample_id':Path(p).stem,'group_id':Path(p).stem,'image_path':p.replace(benchmark,'/mnt/huawei_deepcad/benchmark',1),'mask_path':m.replace(benchmark,'/mnt/huawei_deepcad/benchmark',1)} for p,m in zip(images,masks)]
   manifests[split]=rows
 else:
  from unified_paper_center_inputs import load_verified,manifest_path,sha
  canonical=load_verified(a.dataset,benchmark);manifests=canonical['records']
  cache=OUT/'prepared_inputs/center_probes'/a.dataset/'cache.json'
  if cache.exists():
   cache_used=True
   from unified_paper_center_inputs import CachedCenterDataset
   original_builder=center_probe.build_center_dataset
   def cached_builder(dataset,benchmark_root,split,image_size,max_samples,seed,conic_split_protocol='official-baseline-fold0-nested-v1'):
    assert dataset==a.dataset and image_size==224 and max_samples==0
    return CachedCenterDataset(dataset,split)
   center_probe.build_center_dataset=cached_builder
   print('EXACT SHARED INPUT TENSOR CACHE',cache,flush=True)
 for x,y in [('train','test'),('val','test'),('train','val')]:
  assert not set(r['sample_id'] for r in manifests[x])&set(r['sample_id'] for r in manifests[y])
 (out/'sample_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
 if benchmark!='/mnt/huawei_deepcad/benchmark'and a.dataset=='bbbc038':
  proof=json.loads((OUT/'validation/ai_bbbc038_resident.json').read_text())
  assert proof['membership_sha256']==hashlib.sha256(json.dumps(manifests,sort_keys=True).encode()).hexdigest()
  for path,expected in proof['source_fingerprints'].items():
   stat=Path(path).stat();signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
   if proof['source_stats'].get(path)==signature:continue
   with Path(path).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
 if a.recover_existing_predictions:
  d=json.loads((out/'results_bio_detection.json').read_text());assert d['epochs']==5 and d['batch_size']==8 and d['seed']==a.seed and d['dataset']==a.dataset and d['image_size']==224
  saved_head=torch.load(out/'final_head.pth',map_location='cpu',weights_only=True);assert saved_head['seed']==a.seed
  path=out/'test_patch_predictions.npz'
  with np.load(path)as z:payload={k:z[k]for k in z.files}
  assert np.isfinite(payload['logits']).all()and float(payload['threshold'])==.5
  # For the observed failure exactly one patch can change under float32
  # sigmoid rounding. All four original confusion-matrix metrics uniquely
  # determine that patch's label; preserve the original compressed artifact.
  assert np.count_nonzero(np.abs(payload['logits'])<1e-6)<=1,'Multiple ambiguous patches require exact original-device prediction replay'
  payload['predicted']=(torch.sigmoid(torch.from_numpy(payload['logits']))>=.5).numpy();y=payload['labels']>=.5;v=payload['predicted'];tp=np.sum(y&v);fp=np.sum(~y&v);fn=np.sum(y&~v);tn=np.sum(~y&~v)
  recovered={'test_patch_precision':100*tp/max(tp+fp,1),'test_patch_recall':100*tp/max(tp+fn,1),'test_patch_accuracy':100*(tp+tn)/(tp+tn+fp+fn),'test_patch_f1':200*tp/max(2*tp+fp+fn,1)}
  assert all(abs(recovered[k]-d[k])<1e-6 for k in recovered)
  import shutil
  archive=out/'test_patch_predictions_before_threshold_recovery.npz'
  if not archive.exists():shutil.copy2(path,archive)
  pending=out/'test_patch_predictions.recovered.npz';np.savez_compressed(pending,**payload);pending.replace(path)
  d['artifact_recovery']='No retraining: single ambiguous float32 sigmoid threshold decision uniquely reproduced against original accuracy/precision/recall/F1; old compressed logits retained'
 else:
  d=center_probe.run_bio_detection_eval(arch='vit_base',weights='',checkpoint=checkpoint,train_config=config,
   benchmark_root=benchmark,output_dir=str(out),dataset=a.dataset,batch_size=8,num_workers=2,
   image_size=224,epochs=5,lr=1e-3,autocast_dtype=torch.bfloat16,channel_policy='auto',max_samples_per_split=0,seed=a.seed,save_predictions=True)
 pred=np.load(out/'test_patch_predictions.npz');assert len(pred['labels'])==len(manifests['test'])
 label=pred['labels']>=.5;predicted=pred['predicted']if 'predicted'in pred.files else(torch.sigmoid(torch.from_numpy(pred['logits']))>=.5).numpy();tp=np.sum(label&predicted);fp=np.sum(~label&predicted);fn=np.sum(label&~predicted)
 assert abs(float(2*tp/max(2*tp+fp+fn,1)*100)-d['test_patch_f1'])<1e-6
 d.update(model=a.model,checkpoint=checkpoint,task='detection_proxy',protocol='bio-eval-union-v4',seed=a.seed,
   seed_semantics='training_randomness: head initialization and shuffle; fixed membership',
   sample_manifest=str(out/'sample_manifest.json'),prediction_path=str(out/'test_patch_predictions.npz'),
   test_ids_sha256=hashlib.sha256(json.dumps(manifests['test'],sort_keys=True).encode()).hexdigest(),
   prediction_schema='row-order sample_manifest.test; one row per source image, columns spatial patches; logits and true labels; fold=fixed; checkpoint/model/seed in this result')
 if canonical is not None:d.update(canonical_manifest=str(manifest_path(a.dataset)),manifest_sha256=sha(manifest_path(a.dataset)),grouping=canonical['grouping'],primary_grouping_verified=canonical['primary_grouping_verified'],split_id=canonical['split_id'],input_tensor_cache=str(cache)if cache_used else'not_used')
 tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(result);print('COMPLETE',a.model,a.seed,d['test_patch_f1'],flush=True)
if __name__=='__main__':main()
