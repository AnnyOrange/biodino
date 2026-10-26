#!/usr/bin/env python3
"""One FM native CTC fold and downstream seed, reusing the shared HoVerNet flow."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1]
os.environ['DINOV3_CODE_ROOT']=str(ROOT)
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import torch
import run_ctc_native_full_hs6 as native
import unified_paper_ctc_runner as companion
from unified_paper_ctc_runner import ctcmetrics_commit

def new_spatial_backbone_class(base):
 class NewCTCSpatialBackbone(base):
  """Published 224-pixel encoder input, common stride16 HoVerNet output grid."""
  def __init__(self,name):
   super().__init__(name)
   from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN,MICRO_RGB_STD
   self.input_mean,self.input_std=MICRO_RGB_MEAN,MICRO_RGB_STD
  @torch.inference_mode()
  def get_intermediate_layers(self,images,n=1,reshape=True,**kwargs):
   grid=(images.shape[-2]//16,images.shape[-1]//16)
   images=torch.nn.functional.interpolate(images,size=(224,224),mode='bilinear',align_corners=False)
   feature=super().get_intermediate_layers(images,n=n,reshape=reshape,**kwargs)[0]
   return [torch.nn.functional.interpolate(feature,size=grid,mode='bilinear',align_corners=False)]
 return NewCTCSpatialBackbone

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--seed',type=int,choices=range(5),required=True);a=p.parse_args()
 specification=next(m for m in json.loads((ROOT/'outputs/02_eval_runs/unified_paper/config.json').read_text())['models'] if m['model']==a.model)
 MODEL_REGISTRY={};NEW_MODELS={}
 if specification['kind']!='biodino':
  from benchmark_eval.encoders import MODEL_REGISTRY
  from run_fm_companion_rules import FMSpatialBackbone
  from run_new_fm_v4_detection_20261016 import NewSpatialBackbone,MODELS as NEW_MODELS
  companion.FMSpatialBackbone=new_spatial_backbone_class(NewSpatialBackbone) if a.model in NEW_MODELS else FMSpatialBackbone
 torch.set_num_threads(2)
 out=ROOT/'outputs/02_eval_runs/unified_paper/results'/a.model/'ctc'/f'fold{a.fold}'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'component_result.json').exists():return
 companion.verify_resident_inputs()
 # This entry is single-process; distributed scheduling uses atomic directory claims.
 native.base.ctcmetrics_commit=ctcmetrics_commit
 old=native.base.set_determinism
 native.base.SEED=a.seed
 native.base.set_determinism=lambda seed=a.seed:old(seed)
 data=native.load_cache_manifest(native.CACHE/'data_manifest.json')
 if data['source_split_manifest_sha256']!='7a0f5bc2579f6ae22a2b8f2b16103530cd678466a4b2bbd1f617692b8e3cac27':raise RuntimeError('CTC split changed')
 path=MODEL_REGISTRY[a.model].path if a.model in MODEL_REGISTRY else Path(specification['checkpoint'])
 files=[path] if path.is_file() else sorted(f for f in path.rglob('*') if f.is_file() and f.suffix in ('.pth','.pt','.bin','.safetensors','.h5','.ckpt'))
 if not files:raise RuntimeError('No checkpoint weight file')
 fingerprints={str(f):native.sha256(f) for f in files}
 checkpoint_digest=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
 fold=native.fold_record(data,a.fold)
 invocation={'asset_kind':'external','model':a.model,'seed':a.seed,'seed_semantics':'training_randomness',
   'checkpoint':{'model_id':a.model,'path':str(path),'sha256':checkpoint_digest,'sha256_definition':'SHA256 of sorted weight-file SHA256 mapping','weight_file_hashes':fingerprints},
   'dataset':{'fold':a.fold,'test_domains':fold['test_domains'],'split_sha256':data['source_split_manifest_sha256']},
   'recipe':{'epochs':50,'batch_size':8,'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,'backbone':'frozen','decoder':'shared DINOHoVerNet','selection':'epoch50','seed_controls':['head_initialization','shuffle','training_crop']}}
 if specification['kind']=='biodino':
  invocation['asset_kind']='dinov3'
  invocation['checkpoint']['config_path']=specification['train_config']
 if a.model in NEW_MODELS:
  invocation['input_adaptation']='Undo shared micro normalization; published encoder normalization/channel routing at224; final map bilinear to common stride16 grid; frozen inference microbatches preserve B8 decoder training'
 (out/'invocation_manifest.json').write_text(json.dumps(invocation,indent=2)+'\n')
 print(f'START native CTC {a.model} fold={a.fold} seed={a.seed} 50epochs B8',flush=True)
 result=companion.run_ctc(invocation,out)
 if result['n_domains']!=4 or not all(int(r['ctc_metrics']['Valid'])==1 for r in result['domain_rows']):raise RuntimeError('Incomplete CTC domain score')
 result.update(model=a.model,seed=a.seed,fold=a.fold,invocation=str(out/'invocation_manifest.json'))
 native.atomic_json(out/'component_result.json',result)
 print('COMPLETE',result['macro'],flush=True)
if __name__=='__main__':main()
