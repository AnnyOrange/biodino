#!/usr/bin/env python3
"""Actual official validation bone-age regression; shared deterministic ridge."""
import argparse,json,os,sys,hashlib,socket
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from run_unified_paper_global import make_encoder
OUT=ROOT/'outputs/02_eval_runs/unified_paper';MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json'
def dump(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--rescore',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 if not a.rescore:reserve_for_current_command()
 cfg=json.loads(MANIFEST.read_text());h=hashlib.sha256(MANIFEST.read_bytes()).hexdigest();spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 packed_root=os.environ.get('UNIFIED_PAPER_BONE_AGE_PACK');packed=None;packed_meta=None
 if packed_root:
  if spec['kind']!='biodino':raise ValueError('Prepared tensor transport is certified only for this DINOv3 input path')
  packed_root=Path(packed_root);packed_meta=json.loads((packed_root/'manifest.json').read_text());assert packed_meta['canonical_manifest_sha256']==h
  assert packed_meta['n_records']==len(cfg['records']) and packed_meta['all_spatial_channels_exact']
  for name,digest in packed_meta['archive_sha256'].items():
   with (packed_root/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
  packed=np.load(packed_root/'spatial_inputs.npz');positions={r['path']:i for i,r in enumerate(cfg['records'])}
 out=OUT/'results'/a.model/'bone_age';out.mkdir(exist_ok=True,parents=True);target=out/'component_result.json'
 if target.exists() and not a.rescore:return
 if a.rescore:
  import shutil
  for name in ['component_result.json','validation_predictions.json']:
   old=out/name
   if old.exists() and not (out/(name+'.float32_archive')).exists():shutil.copy2(old,out/(name+'.float32_archive'))
 encoder=None;banks={}
 for split in ['train','official_validation']:
  records=[r for r in cfg['records'] if r['split']==split];cache=out/f'{split}_features.npy';meta=out/f'{split}_features.json'
  if cache.exists() and meta.exists():
   assert json.loads(meta.read_text())['manifest_sha256']==h;banks[split]=np.load(cache);continue
  if a.rescore:raise RuntimeError('CPU rescore requires existing verified feature caches')
  if encoder is None:
   encoder=make_encoder(spec,'xray_bone',224,256)
   if packed is not None:
    with np.load(packed_root/'raw_validation_examples.npz') as examples:
     raw=[torch.from_numpy(examples[f'raw_{i}'].astype('float32')/255).expand(3,-1,-1).contiguous() for i in range(2)]
    staged=[torch.from_numpy(packed[f'image_{i:06d}']).expand(3,-1,-1).contiguous() for i in range(2)]
    original=encoder.encode_images(raw);prepared=encoder.encode_prepared_tensor(staged)
    assert np.array_equal(original,prepared),'Resident encoder raw/prepared features differ'
    dump(out/'input_transport_validation.json',dict(checked_examples=2,features_elementwise_equal=True,all_input_spatial_tensors_verified_on_preparation=True,canonical_manifest_sha256=h,pack_manifest_sha256=hashlib.sha256((packed_root/'manifest.json').read_bytes()).hexdigest()))
  chunks=[]
  for start in range(0,len(records),4):
   images=[]
   for r in records[start:start+4]:
    if packed is not None:images.append(torch.from_numpy(packed[f"image_{positions[r['path']]:06d}"]).expand(3,-1,-1).contiguous())
    else:
     with Image.open(r['path']) as im:
      raw=np.asarray(im);assert raw.ndim==2 and raw.dtype==np.uint8
      images.append(torch.from_numpy(raw.astype('float32')/255).expand(3,-1,-1).contiguous())
   chunks.append(encoder.encode_prepared_tensor(images) if packed is not None else encoder.encode_images(images))
   if start%400==0:print('REAL BONE AGE ENCODE',a.model,split,start+len(images),len(records),flush=True)
  features=np.concatenate(chunks).astype('float32');assert np.isfinite(features).all();temp=cache.with_suffix('.tmp')
  with temp.open('wb') as f:np.save(f,features)
  temp.replace(cache);dump(meta,dict(manifest_sha256=h,model=a.model,checkpoint=spec['checkpoint'],shape=list(features.shape)));banks[split]=features
 del encoder;torch.cuda.empty_cache();tr=[r for r in cfg['records'] if r['split']=='train'];te=[r for r in cfg['records'] if r['split']=='official_validation'];y=np.array([r['age_months'] for r in tr]);yt=np.array([r['age_months'] for r in te])
 from unified_paper_gpu_admission import release_current_reservation
 release_current_reservation()
 probe=make_pipeline(StandardScaler(),Ridge(alpha=1.));probe.fit(banks['train'].astype('float64'),y);pred=probe.predict(banks['official_validation'].astype('float64'));metrics=dict(mae=float(mean_absolute_error(yt,pred)),rmse=float(np.sqrt(mean_squared_error(yt,pred))),r2=float(r2_score(yt,pred)))
 pp=out/'validation_predictions.json';dump(pp,dict(sample_id=[r['sample_id'] for r in te],group_id=[r['group_id'] for r in te],y_true=yt.tolist(),y_pred=pred.tolist(),unit='months'));dump(target,dict(model=a.model,checkpoint=spec['checkpoint'],dataset='xray_bone',task='bone_age_regression',protocol=cfg['protocol'],manifest=str(MANIFEST),manifest_sha256=h,metrics=metrics,prediction_path=str(pp),n_train=len(tr),n_test=len(te),evaluation_cohort='official_validation',seed_applicable=False,grouping=cfg['grouping'],probe=cfg['probe'],solver_precision='float64',host=socket.gethostname(),pid=os.getpid()));print('COMPLETE BONE AGE',a.model,metrics,flush=True)
if __name__=='__main__':main()
