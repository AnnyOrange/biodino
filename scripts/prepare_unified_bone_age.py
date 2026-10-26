#!/usr/bin/env python3
"""Validate official RSNA validation payload; never replace absent ages with sex/IDs."""
import argparse,json,os,time,zipfile,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';DATA=Path('/mnt/huawei_deepcad/benchmark/ood/ood_regression/datasets/RSNA_Bone_Age');ZIP=DATA/'raw/official_validation_verified.zip';DEST=DATA/'extracted/official_validation';STATE=OUT/'bone_age_readiness.json'
def run():
 state={'pid':os.getpid(),'updated':time.time(),'source_url':'https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip','archive':str(ZIP),'official_validation_expected':1425,'official_test_expected':200,'training_expected':12611}
 if ZIP.with_suffix('.zip.aria2').exists() or not ZIP.exists():state.update(status='DOWNLOADING_OFFICIAL_VALIDATION',blocker='Official validation ZIP is incomplete; do not open preallocated1.1GB file as completed data')
 else:
  marker=DEST/'validated_archive.json'
  if not marker.exists():
   with zipfile.ZipFile(ZIP) as z:
    error=z.testzip()
    if error:raise ValueError(f'Corrupt official member {error}')
    for name in z.namelist():
     target=(DEST/name).resolve()
     if not target.is_relative_to(DEST.resolve()):raise ValueError('Unsafe archive path')
    z.extractall(DEST)
   marker.write_text(json.dumps({'sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'members':z.namelist(),'crc':'PASS'}))
  for nested in DEST.rglob('*.zip'):
   nested_marker=nested.with_suffix('.validated.json')
   if not nested_marker.exists():
    with zipfile.ZipFile(nested) as z:
     if z.testzip():raise ValueError('Corrupt nested validation archive')
     for name in z.namelist():
      if not (nested.parent/name).resolve().is_relative_to(DEST.resolve()):raise ValueError('Unsafe nested archive path')
     z.extractall(nested.parent)
    nested_marker.write_text(json.dumps({'crc':'PASS','sha256':hashlib.sha256(nested.read_bytes()).hexdigest()}))
  images=[p for p in DEST.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff','.dcm'] and not p.name.startswith('._')];tables=[];age_tables=[]
  for p in DATA.rglob('*.csv'):
   try:
    df=pd.read_csv(p);cols=df.columns.tolist();tables.append(dict(path=str(p),columns=cols,rows=len(df)))
    if any('boneage' in c.lower().replace('_','').replace(' ','') or c.lower().replace(' ','') in ['skeletalage','ageinmonths'] for c in cols):age_tables.append(str(p))
   except Exception:continue
  official_age=[p for p in age_tables if str(DEST) in p]
  state.update(official_validation_images=len(images),annotation_tables=tables,age_tables=age_tables,official_validation_age_tables=official_age,archive_verified=True)
  if not official_age:state.update(status='BLOCKED_OFFICIAL_VALIDATION_AGE_LABELS',blocker='RSNA official validation ZIP contains images but no age-labelled CSV. Existing200-case test CSV has Case ID/Sex only; training age CSV cannot label held-out validation/test. No random training holdout substituted.')
  else:
   val=pd.read_csv(official_age[0]);trainpath=DATA/'extracted/boneage-training-dataset.csv';train=pd.read_csv(trainpath);vi={p.stem:p for p in images};ti={p.stem:p for p in (DATA/'extracted/boneage-training-dataset').rglob('*.png')}
   assert len(vi)==1425 and len(ti)==12611 and set(vi)==set(val['Image ID'].astype(str)) and set(ti)==set(train.id.astype(str)) and not set(vi)&set(ti)
   records=[]
   for split,df,col,age,paths in [('train',train,'id','boneage',ti),('official_validation',val,'Image ID','Bone Age (months)',vi)]:
    for _,row in df.iterrows():
     sid=str(int(row[col]));records.append(dict(sample_id=sid,group_id=sid,path=str(paths[sid]),split=split,age_months=float(row[age])))
   manifest=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json'
   payload=dict(protocol='rsna-bone-age-official-validation-ridge-v1',records=records,n_train=12611,n_evaluation=1425,evaluation_cohort='Official validation; not hidden200test',metric='MAE in months',probe='Frozen final CLS+patch mean; train-only StandardScaler+Ridge alpha1, image-only; deterministic, no fake training seeds',label_paths=[str(trainpath),official_age[0]],grouping='Released caseID; train/validation IDs disjoint; unavailable repeated-patient linkage explicitly not certified')
   if manifest.exists():assert json.loads(manifest.read_text())==payload
   else:manifest.write_text(json.dumps(payload,indent=2))
   state.update(status='READY_OFFICIAL_VALIDATION',blocker='',manifest=str(manifest),official_validation_images=len(vi),training_images=len(ti))
 STATE.write_text(json.dumps(state,indent=2));print(state['status'],flush=True);return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  s=run()
  if not a.watch or s['status']!='DOWNLOADING_OFFICIAL_VALIDATION':break
  time.sleep(45)
