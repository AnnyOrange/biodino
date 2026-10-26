"""Lock existing complete within-set cohorts and validate optional sample outputs."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from dinov3.eval.bio_frozen_eval.retrieval_clustering import build_retrieval_dataset,retrieval_metrics,clustering_metrics
from benchmark_eval.retrieval_clustering import retrieval_metrics as old_retrieval,clustering_metrics as old_clustering
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['nct-crc-he-1k','crc-val-he-7k'],choices=['nct-crc-he-1k','crc-val-he-7k','nct-crc-he-100','lc25000']);args=parser.parse_args()
 torch.set_num_threads(2);checks=[]
 rng=np.random.default_rng(20261021)
 for tied in [False,True]:
  x=rng.normal(size=(90,12)).astype(np.float32);y=np.arange(90)%3
  if tied:x[1::3]=x[::3]
  previous=old_retrieval(x.copy(),y);current,rows=retrieval_metrics(x.copy(),y,return_samples=True)
  assert previous==current
  for key,v in current.items():assert abs(np.mean([r[key] for r in rows])-v)<1e-12
  for seed in [0,1]:
   previous=old_clustering(x.copy(),y,seed=seed);current,pred=clustering_metrics(x.copy(),y,seed=seed,return_assignments=True)
   assert previous==current and len(pred)==len(y)
  checks.append({'tied_features':tied,'unchanged_metrics':True})
 cohorts=[]
 for name in args.datasets:
  dataset,classes=build_retrieval_dataset(name,benchmark_root='/mnt/huawei_deepcad/benchmark');records=[]
  source=[(Path(p).read_bytes(),label,str(p))for p,label in dataset.samples]if hasattr(dataset,'samples')else dataset.rows
  for encoded,label,sample in source:
   digest=hashlib.sha256(encoded).hexdigest();records.append({'sample_id':str(sample),'label':int(label),'image_sha256':digest,'group_id':digest})
  assert len({r['sample_id'] for r in records})==len(records)
  manifest={'dataset':name,'protocol':'existing v4 within-set leave-one-out; no new supervised split','records':records,'classes':classes,
   'scope':'All rows loaded by the existing published-parquet v4 loader, no sample cap','grouping':'Exact source-image byte identity; duplicate byte-identical images share a group',
   'group_limitation':'Parquet patch identifiers do not supply verified patient/slide membership. Do not label image-conditional uncertainty as patient-level uncertainty; patient-level statistics await a mapping.'}
  if name=='lc25000':
   manifest.update(scope='All25000images in the existing v4 class-folder loader; within-set descriptive retrieval, not independent heldout generalization',group_limitation='LC25000 contains augmented descendants; original-source mapping unavailable. Exact-byte groups do not establish independent origins. Descriptive full-set protocol only; no patient/source-level significance claim.')
  expected={'lc25000':25000,'nct-crc-he-100':99,'nct-crc-he-1k':999,'crc-val-he-7k':7180}
  assert len(records)==expected[name],(name,len(records))
  p=ROOT/'outputs/02_eval_inputs/formal_v4'/f'{name}_retrieval_manifest.json'
  if p.exists():assert json.loads(p.read_text())==manifest
  else:p.write_text(json.dumps(manifest,indent=2)+'\n')
  cohorts.append({'dataset':name,'n':len(records),'unique_source_images':len({r['group_id'] for r in records}),'manifest':str(p)});print(cohorts[-1],flush=True)
 result={'status':'passed','checks_against_existing_fm_metrics':checks,'cohorts':cohorts,'benchmark_result':False}
 (OUT/'validation'/('retrieval_sample_outputs.json'if args.datasets==['nct-crc-he-1k','crc-val-he-7k']else'retrieval_additional_cohorts.json')).write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
