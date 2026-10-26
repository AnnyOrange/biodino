#!/usr/bin/env python3
"""Existing within-set retrieval and five MiniBatchKMeans fits with retained outputs."""
import argparse,json,os,sys,hashlib
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder,extract_features
from dinov3.eval.bio_frozen_eval.retrieval_clustering import build_retrieval_dataset,retrieval_metrics,clustering_metrics
from unified_paper_resident_retrieval_inputs import canonical_path,CanonicalImageIds

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['nct-crc-he-1k','crc-val-he-7k','nct-crc-he-100','lc25000'],required=True);a=p.parse_args()
 torch.set_num_threads(2);np.random.seed(0);torch.manual_seed(0)
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 out=OUT/'results'/a.model/'retrieval_clustering'/a.dataset;out.mkdir(parents=True,exist_ok=True);result=out/'component_result.json'
 if result.exists():return
 manifest_path=ROOT/'outputs/02_eval_inputs/formal_v4'/f'{a.dataset}_retrieval_manifest.json';manifest=json.loads(manifest_path.read_text())
 dataset,classes=build_retrieval_dataset(a.dataset,benchmark_root=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark'))
 if hasattr(dataset,'samples'):
  ids=[canonical_path(row[0]) for row in dataset.samples];labels=np.array([row[1] for row in dataset.samples],dtype=int)
  image_hashes=[sha(row[0]) for row in dataset.samples]
 else:
  ids=[str(row[2]) for row in dataset.rows];labels=np.array([row[1] for row in dataset.rows],dtype=int)
  image_hashes=[hashlib.sha256(row[0]).hexdigest() for row in dataset.rows]
 assert ids==[r['sample_id'] for r in manifest['records']] and labels.tolist()==[r['label'] for r in manifest['records']]
 assert image_hashes==[r['image_sha256'] for r in manifest['records']]
 features_path=out/'features.npz'
 if not features_path.exists():
  if spec['kind']=='biodino':
   encoder=Dinov3CkptEncoder(checkpoint=Path(spec['checkpoint']),train_config=Path(spec['train_config']),device='cuda:0',n_last_blocks=1,use_avgpool=True,autocast_dtype=torch.bfloat16,image_size=224,resize_size=256,channel_policy='auto')
  else:
   from run_unified_paper_global import make_encoder
   encoder=make_encoder(spec,a.dataset,224,256)
  original=encoder.encode_images;batch=4 if spec['min_free_mib']>=16000 else 16
  encoder.encode_images=lambda images:np.concatenate([original(images[i:i+batch]) for i in range(0,len(images),batch)])
  feature_temp=out/'features.pending.npz'
  features,y=extract_features(CanonicalImageIds(dataset),encoder,feature_temp,64,2,True,a.model,save_features=True,save_paths=True)
  feature_temp.replace(features_path)
  del encoder,original
  import gc
  gc.collect();torch.cuda.empty_cache()
 else:
  pack=np.load(features_path);features=pack['features'].astype(np.float32);y=pack['labels'];assert pack['paths'].tolist()==ids
 assert np.array_equal(y,labels);features=features.astype(np.float32)
 # Existing runner normalizes features during retrieval before clustering.
 retrieval,samples=retrieval_metrics(features,labels,return_samples=True)
 for key,value in retrieval.items():assert abs(np.mean([r[key] for r in samples])-value)<1e-12
 pred=out/'retrieval_predictions.npz'
 np.savez_compressed(pred,sample_id=np.array(ids),group_id=np.array([r['group_id'] for r in manifest['records']]),labels=labels,
  neighbor_indices=np.array([r['neighbor_indices'] for r in samples]),neighbor_scores=np.array([r['neighbor_scores'] for r in samples]),
  **{key:np.array([r[key] for r in samples]) for key in retrieval})
 base={'model':a.model,'checkpoint':spec['checkpoint'],'dataset':a.dataset,'protocol':'bio-eval-union-v4','split':'within-set-leave-one-out',
  'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'n_samples':len(labels),'n_classes':len(classes),'feature_cache':str(features_path),'feature_sha256':sha(features_path),
  'grouping':manifest['grouping'],'group_limitation':manifest['group_limitation'],'feature_readout':'final CLS + patch mean; published no-CLS fallback','image_size':224,'resize_size':256}
 dump(out/'retrieval_result.json',{**base,'task':'retrieval','required_seeds':0,'seed':'not_applicable','metrics':retrieval,'prediction_path':str(pred),'prediction_sha256':sha(pred)})
 clustering=[]
 for seed in range(5):
  directory=out/f'seed{seed}';directory.mkdir(exist_ok=True);target=directory/'clustering_result.json'
  if target.exists():clustering.append(json.loads(target.read_text()));continue
  metrics,assignment=clustering_metrics(features.copy(),labels,seed=seed,return_assignments=True,silhouette_seed=0)
  pred=directory/'cluster_predictions.npz';np.savez_compressed(pred,sample_id=np.array(ids),group_id=np.array([r['group_id'] for r in manifest['records']]),labels=labels,cluster=assignment)
  row={**base,'task':'clustering','required_seeds':5,'seed':seed,'seed_semantics':'MiniBatchKMeans training randomness: initialization and minibatch sampling; fixed silhouette evaluation subset seed0','silhouette_evaluation_seed':0,'metrics':metrics,'prediction_path':str(pred),'prediction_sha256':sha(pred),
   'solver':'L2-normalized float32 features; MiniBatchKMeans K=number of declared classes, batch_size2048,n_init=auto; sklearn1.8.0; original v4 settings'}
  dump(target,row);clustering.append(row);print('COMPLETE CLUSTER',a.model,a.dataset,seed,metrics,flush=True)
 dump(result,{**base,'task':'retrieval_clustering','retrieval_result':str(out/'retrieval_result.json'),'clustering_results':[str(out/f'seed{s}'/'clustering_result.json') for s in range(5)],'completed_clustering_seeds':5,'retrieval_metrics':retrieval})
 print('COMPLETE RETRIEVAL CLUSTERING',a.model,a.dataset,flush=True)
if __name__=='__main__':main()
