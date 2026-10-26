"""Existing all734 eligible-gene v4 solver, with retained sample predictions."""
import argparse,hashlib,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from PIL import Image
from run_external4_fixedbudget_model import rxrx3_result
from run_unified_paper_global import make_encoder
OUT=ROOT/'outputs/02_eval_runs/unified_paper';DATA=ROOT/'outputs/02_eval_inputs/formal_v3/rxrx3-core'
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/rxrx3-core_retrieval_manifest.json'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 from unified_paper_gpu_admission import reserve_for_current_command,release_current_reservation
 reserve_for_current_command();parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);a=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models']if m['model']==a.model);manifest=json.loads(MANIFEST.read_text())
 for name,digest in manifest['source_hashes'].items():assert sha(DATA/name)==digest
 meta=json.loads((DATA/'metadata.json').read_text());y=np.load(DATA/'labels.npy');images=np.load(DATA/'images.npy',mmap_mode='r');rows=meta['rows']
 assert y.tolist()==[r['gene']for r in rows]and images.shape==(1468,224,224,3)
 out=OUT/'results'/a.model/'retrieval_clustering/rxrx3-core';out.mkdir(parents=True,exist_ok=True)
 if(out/'component_result.json').exists():return
 path=out/'features.npy'
 if not path.exists():
  encoder=make_encoder(spec,'rxrx3-core',224,256);batch=4 if spec['min_free_mib']>=16000 else 16;chunks=[]
  for start in range(0,len(images),batch):
   chunks.append(np.asarray(encoder.encode_images([Image.fromarray(np.asarray(im))for im in images[start:start+batch]]),dtype=np.float32))
   if start%256==0:print('RXRX3 FEATURES',a.model,start,len(images),flush=True)
  x=np.concatenate(chunks);pending=out/'features.pending.npy';np.save(pending,x);pending.replace(path);del encoder
  import gc
  gc.collect();torch.cuda.empty_cache()
 else:x=np.load(path)
 assert len(x)==1468 and np.isfinite(x).all();release_current_reservation()
 gallery=[r for r in rows if r['split']=='gallery'];query=[r for r in rows if r['split']=='query'];ordered=gallery+query
 common=dict(model=a.model,checkpoint=spec['checkpoint'],dataset='rxrx3-core',protocol=meta['protocol_id'],manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),feature_readout='Final frozen CLS plus patch mean; published no-CLS fallback',feature_cache=str(path),feature_sha256=sha(path),image_size=224,resize_size=256,grouping=manifest['grouping'],group_limitation=manifest['group_limitation'],primary_grouping_verified=True,split='fixed-guide-plate-disjoint-query-gallery')
 for seed in range(5):
  directory=out/f'seed{seed}';directory.mkdir(exist_ok=True);target=directory/'clustering_result.json'
  if target.exists():continue
  result,samples=rxrx3_result(x,y,meta,seed=seed,return_samples=True)
  if seed==0:
   rank=samples['ranks'];metrics={k:result[k]for k in ['recall_at_1','recall_at_5','recall_at_10','mrr_at_10','mrr','map']};pred=out/'retrieval_predictions.npz'
   np.savez_compressed(pred,sample_id=np.array([r['key']for r in query]),group_id=np.array([r['well_id']for r in query]),labels=np.array([r['gene']for r in query]),ranks=rank,recall_at_1=(rank<=1).astype(float),recall_at_5=(rank<=5).astype(float),recall_at_10=(rank<=10).astype(float),mrr=1/rank,neighbor_indices=samples['neighbor_indices'],neighbor_scores=samples['neighbor_scores'])
   dump(out/'retrieval_result.json',{**common,'task':'retrieval','n_samples':734,'n_gallery':734,'seed':'not_applicable','required_seeds':0,'metrics':metrics,'prediction_path':str(pred),'prediction_sha256':sha(pred)})
  pred=directory/'cluster_predictions.npz';np.savez_compressed(pred,sample_id=np.array([r['key']for r in ordered]),group_id=np.array([r['well_id']for r in ordered]),labels=samples['labels'],cluster=samples['cluster'])
  dump(target,{**common,'task':'clustering','n_samples':1468,'seed':seed,'required_seeds':5,'metrics':{'nmi':result['nmi']},'solver':result['clustering_method']+'; batch_size1024; cosine-normalized float32;734clusters','seed_semantics':'MiniBatchKMeans initialization and minibatch sampling; unchanged existing v4 solver budget','prediction_path':str(pred),'prediction_sha256':sha(pred)})
  print('RXRX3 CLUSTER COMPLETE',a.model,seed,result['nmi'],flush=True)
 dump(out/'component_result.json',{**common,'task':'retrieval_clustering','n_samples':1468,'n_query':734,'n_gallery':734,'completed_clustering_seeds':5,'retrieval_result':str(out/'retrieval_result.json'),'clustering_results':[str(out/f'seed{s}/clustering_result.json')for s in range(5)]})
if __name__=='__main__':main()
