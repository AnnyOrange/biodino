"""Exact v4 RxRx1 core, all retrieval aggregations and five clustering fits."""
import argparse,hashlib,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from dinov3.eval.bio_frozen_eval.retrieval_clustering import RxRx1ZipDataset,query_gallery_metrics,clustering_metrics,remap_labels
from dinov3.eval.bio_frozen_eval.encoder import extract_features
from run_unified_paper_global import make_encoder
from prepare_unified_rxrx1 import MANIFEST,sha,signature
from unified_paper_resident_retrieval_inputs import actual_path
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def resident_project_path(path):
 canonical='/mnt/huawei_deepcad/dinov3/'
 return ROOT/str(path)[len(canonical):] if str(path).startswith(canonical) else Path(path)
def dump(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 from unified_paper_gpu_admission import reserve_for_current_command,release_current_reservation
 reserve_for_current_command();p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models']if m['model']==a.model);manifest=json.loads(MANIFEST.read_text());archive=resident_project_path(manifest['packed_archive']);before=signature(archive);source_csv=actual_path(manifest['source_csv'])
 assert sha(source_csv)==manifest['source_csv_sha256']and sha(resident_project_path(manifest['member_index']))==manifest['member_index_sha256']
 if before!=manifest['packed_archive_stat']:assert sha(archive)==manifest['packed_archive_sha256']
 out=OUT/'results'/a.model/'retrieval_clustering/rxrx1-cross';out.mkdir(parents=True,exist_ok=True)
 if(out/'component_result.json').exists():return
 data={};encoder=None
 for role in ['gallery','query']:
  dataset=RxRx1ZipDataset(archive,source_csv,role);records=manifest['records'][role];assert len(dataset)==8864 and [r['site_id']for r in dataset.rows]==[r['sample_id']for r in records]and dataset.labels.tolist()==[r['label']for r in records]
  path=out/f'{role}_features.npz'
  if not path.exists():
   if encoder is None:
    encoder=make_encoder(spec,'rxrx1-cross',224,256);original=encoder.encode_images;batch=4 if spec['kind']=='biodino'or spec['min_free_mib']>=16000 else 16
    encoder.encode_images=lambda images:np.concatenate([original(images[i:i+batch])for i in range(0,len(images),batch)])
   pending=out/f'{role}_features.pending.npz';extract_features(dataset,encoder,pending,64,2,True,a.model,save_features=True,save_paths=True);pending.replace(path)
  with np.load(path)as z:
   assert z['paths'].tolist()==[r['sample_id']for r in records]and z['labels'].tolist()==[r['label']for r in records]
   data[role]=(z['features'].astype('float32'),z['labels'].copy(),dataset.cell_types.copy())
 assert signature(archive)==before,'Core archive changed during extraction'
 if encoder is not None:
  del encoder,original
  import gc
  gc.collect();torch.cuda.empty_cache()
 release_current_reservation()
 common=dict(model=a.model,checkpoint=spec['checkpoint'],dataset='rxrx1-cross',protocol=manifest['protocol'],manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),grouping=manifest['grouping'],group_limitation=manifest['group_limitation'],primary_grouping_verified=True,split='official-cross-experiment-core',feature_readout='Final frozen CLS plus patch mean; published no-CLS fallback',input_policy='All six original floating channels retained to the common encoder adapter; native multichannel BioDINO where supported, published FM channel mapping and normalization',image_size=224,resize_size=256)
 gx,gy,gcell=data['gallery'];qx,qy,qcell=data['query'];query=manifest['records']['query']
 metrics,samples=query_gallery_metrics(gx,gy,qx,qy,metric_device='cpu',return_samples=True)
 for k,v in metrics.items():assert abs(v-np.mean([r[k]for r in samples]))<1e-12
 pred=out/'retrieval_predictions.npz';np.savez_compressed(pred,sample_id=np.array([r['sample_id']for r in query]),group_id=np.array([r['group_id']for r in query]),labels=qy,cell_type=qcell,neighbor_indices=np.array([r['neighbor_indices']for r in samples]),neighbor_scores=np.array([r['neighbor_scores']for r in samples]),**{k:np.array([r[k]for r in samples])for k in metrics})
 secondary=[]
 for cell in sorted(set(qcell)):
  gi=np.flatnonzero(gcell==cell);qi=np.flatnonzero(qcell==cell);values,perquery=query_gallery_metrics(gx[gi],gy[gi],qx[qi],qy[qi],metric_device='cpu',return_samples=True);path=out/f'retrieval_{cell}_predictions.npz'
  np.savez_compressed(path,sample_id=np.array([query[i]['sample_id']for i in qi]),group_id=np.array([query[i]['group_id']for i in qi]),labels=qy[qi],neighbor_indices=gi[np.array([r['neighbor_indices']for r in perquery])],neighbor_scores=np.array([r['neighbor_scores']for r in perquery]),**{k:np.array([r[k]for r in perquery])for k in values})
  secondary.append(dict(aggregation=cell,metrics=values,n_query=len(qi),n_gallery=len(gi),prediction_path=str(path),prediction_sha256=sha(path)))
 macro={k:float(np.mean([r['metrics'][k]for r in secondary]))for k in metrics}
 dump(out/'retrieval_result.json',{**common,'task':'retrieval','seed':'not_applicable','required_seeds':0,'n_samples':8864,'n_gallery':8864,'aggregation':'global','metrics':metrics,'prediction_path':str(pred),'prediction_sha256':sha(pred),'secondary_cell_types':secondary,'macro_cell_type_metrics':macro})
 x=np.concatenate([gx,qx]);labels=remap_labels(np.concatenate([gy,qy]));records=manifest['records']['gallery']+query
 for seed in range(5):
  directory=out/f'seed{seed}';directory.mkdir(exist_ok=True);target=directory/'clustering_result.json'
  if target.exists():continue
  values,assignment=clustering_metrics(x.copy(),labels,seed,return_assignments=True,silhouette_seed=0);pred=directory/'cluster_predictions.npz'
  np.savez_compressed(pred,sample_id=np.array([r['sample_id']for r in records]),group_id=np.array([r['group_id']for r in records]),labels=labels,cluster=assignment)
  dump(target,{**common,'task':'clustering','seed':seed,'required_seeds':5,'n_samples':17728,'aggregation':'global-perturbation','metrics':values,'prediction_path':str(pred),'prediction_sha256':sha(pred),'seed_semantics':'MiniBatchKMeans initialization and minibatch sampling; fixed silhouette subset seed0','silhouette_evaluation_seed':0,'solver':'Original v4 MiniBatchKMeans K=1108,n_init=auto,batch_size2048; L2-normalized float32'})
  print('RXRX1 CLUSTER COMPLETE',a.model,seed,values,flush=True)
 dump(out/'component_result.json',{**common,'task':'retrieval_clustering','n_samples':17728,'n_query':8864,'n_gallery':8864,'completed_clustering_seeds':5,'retrieval_result':str(out/'retrieval_result.json'),'clustering_results':[str(out/f'seed{s}/clustering_result.json')for s in range(5)]})
 print('RXRX1 COMPLETE',a.model,metrics,flush=True)
if __name__=='__main__':main()
