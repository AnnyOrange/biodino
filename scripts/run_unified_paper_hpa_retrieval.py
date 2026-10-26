"""Existing HPA v4 task with shared frozen features and all sample outputs."""
import argparse,hashlib,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from prepare_unified_hpa_retrieval import MANIFEST,datasets,sha
from dinov3.eval.bio_frozen_eval.encoder import extract_features
from dinov3.eval.bio_frozen_eval.retrieval_clustering import query_gallery_metrics,clustering_metrics,remap_labels
from run_unified_paper_global import make_encoder
from unified_paper_resident_retrieval_inputs import actual_path,canonical_path,CanonicalImageIds
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def main():
    from unified_paper_gpu_admission import reserve_for_current_command,release_current_reservation
    reserve_for_current_command()
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();torch.set_num_threads(2);np.random.seed(0);torch.manual_seed(0)
    spec=next(m for m in json.loads((OUT/'config.json').read_text())['models']if m['model']==a.model)
    out=OUT/'results'/a.model/'retrieval_clustering/hpa-subcellular';out.mkdir(parents=True,exist_ok=True)
    if(out/'component_result.json').exists():return
    manifest=json.loads(MANIFEST.read_text());ds=datasets();arrays={};encoder=None
    for path,digest in manifest['source_hashes'].items():assert sha(actual_path(path))==digest
    for role,dataset in ds.items():
        rows=manifest['records'][role];assert len(dataset)==len(rows)
        assert [canonical_path(dataset.root/r['image_path'])for r in dataset.rows]==[r['sample_id']for r in rows]
        assert dataset.labels.tolist()==[r['label']for r in rows]
        for r in rows:assert sha(actual_path(r['sample_id']))==r['image_sha256']
        cache=out/f'{role}_features.npz'
        if not cache.exists():
            if encoder is None:
                encoder=make_encoder(spec,'hpa-subcellular',224,256);original=encoder.encode_images;batch=4 if spec['min_free_mib']>=16000 else 16
                encoder.encode_images=lambda images:np.concatenate([original(images[i:i+batch])for i in range(0,len(images),batch)])
            pending=cache.with_suffix('.pending.npz');x,y=extract_features(CanonicalImageIds(dataset),encoder,pending,64,2,True,a.model,save_features=True,save_paths=True);pending.replace(cache)
        with np.load(cache)as z:
            assert z['paths'].tolist()==[r['sample_id']for r in rows]and z['labels'].tolist()==[r['label']for r in rows]
            arrays[role]=(z['features'].astype('float32'),z['labels'].copy())
    if encoder is not None:
        del encoder,original
        import gc
        gc.collect();torch.cuda.empty_cache()
    release_current_reservation()
    common=dict(model=a.model,checkpoint=spec['checkpoint'],dataset='hpa-subcellular',manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),feature_readout='Final frozen CLS plus patch mean; published no-CLS fallback',image_size=224,resize_size=256,grouping=manifest['grouping'],group_limitation=manifest['group_limitation'],primary_grouping_verified=manifest['primary_grouping_verified'])
    metric,samples=query_gallery_metrics(*arrays['gallery'],*arrays['query'],metric_device='cpu',return_samples=True)
    for k,v in metric.items():assert abs(v-np.mean([r[k]for r in samples]))<1e-12
    query=manifest['records']['query'];pred=out/'retrieval_predictions.npz'
    np.savez_compressed(pred,sample_id=np.array([r['sample_id']for r in query]),group_id=np.array([r['group_id']for r in query]),labels=arrays['query'][1],neighbor_indices=np.array([r['neighbor_indices']for r in samples]),neighbor_scores=np.array([r['neighbor_scores']for r in samples]),**{k:np.array([r[k]for r in samples])for k in metric})
    dump(out/'retrieval_result.json',{**common,'task':'retrieval','protocol':'custom-v1-same-gene-query-gallery','split':'fixed-query-gallery','seed':'not_applicable','required_seeds':0,'n_samples':len(query),'n_gallery':len(arrays['gallery'][1]),'metrics':metric,'prediction_path':str(pred),'prediction_sha256':sha(pred)})
    x,y=arrays['cluster'];labels=remap_labels(y);cluster=manifest['records']['cluster'];robust=np.array([r['robust_ge10']=='1'for r in cluster])
    for seed in range(5):
        directory=out/f'seed{seed}';directory.mkdir(exist_ok=True);path=directory/'clustering_result.json'
        if path.exists():continue
        values,assignment=clustering_metrics(x.copy(),labels,seed,return_assignments=True,silhouette_seed=0);pred=directory/'cluster_predictions.npz'
        np.savez_compressed(pred,sample_id=np.array([r['sample_id']for r in cluster]),group_id=np.array([r['group_id']for r in cluster]),labels=labels,cluster=assignment)
        robust_values,robust_assignment=clustering_metrics(x[robust].copy(),remap_labels(y[robust]),seed,return_assignments=True,silhouette_seed=0)
        secondary=directory/'robust_cluster_predictions.npz';np.savez_compressed(secondary,sample_id=np.array([r['sample_id']for r in cluster])[robust],group_id=np.array([r['group_id']for r in cluster])[robust],labels=remap_labels(y[robust]),cluster=robust_assignment)
        dump(path,{**common,'task':'clustering','protocol':'custom-v1-single-location-all41','split':'full-single-location-cohort','seed':seed,'required_seeds':5,'n_samples':len(labels),'metrics':values,'prediction_path':str(pred),'prediction_sha256':sha(pred),'seed_semantics':'MiniBatchKMeans initialization and minibatch sampling; fixed silhouette subset seed0','silhouette_evaluation_seed':0,'secondary_ge10_34':dict(protocol='custom-v1-single-location-ge10-34',n_samples=int(robust.sum()),metrics=robust_values,prediction_path=str(secondary),prediction_sha256=sha(secondary))})
        print('HPA CLUSTER COMPLETE',a.model,seed,values,flush=True)
    dump(out/'component_result.json',{**common,'task':'retrieval_clustering','protocol':manifest['protocol'],'n_samples':len(labels),'n_query':len(query),'n_gallery':len(arrays['gallery'][1]),'completed_clustering_seeds':5,'retrieval_result':str(out/'retrieval_result.json'),'clustering_results':[str(out/f'seed{s}/clustering_result.json')for s in range(5)],'retrieval_metrics':metric})
    print('HPA COMPLETE',a.model,metric,flush=True)
if __name__=='__main__':main()
