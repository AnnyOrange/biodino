"""One preregistered, label-free prototype-Fisher refinement; CPU bank diagnostic."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans

REPO=Path('/mnt/huawei_deepcad/dinov3')
PREVIOUS=REPO/'outputs/00_reports/gram_replacement_design_20260924/stage_a/refined_controls'
OUT=PREVIOUS.parent/'prototype_fisher'
spec=importlib.util.spec_from_file_location('drc_frozen_validation',PREVIOUS/'validation_source.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=OUT
original_fit=base.fit_directions


def functional_fit(x,y,xc,yc,rank=4):
    mx,my,scale,D,_,_,previous=original_fit(x,y,xc,yc,rank)
    cluster=MiniBatchKMeans(n_clusters=64,n_init=3,max_iter=100,batch_size=1024,random_state=0).fit(y)
    centers=cluster.cluster_centers_
    squared=np.maximum(np.sum(y*y,axis=1,keepdims=True)+np.sum(centers*centers,axis=1)[None,:]-2*y@centers.T,0)
    tau=max(float(np.median(np.min(squared,axis=1))),1e-10)
    probs=softmax((yc@centers.T-.5*np.sum(centers*centers,axis=1)[None,:])/tau,axis=1)
    covp=np.diag(probs.mean(0))-probs.T@probs/len(probs)
    ev,Q=eigh(covp);L=(scale[:,None]*centers.T/tau)@(Q*np.sqrt(np.maximum(ev,0))[None,:])
    residual=(yc-my)/scale-(xc-mx)@D
    functional=residual@L
    cov=functional.T@functional/len(functional)
    vals,V=eigh(cov,subset_by_index=[len(cov)-rank,len(cov)-1])
    U,_=np.linalg.qr(L@V[:,::-1])
    energy=np.mean(np.sum((residual@U)**2,axis=1))
    controls=[]
    for seed in range(8):
        Z,_=np.linalg.qr(np.random.default_rng(1000+seed).normal(size=(y.shape[1],rank)))
        re=np.mean(np.sum((residual@Z)**2,axis=1))
        controls.append((Z,float(np.sqrt(energy/max(re,1e-30)))))
    stats={'method':'prototype_fisher','rank':rank,'prototypes':64,'prototype_seed':0,'temperature':tau,
           'mean_assignment_entropy':float(np.mean(-np.sum(probs*np.log(np.maximum(probs,1e-30)),axis=1))),
           'top_eigenvalues':vals[::-1].tolist(),'eigenvalue_space':'historical prototype predictive-risk covariance',
           'trace_error':previous['trace_error'],'top_energy_fraction':float(energy/previous['trace_error']),
           'decoder_operator_norm':previous['decoder_operator_norm'],
           'random_energy_multipliers':[v[1] for v in controls],
           'label_use':'none for decoder, prototypes, temperature, Fisher or direction selection'}
    return mx,my,scale,D,U,controls,stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['cellpose','chammi-cp-task3','chammi-hpa-task2','chammi-allen-task2'])
    p.add_argument('--base',default='M',choices=['M']);p.add_argument('--seed',type=int,choices=[0,17],required=True)
    args=p.parse_args();OUT.mkdir(exist_ok=True);folder=OUT/f'{args.dataset}_{args.base}_seed{args.seed}'
    if (folder/'results.json').exists():raise SystemExit('Already complete; refusing to overwrite')
    base.fit_directions=functional_fit
    if args.dataset!='cellpose':
        prior=PREVIOUS/f'{args.dataset}_{args.base}_seed{args.seed}'
        cached=np.load(prior/'fitted_maps.npz',allow_pickle=False)
        manifest=json.loads((prior/'manifest.json').read_text())
        count=[0]
        def reuse_head(x,y):
            assert len(x)==manifest['counts']['fit']+manifest['counts']['discovery']
            assert np.array_equal(np.unique(y),cached['classes'])
            index=count[0];count[0]+=1;assert index in (0,1)
            return ((cached['head_W'],cached['head_b'],cached['classes']) if index==0 else
                    (cached['base_head_W'],cached['base_head_b'],cached['classes']))
        base.linear_probe=reuse_head
    start=time.time()
    try:
        base.segmentation(args) if args.dataset=='cellpose' else base.classification(args)
    except Exception as exc:
        base.write(folder/'failure.json',{'type':type(exc).__name__,'error':str(exc)})
        raise
    print('elapsed_seconds',time.time()-start,flush=True)
