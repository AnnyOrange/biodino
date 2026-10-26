"""Matched source-group tests for retained frozen semantic probes."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def run_pairs(cells,settings):
    from unified_paper_segmentation_statistics import count_path,metric,dump
    models=json.loads((OUT/'config.json').read_text())['models'];available={}
    for cell in cells:
        if cell.get('superseded_protocol'):continue
        if cell['dataset']=='multimodal_cellseg' and cell['split']!='neurips22-author-public52-v2':continue
        for budget in [20,50]:
            path=count_path(cell,budget)
            if path.exists():available[cell['model'],cell['dataset'],budget,cell['split']]=(path,cell)
    for dataset,split in sorted({(key[1],key[3]) for key in available}):
        for budget in [20,50]:
            for candidate in [m['model'] for m in models if m['kind']=='biodino']:
                for reference in [m['model'] for m in models if m['kind']=='fm']:
                    if (candidate,dataset,budget,split) not in available or (reference,dataset,budget,split) not in available:continue
                    apath,acell=available[candidate,dataset,budget,split];bpath,bcell=available[reference,dataset,budget,split]
                    fold=split if dataset=='pannuke' else 'fixed';foldtag=f'__{split}' if dataset=='pannuke' else ''
                    target=OUT/'statistics'/f'{candidate}__vs__{reference}__segmentation__{dataset}{foldtag}__E{budget}.json'
                    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [apath,bpath]]
                    if target.exists() and json.loads(target.read_text()).get('counts_sha256')==hashes:continue
                    with np.load(apath) as z:a={k:z[k] for k in z.files}
                    with np.load(bpath) as z:b={k:z[k] for k in z.files}
                    if dataset=='livecell':
                        from unified_paper_livecell_groups import VERSION
                        if str(a.get('grouping_version',''))!=VERSION or str(b.get('grouping_version',''))!=VERSION:continue
                    # A bijection handles harmless cache-order differences; labels and groups must agree exactly.
                    assert len(a['sample_identity'])==len(b['sample_identity'])
                    assert sorted(map(tuple,a['sample_identity']))==sorted(map(tuple,b['sample_identity'])),'Segmentation paired source identities/labels differ'
                    assert np.array_equal(a['groups'],b['groups']) and np.array_equal(a['strata'],b['strata'])
                    aa,bb=a['counts'],b['counts'];assert aa.shape==bb.shape and aa.shape[0]==5
                    assert np.array_equal(aa.sum(-1),bb.sum(-1)),'Paired true pixel counts differ'
                    n=len(a['groups']);rng=np.random.default_rng(settings['random_seed']);draw=[]
                    positions=[np.flatnonzero(a['strata']==s) for s in np.unique(a['strata'])]
                    for start in range(0,settings['bootstrap_replicates'],128):
                        size=min(128,settings['bootstrap_replicates']-start);weights=np.zeros((size,n))
                        for pos in positions:weights[:,pos]=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)),size=size)
                        draw.extend((metric(np.einsum('bn,snij->bsij',weights,aa))-metric(np.einsum('bn,snij->bsij',weights,bb))).tolist())
                    low,high=np.quantile(draw,[.025,.975]);delta=float(metric(aa.sum(1))-metric(bb.sum(1)))
                    rng=np.random.default_rng(settings['random_seed']);extreme=0;difference=aa-bb;asum=aa.sum(1);bsum=bb.sum(1)
                    for start in range(0,settings['permutation_replicates'],128):
                        swaps=rng.integers(2,size=(min(128,settings['permutation_replicates']-start),n))
                        moved=np.einsum('bn,snij->bsij',swaps,difference);d=metric(asum[None]-moved)-metric(bsum[None]+moved)
                        extreme+=np.count_nonzero(np.abs(d)>=abs(delta)-1e-12)
                    if target.exists():
                        from unified_paper_livecell_groups import archive_previous
                        archive_previous(target)
                    dump(target,{'candidate':candidate,'reference':reference,'dataset':dataset,'task':'segmentation','budget':budget,'fold':fold,'split':split,
                        'n_test':len(a['sample_identity']),'n_groups':n,'training_seeds':list(range(5)),'counts_sha256':hashes,'counts_paths':[str(apath),str(bpath)],
                        'bootstrap_replicates':settings['bootstrap_replicates'],'permutation_replicates':settings['permutation_replicates'],'random_seed':settings['random_seed'],
                        'planned_comparisons':sum(m['kind']=='biodino' for m in models)*sum(m['kind']=='fm' for m in models),
                        'primary_independent_groups_verified':dataset not in ['pannuke','tissuenet','multimodal_cellseg'],
                        'primary_grouping_verified':dataset not in ['pannuke','tissuenet','multimodal_cellseg'],
                        'grouping_version':str(a.get('grouping_version','source-manifest-v1')),
                        'test_definition':'Two-sided paired source-group swap of all five individual fitted-head predictions; recompute pooled-pixel class-macro Dice for each seed then mean. Same group bootstrap weights for both models; CoNIC stratified by source cohort. Separate fold/budget families with Holm. Missing donor/acquisition mapping yields exploratory image-conditional inference, never additional independent biological samples.',
                        'metrics':{'mDice':{'difference':delta,'ci_low':float(low),'ci_high':float(high),'p_two_sided':float((extreme+1)/(settings['permutation_replicates']+1))}}})
                    print('SEG PAIR',candidate,reference,dataset,budget,flush=True)
