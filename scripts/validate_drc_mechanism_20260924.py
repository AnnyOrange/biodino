"""CPU counterfactual repairs on existing TRAIN/VALIDATION banks; never reads test.

Diagnostic only: this is neither a trained backbone nor a formal v4 test score.
Decoder fit, direction discovery, and validation have disjoint source groups.
Random controls have equal rank and discovery-set correction energy.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

REPO = Path('/mnt/huawei_deepcad/dinov3')
BANKS = REPO / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
OUT = REPO / 'outputs/00_reports/gram_replacement_design_20260924/stage_a/refined_controls'
ROLES = {'E': ('early', 12687), 'M': ('middle', 20007), 'L': ('late', 29279)}


def digest(x):
    return hashlib.sha256('\n'.join(map(str, x)).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def group_split(groups, seed):
    idx = np.arange(len(groups))
    train, val = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(idx, groups=groups))
    fi, ca = next(GroupShuffleSplit(1, test_size=.25, random_state=seed+1).split(train, groups=groups[train]))
    fit, calibration = train[fi], train[ca]
    sets = [set(groups[i]) for i in [fit, calibration, val]]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    return fit, calibration, val


def pure_class_group_split(groups, labels, seed):
    """CP wells have one treatment each; stratify source groups, never cells."""
    rng=np.random.default_rng(seed)
    partitions=[[],[],[]]
    unique=np.unique(groups)
    group_label={g:np.unique(labels[groups==g]) for g in unique}
    assert all(len(v)==1 for v in group_label.values())
    for label in np.unique(labels):
        gg=np.array([g for g in unique if group_label[g][0]==label]);rng.shuffle(gg)
        assert len(gg)>=5,'Insufficient independent treatment groups for three partitions'
        n=max(1,round(.2*len(gg)))
        for dest,selected in zip(partitions,[gg[2*n:],gg[n:2*n],gg[:n]]):
            dest.extend(np.flatnonzero(np.isin(groups,selected)).tolist())
    return tuple(np.sort(v) for v in partitions)


def fit_directions(x, y, xc, yc, rank=4):
    """Row convention: normalized historical target y_hat = (x-mx) @ D."""
    mx, my = x.mean(0), y.mean(0)
    var = y.var(0)
    scale = np.sqrt(np.maximum(var, .05 * var.mean()))
    xx = np.asarray(x - mx, np.float64)
    yy = np.asarray((y - my) / scale, np.float64)
    gram = xx.T @ xx
    ridge = 1e-3 * np.trace(gram) / len(gram)
    gram.flat[::len(gram)+1] += ridge
    D = solve(gram, xx.T @ yy, assume_a='pos')
    residual = (yc-my)/scale - (xc-mx) @ D
    cov = residual.T @ residual / len(residual)
    vals, vecs = eigh(cov, subset_by_index=[len(cov)-rank, len(cov)-1])
    U = vecs[:, ::-1]
    top_energy = np.mean(np.sum((residual @ U)**2, axis=1))
    controls = []
    for seed in range(8):
        V, _ = np.linalg.qr(np.random.default_rng(1000+seed).normal(size=(y.shape[1], rank)))
        random_energy = np.mean(np.sum((residual @ V)**2, axis=1))
        controls.append((V, float(np.sqrt(top_energy / max(random_energy, 1e-30)))))
    spec = {'rank': rank, 'ridge': float(ridge), 'trace_error': float(np.trace(cov)),
            'top_eigenvalues': vals[::-1].tolist(), 'top_energy_fraction': float(top_energy/np.trace(cov)),
            'random_energy_multipliers': [c[1] for c in controls],
            'decoder_operator_norm': float(np.sqrt(eigh(D.T@D, subset_by_index=[len(cov)-1, len(cov)-1], eigvals_only=True)[0]))}
    return mx, my, scale, D, U, controls, spec


def variant_logits(x, y, weights, bias, fitted):
    mx, my, scale, D, U, controls, _ = fitted
    target_weights = scale[:, None] * weights
    center = my @ weights + bias
    xcenter = x - mx
    native = y @ weights + bias
    mapped = xcenter @ (D @ target_weights) + center
    variants = {'anchor_native': native, 'mapped_base': mapped}
    # Equal-energy comparison SHRINKS the selected repair; it never amplifies
    # random residuals beyond their true historical target (an unfair control).
    repairs=[('top4',U,1.)]
    for i,(V,factor) in enumerate(controls):
        repairs.extend([(f'random4_{i}',V,1.),(f'top4_energy_matched_{i}',U,1./factor)])
    for name, V, factor in repairs:
        residual_proj = ((y-my)/scale) @ V - xcenter @ (D @ V)
        variants[name] = mapped + factor * residual_proj @ (V.T @ target_weights)
    return variants


def load_global(ds, role):
    paths = list((BANKS / 'frozen' / ds / role / 'features' / ds).glob('*_train.npz'))
    assert len(paths) == 1, paths
    with np.load(paths[0], allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in ['features', 'labels', 'paths']}, paths[0]


def metadata_groups(ds, paths):
    segment = {'chammi-cp-task3': 'CP', 'chammi-hpa-task2': 'HPA', 'chammi-allen-task2': 'Allen'}[ds]
    root = Path('/mnt/huawei_deepcad/benchmark/Classification/CHAMMI')
    mapping = {}
    with (root / segment / 'enriched_meta.csv').open() as f:
        for row in csv.DictReader(f):
            if row['train_test_split'] != 'Train':
                continue
            if segment == 'CP':
                group = row['source'] + '/' + row['Plate'] + '/' + row['Well']
            elif segment == 'HPA':
                group = row['ID']
            else:
                group = row['PlateId'] + '/' + row['WellId']
            mapping[str(root / row['file_path'])] = group
    return np.array([mapping[str(p)] for p in paths]), {'CP':'source/plate/well', 'HPA':'original image ID', 'Allen':'PlateId/WellId'}[segment]


def linear_probe(x, labels):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1., class_weight='balanced', max_iter=10000, random_state=0).fit(scaler.transform(x), labels)
    W = model.coef_.T / scaler.scale_[:, None]
    b = model.intercept_ - scaler.mean_ @ W
    return W, b, model.classes_


def classification(args):
    anchor, ap = load_global(args.dataset, 'E')
    base, bp = load_global(args.dataset, args.base)
    assert np.array_equal(anchor['paths'], base['paths']) and np.array_equal(anchor['labels'], base['labels'])
    groups, rule = metadata_groups(args.dataset, anchor['paths'])
    labels = anchor['labels'].reshape(-1)
    fit, calibration, val = (pure_class_group_split(groups,labels,args.seed) if args.dataset=='chammi-cp-task3'
                             else group_split(groups,args.seed))
    train = np.sort(np.r_[fit, calibration])
    assert set(labels[train]) == set(labels[val]), 'Class coverage differs across source-group development split'
    config = {'inputs': [str(ap), str(bp)], 'scope': 'official TRAIN only; new group-disjoint inner validation',
              'group_rule': rule, 'group_class_stratified':args.dataset=='chammi-cp-task3',
              'ordered_paths_sha256': digest(anchor['paths']),
              'indices': {k: v.tolist() for k,v in [('fit',fit),('direction_discovery',calibration),('validation',val)]},
              'counts': {'fit': len(fit), 'discovery': len(calibration), 'validation': len(val)}, 'test_loaded': False}
    target = OUT / f'{args.dataset}_{args.base}_seed{args.seed}'
    write(target / 'manifest.json', config)
    # Label-free decoder fit capped to the proposal's 4096 source samples.
    fit = np.sort(np.random.default_rng(0).choice(fit, min(4096,len(fit)), replace=False))
    X, Y = base['features'].astype(np.float64), anchor['features'].astype(np.float64)
    fitted = fit_directions(X[fit], Y[fit], X[calibration], Y[calibration])
    print('decoder/directions complete', args.dataset, fitted[-1], flush=True)
    W, b, classes = linear_probe(Y[train], labels[train])
    BW, bb, bclasses = linear_probe(X[train], labels[train])
    assert np.array_equal(classes, bclasses)
    np.savez_compressed(target/'fitted_maps.npz',mx=fitted[0],my=fitted[1],scale=fitted[2],decoder=fitted[3],
                        directions=fitted[4],head_W=W,head_b=b,base_head_W=BW,base_head_b=bb,classes=classes)
    logits = variant_logits(X[val], Y[val], W, b, fitted)
    logits['base_native'] = X[val] @ BW + bb
    predictions = {k: classes[np.argmax(v, axis=1)] if len(classes)>2 else classes[(v[:,0]>0).astype(int)] for k,v in logits.items()}
    scores = {k: float(balanced_accuracy_score(labels[val], p)) for k,p in predictions.items()}
    mse = {k: float(np.mean((v-logits['anchor_native'])**2)) for k,v in logits.items() if k!='base_native'}
    np.savez_compressed(target / 'validation_predictions.npz', labels=labels[val], groups=groups[val], **predictions)
    result = {'dataset':args.dataset,'base':args.base,'seed':args.seed,'diagnostic_only':True,
              'scores_balanced_accuracy':scores,'anchor_logit_mse':mse,'direction_statistics':fitted[-1],
              'interpretation':'Repairs access old E features as a counterfactual. No backbone was trained; not a deployment result.'}
    write(target/'results.json', result)
    print(json.dumps(result,indent=1),flush=True)


def dense_path(role, split):
    label, step = ROLES[role]
    tag = 'last1__pad__s512_spformal_static_v1'
    run = f'hs6_l5_{label}_ampbf16_b32__best__{tag}'
    file_tag = tag.replace('__', '_')
    return BANKS/'segmentation/cache'/run/'cellpose'/str(step)/f'cellpose_{split}_config_{file_tag}_ampbf16_b32.npz'


def dense_head(role):
    import torch
    label, step = ROLES[role]
    paths=list((BANKS/'segmentation/results').glob(f'hs6_l5_{label}_*/budget50/seed0/cellpose/{step}/best_head.pth'))
    assert len(paths)==1, paths
    state=torch.load(paths[0],map_location='cpu',weights_only=True)
    s={k:v.numpy().astype(np.float64) for k,v in state.items()}
    conv=s['head.2.weight'][:,:,0,0].T
    norm=s['head.1.weight']/np.sqrt(s['head.1.running_var']+1e-5)
    W=norm[:,None]*conv
    b=(s['head.1.bias']-s['head.1.running_mean']*norm)@conv+s['head.2.bias']
    reference=json.loads(paths[0].with_name('results.json').read_text())['val']['mIoU']
    return W,b,float(reference),str(paths[0])


def load_dense(role, split):
    path=dense_path(role,split)
    with np.load(path,allow_pickle=False) as data:
        assert int(data['chunked'])==0
        return {k:np.asarray(data[k]) for k in ['features','sem_masks']},path


def sampled_patches(bank, ids, count):
    feat=bank['features']; n,d,h,w=feat.shape
    rng=np.random.default_rng(0);rows=[]
    for i in ids:
        centers=bank['sem_masks'][i,8::16,8::16].reshape(-1)
        valid=np.flatnonzero(centers!=255)
        chosen=rng.choice(valid,min(count,len(valid)),replace=False)
        rows.append(feat[i].reshape(d,h*w)[:,chosen].T)
    return np.concatenate(rows).astype(np.float64)


def segmentation(args):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    A,ap=load_dense('E','train');B,bp=load_dense(args.base,'train')
    assert np.array_equal(A['sem_masks'],B['sem_masks']) and A['features'].shape==B['features'].shape
    ids=np.arange(len(A['features']));rng=np.random.default_rng(args.seed);rng.shuffle(ids)
    fit=ids[:int(.7*len(ids))];cal=ids[int(.7*len(ids)):]
    target=OUT/f'cellpose_{args.base}_seed{args.seed}'
    write(target/'manifest.json',{'inputs':[str(ap),str(bp)],'fit_images':fit.tolist(),'discovery_images':cal.tolist(),
                                'validation':'existing v4 Cellpose validation only','test_loaded':False,'rank':4})
    ax,bx=sampled_patches(A,fit,16),sampled_patches(B,fit,16)
    pick=np.sort(np.random.default_rng(0).choice(len(ax),min(4096,len(ax)),replace=False))
    fitted=fit_directions(bx[pick],ax[pick],sampled_patches(B,cal,32),sampled_patches(A,cal,32))
    del A,B,ax,bx
    print('decoder/directions complete cellpose', fitted[-1],flush=True)
    A,ap=load_dense('E','val');B,bp=load_dense(args.base,'val')
    assert np.array_equal(A['sem_masks'],B['sem_masks'])
    W,b,ref,headpath=dense_head('E');BW,bb,bref,bheadpath=dense_head(args.base)
    confusions={};sums={};npatch=0
    for i in range(len(A['features'])):
        af=A['features'][i];bf=B['features'][i];d,h,w=af.shape
        Y=af.reshape(d,-1).T.astype(np.float64);X=bf.reshape(d,-1).T.astype(np.float64)
        variants=variant_logits(X,Y,W,b,fitted);variants['base_native']=X@BW+bb
        names=list(variants);low=np.stack([variants[k].T.reshape(2,h,w) for k in names]).astype(np.float32)
        logits=F.interpolate(torch.from_numpy(low),size=A['sem_masks'][i].shape,mode='bilinear',align_corners=False)
        pred=logits.argmax(dim=1).numpy();gt=A['sem_masks'][i];valid=gt!=255
        for j,name in enumerate(names):
            cm=np.bincount(2*gt[valid].astype(int)+pred[j][valid],minlength=4).reshape(2,2)
            confusions.setdefault(name,[]).append(cm)
            if name!='base_native':sums[name]=sums.get(name,0.)+float(np.sum((variants[name]-variants['anchor_native'])**2))
        npatch+=len(X)*2
    def miou(cms):
        cm=np.sum(cms,axis=0);return float(np.mean(np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))))
    scores={k:miou(v) for k,v in confusions.items()}
    parity={'anchor_difference':scores['anchor_native']-ref,'base_difference':scores['base_native']-bref}
    assert max(abs(v) for v in parity.values())<1e-4,parity
    np.savez_compressed(target/'validation_confusions.npz',**{k:np.asarray(v) for k,v in confusions.items()})
    result={'dataset':'cellpose','base':args.base,'seed':args.seed,'diagnostic_only':True,'scores_mIoU':scores,
            'canonical_validation_parity':parity,'anchor_logit_mse':{k:v/npatch for k,v in sums.items()},
            'direction_statistics':fitted[-1],'heads':[headpath,bheadpath],
            'interpretation':'Saved seed0 E50 heads; counterfactual old-feature access, not trained model results. No instance metric claim.'}
    write(target/'results.json',result);print(json.dumps(result,indent=1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['cellpose','chammi-cp-task3','chammi-hpa-task2','chammi-allen-task2'])
    parser.add_argument('--base',default='M',choices=['M','L']);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f'{args.dataset}_{args.base}_seed{args.seed}'
    if (target/'results.json').exists():raise SystemExit('Already complete; refusing to overwrite results')
    start=time.time()
    try:
        segmentation(args) if args.dataset=='cellpose' else classification(args)
    except Exception as exc:
        write(target/'failure.json',{'type':type(exc).__name__,'error':str(exc)})
        raise
    print('elapsed_seconds',time.time()-start,flush=True)
