#!/usr/bin/env python3
"""Nested/grouped train-only ridge tuning; unchanged v4 outer identities.

Fixed alpha=1 is retained alongside the explicitly named tuned-readout extension.
The alpha grid is evaluated ONLY inside outer training data.
"""
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from dinov3.eval.bio_frozen_eval.coexistence import balanced_concat, load_feature_bank
from dinov3.eval.bio_frozen_eval.run_coexistence_classification import _input_banks, _subset
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
from dinov3.eval.bio_frozen_eval.registry import build_dataset

REPO=Path('/mnt/huawei_deepcad/dinov3')
BENCH=Path('/mnt/huawei_deepcad/benchmark')
FROZEN=REPO/'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen'
OUT=REPO/'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923/regression_tuning'
GRID=np.logspace(-4,6,11)

def predictions(x,y,z,alphas):
    # One eigensystem per fold, exact StandardScaler + Ridge with intercept.
    x=np.asarray(x,np.float64);z=np.asarray(z,np.float64);y=np.asarray(y,np.float64)
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1
    x=(x-mu)/sd;z=(z-mu)/sd;ym=y.mean();yc=y-ym
    if len(x)<x.shape[1]:
        ev,u=np.linalg.eigh(x@x.T);ev=np.maximum(ev,0)
        return (z@x.T@u)@((u.T@yc)[:,None]/(ev[:,None]+np.asarray(alphas)[None,:]))+ym
    ev,u=np.linalg.eigh(x.T@x);ev=np.maximum(ev,0)
    return (z@u)@((u.T@x.T@yc)[:,None]/(ev[:,None]+np.asarray(alphas)[None,:]))+ym

def choose(x,y,groups):
    unique=np.unique(groups);folds=min(5,len(unique))
    if folds<2:raise ValueError('Need >=2 independent train groups')
    splits=list(GroupKFold(folds).split(x,y,groups));scores=[];fold_ids=[]
    for tr,va in splits:
        assert not set(groups[tr]) & set(groups[va])
        p=predictions(x[tr],y[tr],x[va],GRID)
        scores.append([r2_score(y[va],p[:,i]) for i in range(len(GRID))])
        fold_ids.append({'train':tr.tolist(),'validation':va.tolist()})
    means=np.mean(scores,axis=0);best=int(np.argmax(means))
    return float(GRID[best]),{'alpha_grid':GRID.tolist(),'cv_r2':means.tolist(),'folds':fold_ids}

def metrics(y,p):
    return {'r2':float(r2_score(y,p)),'mae':float(mean_absolute_error(y,p)),
            'spearman':float(spearmanr(y,p).statistic)}

def canonical_predictions(x,y,z,alphas):
    # Outer reporting uses the exact canonical v4 sklearn estimator/dtype path.
    return np.stack([make_pipeline(StandardScaler(),Ridge(alpha=a)).fit(x,y).predict(z)
                     for a in alphas],axis=1)

def groups_for(ds,paths):
    if ds=='bbbc005':
        return np.array([re.search(r'_([A-Za-z]\d+_C\d+_F\d+)_',Path(p).name).group(1) for p in paths])
    if ds=='conic-cell-count':
        with (BENCH/'Regression/CoNIC_Cell_Count/conic_cell_count.csv').open() as f:
            mapping={int(r['image_index']):r['source_image'] for r in csv.DictReader(f)}
        return np.array([mapping[int(p.rsplit(':',1)[1])] for p in paths])
    if ds=='livecell-cell-count':
        result=[]
        for p in paths:
            name=Path(p).stem
            # Group a field's time series, preserving plate/well and site.
            m=re.match(r'(.+)_\d+d\d+h\d+m_(\d+)$',name)
            if not m:raise ValueError('Unrecognized LIVECell time-series identity: '+name)
            result.append(m.group(1)+'_site'+m.group(2))
        return np.array(result)
    raise ValueError(ds)

def evaluate(ds, arm=None, campaign=None):
    global OUT
    if arm:
        root=Path(campaign);record=root/'frozen'/ds/arm/'last_result.json'
        info=json.loads(record.read_text());feature=Path(info['feature_file'])
        assert info['dataset']==ds and info['batch_size']==64
        prov={'result':str(record),'feature_file':str(feature),
              'feature_sha256':hashlib.sha256(feature.read_bytes()).hexdigest(),
              'checkpoint':info['checkpoint'],'train_config':info['train_config']}
        bank=load_feature_bank(feature)
        if ds=='bbbc005':
            dataset,_=build_dataset(ds,'train',None,None,benchmark_root=BENCH)
            assert np.array_equal(np.array([str(s.image_path) for s in dataset.samples]),bank.paths)
            tr,te=group_split_indices(ds,dataset,BENCH)
            train={arm:_subset(bank,tr)};test={arm:_subset(bank,te)}
        elif ds=='bbbc013':train=test={arm:bank}
        else:
            assert feature.stem.endswith('_train')
            test_file=feature.with_name(feature.stem[:-6]+'_test.npz')
            prov['test_sha256']=hashlib.sha256(test_file.read_bytes()).hexdigest()
            train={arm:bank};test={arm:load_feature_bank(test_file)}
            assert not set(bank.paths)&set(test[arm].paths)
        OUT=root/'regression_tuning/arms'/arm
        return evaluate_prepared(ds,train,test,prov)
    if ds in ('bbbc005','bbbc013'):
        allbanks,prov=_input_banks(FROZEN,ds,'whole')
        if ds=='bbbc005':
            dataset,_=build_dataset(ds,'train',None,None,benchmark_root=BENCH)
            assert np.array_equal(np.array([str(s.image_path) for s in dataset.samples]),allbanks['E'].paths)
            tr,te=group_split_indices(ds,dataset,BENCH)
            train={k:_subset(v,tr) for k,v in allbanks.items()};test={k:_subset(v,te) for k,v in allbanks.items()}
        else:train=test=allbanks
    else:
        train,p1=_input_banks(FROZEN,ds,'train');test,p2=_input_banks(FROZEN,ds,'test');prov={'train':p1,'test':p2}
        assert not set(train['E'].paths)&set(test['E'].paths)
    for banks in (train,) if train is test else (train,test):
        banks['E+L']=balanced_concat(banks['E'],banks['L']);banks['M+L']=balanced_concat(banks['M'],banks['L'])
    return evaluate_prepared(ds,train,test,prov)

def evaluate_prepared(ds,train,test,prov):
    result={'dataset':ds,'protocol':'bio-eval-union-v4 outer splits + train-only tuned ridge extension',
            'provenance':prov,'selection':'inner grouped CV only','results':{}}
    for arm,bank in train.items():
        x,y=bank.features,bank.labels.astype(float)
        if ds=='bbbc013':
            groups=np.array([re.search(r'Channel\d+-\d+-([A-H])-',Path(p).name,re.I).group(1).upper() for p in bank.paths])
            y=np.log1p(y);pred=np.full(len(y),np.nan);fixed=pred.copy();selections=[];compound=[]
            for name,rows in [('wortmannin',list('ABCD')),('ly294002',list('EFGH'))]:
                for row in rows:
                    tr=np.flatnonzero(np.isin(groups,rows)&(groups!=row));te=np.flatnonzero(groups==row)
                    assert len(tr)==36 and len(te)==12
                    alpha,cv=choose(x[tr],y[tr],groups[tr]);pp=canonical_predictions(x[tr],y[tr],x[te],[1.,alpha])
                    fixed[te],pred[te]=pp[:,0],pp[:,1]
                    selections.append({'compound':name,'test_row':row,'alpha':alpha,'inner':cv})
                ix=np.isin(groups,rows);compound.append({'name':name,'fixed':metrics(y[ix],fixed[ix]),'tuned':metrics(y[ix],pred[ix])})
            scores={kind:{k:float(np.mean([c[kind][k] for c in compound])) for k in ['r2','mae','spearman']} for kind in ['fixed','tuned']}
            scores.update(compounds=compound,selections=selections)
        else:
            groups=groups_for(ds,bank.paths);alpha,cv=choose(x,y,groups)
            OUT.mkdir(parents=True,exist_ok=True)
            # Selection artifact committed before outer test is evaluated.
            selection={'dataset':ds,'arm':arm,'alpha':alpha,'inner':cv,'train_paths_sha256':hashlib.sha256('\n'.join(bank.paths).encode()).hexdigest()}
            (OUT/f'{ds}_{arm.replace("+","_")}_selection.json').write_text(json.dumps(selection))
            pp=canonical_predictions(x,y,test[arm].features,[1.,alpha]);yy=test[arm].labels
            scores={'fixed':metrics(yy,pp[:,0]),'tuned':metrics(yy,pp[:,1]),'alpha':alpha,'inner':cv}
        result['results'][arm]=scores
        print(ds,arm,'fixed',scores['fixed'],'tuned',scores['tuned'],flush=True)
        OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{ds}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True)
    ap.add_argument('--arm');ap.add_argument('--campaign');a=ap.parse_args()
    if a.arm and not a.campaign:ap.error('--arm requires --campaign')
    evaluate(a.dataset,a.arm,a.campaign)
