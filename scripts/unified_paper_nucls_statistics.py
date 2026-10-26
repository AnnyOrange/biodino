"""NuCLS patient-cluster inference, with folds separate from downstream seeds."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import unified_paper_native_statistics as native
METRICS=native.METRICS+['matched_mask_dice_pooled','matched_mask_dice_mean','matched_classification_accuracy','detection_f1_iou50']
VERSION=1

def prepare(payload):
    # Category8 is ambiguous and is excluded by the shared evaluator's catIds.
    # Its crowd objects cannot match predictions of any other category.
    records=[]
    for row in payload['records']:
        objects=[obj for obj in row['objects'] if not obj['ambiguous']]
        records.append({**row,'boxes':[o['box'] for o in objects],'labels':[o['label'] for o in objects]})
    cache=native.prepare({**payload,'records':records,'classes':payload['classes'][:7]})
    counts=[]
    for row,sample in zip(payload['records'],payload['sample_metrics']):
        assert row['sample_id']==sample['sample_id'] and row['group_id']==sample['group_id']
        confusion=np.asarray(sample['classification_confusion'])
        assert confusion.sum()==sample['matched_nuclei']
        counts.append([sample['mask_intersection'],sample['mask_area_sum'],sample['mask_dice_sum'],sample['matched_genuine_masks'],
                       np.trace(confusion),sample['matched_nuclei'],sample['true_nuclei'],sample['predicted_nuclei']])
    return {'matches':cache,'counts':np.asarray(counts,dtype=float)}

def one_metric(cache,indices,other=None,swaps=None):
    bbox=native.ap(cache['matches'],indices,None if other is None else other['matches'],swaps)
    rows=cache['counts'] if other is None else np.where(np.asarray(swaps)[:,None],other['counts'],cache['counts'])
    overlap,area,dice,nmask,correct,matched,ntrue,npred=rows[indices].sum(0)
    numerator=np.array([2*overlap,dice,correct,2*matched]);denominator=np.array([area,nmask,matched,ntrue+npred])
    secondary=np.divide(numerator,denominator,out=np.full(4,np.nan),where=denominator>0)
    return np.r_[bbox,secondary]

def metric(a,indices,b=None,swaps=None):
    # Ordinary mean: a seed with an undefined conditional metric is not dropped.
    return np.mean([one_metric(c,indices,None if b is None else b['caches'][i],swaps) for i,c in enumerate(a['caches'])],axis=0)

def load(paths):
    caches=[];hashes=[];values=[];seeds=[];ids=groups=signature=None
    for path in sorted(paths,key=lambda p:int(p.parent.name.removeprefix('seed'))):
        raw=json.loads(path.read_text());pred=Path(raw['prediction_path']);digest=native.sha(pred)
        assert digest==raw['prediction_sha256']
        payload=torch.load(pred,map_location='cpu',weights_only=False)
        this_ids=[r['sample_id'] for r in payload['records']];this_groups=[r['group_id'] for r in payload['records']]
        record_hash=hashlib.sha256(json.dumps(payload['records'],sort_keys=True).encode()).hexdigest()
        if ids is not None:assert ids==this_ids and groups==this_groups and signature==record_hash,'Unpaired NuCLS seeds'
        ids,groups,signature=this_ids,this_groups,record_hash
        cache_path=pred.with_name('nucls_per_fov_statistics.pth')
        saved=torch.load(cache_path,map_location='cpu',weights_only=False) if cache_path.exists() else {}
        if saved.get('version')!=VERSION or saved.get('prediction_sha256')!=digest:
            saved={'version':VERSION,'prediction_sha256':digest,**prepare(payload)}
            tmp=cache_path.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(cache_path)
        value=one_metric(saved,np.arange(len(ids)));expected=[raw['metrics'][k] if raw['metrics'][k] is not None else np.nan for k in METRICS]
        assert np.allclose(value,expected,atol=1e-12,rtol=0,equal_nan=True),(value,expected)
        caches.append(saved);hashes.append(digest);values.append(value);seeds.append(raw['seed'])
    return {'caches':caches,'ids':ids,'groups':groups,'target_sha256':signature,'prediction_sha256':hashes,'values':np.asarray(values),'seeds':seeds}

def aligned(a,b):
    for k in ['ids','groups','target_sha256','seeds']:assert a[k]==b[k],f'NuCLS comparison differs: {k}'

def group_members(a):
    return [np.flatnonzero(np.asarray(a['groups'])==g) for g in sorted(set(a['groups']))]

def intervals(samples):
    samples=np.asarray(samples);output=[]
    for values in samples.T:
        finite=np.isfinite(values);fraction=float(finite.mean())
        # Empty matched-mask/classification denominators remain undefined.
        lo,hi=np.quantile(values[finite],[.025,.975]) if fraction>=.95 else (None,None)
        output.append({'ci_low':None if lo is None else float(lo),'ci_high':None if hi is None else float(hi),'valid_bootstrap_fraction':fraction})
    return output

def bootstrap(folds,others,repeats,seed):
    rng=np.random.default_rng(seed);members={f:group_members(a) for f,a in folds.items()};samples=[]
    for _ in range(repeats):
        values=[]
        for f,a in folds.items():
            groups=members[f];indices=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
            value=metric(a,indices)
            if others is not None:value-=metric(others[f],indices)
            values.append(value)
        samples.append(np.mean(values,axis=0))
    return intervals(samples)

def permutation(folds,others,repeats,seed):
    rng=np.random.default_rng(seed);mappings={}
    for f,a in folds.items():
        aligned(a,others[f]);_,mappings[f]=np.unique(a['groups'],return_inverse=True)
    delta=np.mean([metric(a,np.arange(len(a['ids'])))-metric(others[f],np.arange(len(a['ids']))) for f,a in folds.items()],axis=0)
    extreme=np.zeros(len(METRICS),int);valid=np.zeros(len(METRICS),int)
    for _ in range(repeats):
        values=[]
        for f,a in folds.items():
            mapping=mappings[f];swaps=rng.integers(2,size=int(mapping.max())+1)[mapping].astype(bool);ids=np.arange(len(a['ids']))
            values.append(metric(a,ids,others[f],swaps)-metric(others[f],ids,a,swaps))
        difference=np.mean(values,axis=0);finite=np.isfinite(difference);valid+=finite
        extreme+=finite&(np.abs(difference)>=np.abs(delta)-1e-12)
    # Never silently omit undefined randomizations: those tests have no p-value.
    p=np.where((valid==repeats)&np.isfinite(delta),(extreme+1)/(repeats+1),np.nan)
    return delta,p,valid

def serial(value):
    value=float(value);return value if np.isfinite(value) else None

def ci_result(model,folds,s,scope):
    seed_sets={tuple(a['seeds']) for a in folds.values()};assert len(seed_sets)==1
    seeds=next(iter(seed_sets));values=np.mean([a['values'] for a in folds.values()],axis=0)
    ci=bootstrap(folds,None,s['bootstrap_replicates'],s['random_seed'])
    return {'version':VERSION,'model':model,'dataset':'nucls_qc','task':'nuclei_detection_segmentation','fold':scope,
        'evaluation_folds':list(folds),'n_test':sum(len(a['ids']) for a in folds.values()),'n_groups':sum(len(set(a['groups'])) for a in folds.values()),
        'training_seeds':list(seeds),'required_seeds':5,'complete_training':len(seeds)==5,'per_seed_values':[[serial(v) for v in row] for row in values],'metric_order':METRICS,
        'seed_sd':[serial(x) for x in np.std(values,axis=0,ddof=1)] if len(seeds)>1 else [None]*len(METRICS),
        'prediction_sha256':[h for a in folds.values() for h in a['prediction_sha256']],
        'bootstrap_replicates':s['bootstrap_replicates'],'random_seed':s['random_seed'],
        'uncertainty':'TCGA patient bootstrap within each official fold, synchronized across seeds; recompute AP and pooled matching metrics per fit, then equal-weight mean of folds and seeds. Conditional on fitted heads; no prediction ensemble. Partial seed sets explicitly labelled.',
        'metrics':{k:{'value':serial(values[:,i].mean()),**ci[i]} for i,k in enumerate(METRICS)}}

def run():
    cfg=json.loads((OUT/'config.json').read_text());s=json.loads((OUT/'nucls_statistics_config.json').read_text());store=OUT/'statistics/nucls';store.mkdir(exist_ok=True);available={}
    for spec in cfg['models']:
        model=spec['model'];folds={}
        for fold in s['evaluation_folds']:
            paths=list((OUT/'results'/model/'nucls_qc'/f'fold{fold}').glob('seed*/component_result.json'))
            if not paths:continue
            try:
                a=load(paths);folds[fold]=a;target=store/f'{model}__fold{fold}__ci.json'
                old=json.loads(target.read_text()) if target.exists() else {}
                if old.get('prediction_sha256')!=a['prediction_sha256'] or old.get('version')!=VERSION:
                    native.dump(target,ci_result(model,{fold:a},s,fold));print('NUCLS PATIENT CI',model,fold,a['seeds'],flush=True)
            except Exception as error:
                native.dump(store/f'{model}__fold{fold}__error.json',{'error':repr(error),'updated':time.time()});print('NUCLS STATS ERROR',model,fold,repr(error),flush=True)
        if len(folds)!=4 or any(a['seeds']!=list(range(5)) for a in folds.values()):continue
        patient_sets=[set(a['groups']) for a in folds.values()]
        assert sum(map(len,patient_sets))==len(set.union(*patient_sets)),'Evaluation folds share patients'
        available[model]=folds;target=store/f'{model}__mean_2_5__ci.json';hashes=[h for a in folds.values() for h in a['prediction_sha256']]
        old=json.loads(target.read_text()) if target.exists() else {}
        if old.get('prediction_sha256')!=hashes or old.get('version')!=VERSION:
            native.dump(target,ci_result(model,folds,s,'mean_2_5'));print('NUCLS FOUR-FOLD CI',model,flush=True)
    for candidate in [m['model'] for m in cfg['models'] if m['kind']=='biodino']:
        for reference in [m['model'] for m in cfg['models'] if m['kind']=='fm']:
            if candidate not in available or reference not in available:continue
            a,b=available[candidate],available[reference];hashes=[h for f in a for h in a[f]['prediction_sha256']+b[f]['prediction_sha256']]
            target=store/f'{candidate}__vs__{reference}__mean_2_5.json';old=json.loads(target.read_text()) if target.exists() else {}
            if old.get('prediction_sha256')==hashes and old.get('version')==VERSION:continue
            for f in a:aligned(a[f],b[f])
            ci=bootstrap(a,b,s['bootstrap_replicates'],s['random_seed']);delta,p,valid=permutation(a,b,s['permutation_replicates'],s['random_seed'])
            native.dump(target,{'version':VERSION,'candidate':candidate,'reference':reference,'dataset':'nucls_qc','task':'nuclei_detection_segmentation','fold':'mean_2_5',
                'n_test':sum(len(x['ids']) for x in a.values()),'n_groups':sum(len(set(x['groups'])) for x in a.values()),'training_seeds':list(range(5)),
                'prediction_sha256':hashes,'bootstrap_replicates':s['bootstrap_replicates'],'permutation_replicates':s['permutation_replicates'],'random_seed':s['random_seed'],
                'test_definition':s['paired_test'],'primary_grouping_verified':True,
                'metrics':{k:{'difference':serial(delta[i]),**ci[i],'p_two_sided':serial(p[i]),'valid_permutations':int(valid[i])} for i,k in enumerate(METRICS)}})
            print('NUCLS FOUR-FOLD PAIR',candidate,reference,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args();torch.set_num_threads(2)
    with (OUT/'locks/nucls_statistics.local.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            run()
            if not args.watch:break
            time.sleep(60)
