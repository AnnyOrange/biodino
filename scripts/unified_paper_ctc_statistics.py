"""CTC sequence/domain blocks: retained frame index, five-fit CI and paired tests."""
import argparse,csv,fcntl,hashlib,itertools,json,os,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
DATA=ROOT/'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json'
METRICS=['TRA','DET','SEG'];STORE=OUT/'statistics/ctc'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def index_frames(path,data):
    result=read(path);invocation=read(Path(result['invocation']));target=path.with_name('test_frame_index.csv');audit=target.with_suffix('.json')
    provenance=dict(result_sha256=sha(path),data_manifest_sha256=sha(DATA))
    if target.exists() and audit.exists() and all(read(audit).get(k)==v for k,v in provenance.items()):return
    fold=int(result['fold']);expected=next(f['test_domains'] for f in data['folds'] if f['fold']==fold)
    assert sorted(r['domain'] for r in result['domain_rows'])==sorted(expected)
    rows=[]
    for score in result['domain_rows']:
        domain=score['domain'];spec=data['domains'][domain];pred=path.parent/'domains'/domain/'02_RES';gt=Path(spec['test_gt_dir'])
        assert score['ctc_metrics']['Valid']==1 and len(spec['test_frames'])==score['test_frames']
        assert (pred/'res_track.txt').is_file() and (gt/'TRA/man_track.txt').is_file()
        for frame in spec['test_frames']:
            prediction=pred/f"mask{frame['token']}.tif";truth=gt/'TRA'/f"man_track{frame['token']}.tif"
            assert prediction.is_file() and truth.is_file(),(prediction,truth)
            rows.append(dict(model=result['model'],checkpoint=invocation['checkpoint']['path'],dataset='ctc',task='cell_tracking',fold=fold,seed=result['seed'],sample_id=f"{domain}/02/{frame['token']}",group_id=domain,sequence_id=f'{domain}/02',frame=frame['frame'],image_path=frame['image'],true_label_path=str(truth),prediction_path=str(prediction),true_lineage_path=str(gt/'TRA/man_track.txt'),predicted_lineage_path=str(pred/'res_track.txt')))
    tmp=target.with_suffix('.tmp.csv')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(target)
    write(audit,dict(**provenance,model=result['model'],fold=fold,seed=result['seed'],n_frames=len(rows),n_domain_blocks=len(expected),index=str(target),index_sha256=sha(target),group_unit='Whole held-out challenge dataset/domain; entire sequence, frames, tracks and all training seeds stay together',protocol='Released labelled sequence02 from held-out domains, not hidden official test leaderboard',label_storage='uint32 local scoring extension; not official uint16 submission',source_manifest=str(DATA)))

def load_complete(model,folds,data):
    per_seed=[];source=[];domains=[];strata=[]
    for fold in folds:
        names=next(f['test_domains'] for f in data['folds'] if f['fold']==fold);values=[]
        for seed in range(5):
            path=OUT/'results'/model/'ctc'/f'fold{fold}'/f'seed{seed}'/'component_result.json'
            if not path.exists() or not path.with_name('test_frame_index.json').exists():return None
            result=read(path);by_domain={r['domain']:r for r in result['domain_rows']}
            assert result['seed']==seed and result['fold']==fold and set(by_domain)==set(names)
            current=np.array([[by_domain[d]['ctc_metrics'][m] for m in METRICS] for d in names],float)
            assert np.isfinite(current).all() and all(by_domain[d]['ctc_metrics']['Valid']==1 for d in names)
            assert np.allclose(current.mean(0),[result['macro'][m] for m in METRICS],atol=1e-12,rtol=0)
            values.append(current);source.append(dict(path=str(path),sha256=sha(path),frame_index=str(path.with_name('test_frame_index.csv'))))
        per_seed.append(np.stack(values));domains.extend(names);strata.extend([fold]*len(names))
    assert len(set(domains))==len(domains)
    return dict(values=np.concatenate(per_seed,axis=1),domains=domains,strata=np.array(strata),source=source,n_test=sum(len(data['domains'][d]['test_frames']) for d in domains))

def bootstrap(values,strata,settings):
    # values are domain metrics, averaged across separate fits, never ensembles.
    rng=np.random.default_rng(settings['random_seed']);positions=[np.flatnonzero(strata==s) for s in np.unique(strata)];draw=[]
    for _ in range(settings['bootstrap_replicates']):
        indices=np.concatenate([rng.choice(pos,len(pos),replace=True) for pos in positions]);draw.append(values[indices].mean(0))
    return np.quantile(draw,[.025,.975],axis=0)

def permutation(difference,settings):
    delta=difference.mean(0);n=len(difference)
    if n<=16:
        signs=np.array(list(itertools.product([-1,1],repeat=n)),float);draw=signs@difference/n
        return (np.abs(draw)>=np.abs(delta)-1e-12).mean(0),len(signs),'exact domain-block sign permutation'
    rng=np.random.default_rng(settings['random_seed']);b=settings['permutation_replicates'];extreme=np.zeros(len(METRICS),int)
    for start in range(0,b,512):
        signs=rng.integers(0,2,size=(min(512,b-start),n))*2-1;draw=signs@difference/n;extreme+=(np.abs(draw)>=np.abs(delta)-1e-12).sum(0)
    return (extreme+1)/(b+1),b,'Monte Carlo domain-block sign permutation with plus-one correction'

def run():
    cfg=read(OUT/'config.json');settings=cfg['statistics'];data=read(DATA);errors=[];indexed=0
    assert data['source_split_manifest_sha256']=='7a0f5bc2579f6ae22a2b8f2b16103530cd678466a4b2bbd1f617692b8e3cac27'
    for path in (OUT/'results').glob('*/ctc/fold*/seed*/component_result.json'):
        try:index_frames(path,data);indexed+=1
        except Exception as e:errors.append(dict(path=str(path),error=repr(e)));traceback.print_exc()
    available={}
    for model in cfg['models']:
        for fold in [0,1,2,3,4,'all']:
            value=load_complete(model['model'],list(range(5)) if fold=='all' else [fold],data)
            if value is None:continue
            available[model['model'],fold]=value;target=STORE/f"{model['model']}__fold{fold}__ci.json"
            if target.exists() and read(target).get('source')==value['source'] and read(target).get('n_test')==value['n_test']:continue
            a=value['values'];ci=bootstrap(a.mean(0),value['strata'],settings);scores=a.mean(1)
            write(target,dict(model=model['model'],task='cell_tracking',dataset='ctc',fold=str(fold),training_seeds=list(range(5)),folds=list(range(5)) if fold=='all' else [fold],n_test=value['n_test'],n_groups=len(value['domains']),groups=value['domains'],source=value['source'],primary_independent_groups_verified=True,group_unit='Published held-out CTC dataset/domain, keeping whole sequence and all seeds together',uncertainty='Domain bootstrap stratified by fixed fold, conditional on fitted heads and selected challenge domains; five separate fit metrics averaged, no prediction ensemble; fold and seed are distinct',bootstrap_replicates=settings['bootstrap_replicates'],metrics={m:dict(value=float(scores[:,i].mean()),seed_sd=float(scores[:,i].std(ddof=1)),per_seed_values=scores[:,i].tolist(),ci_low=float(ci[0,i]),ci_high=float(ci[1,i])) for i,m in enumerate(METRICS)}))
            print('CTC CI',model['model'],fold,flush=True)
    for fold in [0,1,2,3,4,'all']:
        for candidate in [m['model'] for m in cfg['models'] if m['kind']=='biodino']:
            for reference in [m['model'] for m in cfg['models'] if m['kind']=='fm']:
                if (candidate,fold) not in available or (reference,fold) not in available:continue
                a,b=available[candidate,fold],available[reference,fold];assert a['domains']==b['domains'] and np.array_equal(a['strata'],b['strata'])
                target=OUT/'statistics'/f'{candidate}__vs__{reference}__ctc__fold{fold}.json';source=[a['source'],b['source']]
                if target.exists() and read(target).get('source')==source and read(target).get('n_test')==a['n_test']:continue
                difference=(a['values']-b['values']).mean(0);ci=bootstrap(difference,a['strata'],settings);p,n,method=permutation(difference,settings)
                write(target,dict(candidate=candidate,reference=reference,dataset='ctc',task='cell_tracking',fold=str(fold),budget='',source=source,n_test=a['n_test'],n_groups=len(a['domains']),groups=a['domains'],training_seeds=list(range(5)),primary_independent_groups_verified=True,planned_comparisons=settings['planned_comparisons_per_family'],bootstrap_replicates=settings['bootstrap_replicates'],permutation_replicates=n,test_definition='Two-sided paired whole-domain model swap, synchronized over all five fits and every frame/track; fold-stratified domain bootstrap; separate Holm family per metric and fold or all-fold aggregate',permutation_method=method,metrics={m:dict(difference=float(difference[:,i].mean()),ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),p_two_sided=float(p[i])) for i,m in enumerate(METRICS)}))
    write(OUT/'states/ctc_statistics.json',dict(pid=os.getpid(),updated=time.time(),indexed_fits=indexed,complete_model_fold_sets=len(available),errors=errors))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with (OUT/'locks/ctc_statistics.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            run()
            if not args.watch:break
            time.sleep(60)
