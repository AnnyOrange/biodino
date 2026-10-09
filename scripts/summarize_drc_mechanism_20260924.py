"""Paired source-group bootstrap and predeclared mechanism gate; CPU only."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/gram_replacement_design_20260924/stage_a/refined_controls')


def interval(x):
    return [float(v) for v in np.quantile(x,[.025,.975])]


def draws_for_classification(folder,names):
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:
        y=z['labels'];groups=z['groups'];preds={k:z[k] for k in names}
    classes,ci=np.unique(y,return_inverse=True);gg,gi=np.unique(groups,return_inverse=True)
    g,c=len(gg),len(classes)
    total=np.bincount(gi*c+ci,minlength=g*c).reshape(g,c)
    correct=np.stack([np.bincount(gi*c+ci,weights=(preds[k]==y),minlength=g*c).reshape(g,c) for k in names],axis=1)
    rng=np.random.default_rng(197);weights=np.zeros((2000,g))
    pure=(np.count_nonzero(total,axis=1)==1).all()
    strata=[np.flatnonzero(total[:,j]) for j in range(c)] if pure else [np.arange(g)]
    for indices in strata:
        weights[:,indices]=rng.multinomial(len(indices),np.ones(len(indices))/len(indices),size=2000)
    den=weights@total
    valid=(den>0).all(1)
    num=(weights@correct.reshape(g,-1)).reshape(2000,len(names),c)
    values=np.mean(num[valid]/den[valid,None,:],axis=2)
    return values,{'unit':'source group','n_groups':g,'class_stratified_groups':bool(pure),'valid_bootstrap_draws':int(valid.sum())}


def draws_for_segmentation(folder,names):
    with np.load(folder/'validation_confusions.npz',allow_pickle=False) as z:
        counts=np.stack([z[k] for k in names],axis=1)
    n=len(counts);rng=np.random.default_rng(197)
    weights=rng.multinomial(n,np.ones(n)/n,size=2000)
    cm=(weights@counts.reshape(n,-1)).reshape(2000,len(names),2,2)
    tp=np.diagonal(cm,axis1=2,axis2=3)
    den=cm.sum(2)+cm.sum(3)-tp
    return np.mean(tp/den,axis=2),{'unit':'v4 validation image','n_groups':n,'valid_bootstrap_draws':2000,
                                 'scope':'conditional on saved seed0 E50 heads; not training-seed uncertainty'}


def main():
    out={'status':'COMPLETE','backbone_training_performed':False,'test_data_used_for_selection':False,'datasets':{}}
    for folder in sorted(ROOT.iterdir()):
        p=folder/'results.json'
        if not p.is_file():continue
        row=json.loads(p.read_text());score=row.get('scores_balanced_accuracy',row.get('scores_mIoU'));names=list(score)
        values,meta=draws_for_segmentation(folder,names) if row['dataset']=='cellpose' else draws_for_classification(folder,names)
        col={k:i for i,k in enumerate(names)}
        equal=values[:,[col[f'top4_energy_matched_{i}'] for i in range(8)]].mean(1)-values[:,[col[f'random4_{i}'] for i in range(8)]].mean(1)
        gain=values[:,col['top4']]-values[:,col['mapped_base']]
        decline=values[:,col['anchor_native']]-values[:,col['base_native']]
        point_equal=np.mean([score[f'top4_energy_matched_{i}']-score[f'random4_{i}'] for i in range(8)])
        item={'dataset':row['dataset'],'seed':row['seed'],'metric':'mIoU' if row['dataset']=='cellpose' else 'balanced_accuracy',
              'native_E':score['anchor_native'],'native_M':score['base_native'],
              'mapped_M_to_E':score['mapped_base'],'top4_repaired':score['top4'],
              'native_decline':score['anchor_native']-score['base_native'],'native_decline_CI95':interval(decline),
              'top4_repair_gain':score['top4']-score['mapped_base'],'top4_repair_gain_CI95':interval(gain),
              'equal_energy_advantage_over_random':float(point_equal),'equal_energy_advantage_CI95':interval(equal),
              'feature_error_fraction_top4':row['direction_statistics']['top_energy_fraction'],
              'readout_logit_error_fraction_removed':1-row['anchor_logit_mse']['top4']/row['anchor_logit_mse']['mapped_base'],
              'bootstrap':meta}
        item['individual_gate_passed']=bool(item['native_decline']>0 and item['top4_repair_gain']>0 and interval(gain)[0]>0 and interval(equal)[0]>0)
        out['datasets'][folder.name]=item
    assert len(out['datasets'])==8,'Wait for all predeclared datasets/splits'
    cp=all(out['datasets'][f'chammi-cp-task3_M_seed{s}']['individual_gate_passed'] for s in [0,17])
    seg=all(out['datasets'][f'cellpose_M_seed{s}']['individual_gate_passed'] for s in [0,17])
    out.update(primary_declining_classification_passed=cp,segmentation_passed=seg,stage_a_passed=bool(cp and seg),
               decision='PROCEED_TO_UNLABELED_CALIBRATION' if cp and seg else 'DO_NOT_LAUNCH_DRC_BACKBONE_TRAINING',
               limitation='Counterfactual repair has access to historical E features. It is a mechanism screen, not deployment or a comparison to Gram.')
    (ROOT/'MECHANISM_GATE.json').write_text(json.dumps(out,indent=2))
    for k,r in out['datasets'].items():
        print(k,'decline',round(r['native_decline'],6),'repair',round(r['top4_repair_gain'],6),
              'matched advantage',round(r['equal_energy_advantage_over_random'],6),'CI',r['equal_energy_advantage_CI95'],'pass',r['individual_gate_passed'])
    print(out['decision'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    ROOT=parser.parse_args().root
    main()
