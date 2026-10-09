"""Matched completed-v4 observations; no missing-cell imputation or test selection."""
import csv,json,math,statistics,hashlib
from pathlib import Path
from datetime import datetime,timezone
REPO=Path('/mnt/huawei_deepcad/dinov3');NEW=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927';OLD=REPO/'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923';OUT=REPO/'outputs/00_reports/deepcad_method_20260927/results_20260929'

def collect(root):
    models={}
    for p in (root/'tasks').glob('*.json'):
        t=json.loads(p.read_text());arm=t['arm'];st=root/'claims'/t['id']/'status.json'
        if not st.exists() or json.loads(st.read_text()).get('state')!='DONE':continue
        model=models.setdefault(arm,dict(classification={},metrics={},sources={},seg_seeds={}))
        done=t['done'];ds=t['dataset'];kind=done['type'];f=Path(done.get('path','/missing'))
        if kind=='summary_csv' and f.is_file():
            rows=[r for r in csv.DictReader(f.open()) if not r.get('error')]
            for r in rows:
                if r.get('balanced_accuracy') and ds!='lc25000':
                    model['classification'][ds]=float(r['balanced_accuracy'])*100
                if r.get('accuracy') and ds=='chammi-cp-task3':model['metrics']['CP Task3 accuracy (%)']=float(r['accuracy'])*100
                if r.get('r2'):model['metrics'][ds+' R2']=float(r['r2'])
                if r.get('recall_at_1') and r.get('task')=='retrieval':model['metrics'][ds+' R@1 (%)']=float(r['recall_at_1'])*100
                if r.get('result_file'):
                    rf=Path(r['result_file']);x=json.loads(rf.read_text());rx=x.get('tests',{}).get('rxrx3',{})
                    if 'map' in rx:model['metrics']['RxRx3 mAP (%)']=float(rx['map'])*100
            model['sources'][t['id']]=str(f)
        elif kind=='detection_json' and f.is_file():
            x=json.loads(f.read_text());model['metrics'][ds+' patch F1 (%)']=x['test_patch_f1'];model['sources'][t['id']]=str(f)
        elif kind=='seg_results':
            rr=Path(done['root']);files=[]
            for base in rr.glob(done['run_prefix']+'*'):
                files+=list(base.glob(f'**/budget50/seed*/{ds}/{done["ckpt_id"]}/results.json'))
                files+=list(base.glob(f'budget50/seed*/{ds}/{done["ckpt_id"]}/results.json'))
            files=sorted(set(files));vals=[];valseeds=[]
            for file in files:
                x=json.loads(file.read_text())
                if 'mIoU' in x.get('test',{}):vals.append(x['test']['mIoU']*100);valseeds.append(x['val']['mIoU']*100)
            if vals:
                model['metrics'][ds+' E50 mIoU (%)']=statistics.mean(vals);model['seg_seeds'][ds]=dict(test=vals,val=valseeds,n=len(vals));model['sources'][t['id']]=[str(f) for f in files]
    return models

def contrast(models,a,b,common=None):
    aa,bb=models[a],models[b];shared=sorted(set(aa['classification'])&set(bb['classification'])) if common is None else common
    rows=[]
    if shared:
        av=statistics.mean(aa['classification'][d] for d in shared);bv=statistics.mean(bb['classification'][d] for d in shared)
        rows.append(dict(metric=f'Classification common{len(shared)} BA (%)',a=av,b=bv,delta_b_minus_a=bv-av))
    for key in sorted(aa['metrics'].keys()&bb['metrics'].keys()):rows.append(dict(metric=key,a=aa['metrics'][key],b=bb['metrics'][key],delta_b_minus_a=bb['metrics'][key]-aa['metrics'][key]))
    return dict(a=a,b=b,classification_datasets=shared,rows=rows)

def main():
    models=collect(OLD);models.update(collect(NEW));pairs={}
    ck_arms=[f'ck_{m}_e12687_formal_ck{s}' for s in [12931,13175,13663] for m in ['c','k']]
    common=sorted(set.intersection(*(set(models[a]['classification']) for a in ck_arms)))
    for step in [12931,13175,13663]:pairs[f'K_minus_C_{step}']=contrast(models,f'ck_c_e12687_formal_ck{step}',f'ck_k_e12687_formal_ck{step}',common)
    reference_arms=[f'{m}_formal_ck{s}' for s in [13175,13663] for m in ['adaptive','vanilla','gram']]+[f'ck_{m}_e12687_formal_ck{s}' for s in [13175,13663] for m in ['c','k']]
    common_ref=sorted(set.intersection(*(set(models[a]['classification']) for a in reference_arms)))
    references=[]
    for step in [13175,13663]:
        for method,arm in [('vanilla',f'vanilla_formal_ck{step}'),('Gram',f'gram_formal_ck{step}'),('Adaptive',f'adaptive_formal_ck{step}'),('C',f'ck_c_e12687_formal_ck{step}'),('K',f'ck_k_e12687_formal_ck{step}')]:
            references.append(dict(step=step,method=method,n=len(common_ref),BA=statistics.mean(models[arm]['classification'][x] for x in common_ref)))
    pairs['EM_minus_adaptive_mid']=contrast(models,'adaptive_mid_retry2_formal_ck20495','metric_mid_formal_ck20495')
    longarms=['adaptive_formal_ck15127','adaptive_continue_v2_gpu2_ck15615','adaptive_continue_v2_gpu2_ck16103']+[f'adaptive_continue_resume16103_gpu3_formal_ck{s}' for s in [16591,17079,17567]]
    commonlong=sorted(set.intersection(*(set(models[a]['classification']) for a in longarms)))
    longrows=[dict(arm=a,n=len(commonlong),BA=statistics.mean(models[a]['classification'][d] for d in commonlong),metrics=models[a]['metrics']) for a in longarms]
    diagnostics={};raw={}
    train=REPO/'outputs/01_training_runs/hs6_l5_deepcad_method_20260927'
    for a in ['ck_c_e12687_formal','ck_k_e12687_formal','adaptive_continue_resume16103_gpu3_formal']:
        rr=[json.loads(x) for x in (train/a/'raw_loss_metrics.jsonl').read_text().splitlines()];raw[a]={r['optimizer_update']:r for r in rr}
        tail=rr[-100:];keys=['recovery_global_gate','recovery_local_gate','recovery_loss','recovery_global_error','recovery_local_error','recovery_global_budget','recovery_local_budget','recovery_global_decoder_gain','recovery_local_decoder_gain','recovery_global_gain_clipped_fraction','recovery_local_gain_clipped_fraction','recovery_global_student_feature_rms','recovery_local_student_feature_rms']
        diagnostics[a]=dict(last_step=rr[-1]['optimizer_update'],rows=len(rr),last100={k:statistics.mean(r[k] for r in tail if k in r) for k in keys if k in tail[-1]})
    c,k=raw['ck_c_e12687_formal'],raw['ck_k_e12687_formal'];steps=sorted(c.keys()&k.keys());paired=dict(updates=len(steps),matching=sum(c[s]['batch_sample_key_digest']==k[s]['batch_sample_key_digest'] for s in steps))
    result=dict(time=datetime.now(timezone.utc).isoformat(),models=models,contrasts=pairs,reference_classification=references,reference_common=common_ref,long_adaptive=longrows,long_common=commonlong,diagnostics=diagnostics,paired_stream=paired,inventory=json.loads((NEW/'FULL_V4_INVENTORY.json').read_text())['counts'],caveats=['Complete v4 not available for any arm yet','C/K paired single rank; historical early references used two ranks','Classification intersection excludes LC25000 and non-BA metrics; not all-task score','E50 segmentation is a specified budget, not all-budget summary','One representation training seed; no significance claim'])
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['models','inventory']},indent=2))
if __name__=='__main__':main()
