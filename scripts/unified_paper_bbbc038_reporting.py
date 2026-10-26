"""The same five fitted heads evaluated separately on both released scoring sets."""
import hashlib,json
from pathlib import Path
import numpy as np
def apply_bbbc038_results(root,rows,runs,models):
    out=root/'outputs/02_eval_runs/unified_paper';manifest=root/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json'
    if not manifest.exists():return
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest();data=json.loads(manifest.read_text())
    for model,spec in models.items():
        found=[]
        for seed in range(5):
            path=out/'results'/model/'bbbc038_official'/f'seed{seed}'/'component_result.json'
            if not path.exists():continue
            raw=json.loads(path.read_text());pred=Path(raw['prediction_path'])
            assert raw['model']==model and raw['seed']==seed and raw['checkpoint']==spec['checkpoint']and raw['manifest_sha256']==digest and raw['completed_epochs']==50
            assert hashlib.sha256(pred.read_bytes()).hexdigest()==raw['prediction_sha256'];predictions=json.loads(pred.read_text())
            for stage in ['stage1','stage2']:
                selected=[r for r in predictions if r['fold']==stage]
                assert [r['sample_id']for r in selected]==[r['sample_id']for r in data['splits'][stage]]
                assert abs(np.mean([r['metrics']['dsb_mean_precision']for r in selected])-raw['stages'][stage]['metrics']['dsb_mean_precision'])<1e-12
            found.append((seed,path,raw))
        for stage in ['stage1','stage2']:
            key=(model,'instance_segmentation','bbbc038-official','',stage)
            if key not in rows:continue
            row=rows[key];row.update(metric='dsb_mean_precision',required_seeds=5,completed_seeds=len(found),data_complete='是',sample_manifest=str(manifest),split_sha256=digest,split_id=data['protocol']+':'+stage,patient_group_verified=False,uncertainty_group_scope=data['grouping'])
            if not found:continue
            values=[r['stages'][stage]['metrics']['dsb_mean_precision']for _,_,r in found]
            row.update(value=float(np.mean(values)),sd=float(np.std(values,ddof=1))if len(values)>1 else np.nan,seed_values=json.dumps(values),seed_ids=json.dumps([s for s,_,_ in found]),protocol_verified=True,protocol=data['protocol'],raw_result_path=json.dumps([str(p)for _,p,_ in found]),prediction_path=json.dumps([r['prediction_path']for _,_,r in found]),seed_semantics=found[0][2]['seed_semantics'],reason=f'{len(found)}/5 fixed50epoch frozen decoder fits; complete official {stage} scoring objects; local strict-IoU scorer; not a leaderboard submission',validation_basis='Full670training cohort, no test selection; per-image masks and scored IDs retained. One fit evaluated in both stages, never counted as two trainings.')
            ci=out/'statistics'/f'{model}__bbbc038-official__{stage}__ci.json'
            if ci.exists()and len(found)==5:
                stat=json.loads(ci.read_text())
                if stat['prediction_sha256']==[r['prediction_sha256']for _,_,r in found]:
                    value=stat['metrics']['dsb_mean_precision'];row.update(ci_low=value['ci_low'],ci_high=value['ci_high'],statistics_path=str(ci))
            for seed,path,raw in found:runs.append({**row,'seed':seed,'run_value':raw['stages'][stage]['metrics']['dsb_mean_precision'],'raw_result_path':str(path),'prediction_path':raw['prediction_path']})
