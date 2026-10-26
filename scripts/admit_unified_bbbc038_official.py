"""Release the full fixed-checkpoint matrix only after a real full-budget pilot."""
import argparse,fcntl,json,os,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def job(spec,seed):
    model=spec['model'];key=f'{model}__bbbc038_official__seed{seed}'
    return dict(key=key,model=model,task='instance_segmentation',dataset='bbbc038-official',fold='stage1+stage2',seed=seed,
        entrypoint='run_unified_paper_bbbc038_official.py',args=['--model',model,'--seed',str(seed)],claim_key=key,
        min_free_mib=8000 if model in ['biodino_1tb_splus','dinov2']else max(12000,spec['min_free_mib']),expected_peak_mib=6000 if model in ['biodino_1tb_splus','dinov2']else max(10000,spec['min_free_mib']-3000),
        result=str(OUT/'results'/model/'bbbc038_official'/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
def run():
    with (OUT/'locks/model_admission.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=json.loads((OUT/'config.json').read_text());pilot_spec=next(m for m in cfg['models']if m['model']=='biodino_1tb_splus');pilot=job(pilot_spec,0)
        existing=json.loads((OUT/'bbbc038_official_queue.json').read_text())if(OUT/'bbbc038_official_queue.json').exists()else{'tasks':[pilot]}
        result=Path(pilot['result']);passed=False
        if result.exists():
            raw=json.loads(result.read_text());pred=Path(raw['prediction_path'])
            assert raw['completed_epochs']==50 and raw['seed']==0 and raw['checkpoint']==pilot_spec['checkpoint']
            assert {s:v['n_test']for s,v in raw['stages'].items()}==dict(stage1=65,stage2=106)
            assert hashlib.sha256(pred.read_bytes()).hexdigest()==raw['prediction_sha256']
            for r in json.loads(pred.read_text()):assert Path(r['prediction_path']).exists()and Path(r['truth_path']).exists()
            passed=True
        if passed:
            jobs={j['key']:j for j in existing['tasks']}
            for spec in cfg['models']:
                for seed in range(5):
                    j=job(spec,seed)
                    if spec.get('execution_host')=='5090-lyx-xr' or spec['model']in ['biodino_20tb_rebalanced_best','biodino_20tb_residual_best']:
                        j['blocked_reason']='Resident official DSB input preparation not yet complete; no checkpoint transfer permitted'
                    jobs.setdefault(j['key'],j)
            existing['tasks']=list(jobs.values())
        existing.update(stage='full_budget_pilot_passed'if passed else'one_real_full_budget_pilot',recipe='shared frozen HoVerNet50epochs B8, final epoch, all670official train, scored stage1/2 separately')
        dump(OUT/'bbbc038_official_queue.json',existing)
        dump(OUT/'states/bbbc038_official_admission.json',dict(pid=os.getpid(),updated=time.time(),status=existing['stage'],jobs=len(existing['tasks']),pilot_result=str(result)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with (OUT/'locks/bbbc038_official_admission.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            run()
            if not args.watch:break
            time.sleep(60)
