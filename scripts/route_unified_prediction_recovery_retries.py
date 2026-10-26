"""Route failed CPU retained-head reproduction to the live GPU queue once."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,d):
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def run():
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=OUT/'segmentation_recovery_queue.json';target=OUT/'segmentation_queue.json'
        cpu=read(source);gpu=read(target);keys={j['key'] for j in gpu['tasks']};keep=[];added=[];dependency_repairs=[];dependencies_changed=False
        inventories={}
        for job in cpu['tasks']+gpu['tasks']:
            if job.get('entrypoint')!='run_unified_paper_segmentation.py':continue
            args=job['args'];path=Path(args[args.index('--inventory-file')+1]) if '--inventory-file' in args else OUT/'biodino_segmentation_inventory.json'
            if str(path) not in inventories:inventories[str(path)]=read(path)['cells']
            cell=inventories[str(path)][int(args[args.index('--cell-index')+1])]
            if cell.get('superseded_protocol'):continue
            recovery='--recover-existing' in args
            if recovery and cell.get('requires_fresh_training'):
                reason='Retained-head recovery superseded by certified fresh five-seed fits; original outputs preserved'
                if job.get('blocked_reason')!=reason:job['blocked_reason']=reason;dependencies_changed=True
                continue
            required=[cell['caches']['test']] if recovery else list(cell['caches'].values())
            if job.get('requires')!=required:job['requires']=required;dependencies_changed=True
            attempt_path=OUT/'states'/f"{job['key']}_attempts.json";attempt=read(attempt_path)
            if attempt.get('attempts',0)<2 or not all(Path(f).is_file() for f in required):continue
            if Path(job['result']).exists() or (OUT/'locks'/f"{job.get('claim_key',job['model'])}.claimdir").exists():continue
            log=Path(job['log']).read_text() if Path(job['log']).exists() else ''
            failure=log.rstrip().splitlines()[-1] if log.rstrip() else ''
            if failure not in ['ValueError: Frozen split caches unavailable','ValueError: Test cache unavailable for retained-head recovery']:continue
            record=dict(job=job,prior_attempt=attempt,at=time.time(),reason='Previous attempts failed before training/inference because required frozen caches were missing. Exact required files now exist; enforce dependencies and retry unchanged fit/recovery.')
            write(OUT/'corrections'/f"{job['key']}__dependency_repair.json",record)
            write(attempt_path,dict(attempts=0,previous_dependency_failures=attempt,repaired_at=time.time()));dependency_repairs.append(job['key'])
        for job in cpu['tasks']:
            attempt=read(OUT/'states'/f"{job['key']}_attempts.json")
            audit=read(Path(job['result']).with_name('prediction_recovery_audit.json'))
            if Path(job['result']).exists() or attempt.get('attempts',0)<2 or audit.get('pass',True) or audit.get('inference_device')!='cpu':
                keep.append(job);continue
            new=dict(job);new['key']=job['key']+'__gpu_retry';new['log']=str(OUT/'logs'/f"{new['key']}.log");new['min_free_mib']=3500
            if new['key'] not in keys:added.append(new);keys.add(new['key'])
            record={'source_job':job,'attempt':attempt,'failed_audit':audit,'gpu_job':new,'at':time.time(),
                'reason':'Original head GPU inference retry; tolerance unchanged, failed artifacts preserved; no additional training seed'}
            write(OUT/'corrections'/f"{job['key']}__gpu_retry.json",record)
        if len(keep)!=len(cpu['tasks']) or dependencies_changed:
            cpu['tasks']=keep;gpu['tasks']=added+gpu['tasks'];write(source,cpu);write(target,gpu)
        write(OUT/'states/prediction_recovery_routing.json',{'pid':os.getpid(),'updated':time.time(),'added_gpu_retries':[j['key'] for j in added],'dependency_repairs':dependency_repairs})
        if dependency_repairs:print('CACHE DEPENDENCIES REPAIRED',dependency_repairs,flush=True)
        if added:print('GPU RECOVERY RETRIES',[j['key'] for j in added],flush=True)
    # A certified GPU reproduction failure has a concrete recovery path; do not
    # leave it permanently blocked or relax the predeclared metric tolerance.
    for inventory_name in ['biodino_segmentation_inventory.json','fm_segmentation_inventory.json']:
        inventory_path=OUT/inventory_name
        for cell in read(inventory_path).get('cells',[]):
            if cell.get('requires_fresh_training') or cell.get('superseded_protocol') or cell.get('execution_host'):continue
            if cell['dataset']=='multimodal_cellseg':continue
            folder=OUT/'results'/cell['model']/'segmentation'/cell['dataset']/cell['split']
            evidence=next((f for f in folder.glob('E*/seed*/prediction_recovery_audit.json') if read(f).get('pass') is False and read(f).get('inference_device','').startswith('cuda')),None)
            if evidence is None:continue
            if all(Path(p).is_file() for p in cell['caches'].values()):
                cmd=['admit_unified_retained_cache_refit.py','--model',cell['model'],'--dataset',cell['dataset'],'--split',cell['split'],'--inventory-file',str(inventory_path)]
            elif 'original_test_cache' in cell:
                cmd=['admit_unified_segmentation_refit.py','--model',cell['model'],'--dataset',cell['dataset'],'--evidence',str(evidence)]
            else:continue
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/cmd[0]),*cmd[1:]],capture_output=True,text=True,timeout=120)
            print('GPU MISMATCH REFIT',cell['model'],cell['dataset'],cell['split'],result.returncode,result.stdout,result.stderr,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    with (OUT/'locks/prediction_recovery_routing.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        while True:
            run()
            if not a.watch:return
            time.sleep(45)
if __name__=='__main__':main()
