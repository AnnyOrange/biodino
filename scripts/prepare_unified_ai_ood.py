"""Publish verified lossless input packs and real resident-host OOD jobs."""
import fcntl,json,os,subprocess,time
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,PY
def sync(files):
    subprocess.run(['rsync','-aR','--partial',*map(str,files),HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True,timeout=43200)
def admit(task):
    code='''import json,pathlib,fcntl
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';task=__TASK__
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'ood_repair_queue.json';q=json.loads(path.read_text()) if path.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']};models=json.loads((out/'config.json').read_text())['models']
 for m in models:
  if m['model'] not in ['biodino_20tb_rebalanced_best','biodino_20tb_residual_best']:continue
  key=m['model']+'__ood__'+task+'__groupfix_v3'
  if key in known:continue
  q['tasks'].append(dict(key=key,model=m['model'],dataset=task,task='ood',entrypoint='run_unified_paper_ood.py',args=['--model',m['model'],'--task',task,'--checkpoint',m['checkpoint'],'--train-config',m['train_config'],'--input-fix-v2','--group-fix-v3','--batch-size','32','--workers','2'],claim_key=m['model']+'__resident_ood',min_free_mib=10000,expected_peak_mib=5500,result=str(out/'results'/m['model']/'ood_groupfix_v3'/f'{task}_result.json'),log=str(out/'logs'/f'{key}.log'),requires=[str(out/'prepared_inputs/ood_fixed_id/manifest.json'),str(out/'validation/ood_input_fingerprints'/f'{task}.json')]+([str(out/'prepared_inputs/ood_fixed_cryo_spatial/manifest.json')] if task=='cryo' else [])))
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(path);print('RESIDENT OOD JOBS',task,len(q['tasks']),flush=True)
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__TASK__',repr(task))
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'python3 -u -'],input=code,text=True,check=True,timeout=60)
def main():
    with (OUT/'locks/ai_ood_prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_pid=os.environ.get('UNIFIED_OOD_WAIT_TRANSFER_PID')
        while old_pid and Path(f'/proc/{old_pid}/cmdline').exists():
            try:active=b'rsync' in Path(f'/proc/{old_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if not active:break
            print('WAIT EXISTING ID TRANSFER',old_pid,flush=True);time.sleep(30)
        sync(['scripts/run_unified_paper_ood.py','scripts/unified_paper_ood_packed.py','scripts/verify_unified_ood_inputs.py','scripts/ood_input_fix.py','dinov3/eval/bio_frozen_eval/encoder.py'])
        published=[]
        for task in ['xray','cryo']:
            needed=[OUT/'prepared_inputs/ood_fixed_id/manifest.json',OUT/'validation/ood_input_fingerprints/id.json',OUT/'validation/ood_input_fingerprints'/f'{task}.json']
            if task=='cryo':needed.append(OUT/'prepared_inputs/ood_fixed_cryo_spatial/manifest.json')
            while not all(p.exists() for p in needed):
                print('WAIT DATA',task,[str(p) for p in needed if not p.exists()],flush=True);time.sleep(30)
            files=[str(p.relative_to(ROOT)) for p in needed]
            files += [str((OUT/'prepared_inputs/ood_fixed_id/id_inputs.npz').relative_to(ROOT))] if task=='xray' else [str((OUT/'prepared_inputs/ood_fixed_cryo_spatial'/name).relative_to(ROOT)) for name in ['cryo_inputs.npz','raw_validation_examples.npz']]
            # Transfer complete packs before allowing a worker to claim the task.
            sync(files);admit(task);published.append(task)
            state=OUT/'states/ai_ood_preparation.json';tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),published=published,host=HOST,input_only_transfer=True))+'\n');tmp.replace(state)
if __name__=='__main__':main()
