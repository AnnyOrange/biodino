"""Append the three missing v4 cells for every admitted method checkpoint."""
import argparse,collections,csv,fcntl,json,time
from pathlib import Path
import selective_retention_eval_queue_20260923 as q

ROOT=Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927')
PY='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'

def requeue_resource_interruptions():
    # Worker-observed terminated processes are retryable; retain every attempt.
    attempts=ROOT/'attempts';attempts.mkdir(exist_ok=True)
    for p in (ROOT/'claims').glob('*/status.json'):
        st=json.loads(p.read_text())
        if st.get('state')!='FAILED' or st.get('returncode')!=-15:continue
        if not (p.parent/'resource_interruption.json').exists():continue
        tid=p.parent.name
        if len(list(attempts.glob(tid+'__resource_retry_*')))>=3:continue
        q.atomic(p.parent/'retry_reason.json',dict(time=q.now(),reason='Owned worker terminated test after new foreign GPU occupancy; preserve outputs and retry elsewhere'))
        p.parent.rename(attempts/f'{tid}__resource_retry_{time.time_ns()}')

def register():
    gate=ROOT/'FULL_V4_ADMISSION.json'
    if not gate.exists():return
    admission=json.loads(gate.read_text());assert admission['status']=='READY_FOR_FULL_PROTOCOL_EXECUTION'
    source=Path(admission['source']);ready=ROOT/'full_registered';ready.mkdir(exist_ok=True)
    for marker in (ROOT/'registered').glob('*.json'):
        arm=marker.stem
        if (ready/marker.name).exists():continue
        original=json.loads(marker.read_text());ck=Path(original['checkpoint']);config=ck.parents[2]/'config.yaml'
        step=int(ck.parent.name.split('_')[-1]);assert q.sha(config)==original['train_config_sha256']
        common=dict(arm=arm,cwd=str(source),pythonpath=str(source),checkpoint_sha256=original['checkpoint_sha256'],
                    train_config_sha256=original['train_config_sha256'],protocol='bio-eval-union-v4',priority=0,
                    created=q.now(),admission_sha256=q.sha(gate),teacher_branch=True)
        ctc_root=ROOT/'ctc'/arm
        vendor=str((ROOT.parents[2]/'outputs/02_eval_runtime/py-ctcmetrics').resolve())
        bootstrap=(f"import os,sys,runpy;os.environ.update({{'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':{vendor!r}}});"
                   f"sys.path.insert(0,{str(source/'scripts')!r});runpy.run_path({str(source/'scripts/run_method_native_ctc_20260928.py')!r},run_name='__main__')")
        tasks=[dict(common,id=f'ctc__native__{arm}',family='cell_tracking',dataset='ctc',order=100,
            heavy=True,expected_memory_mib=9000,
            cmd=[PY,'-B','-c',bootstrap,'--arm',arm,'--checkpoint',str(ck),
                 '--config',str(config),'--step',str(step),'--output',str(ctc_root)],
            done=dict(type='ctc_native',path=str(ctc_root/'models'/arm/'results.json'),dataset='ctc'))]
        linkdir=ROOT/'ood_adapters'/arm/str(step);linkdir.mkdir(parents=True,exist_ok=True);link=linkdir/'checkpoint.pth'
        if not link.exists():link.symlink_to(ck)
        assert link.resolve()==ck.resolve()
        for index,ds in enumerate(['xray','cryo']):
            label=f'{arm}_{ds}';out=ROOT/'ood'/ds
            cmd=[PY,'-B','-m','dinov3.eval.eval_ood.dinov3_runner','--model-name',label,'--ckpt-root',str(linkdir.parent),
                 '--ckpt-iter',str(step),'--train-config',str(config),'--output-dir',str(out),'--benchmark-root','/mnt/huawei_deepcad/benchmark',
                 '--ood-root','/mnt/huawei_deepcad/benchmark/ood','--tasks',ds,'--device','cuda:0','--batch-size','64','--num-workers','2',
                 '--seed','0','--n-last-blocks','1','--autocast-dtype','bf16','--resize-size','256','--crop-size','224',
                 '--percentile-low','0.5','--percentile-high','99.5','--xray-input-mode','three_slices','--xray-slices-per-volume','8',
                 '--cryo-max-particles-per-project','20000','--id-max-samples','3000','--id-datasets','bloodmnist','bbbc048','cyclops','--phase','all']
            tasks.append(dict(common,id=f'ood__{ds}__{arm}',family='ood',dataset=ds,order=101+index,
                              expected_memory_mib=4200,cmd=cmd,
                              done=dict(type='ood_json',dataset=ds,path=str(out/label/str(step)/'last_result.json'),expected_ood=992 if ds=='xray' else 80000)))
        for task in tasks:q.atomic(ROOT/'tasks'/f'{task["id"]}.json',task)
        original.update(n_tasks=49,extension_tasks=3,blocked=[],
                        note='56 task-dataset cells scheduled; validated outputs required. LC25000 classification remains provisional.')
        q.atomic(marker,original)
        q.atomic(ready/marker.name,dict(time=q.now(),arm=arm,added_tasks=[t['id'] for t in tasks],expected_cells=56,
                 note='All v4 cells now scheduled; completion requires actual validated results. LC25000 remains provisional.'))
        print(q.now(),'REGISTER_FULL',arm,flush=True)

def inventory():
    spec=json.loads((ROOT.parents[2]/'Evaluation Rules/protocol_v4.json').read_text())
    families={}
    for family in ['classification','regression','retrieval','segmentation','cell_tracking','ood']:
        families[family]=sum([spec.get(section,{}).get(family,[]) for section in ['tier_a','tier_b','union_extension']],[])
    families['clustering']=families['retrieval'];families['detection_proxy']=spec['union_extension']['detection_proxy']
    tasks=[json.loads(p.read_text()) for p in (ROOT/'tasks').glob('*.json')];rows=[]
    for marker in (ROOT/'registered').glob('*.json'):
        arm=marker.stem
        for family,datasets in families.items():
            for dataset in datasets:
                actual='classification' if family=='regression' else 'retrieval' if family=='clustering' else family
                match=[t for t in tasks if t['arm']==arm and t['dataset']==dataset and t['family']==actual]
                assert len(match)<=1,(arm,family,dataset)
                state='PREFLIGHT_PENDING';task_id=None
                if match:
                    task_id=match[0]['id'];p=ROOT/'claims'/task_id/'status.json'
                    state=json.loads(p.read_text())['state'] if p.exists() else 'QUEUED'
                rows.append(dict(arm=arm,family=family,dataset=dataset,state=state,task_id=task_id,
                                 provisional=family=='classification' and dataset=='lc25000'))
    counts={arm:dict(collections.Counter(r['state'] for r in rows if r['arm']==arm)) for arm in sorted({r['arm'] for r in rows})}
    q.atomic(ROOT/'FULL_V4_INVENTORY.json',dict(time=q.now(),expected_cells_per_checkpoint=56,counts=counts,cells=rows,
            execution_complete=bool(rows) and all(r['state']=='DONE' for r in rows),strict_aggregate_allowed=False,
            note='All cells stay in inventory. LC25000 classification retains the protocol provisional label.'))
    if (ROOT/'FULL_V4_ADMISSION.json').exists():
        path=ROOT/'EXPECTED_INVENTORY.json';data=json.loads(path.read_text())
        worker=ROOT/'WORKER.json'
        deepcad_gpus=json.loads(worker.read_text()).get('gpus',[]) if worker.exists() else []
        data.update(blocked=[],target_hosts={'local5090':list(range(8)),'deepcad':deepcad_gpus,'cpu15':[0]},
                    execution_scope='All 56 v4 cells scheduled; see FULL_V4_INVENTORY.json for actual completion.')
        for manifest in (ROOT/'fleet').glob('deepcad_gpu*/manifest.json'):
            if (manifest.parent/'RELOCATED.json').exists():continue
            worker_info=json.loads(manifest.read_text())
            data['target_hosts'][manifest.parent.name]=worker_info['gpus']
        q.atomic(path,data)

def main():
    lock=(ROOT/'full_register.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    for _ in range(60*60):
        # Discover new C/K checkpoints before appending the full v4 extension.
        import deepcad_method_campaign_20260927 as campaign
        campaign.register()
        register()
        requeue_resource_interruptions()
        inventory()
        if a.once:return
        time.sleep(60)

if __name__=='__main__':main()
