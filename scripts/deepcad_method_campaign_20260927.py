"""Independent training discovery and persistent v4 evaluation on free GPUs."""
import argparse, csv, fcntl, hashlib, json, math, os, signal, socket, subprocess, sys, time
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

REPO=Path('/mnt/huawei_deepcad/dinov3')
TRAIN=REPO/'outputs/01_training_runs/hs6_l5_deepcad_method_20260927'
REPORT=REPO/'outputs/00_reports/deepcad_method_20260927'
EVAL=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'
OLD=REPO/'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923'
OLDTRAIN=REPO/'outputs/01_training_runs/hs6_l5_selective_retention_20260923/adaptive_formal'
TAG='adaptive_formal_ck14151'
PY='/home/deepcad/anaconda3/envs/dinov3/bin/python'
RUNS=['adaptive_continue_v2_formal','adaptive_continue_v2_relocated','adaptive_continue_v2_gpu2',
      'metric_mid_formal','adaptive_mid_formal','adaptive_mid_retry_formal']
sys.path.insert(0,str(REPO/'scripts'))
import run_deepcad_method_20260927 as train
import selective_retention_eval_queue_20260923 as queue

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return queue.sha(p)
def atomic(p,d):return queue.atomic(p,d)
def known_runs():return sorted(set(RUNS)|{p.name for pattern in ['adaptive_mid_retry*_formal','adaptive_continue_resume*_formal','ck_c_e12687_formal','ck_k_e12687_formal'] for p in TRAIN.glob(pattern)})

def admit_smoke():
    p=TRAIN/'metric_mid_smoke_b128'
    if not (p/'exit.json').exists():return False
    assert json.loads((p/'exit.json').read_text())['returncode']==0,'DDP smoke failed'
    rows=[json.loads(x) for x in (p/'raw_loss_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==16
    for r in rows:
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,float))
        assert 0<r['total_loss']<50 and 0<r['backbone_grad_norm']<100
    assert all(r['metric_loss']>0 for r in rows[-4:]),'No active metric gradient after calibration'
    mechanism=json.loads((REPORT/'MECHANISM_CHECKS.json').read_text())
    assert mechanism['engineering_pilot_allowed']
    atomic(REPORT/'PILOT_ADMISSION.json',dict(time=now(),status='ADMITTED_EXPLORATORY_FIXED_488_UPDATES',
        smoke_steps=len(rows),last_loss=rows[-1]['total_loss'],last_metric_loss=rows[-1]['metric_loss'],
        mathematical_checks=mechanism['mathematical_checks'],method_design_sha256=sha(REPORT/'METHOD_DESIGN.md'),
        full_v4_aggregate_allowed=False,task_improvement_established=False))
    return True

def register():
    # The watcher and evaluation worker discover independently; serialize claims.
    with (EVAL/'register.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _register()

def _register():
    templates=[json.loads(p.read_text()) for p in (OLD/'tasks').glob('*'+TAG+'*')]
    # Prefer canonical native-final, never launch custom multi-layer companions.
    templates=[t for t in templates if not (t['id'].startswith('seg__') and t['dataset'] in ['conic','livecell','pannuke'])]
    source_manifest={}
    for run in known_runs():
        path=TRAIN/run
        if (path/'INVALID_RESUME.json').exists():continue
        for ck in sorted((path/'eval').glob('training_*/teacher_checkpoint.pth')):
            if time.time()-ck.stat().st_mtime<60:continue
            step=int(ck.parent.name.split('_')[-1]);arm=f'{run}_ck{step}'
            marker=EVAL/'registered'/f'{arm}.json'
            if marker.exists():continue
            config=path/'config.yaml'
            import torch
            state=torch.load(ck,map_location='cpu',mmap=True,weights_only=False)
            assert isinstance(state.get('teacher'),dict) and any(k.startswith('backbone.') for k in state['teacher'])
            cksha=sha(ck);configsha=sha(config);del state
            def replace(value):
                if isinstance(value,str):
                    return value.replace(str(OLDTRAIN/'eval/training_14151/teacher_checkpoint.pth'),str(ck)).replace(str(OLDTRAIN/'config.yaml'),str(config)).replace(str(OLD),str(EVAL)).replace(TAG,arm)
                if isinstance(value,list):return [replace(x) for x in value]
                if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
                return value
            for template in templates:
                t=replace(template);cmd=t['cmd']
                if '--checkpoint-id' in cmd:cmd[cmd.index('--checkpoint-id')+1]=str(step)
                if 'ckpt_id' in t['done']:t['done']['ckpt_id']=step
                if t['dataset']=='rxrx3-core':
                    cmd[1]=str(EVAL/'runtime/run_rxrx3.py')
                if t['family']=='detection_proxy':
                    adapter=EVAL/'runtime/run_detection_single_rank.py'
                    cmd[2:4]=[str(adapter)]
                    t['runtime_adapter_sha256']=sha(adapter)
                for flag,value in [('--batch-size','8' if t['family']=='detection_proxy' else '64'),('--feature-batch-size','32'),('--probe-batch-size','32')]:
                    if flag in cmd:assert cmd[cmd.index(flag)+1]==value,(t['id'],flag)
                t.update(checkpoint_sha256=cksha,train_config_sha256=configsha,created=now(),
                         priority=0 if t['dataset'] in ['chammi-cp-task3','chammi-hpa-task2','cellpose','bbbc038','hpa-subcellular','rxrx3-core'] else 1)
                cwd=Path(t['cwd']);assert cwd.is_dir()
                source_manifest[str(cwd)]=None
                t['template_sha256']=hashlib.sha256(json.dumps(template,sort_keys=True).encode()).hexdigest()
                atomic(EVAL/'tasks'/f'{t["id"]}.json',t)
            atomic(marker,dict(time=now(),checkpoint=str(ck),checkpoint_sha256=cksha,train_config_sha256=configsha,
                n_tasks=len(templates),expected_cells=56,full_v4_aggregate_allowed=False,
                blocked=['cell_tracking/ctc: native new-arm admission pending','ood/xray: not scheduled','ood/cryo: not scheduled'],
                note='29 frozen tasks = 25 classification + 4 regression; 7 retrieval tasks also produce clustering. 7 segmentation tasks include all budgets/rotations/seeds.'))
    for source in source_manifest:
        sid=hashlib.sha256(source.encode()).hexdigest()[:16]
        p=EVAL/'provenance'/f'source_{sid}.json'
        if not p.exists():atomic(p,dict(source=source,files=train.hash_files(Path(source)),time=now()))

def prepare():
    for sub in ['tasks','claims','registered','logs','runtime','provenance']:(EVAL/sub).mkdir(parents=True,exist_ok=True)
    src=(REPO/'scripts/run_selective_rxrx3_20260923.py').read_text()
    src=src.replace('import selective_retention_eval_queue_20260923 as q',
        'import sys\nsys.path.insert(0,"'+str(REPO/'scripts')+'")\nimport selective_retention_eval_queue_20260923 as q\nq.ROOT=Path("'+str(EVAL)+'")')
    # Path must exist before the injected root assignment.
    assert 'from pathlib import Path' in src
    (EVAL/'runtime/run_rxrx3.py').write_text(src)
    (EVAL/'runtime/run_detection_single_rank.py').write_text((REPO/'scripts/run_deepcad_detection_single_rank_20260928.py').read_text())
    spec=json.loads((REPO/'Evaluation Rules/protocol_v4.json').read_text())
    atomic(EVAL/'EXPECTED_INVENTORY.json',dict(protocol=spec,expected_per_arm=56,
        families=dict(classification=25,regression=4,retrieval=7,clustering=7,segmentation=7,detection_proxy=3,cell_tracking=1,ood=2),
        full_v4_aggregate_allowed=False,classification_strict_mean_excludes=['lc25000'],
        blocked=['CTC native evaluator new-arm admission','OOD xray/cryo not scheduled'],
        source_template=str(OLD),target_gpus=[4,5],test_min_slots=5,target_vram_fraction=.70))

def evaluate(hours=60,gpus=None):
    from run_method_v4_fleet_20260928 import validate as validate_task
    gpus=[5] if gpus is None else gpus
    prepare();active={};started=time.monotonic();laststart={g:0. for g in gpus};failures=0
    recently_owned={};foreign_seen={};yielded=set()
    worker_lock=(EVAL/'eval.lock').open('a');fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks=[]
    for g in gpus:
        f=open(f'/tmp/dinov3_deepcad_gpu_{g}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    atomic(EVAL/'WORKER.json',dict(time=now(),pid=os.getpid(),gpus=gpus,hours=hours))
    # Preserve in-flight evaluations if a supervisor is restarted.
    for status_file in (EVAL/'claims').glob('*/status.json'):
        st=json.loads(status_file.read_text())
        if st['state']!='RUNNING':continue
        if st.get('host') not in (None,'deepcad',socket.gethostname()):continue
        tid=status_file.parent.name;t=json.loads((EVAL/'tasks'/f'{tid}.json').read_text())
        cmdline=Path(f'/proc/{st["pid"]}/cmdline')
        actual=cmdline.read_bytes().rstrip(b'\0').decode().split('\0') if cmdline.exists() else []
        if actual!=st.get('command'):
            ok,why=validate_task(t)
            if not ok and (status_file.parent/'yield_reason.json').exists():
                archive=EVAL/'attempts'/f'{tid}__yield_reconciled_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True)
                status_file.parent.rename(archive)
                continue
            atomic(status_file,dict(st,state='DONE' if ok else 'FAILED',validation=why,reconciled=now()))
            continue
        assert st['gpu'] in gpus,'Live evaluations on another GPU; retain their supervisor'
        proc=queue.AdoptedProcess(st['pid'],status_file.parent/'exit.json')
        log=(EVAL/'logs'/f'{tid}.log').open('a')
        active[tid]=(proc,t,st['gpu'],log,time.monotonic())
    while time.monotonic()-started<hours*3600 or active:
        register()
        # NVIDIA's process list can briefly retain a just-finished CUDA PID.
        # Remember our own descendants before removing completed task handles.
        for proc,*_ in active.values():
            for pid in train.descendants(proc.pid):recently_owned[pid]=time.monotonic()+60
        for tid,(proc,t,g,log,began) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close()
            try:ok,why=validate_task(t)
            except Exception as exc:ok,why=False,repr(exc)
            atomic(EVAL/'claims'/tid/'status.json',dict(state='DONE' if rc==0 and ok else 'FAILED',returncode=rc,validation=why,gpu=g,pid=proc.pid,end=now()))
            if tid in yielded and not ok:
                archive=EVAL/'attempts'/f'{tid}__yield_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True)
                (EVAL/'claims'/tid).rename(archive)
                yielded.remove(tid)
            elif rc or not ok:failures+=1
            del active[tid]
        gs=train.gpu_info();procs=train.gpu_processes();owned=set()
        for proc,*_ in active.values():owned|=train.descendants(proc.pid)
        recently_owned={pid:expiry for pid,expiry in recently_owned.items() if expiry>time.monotonic()}
        foreign={g:[p for p in procs if p['uuid']==gs[g]['uuid'] and p['pid'] not in owned
                    and p['pid'] not in recently_owned and Path(f'/proc/{p["pid"]}').exists()] for g in gpus}
        seen_now={(g,p['pid']) for g in gpus for p in foreign[g]}
        foreign_seen={key:foreign_seen.get(key,0)+1 for key in seen_now}
        # Yield only this campaign's children if another user starts on our GPU.
        for tid,(proc,t,g,log,began) in list(active.items()):
            if any(foreign_seen[(g,p['pid'])]>=2 for p in foreign[g]):
                atomic(EVAL/'claims'/tid/'yield_reason.json',dict(time=now(),foreign=foreign[g]))
                yielded.add(tid);train.stop(proc)
        counts=Counter(v[2] for v in active.values())
        atomic(EVAL/'RESOURCE_STATUS.json',dict(time=now(),supervisor=os.getpid(),gpus={g:gs[g] for g in gpus},running=dict(counts),foreign=foreign,failures=failures))
        if failures>=6 or time.monotonic()-started>=hours*3600:
            if not active:break
            time.sleep(10);continue
        pending=[json.loads(p.read_text()) for p in (EVAL/'tasks').glob('*.json') if not (EVAL/'claims'/p.stem).exists()]
        pending.sort(key=lambda t:(t['priority'],t['order'],t['id']))
        avail=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])//1024
        for g in gpus:
            if foreign[g] or counts[g]>=9 or time.monotonic()-laststart[g]<12:continue
            if counts[g]>=5 and gs[g]['used']/gs[g]['total']>=.70:continue
            for t in pending:
                # Reserve concurrent extraction peaks, including processes still loading.
                peak=max(9000 if t.get('heavy') else 3500,queue.gpu_memory_cost(t))
                committed=sum(max(9000 if v[1].get('heavy') else 3500,queue.gpu_memory_cost(v[1])) for v in active.values() if v[2]==g)
                if committed+peak>gs[g]['total']-3000 or gs[g]['total']-gs[g]['used']<peak+1500:continue
                if t.get('heavy') and any(v[2]==g and v[1].get('heavy') for v in active.values()):continue
                if avail<queue.host_memory_cost(t)+50000:continue
                # Verify immutable evaluator fingerprint before each task.
                sid=hashlib.sha256(t['cwd'].encode()).hexdigest()[:16]
                registered=json.loads((EVAL/'provenance'/f'source_{sid}.json').read_text())['files']
                if train.hash_files(Path(t['cwd']))!=registered:raise RuntimeError('Evaluator source changed: '+t['cwd'])
                if 'runtime_adapter_sha256' in t:
                    assert sha(Path(t['cmd'][2]))==t['runtime_adapter_sha256'],'Runtime adapter changed'
                claim=EVAL/'claims'/t['id']
                try:claim.mkdir()
                except FileExistsError:continue
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=t['pythonpath'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                env['DINOV3_ROOT']=t['cwd']
                # Single-rank evaluators do not need IB/P2P/cuMem transport.
                # Use the host-compatible path; preserve all numerical settings.
                env.update(NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_NET='Socket',
                           NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0',PYTHONFAULTHANDLER='1')
                if t['family']=='detection_proxy':env['NCCL_DEBUG']='INFO'
                cmd=[str(x).replace('{GPU}',str(g)) for x in t['cmd']]
                log=(EVAL/'logs'/f'{t["id"]}.log').open('a')
                proc=subprocess.Popen(cmd,cwd=t['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic(claim/'status.json',dict(state='RUNNING',start=now(),gpu=g,pid=proc.pid,command=cmd,
                      runtime_environment={k:v for k,v in env.items() if k.startswith('NCCL_') or k in ['CUDA_VISIBLE_DEVICES','PYTHONFAULTHANDLER']}))
                active[t['id']]=(proc,t,g,log,time.monotonic());laststart[g]=time.monotonic();counts[g]+=1;pending.remove(t);break
        time.sleep(10)
    atomic(EVAL/'QUEUE_EXIT.json',dict(time=now(),failures=failures))

def campaign():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare()
    deadline=time.time()+1800
    while not admit_smoke():
        if time.time()>deadline:raise TimeoutError('Engineering smoke did not finish')
        atomic(REPORT/'STATUS.json',dict(time=now(),stage='WAITING_DDP_SMOKE'));time.sleep(15)
    selected_gpu=5
    for arm,source,port in [
        ('metric_mid','/mnt/huawei_deepcad/dinov3_metric_recovery_snapshot_20260927',32737),
        ('adaptive_mid','/mnt/huawei_deepcad/dinov3_adaptive_continue_v2_snapshot_20260927',32739)]:
        out=TRAIN/(arm+'_formal')
        if (out/'exit.json').exists():
            assert json.loads((out/'exit.json').read_text())['returncode']==0
            continue
        wait_deadline=time.monotonic()+72*3600
        while True:
            gs=train.gpu_info();occupied={p['uuid'] for p in train.gpu_processes()}
            available=[g for g in [5,4,3,0] if gs[g]['used']<1024 and gs[g]['uuid'] not in occupied]
            if available:selected_gpu=available[0];break
            atomic(REPORT/'STATUS.json',dict(time=now(),stage='WAITING_FREE_GPU',arm=arm,candidate_gpus=[5,4,3,0],
                                           reason='Other users own CUDA processes; do not share or terminate them.'))
            if time.monotonic()>wait_deadline:raise TimeoutError('No candidate GPU became available within72h')
            time.sleep(30)
        atomic(REPORT/'STATUS.json',dict(time=now(),stage='TRAINING',arm=arm,gpus=[selected_gpu],planned_end=20495))
        cmd=[PY,str(REPO/'scripts/run_deepcad_method_20260927.py'),'--arm',arm,'--gpus',str(selected_gpu),'--batch','128','--end','20495','--tag','formal','--port',str(port),'--fresh-mid','--source',source]
        rc=subprocess.call(cmd)
        if rc:raise RuntimeError(f'{arm} failed with {rc}; stopping chain')
        assert (out/'eval/training_20495/teacher_checkpoint.pth').exists()
        register()
    # Exact sample-sequence comparison documents any residual data-stream confound.
    rows=[]
    for arm in ['metric_mid','adaptive_mid']:
        rows.append([json.loads(x) for x in (TRAIN/(arm+'_formal')/'raw_loss_metrics.jsonl').read_text().splitlines()])
    equal=[a['batch_sample_key_digest']==b['batch_sample_key_digest'] for a,b in zip(*rows)]
    atomic(REPORT/'PAIRED_STREAM_AUDIT.json',dict(updates=[len(r) for r in rows],compared=len(equal),matching=sum(equal),all_match=all(equal)))
    atomic(REPORT/'STATUS.json',dict(time=now(),stage='V4_EVALUATION',gpus=[selected_gpu]))
    evaluate(gpus=[selected_gpu])

def watch():
    prepare()
    lock=(REPORT/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(60*72):
        register()
        progress={}
        for run in known_runs():
            p=TRAIN/run;f=p/'raw_loss_metrics.jsonl'
            entry={'status':'NOT_STARTED'}
            if f.exists():
                lines=f.read_text().splitlines()
                if lines:
                    d=json.loads(lines[-1]);entry=dict(status='RUNNING',updates_in_run=len(lines),checkpoint=d['optimizer_update'],loss=d['total_loss'],
                       batch=d['local_batch_size'],effective_batch=d['effective_global_batch_size'])
            if (p/'exit.json').exists():entry.update(status='EXITED',exit=json.loads((p/'exit.json').read_text()))
            if (p/'telemetry.json').exists():entry['resources']=json.loads((p/'telemetry.json').read_text())
            progress[run]=entry
        counts=Counter(json.loads(p.read_text())['state'] for p in (EVAL/'claims').glob('*/status.json'))
        tasks=len(list((EVAL/'tasks').glob('*.json')))
        evaluation=dict(tasks=tasks,states=dict(counts),unclaimed=tasks-sum(counts.values()),full_v4_aggregate_allowed=False)
        resource=EVAL/'RESOURCE_STATUS.json'
        if resource.exists():evaluation['resources']=json.loads(resource.read_text())
        atomic(REPORT/'LIVE_PROGRESS.json',dict(time=now(),runs=progress,evaluation=evaluation,registered_evaluations=len(list((EVAL/'registered').glob('*.json')))))
        atomic(REPORT/'STATUS.json',dict(time=now(),stage='INDEPENDENT_TRAINING_AND_EVALUATION',evaluation=evaluation,
              prior_failure='Serial coordinator stopped on interrupted adaptive_mid; repaired 2026-09-28.',
              note='Task registration is not evaluation completion. Read states and resource timestamp.'))
        controls=[p for p in TRAIN.glob('adaptive_mid_retry*_formal') if (p/'raw_loss_metrics.jsonl').exists()]
        control=max(controls,key=lambda p:(p/'launch_manifest.json').stat().st_mtime).name if controls else 'adaptive_mid_retry_formal'
        pair=[TRAIN/run/'raw_loss_metrics.jsonl' for run in ['metric_mid_formal',control]]
        if all(p.exists() for p in pair):
            rows=[[json.loads(line) for line in p.read_text().splitlines()] for p in pair]
            matches=[a['batch_sample_key_digest']==b['batch_sample_key_digest'] for a,b in zip(*rows)]
            atomic(REPORT/'PAIRED_STREAM_AUDIT.json',dict(time=now(),runs=['metric_mid_formal',control],
                  updates=[len(r) for r in rows],compared=len(matches),matching=sum(matches),
                  all_compared_match=all(matches),complete=len(matches)==488))
        time.sleep(60)

def control_retry():
    """Bounded retries for infrastructure interruptions, on GPU3 only."""
    lock=(REPORT/'control_retry.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+60*3600
    for attempt in range(2,5):
        arm=f'adaptive_mid_retry{attempt}';out=TRAIN/(arm+'_formal')
        if out.exists():
            if (out/'eval/training_20495/teacher_checkpoint.pth').exists():return
            continue
        while True:
            gs=train.gpu_info();occupied={p['uuid'] for p in train.gpu_processes()}
            if gs[3]['used']<1024 and gs[3]['uuid'] not in occupied:break
            atomic(REPORT/'CONTROL_STATUS.json',dict(time=now(),state='WAITING_FREE_GPU3',arm=arm))
            if time.monotonic()>deadline:raise TimeoutError('Control retry deadline')
            time.sleep(30)
        cmd=[PY,str(REPO/'scripts/run_deepcad_method_20260927.py'),'--arm',arm,'--gpus','3',
             '--batch','128','--end','20495','--tag','formal','--port',str(32747+attempt),'--fresh-mid',
             '--source','/mnt/huawei_deepcad/dinov3_adaptive_continue_v2_snapshot_20260927']
        atomic(REPORT/'CONTROL_STATUS.json',dict(time=now(),state='TRAINING',arm=arm,gpu=3,command=cmd))
        rc=subprocess.call(cmd)
        if rc==0:
            atomic(REPORT/'CONTROL_STATUS.json',dict(time=now(),state='COMPLETE',arm=arm));return
        exit_file=out/'exit.json'
        if not exit_file.exists() or json.loads(exit_file.read_text()).get('reason')!='foreign process entered selected GPU':
            atomic(REPORT/'CONTROL_STATUS.json',dict(time=now(),state='FAILED',arm=arm,returncode=rc));return
    atomic(REPORT/'CONTROL_STATUS.json',dict(time=now(),state='RETRY_BUDGET_EXHAUSTED',max_additional_attempts=3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['campaign','register','eval','prepare','watch','control-retry'])
    p.add_argument('--gpus',default='4,5');p.add_argument('--hours',type=float,default=60);a=p.parse_args()
    if a.mode=='campaign':campaign()
    elif a.mode=='register':prepare();register()
    elif a.mode=='eval':evaluate(hours=a.hours,gpus=[int(g) for g in a.gpus.split(',')])
    elif a.mode=='watch':watch()
    elif a.mode=='control-retry':control_retry()
    else:prepare()
