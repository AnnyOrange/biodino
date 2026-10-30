#!/usr/bin/env python3
"""Preserve validated full checkpoints before switching each H100 seed to 512x2."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/data_2/suxin/runs/L_seed_repeats_20261028')
REPO=Path('/home/suxin/project/dinov3')
SCRIPT=str(REPO/'scripts/run_l_seed_repeats_h100_20261028.py')
JOBS=[('1tb',1),('1tb',2),('5tb',1),('5tb',2)]


def write(path,obj):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def processes():
    rows={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
            parent=int((p/'stat').read_text().split(') ')[1].split()[1])
            rows[int(p.name)]=(parent,args)
        except (OSError,ValueError):pass
    return rows


def run_processes(dataset,seed,rows):
    wrappers={pid for pid,(_,a) in rows.items() if SCRIPT in a and 'train' in a and
              '--dataset' in a and a[a.index('--dataset')+1]==dataset and
              '--seed' in a and a[a.index('--seed')+1]==str(seed)}
    result=set(wrappers)
    # Include surviving rank trees when a wrapper exited unexpectedly.
    out=str(ROOT/f'{dataset}_seed{seed}')
    result|={pid for pid,(_,a) in rows.items() if '--output-dir' in a and a[a.index('--output-dir')+1]==out and
             any(x.endswith('/train_entry.py') for x in a)}
    while True:
        extra={pid for pid,(parent,_) in rows.items() if parent in result}-result
        if not extra:break
        result|=extra
    return result


def stop(pids,rows):
    for pid in pids:
        try:
            fd=os.pidfd_open(pid)
            actual=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if actual!=rows[pid][1]:raise RuntimeError(f'PID identity changed: {pid}')
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            os.close(fd)
        except ProcessLookupError:pass


def apply_config(run,checkpoint):
    import yaml
    source=run/'source_config.yaml';plan_path=run/'plan.json'
    plan=json.loads(plan_path.read_text())
    cfg=yaml.safe_load(source.read_text())
    if not (run/'pre_batch512_source_config.yaml').exists():
        (run/'pre_batch512_source_config.yaml').write_text(source.read_text())
        write(run/'pre_batch512_plan.json',plan)
    cfg['train']['batch_size_per_gpu']=512
    cfg['optim']['gradient_accumulation_steps']=1
    assert cfg['train']['seed']==plan['seed']
    temp=source.with_suffix('.yaml.pending');temp.write_text(yaml.safe_dump(cfg,sort_keys=False));temp.replace(source)
    plan.update(microbatch=512,accumulation=1,global_batch=1024,config_sha256=sha(source),
                batch512_switch_checkpoint=checkpoint,
                batch_switch_note='User requested 512 per GPU without gradient accumulation; KoLeo group size changes. Initial segment is retained, not a pure 512 run from initialization.')
    write(plan_path,plan)
    return plan



def throughput(run, since=0):
    path=run/'raw_loss_metrics.jsonl'
    if not path.exists():return dict(status='NOT_STARTED')
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-2000000))
        lines=stream.read().splitlines()
    rows=[]
    for line in lines:
        try:
            row=json.loads(line)
            if row['time_unix']>=since:rows.append(row)
        except (ValueError,KeyError):pass
    if not rows:return dict(status='NO_POST_SWITCH_UPDATES_YET')
    last=rows[-1]
    recent=[row for row in rows if row['time_unix']>=last['time_unix']-1800]
    first=recent[0]
    updates=last['optimizer_updates_completed']-first['optimizer_updates_completed']
    elapsed=last['time_unix']-first['time_unix']
    return dict(status='MEASURED',updates=last['optimizer_updates_completed'],
                local_batch=last['local_batch_size'],accumulation=last['accum_steps'],
                effective_global_batch=last['effective_global_batch_size'],
                batch512_verified=last['local_batch_size']==512 and last['accum_steps']==1 and last['effective_global_batch_size']==1024,
                measurement_updates=updates,measurement_seconds=elapsed,
                seconds_per_update=elapsed/updates if updates>0 else None,
                images_per_second=1024*updates/elapsed if elapsed>0 else None,
                last_update_age_seconds=time.time()-last['time_unix'])


def main():
    with (ROOT/'batch512_switch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        status={}
        for dataset,seed in JOBS:
            name=f'{dataset}_seed{seed}';run=ROOT/name
            marker=run/'BATCH_SWITCH_IN_PROGRESS'
            result=run/'BATCH512_SWITCH.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='CONFIGURED':
                record=json.loads(result.read_text())
                record['runtime']=throughput(run,record['time'])
                status[name]=record;continue
            rows=processes();pids=run_processes(dataset,seed,rows)
            checkpoints=sorted((run/'ckpt').glob('*/checkpoint.pth'),key=lambda p:int(p.parent.name))
            if marker.exists():
                record=json.loads(marker.read_text())
                if pids:
                    status[name]=dict(status='WAIT_FOR_OLD_TRAINER_EXIT',pids=sorted(pids));continue
                plan=apply_config(run,record['checkpoint_iteration'])
                write(run/'status.json',dict(status='READY_TO_RESUME_BATCH512',latest_checkpoint=record['checkpoint_iteration']))
                record.update(status='CONFIGURED',config_sha256=plan['config_sha256'],time=time.time())
                write(result,record);marker.unlink();status[name]=record;continue
            if not pids and not (run/'raw_loss_metrics.jsonl').exists():
                plan=apply_config(run,None)
                record=dict(status='CONFIGURED',checkpoint_iteration=None,seed=seed,batch=512,accumulation=1,global_batch=1024,time=time.time(),note='Not yet started; configured from initialization')
                write(result,record);status[name]=record;continue
            if not checkpoints:
                status[name]=dict(status='WAIT_FOR_FIRST_COMPLETE_CHECKPOINT',next_expected_update=4000,pids=sorted(pids),runtime=throughput(run));continue
            checkpoint=checkpoints[-1]
            if time.time()-checkpoint.stat().st_mtime<15:
                status[name]=dict(status='WAIT_FOR_CHECKPOINT_WRITE');continue
            import torch
            obj=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            iteration=int(checkpoint.parent.name)
            assert obj['iteration']==iteration and obj['model'] and obj['optimizer']
            assert obj['optimizer']['state'], 'Optimizer state is empty'
            del obj
            # Pin the pre-switch config before changing the training configuration.
            pinned=run/f'eval/union_v4/checkpoints/{iteration}/checkpoint.pth'
            pinned.parent.mkdir(parents=True,exist_ok=True)
            if not pinned.exists():os.link(checkpoint,pinned)
            identity=pinned.parent/'identity.json'
            if not identity.exists():
                (pinned.parent/'config.yaml').write_text((run/'source_config.yaml').read_text())
                write(identity,dict(iteration=iteration,bytes=checkpoint.stat().st_size,sha256=sha(checkpoint),config_sha256=sha(pinned.parent/'config.yaml')))
            record=dict(status='CHECKPOINT_VERIFIED_STOPPING',checkpoint=str(checkpoint),checkpoint_iteration=iteration,
                        checkpoint_bytes=checkpoint.stat().st_size,seed=seed,batch=512,accumulation=1,global_batch=1024,
                        stopped_pids=sorted(pids),time=time.time(),before_switch_throughput=throughput(run),note='Resume keeps model and optimizer; updates after this snapshot may be replayed')
            write(marker,record)
            # Supervisor respects this marker and leaves evaluation/other seeds running.
            stop(pids,rows)
            status[name]=record
        overall=json.loads((ROOT/'plan.json').read_text())
        overall['jobs']=[json.loads((ROOT/f'{d}_seed{s}'/'plan.json').read_text()) for d,s in JOBS]
        overall['batch512_request']='User authorized save full checkpoint, then 512x2 GPUs with accumulation1'
        write(ROOT/'plan.json',overall)
        write(ROOT/'batch512_switch_status.json',dict(updated=time.time(),runs=status))
        print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
