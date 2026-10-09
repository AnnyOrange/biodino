#!/usr/bin/env python3
"""Resume the existing Adaptive optimizer checkpoint on four A100 DDP ranks."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_resume17689_gpu3_formal'
OUTPUT = ROOT / 'outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_ddp4_20260929'


def main():
    parent = json.loads((PARENT/'launch_manifest.json').read_text())
    source = Path(parent['source'])
    for name in ('dinov3/train/train.py','dinov3/checkpointer/checkpointer.py'):
        assert (source/name).is_file()
    checkpoints = sorted((int(p.name),p/'checkpoint.pth') for p in (PARENT/'ckpt').iterdir()
                         if p.name.isdigit() and (p/'checkpoint.pth').stat().st_size > 3_000_000_000)
    step, original = checkpoints[-1]
    if step < 18543:
        raise RuntimeError('Latest Adaptive checkpoint is unexpectedly old')
    target = OUTPUT/'ckpt'/str(step)/'checkpoint.pth'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise RuntimeError('Four-rank continuation already prepared')
    os.link(original,target)
    cmd=parent['command'][:]
    cmd[cmd.index('--nproc_per_node=1')]='--nproc_per_node=4'
    cmd[cmd.index('--master_port=32933')]='--master_port=32944'
    cmd[cmd.index('--output-dir')+1]=str(OUTPUT)
    for option,value in [('optim.gradient_accumulation_steps','2'),('train.num_workers','2'),
                         ('train.max_updates','29280')]:
        matches=[i for i,arg in enumerate(cmd) if arg.startswith(option+'=')]
        assert len(matches)==1, option
        cmd[matches[0]]=option+'='+value
    assert 'compute_precision.distributed_mode=ddp' in cmd
    assert 'train.batch_size_per_gpu=128' in cmd
    assert 'optim.gradient_accumulation_steps=2' in cmd
    assert 128*4*2==1024
    env=os.environ.copy()
    env.update(parent['environment'],CUDA_VISIBLE_DEVICES='0,3,4,5',
               PYTHONPATH=str(source),OMP_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_NET='Socket',
               NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0')
    manifest=dict(time_utc=datetime.now(timezone.utc).isoformat(),parent=str(PARENT),
                  parent_checkpoint=str(original),resumed_step=step,
                  checkpoint_is_hardlink=original.stat().st_ino==target.stat().st_ino,
                  source=str(source),command=cmd,cuda_visible_devices='0,3,4,5',
                  effective_global_batch=1024,accumulation=2,world_size=4,
                  target_update=29279,note='Consolidated checkpoint restores model and optimizer; no fresh optimizer restart.')
    (OUTPUT/'launch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (OUTPUT/'launch.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUTPUT/'launcher_status.json').write_text(json.dumps(dict(state='STARTED',pid=process.pid,
                                        time_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    print(json.dumps(dict(pid=process.pid,resumed_step=step,output=str(OUTPUT)),indent=2))


if __name__=='__main__':
    main()
