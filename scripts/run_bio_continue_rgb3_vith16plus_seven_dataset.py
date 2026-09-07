#!/usr/bin/env python3
"""Safe seed-0 scheduler for the new BioDINO seven-dataset comparison."""
from __future__ import annotations
import json, shlex, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = REPO / 'outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_seven_dataset'
LOGS = ROOT / 'logs'; STATUS = ROOT / 'scheduler_status.json'
PYTHON = '/home/inspur/anaconda3/envs/dinov3/bin/python'
CKPT = '/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349'
CFG = '/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/config.yaml'
DATA = {
 'monuseg':'/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted',
 'bbbc038':'/mnt/huawei_deepcad/benchmark/segmentation/bbbc038/extracted',
 'cellpose':'/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted',
 'tissuenet':'/mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted',
 'livecell':'/mnt/huawei_deepcad/benchmark/segmentation/LIVECell',
 'pannuke':'/mnt/huawei_deepcad/benchmark/segmentation/pannuke/extracted',
 'conic':'/mnt/huawei_deepcad/benchmark/segmentation/conic/extracted',
}
THRESH = {'bbbc038':('0.46','0.58'),'tissuenet':('0.57','0.4'),'livecell':('0.5','0.4'),'pannuke':('0.46','0.45'),'conic':('0.46','0.43')}
HOSTS = {'cpu8':'cpu8','cpu10':'cpu10','cpu11':'cpu11'}
lock = threading.Lock(); state = {'created_at':datetime.now(timezone.utc).isoformat(),'checkpoint':CKPT,'tasks':{}}

def write():
    ROOT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(exist_ok=True)
    t=STATUS.with_suffix('.tmp'); t.write_text(json.dumps(state,indent=2)+'\n'); t.replace(STATUS)
def ssh(host, cmd):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,cmd],capture_output=True,text=True)
def audit(host):
    r=ssh(host,"free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0); pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null|sed '/^$/d'|paste -sd, -); printf '%s|%s|%s|%s\\n' \"$free\" \"$util\" \"$uuid\" \"$pids\"")
    try:
        free,util,uuid,pids=r.stdout.strip().split('|',3); d={'host':host,'free_mib':int(free),'utilization':int(util),'uuid':uuid,'compute_pids':pids,'ssh_rc':r.returncode}; return d, r.returncode==0 and int(free)>=22000 and int(util)==0 and not pids
    except Exception: return {'host':host,'raw':r.stdout,'stderr':r.stderr,'ssh_rc':r.returncode},False
def task_cmd(ds, setting, out, gpu):
    base=[PYTHON,'-u','-m','dinov3.eval.bio_segmentation.instance_seg.train','--dataset',ds,'--data-root',DATA[ds],'--checkpoint',CKPT,'--train-config',CFG,'--output-dir',str(out),'--layers','7','15','23','31','--epochs','50','--batch-size','1','--grad-accum-steps','8','--crop-size','256','--stride','192','--lr','1e-3','--weight-decay','1e-4','--amp-dtype','bf16','--feature-size','32','--embed-proj','384','--fusion-mode','bucket_concat','--decoder-variant','current','--num-workers','0','--eval-every','10','--seed','0','--aug','strong','--mosaic-prob','0.3','--np-loss-mode','ce_dice','--fg-thresh','0.5','--energy-thresh','0.4','--skip-test-eval']
    if setting=='no_trick' or setting=='monuseg_validated': base += ['--freeze-backbone']
    else: base += ['--finetune','--backbone-lr','2e-5']
    return f'cd {REPO} && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 {shlex.join(base)}'
def post_cmd(ds, out, head, fg, energy):
    return f'cd {REPO} && {PYTHON} -u -m dinov3.eval.bio_segmentation.instance_seg.eval_full --dataset {ds} --data-root {DATA[ds]} --checkpoint {CKPT} --train-config {CFG} --head-path {head} --checkpoint-kind decoder --output {out}/results.json --layers 7 15 23 31 --feature-size 32 --embed-proj 384 --split val --crop-size 256 --stride 192 --fg-thresh {fg} --energy-thresh {energy}'
def run_task(name, host, cmd, out):
    d,ok=audit(host)
    with lock: state['tasks'][name].update(status='audited',audit=d,host=host); write()
    if not ok:
        with lock: state['tasks'][name]['status']='blocked_gpu'; write()
        return
    ep=out/'exit_code.txt'; log=LOGS/(name+'.log'); out.mkdir(parents=True,exist_ok=True)
    body=f'{cmd}; rc=$?; printf "%s\\n" "$rc" > {ep}.tmp; mv {ep}.tmp {ep}; exit $rc'
    launch=ssh(host,f'nohup bash -lc {shlex.quote(body)} >> {shlex.quote(str(log))} 2>&1 < /dev/null & echo $!')
    if launch.returncode or not launch.stdout.strip().isdigit():
        with lock: state['tasks'][name].update(status='launch_failed',stderr=launch.stderr); write(); return
    pid=int(launch.stdout.strip())
    with lock: state['tasks'][name].update(status='running',pid=pid,log=str(log),output=str(out),command=cmd,started_at=datetime.now(timezone.utc).isoformat(),gpu_uuid=d.get('uuid')); write()
    while ssh(host,f'ps -p {pid} -o pid=').stdout.strip():
        with lock: state['tasks'][name]['last_alive_at']=datetime.now(timezone.utc).isoformat(); write()
        time.sleep(30)
    rc=int(ep.read_text().strip()) if ep.exists() else 255
    with lock: state['tasks'][name].update(status='completed' if rc==0 and (out/'results.json').exists() else 'failed',exit_code=rc,finished_at=datetime.now(timezone.utc).isoformat()); write()
def worker(host, names):
    for name,ds,setting in names:
        out=ROOT/ds/setting
        if (out/'results.json').exists() and (out/'exit_code.txt').exists() and (out/'exit_code.txt').read_text().strip()=='0': continue
        # validated post-processing waits for the corresponding no-trick head; Full FT waits only on its own GPU queue.
        if setting=='validated_post':
            src=ROOT/ds/'no_trick'
            while not (src/'results.json').exists(): time.sleep(30)
            fg,en=THRESH[ds]; cmd=post_cmd(ds,out,src/'best_head.pth',fg,en)
        else: cmd=task_cmd(ds,setting,out,0)
        run_task(f'{ds}_{setting}',host,cmd,out)
def main():
    ROOT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(exist_ok=True)
    queues={'cpu8':[('monuseg','monuseg','no_trick'),('bbbc038','bbbc038','no_trick'),('bbbc038','bbbc038','validated_post'),('monuseg','monuseg','monuseg_validated')], 'cpu10':[('cellpose','cellpose','no_trick'),('cellpose','cellpose','cellpose_validated'),('tissuenet','tissuenet','no_trick'),('tissuenet','tissuenet','validated_post')], 'cpu11':[('livecell','livecell','no_trick'),('livecell','livecell','livecell_validated'),('pannuke','pannuke','no_trick'),('pannuke','pannuke','validated_post'),('conic','conic','no_trick'),('conic','conic','validated_post')]}
    for q in queues.values():
        for _,ds,s in q: state['tasks'][f'{ds}_{s}']={'status':'queued','dataset':ds,'setting':s}
    write(); ts=[threading.Thread(target=worker,args=(h,q),daemon=True) for h,q in queues.items()]
    [t.start() for t in ts]; [t.join() for t in ts]
    state['finished_at']=datetime.now(timezone.utc).isoformat(); write()
if __name__=='__main__': main()
