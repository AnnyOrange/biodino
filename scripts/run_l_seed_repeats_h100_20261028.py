#!/usr/bin/env python3
"""H100 L repeats: independent training and capacity-limited evaluation on GPU 0/1."""
import argparse
import collections
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/home/suxin/project/dinov3')
ROOT = Path('/data_2/suxin/runs/L_seed_repeats_20261028')
PYTHON = '/data_2/suxin/envs/dinov3/bin/python'
SCRIPT = REPO / 'scripts/run_l_seed_repeats_h100_20261028.py'
OLD = {'1tb': Path('/data_2/suxin/runs/HS6_L_1TB_biosafe256_15374_to80000_2xH100_20261020'),
       '5tb': Path('/data_2/suxin/runs/HS6_L_5TB_nogram_41479_to80000_4xH100_20261026')}
JOBS = [('1tb', 1, '2,3'), ('1tb', 2, '2,3'), ('5tb', 1, '4,5'), ('5tb', 2, '6,7')]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    import yaml
    ROOT.mkdir(parents=True, exist_ok=True)
    init = Path('/data_2/suxin/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    assert init.is_file()
    training_source = (REPO/'dinov3/train/train.py').read_text()
    periodic = 'if (iteration + 1) % cfg.checkpointing.period == 0:'
    assert training_source.count(periodic) == 1
    training_source = training_source.replace(periodic, 'if (iteration + 1) % cfg.checkpointing.period == 0 or iteration + 1 == max_iter:')
    entry = ROOT/'train_entry.py'
    if entry.exists() and entry.read_text() != training_source:
        raise ValueError('Frozen training entry differs')
    entry.write_text(training_source)
    source = REPO / 'scripts/run_1tb_vitl16_union_v4_h100_20261019.py'
    frozen = ROOT / 'eval_snapshot.py'
    if not frozen.exists():
        shutil.copy2(source, frozen)
    datasets = {
        '1tb': 'packwds_robust:/data_2/suxin/train-dataset/microscopy-100k-patched/filtered_mixed_train_w*-*.tar::pct=1,99',
        '5tb': 'mixwds_robust:0.3=/data_2/suxin/train-dataset/microscopy-100k-patched/filtered_mixed_train_w*.tar||0.7=/data_2/suxin/train-dataset/filtered_mixed_train*.tar::pct=1,99'}
    pools = {}
    for tag, folder, pattern in [('1tb', '/data_2/suxin/train-dataset/microscopy-100k-patched', 'filtered_mixed_train_w*.tar'), ('5tb', '/data_2/suxin/train-dataset', 'filtered_mixed_train*.tar')]:
        pools[tag] = [dict(path=str(p), bytes=p.stat().st_size) for p in sorted(Path(folder).glob(pattern))]
        assert len(pools[tag]) == {'1tb': 326, '5tb': 1251}[tag]
    write(ROOT / 'data_inventory.json', pools)
    plans = []
    for dataset, seed, gpus in JOBS:
        run = ROOT / f'{dataset}_seed{seed}'
        run.mkdir(exist_ok=True)
        cfg = yaml.safe_load((OLD[dataset] / 'source_config.yaml').read_text())
        updates = 15 * cfg['train']['OFFICIAL_EPOCH_LENGTH']
        batch = 128 if dataset == '1tb' else 64
        cfg['compute_precision']['distributed_mode'] = 'ddp'
        cfg['student']['resume_from_teacher_chkpt'] = str(init)
        cfg['train'].update(output_dir=str(run), dataset_path=datasets[dataset], seed=seed,
                            batch_size_per_gpu=batch, max_updates=updates,
                            checkpointing=True, checkpointing_full=True,
                            wds_deterministic_resampling=True)
        cfg['optim']['gradient_accumulation_steps'] = 1024 // (2 * batch)
        cfg['evaluation']['eval_period_iterations'] = 0
        cfg['checkpointing'].update(period=4000, max_to_keep=None, sharded=False)
        config_path = run / 'source_config.yaml'
        text = yaml.safe_dump(cfg, sort_keys=False)
        if config_path.exists() and config_path.read_text() != text:
            raise ValueError(f'Immutable config differs: {config_path}')
        config_path.write_text(text)
        plan = dict(dataset=dataset, seed=seed, gpus=gpus, target_updates=updates,
                    global_batch=1024, microbatch=batch,
                    accumulation=cfg['optim']['gradient_accumulation_steps'],
                    config_sha256=sha(config_path), initialization=str(init),
                    initialization_sha256=sha(init),
                    checkpoints=[*range(3999, updates, 4000), updates-1],
                    independent_from_initial_weights=True,
                    statistical_note='Old seed0 is not certified matched: historical microbatch/distribution and uncontrolled resampling differ. Do not pool automatically.')
        write(run / 'plan.json', plan)
        plans.append(plan)
    write(ROOT / 'plan.json', dict(jobs=plans, evaluation_gpus=[0, 1],
          eval_snapshot_sha256=sha(frozen), training_entry_sha256=sha(entry), evaluation_interval_updates=4000,
          original_epoch_budget=15, continuation_tail_not_included=True,
          initial_parallel_jobs=['1tb_seed1', '5tb_seed1', '5tb_seed2'],
          sequential_followup='1tb_seed2 on GPU2,3 after seed1'))
    print(json.dumps(plans, indent=2))


def train(dataset, seed):
    run = ROOT / f'{dataset}_seed{seed}'
    plan = json.loads((run / 'plan.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpus'], PYTHONPATH=str(REPO),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    cmd = [PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=2',
           f'--master_port={29700 + (10 if dataset == "5tb" else 0) + seed}',
           str(ROOT/'train_entry.py'), '--config-file', str(run / 'source_config.yaml'),
           '--output-dir', str(run), '--seed', str(seed), f'train.seed={seed}']
    run_lock = (run / 'train.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failures = 0
    while failures < 3:
        checkpoints = [int(p.parent.name) for p in (run / 'ckpt').glob('*/checkpoint.pth')]
        latest = max(checkpoints, default=-1)
        if latest >= plan['target_updates'] - 1:
            write(run / 'status.json', dict(status='COMPLETE', latest_checkpoint=latest))
            return 0
        if sha(ROOT/'train_entry.py') != json.loads((ROOT/'plan.json').read_text())['training_entry_sha256']:
            raise ValueError('Frozen training entry changed')
        if sha(run / 'source_config.yaml') != plan['config_sha256']:
            raise ValueError('Training config changed')
        write(run / 'status.json', dict(status='RUNNING', command=cmd, started=time.time(), latest_checkpoint=latest))
        with (run / 'console.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        if code == 0 and (run / f'ckpt/{plan["target_updates"]-1}/checkpoint.pth').is_file():
            continue
        failures += 1
        write(run / 'status.json', dict(status='RETRY_WAIT' if failures < 3 else 'FAILED', returncode=code, failures=failures))
        if failures < 3:
            time.sleep(30)
    return 1


def module(run, profile, new=False):
    os.environ['DINOV3_1TB_EVAL_PROFILE'] = profile
    spec = importlib.util.spec_from_file_location('eval_' + run.name, ROOT / 'eval_snapshot.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.RUN = m.SOURCE = run
    m.ROOT = run / 'eval/union_v4'
    m.MULTIMODAL_V2_ROOT = m.ROOT / m.MULTIMODAL_V2_PROTOCOL
    if new:
        plan = json.loads((run / 'plan.json').read_text())
        m.CHECKPOINTS = tuple(plan['checkpoints'])
        m.MODEL_TAG = 'hs6_l_' + run.name
        manifest_path = m.ROOT / 'manifest.json'
        if not manifest_path.exists():
            tasks = m.inventory(json.loads(m.PROTOCOL.read_text()))
            for task in tasks:
                if task['dataset'] == 'multimodal_cellseg':
                    task['blocker'] = 'Use corrected Multimodal-v2 runner'
            write(manifest_path, dict(protocol_id='bio-eval-union-v4', protocol_sha256=sha(m.PROTOCOL),
                  tasks=tasks, checkpoints=plan['checkpoints'], full_checkpoints=plan['checkpoints'],
                  seed=0, frozen_batch=64, dense_feature_batch=32, dense_probe_batch=32,
                  detection_batch=8, autocast_dtype='bf16',
                  evaluation_scope='union_v4_with_explicit_blockers_and_corrected_multimodal',
                  backbone_training_seed=plan['seed'], parallel_slots={'0':5,'1':5}))
        if not (m.MULTIMODAL_V2_ROOT / 'manifest.json').exists():
            m.prepare_multimodal_v2()
    from scripts.run_lc25000_h100_grouped_20261029 import install
    install(m)
    return m


def modules():
    result = [module(ROOT / f'{d}_seed{s}', 'biosafe256', True) for d,s,_ in JOBS]
    for d, profile in [('1tb','biosafe256'), ('5tb','5tb_nogram')]:
        result.append(module(OLD[d], profile))
    legacy = Path('/data_2/suxin/runs/HS6_L_1TB_40000_to80000_2xH100_DDP_20261019')
    if (legacy / 'eval/union_v4/manifest.json').exists():
        result.append(module(legacy, 'legacy'))
    return result


def execute(run, profile, new, gpu, iteration, taskid):
    m = module(Path(run), profile, new)
    manifest = json.loads((m.ROOT / 'manifest.json').read_text())
    if sha(m.PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Locked protocol changed')
    task = next(t for t in manifest['tasks'] if t['id'] == taskid)
    ckpt = m.ROOT / f'checkpoints/{iteration}/checkpoint.pth'
    if task['dataset'] == 'multimodal_cellseg':
        state = m.MULTIMODAL_V2_ROOT / f'states/ck{iteration}.json'
        mm = json.loads((m.MULTIMODAL_V2_ROOT / 'manifest.json').read_text())
        m.run_multimodal_v2(gpu, iteration, ckpt, state, mm)
        m.promote_multimodal_v2(iteration)
    else:
        state = m.ROOT / f'states/ck{iteration}/{taskid}.json'
        m.evaluate_task(gpu, task, iteration, manifest, ckpt, state)


def gpu_info():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,uuid','--format=csv,noheader,nounits'], text=True)
    return {int(a):dict(used=int(b), total=int(c), uuid=d.strip()) for a,b,c,d in (line.split(',') for line in rows.splitlines())}


def process_info():
    result = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p/'cmdline').read_bytes().decode().replace('\0',' ')
            parent = int((p/'stat').read_text().split(') ')[1].split()[1])
            result[int(p.name)] = (parent, args)
        except (OSError, ValueError, UnicodeDecodeError):
            pass
    return result


def evaluation_controller():
    controller_lock = (ROOT/'evaluation.lock').open('a')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ms = modules()
    active = {}
    seen = {}
    retry_path = ROOT/'evaluation_attempts.json'
    retry = {tuple(v['key']):v['attempts'] for v in json.loads(retry_path.read_text())} if retry_path.exists() else {}
    report_child = None
    report_log = None
    last_report = 0
    while True:
        if (ROOT/'LC_CORRECTION_MIGRATING').exists():
            time.sleep(10)
            continue
        for key, entry in list(active.items()):
            if entry['child'].poll() is not None:
                entry['log'].close()
                active.pop(key)
        procs = process_info()
        gpu = gpu_info()
        used_by_pid = {}
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            a,b = row.split(',')
            if b.strip().isdigit():
                used_by_pid[int(a)] = int(b)
        reserved = {0:0,1:0}
        count = collections.Counter()
        for entry in active.values():
            descendants = {entry['child'].pid}
            while True:
                more = {pid for pid,(par,_) in procs.items() if par in descendants} - descendants
                if not more:break
                descendants |= more
            actual = sum(used_by_pid.get(pid,0) for pid in descendants)
            reserved[entry['gpu']] += max(0, entry['peak']-actual)
            count[entry['gpu']] += 1
        external = []
        for pid,(parent,args) in procs.items():
            if not (' -m dinov3.eval.' in args or 'run_lc25000_h100_grouped_20261029.py' in args) or (' -m dinov3.eval.' in procs.get(parent,(0,''))[1] or 'run_lc25000_h100_grouped_20261029.py' in procs.get(parent,(0,''))[1]):
                continue
            # Evaluations surviving the old queue shutdown keep their peak reservation.
            ancestor = parent
            own = False
            while ancestor in procs:
                if any(e['child'].pid == ancestor for e in active.values()):
                    own = True
                    break
                ancestor = procs[ancestor][0]
            if own:continue
            try:
                env = dict(item.split('=',1) for item in (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\0') if '=' in item)
                g = int(env.get('CUDA_VISIBLE_DEVICES','-1'))
            except (OSError, ValueError):continue
            if g not in (0,1):continue
            descendants={pid}
            while True:
                more={p for p,(par,_) in procs.items() if par in descendants}-descendants
                if not more:break
                descendants |= more
            actual=sum(used_by_pid.get(p,0) for p in descendants)
            peak=16384 if 'segmentation' in args else 8192
            reserved[g]+=max(0,peak-actual)
            count[g]+=1
            external.append(dict(pid=pid,gpu=g,peak=peak,actual=actual))
        candidates = []
        counts = collections.Counter()
        for m in ms:
            new = m.RUN.parent == ROOT
            manifest = json.loads((m.ROOT / 'manifest.json').read_text())
            if sha(m.PROTOCOL) != manifest['protocol_sha256']:
                raise ValueError(f'Protocol changed for {m.RUN}')
            for step in manifest['checkpoints']:
                ckpt = m.ROOT / f'checkpoints/{step}/checkpoint.pth'
                identity = ckpt.parent/'identity.json'
                if not identity.exists() and new:
                    src = m.RUN/f'ckpt/{step}/checkpoint.pth'
                    if src.exists():
                        stat = src.stat()
                        signature = (stat.st_size, stat.st_mtime_ns)
                        if seen.get(str(src)) == signature and stat.st_size > 3_000_000_000:
                            import torch
                            obj = torch.load(src, map_location='cpu', mmap=True, weights_only=False)
                            assert obj['iteration'] == step and obj['optimizer']
                            del obj
                            ckpt.parent.mkdir(parents=True,exist_ok=True)
                            if not ckpt.exists():os.link(src,ckpt)
                            shutil.copy2(m.RUN/'source_config.yaml',ckpt.parent/'config.yaml')
                            write(identity,dict(iteration=step,bytes=stat.st_size,sha256=sha(ckpt),config_sha256=sha(ckpt.parent/'config.yaml')))
                        seen[str(src)] = signature
                if not identity.exists():continue
                for task in manifest['tasks']:
                    corrected = task['dataset']=='multimodal_cellseg'
                    state_path = (m.MULTIMODAL_V2_ROOT/f'states/ck{step}.json' if corrected else m.ROOT/f'states/ck{step}/{task["id"]}.json')
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    if task['id']=='classification__lc25000' and state.get('status')=='VALID_COMPLETE' and state.get('metrics',{}).get('protocol')!='lc25000-clean-original-tile-80-20-seed0-v1':
                        state=dict(status='PENDING_PROTOCOL_CORRECTION',reason='Legacy random split is not the pinned original-tile split')
                        write(state_path,state)
                    counts[state.get('status','PENDING')] += 1
                    key = (str(m.RUN),step,task['id'])
                    if key in active or state.get('status')=='VALID_COMPLETE':continue
                    if task.get('blocker') and not corrected:
                        write(state_path,dict(status='BLOCKED_NOT_TESTED',reason=task['blocker']))
                        continue
                    if corrected and not (m.MULTIMODAL_V2_ROOT/'manifest.json').exists():continue
                    output = str(m.ROOT/f'ck{step}/{task["id"]}')
                    if corrected:output=str(m.MULTIMODAL_V2_ROOT/f'ck{step}')
                    if any(output in args for _,args in procs.values()):continue
                    attempts = retry.get(key,0)
                    if attempts>=3:continue
                    priority = 0 if new else 1 if task['id']=='classification__lc25000' else 2
                    candidates.append((priority,step,m,task,key))
        candidates.sort(key=lambda v:(v[0],v[1]))
        available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
        total_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))
        for _,step,m,task,key in candidates:
            peak = 16384 if task['family'] in ('segmentation','detection_proxy') else 8192
            if available_kib/1024 - (len(active)+1)*32768 < max(total_kib/1024*.2,8192):break
            choices = [g for g in (0,1) if count[g]<8 and gpu[g]['used']+reserved[g]+peak <= gpu[g]['total']*.8]
            if not choices:continue
            g = min(choices,key=lambda v:gpu[v]['used']+reserved[v])
            # Admission includes real occupancy from all users and reservations until child exits.
            log = (m.ROOT/'seed_dispatch.log').open('a')
            cmd = [PYTHON,'-u',str(SCRIPT),'eval-task','--run',str(m.RUN),'--profile',m.PROFILE,
                   '--gpu',str(g),'--step',str(step),'--task',task['id']]
            if m.RUN.parent == ROOT:cmd.append('--new')
            child = subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(REPO)))
            active[key]=dict(child=child,log=log,gpu=g,peak=peak)
            retry[key]=retry.get(key,0)+1
            write(retry_path,[dict(key=list(k),attempts=v) for k,v in retry.items()])
            reserved[g]+=peak
            count[g]+=1
        write(ROOT/'evaluation_status.json',dict(updated=time.time(),gpu=gpu,
              admitted_per_gpu=dict(count),reserved_mib=reserved,states=dict(counts),external_evaluations=external,
              active=[dict(run=k[0],step=k[1],task=k[2],gpu=e['gpu'],pid=e['child'].pid) for k,e in active.items()],
              floor_limit_reason='GPU0/1 shared with other users; actual occupancy plus task peak reserves must remain <=80%. No training GPUs used.'))
        if report_child is not None and report_child.poll() is not None:
            report_log.close()
            report_child = None
        if report_child is None and time.time()-last_report >= 120:
            report_log = (ROOT/'report.log').open('a')
            report_child = subprocess.Popen([PYTHON,'-u',str(SCRIPT),'report'],cwd=REPO,stdout=report_log,stderr=subprocess.STDOUT)
            last_report = time.time()
        time.sleep(15)


def report():
    progress = {}
    rows = []
    for m in modules():
        manifest = json.loads((m.ROOT/'manifest.json').read_text())
        if m.RUN.parent != ROOT:
            try:
                m.refresh_curves(manifest)
            except Exception as error:
                write(ROOT/'report_errors'/f'{m.RUN.name}.json',dict(error=repr(error),time=time.time()))
        if m.RUN.parent == ROOT:
            path = m.RUN/'raw_loss_metrics.jsonl'
            latest = {}
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-65536))
                    lines = stream.read().splitlines()
                for line in reversed(lines):
                    try:
                        latest = json.loads(line)
                        break
                    except ValueError:pass
            progress[m.RUN.name] = {k:latest.get(k) for k in ('optimizer_updates_completed','total_loss','effective_global_batch_size','batch_sample_key_digest','time_unix')}
            try:
                m.refresh_curves(manifest)
            except Exception as error:
                progress[m.RUN.name]['curve_error']=repr(error)
        for step in manifest['checkpoints']:
            states = []
            for task in manifest['tasks']:
                path = m.ROOT/f'states/ck{step}/{task["id"]}.json'
                state = json.loads(path.read_text()).get('status','PENDING') if path.exists() else 'PENDING'
                states.append(state)
            counts = collections.Counter(states)
            rows.append(dict(run=m.RUN.name,checkpoint=step,updates=step+1,
                exported=(m.ROOT/f'checkpoints/{step}/identity.json').exists(),
                valid_complete=counts['VALID_COMPLETE'],expected=len(states),
                blocked=counts['BLOCKED_NOT_TESTED'],failed=counts['FAILED'],
                fully_complete=counts['VALID_COMPLETE']==len(states)))
    write(ROOT/'training_progress.json',dict(updated=time.time(),runs=progress))
    temp=ROOT/'coverage.csv.tmp'
    with temp.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    temp.replace(ROOT/'coverage.csv')


def supervise():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        children={}
        failures=collections.Counter()
        while True:
            desired=[('eval', ['eval-controller'])]
            for dataset,seed,gpus in JOBS:
                run=ROOT/f'{dataset}_seed{seed}'
                if (run/'BATCH_SWITCH_IN_PROGRESS').exists():continue
                if dataset=='1tb' and seed==2:
                    first=ROOT/'1tb_seed1/status.json'
                    if not first.exists() or json.loads(first.read_text()).get('status')!='COMPLETE':continue
                status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
                if status.get('status') not in ('COMPLETE','FAILED'):
                    desired.append((run.name,['train','--dataset',dataset,'--seed',str(seed)]))
            existing = process_info()
            adopted = {}
            for name,args in desired:
                if name not in children:
                    matching = [pid for pid,(_,cmd) in existing.items() if f'{SCRIPT} '+ ' '.join(args) + ' ' in cmd]
                    if matching:
                        adopted[name] = matching
                        continue
                if name in children and children[name][0].poll() is None:continue
                if name in children:
                    children[name][1].close()
                    failures[name]+=1
                    if failures[name]>=3:continue
                log=(ROOT/f'{name}_supervisor.log').open('a')
                child=subprocess.Popen([PYTHON,'-u',str(SCRIPT),*args],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                children[name]=(child,log)
            write(ROOT/'supervisor_status.json',dict(updated=time.time(),children={k:dict(pid=v[0].pid,returncode=v[0].poll()) for k,v in children.items()},failures=dict(failures),adopted=adopted))
            time.sleep(20)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','supervise','train','eval-controller','eval-task','report'])
    p.add_argument('--dataset');p.add_argument('--seed',type=int)
    p.add_argument('--run');p.add_argument('--profile');p.add_argument('--gpu',type=int)
    p.add_argument('--step',type=int);p.add_argument('--task');p.add_argument('--new',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='supervise':supervise()
    elif a.mode=='train':sys.exit(train(a.dataset,a.seed))
    elif a.mode=='eval-controller':evaluation_controller()
    elif a.mode=='report':report()
    else:execute(a.run,a.profile,a.new,a.gpu,a.step,a.task)

if __name__=='__main__':main()
