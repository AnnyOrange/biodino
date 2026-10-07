#!/usr/bin/env python3
"""Continue 20TB paired evaluation with adaptive slots/GPU and measured memory admission.

The worker lifecycle is derived from the frozen 20260918 queue. Evaluation commands,
validation, source fingerprints, data splits, batches and numerical checks are reused
unchanged. Scheduling permits up to twelve jobs and targets 75% VRAM, with a 60-second
reservation for loading processes and 2 GiB beyond the evaluator's peak reserve. CPU phases may use
less VRAM; worker status explicitly records GPUs below 70%.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918')
OUTPUT = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_retest_fleet_20260918 as fleet
from run_shared_frozen_stage1_20260917 import (
    THREADS, verify_campaign_source, save, resources, project_tests,
    checkpoint_record, verify_source, sha256,
)
command, validate_cell = fleet.command, fleet.validate
RUNS = {
    'noGRAM20tb': REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924',
    'cls_slow2_20tb': REPO / 'outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork36111_20261004/cls_slow2',
}


def assets():
    result = []
    for arm, root in RUNS.items():
        for path in sorted((root / 'eval').glob('training_*/teacher_checkpoint.pth')):
            ck = int(path.parent.name.split('_')[-1])
            if ck <= 36111 or (ck + 1) % 488 or ((ck + 1) // 488) % 4 != 2:
                continue
            stat = path.stat()
            if stat.st_size < 1_000_000_000 or time.time() - stat.st_mtime < 180:
                continue
            result.append(dict(arm=arm, checkpoint_id=str(ck), path=str(path),
                               config=str(root / 'config.yaml'), kind='dinov3', model_id='', reserve_mib=4096))
    return sorted(result, key=lambda a: (int(a['checkpoint_id']), a['arm']))


def prepare(args):
    if (args.output / 'campaign_manifest.json').exists():
        raise FileExistsError('Campaign already prepared')
    ref = json.loads((REPO / 'outputs/02_eval_inputs/shared_fleet_reference_20260930/raw_campaign_manifest.json').read_text())
    fleet.queue.verify_campaign_source(ref)
    source = json.loads((ROOT / 'source_snapshot.json').read_text())
    if source != ref['source_snapshot']:
        raise ValueError('Reference evaluator fingerprint mismatch')
    manifest = {k: ref[k] for k in ('datasets', 'benchmark_root', 'numerical_environment')}
    protocol = REPO / 'Evaluation Rules/protocol_v4.json'
    admitted = assets()
    manifest.update(protocol_id='bio-eval-union-v4', campaign_scope='V4_SHARED_V3_COMPONENTS_ONLY',
        v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
        authorization='2026-10-06 user: continue development; use lyx, deepcad, qi and local; test VRAM above 70%, multiple tests per GPU',
        git_commit=source['git_commit'], source_snapshot=source, source_snapshot_path=str(ROOT),
        external_source_hashes={str(Path(__file__).resolve()): sha256(__file__), str(protocol): sha256(protocol)},
        checkpoint_assets=admitted, training_roots={k:str(v) for k,v in RUNS.items()},
        tasks=fleet.tasks_for(admitted, ref['datasets']), protocol_sha256=sha256(protocol),
        batch_size=64, seed=0, n_last_blocks=1, use_avgpool=True, autocast_dtype='bf16', num_workers=2,
        online_checkpoints=True, isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True,
        checkpoint_selection_rule='ck > 36111; (ck+1)%488 == 0; ((ck+1)//488)%4 == 2',
        scheduling=dict(target_per_gpu=5, memory_target=.75, minimum_target=.70, settle_seconds=60, extra_reserve_mib=2048),
        deepcad_authorized_gpu_indices=list(range(8)), deepcad_authorized_max_gpus=8,
        created_unix=time.time())
    save(args.output / 'campaign_manifest.json', manifest)
    (args.output / '_state/inputs').mkdir(parents=True, exist_ok=True)
    print('PREPARED', len(admitted), 'checkpoints', len(manifest['tasks']), 'tasks', flush=True)


def watch(args):
    state = args.output / '_state'
    with (state / 'online_watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with (state / 'manifest.lock').open('a') as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                manifest = json.loads((args.output / 'campaign_manifest.json').read_text())
                known = {a['path'] for a in manifest['checkpoint_assets']}
                new = [a for a in assets() if a['path'] not in known]
                if new:
                    manifest['checkpoint_assets'] += new
                    manifest['tasks'] += fleet.tasks_for(new, manifest['datasets'])
                    save(args.output / 'campaign_manifest.json', manifest)
                    print('ADMITTED', [(a['arm'], a['checkpoint_id']) for a in new], flush=True)
            time.sleep(60)


def worker(args):
    os.environ.update(THREADS)
    manifest = json.loads((args.output/'campaign_manifest.json').read_text())
    family=getattr(args,'task_family','mixed')
    if family!='mixed':
        manifest['tasks']=[task for task in manifest['tasks'] if
            (task['dataset']['task']=='segmentation')==(family=='segmentation')]
    components = getattr(args, 'components', None)
    if components:
        manifest['tasks'] = [task for task in manifest['tasks'] if task['dataset'].get('component') in components]
    verify_campaign_source(manifest)
    commit = manifest['git_commit']
    import torch
    import sklearn
    import importlib.util
    expected_environment = manifest.get('numerical_environment')
    if expected_environment:
        import importlib.metadata
        actual_environment = {name: importlib.metadata.version(name) for name in expected_environment}
        if actual_environment != expected_environment:
            raise RuntimeError('Numerical dependency fingerprint does not match: ' + str(actual_environment))
        if importlib.metadata.version('torch').split('+')[0] != torch.__version__.split('+')[0]:
            raise RuntimeError('Torch runtime/metadata mismatch; repair isolated environment before testing')
    has_pyarrow = importlib.util.find_spec('pyarrow') is not None
    has_external = all(importlib.util.find_spec(name) is not None for name in ('transformers','timm','safetensors'))
    optional_modules = {'bioclip':('open_clip',), 'conch':('einops',),
                        'cytoself':('h5py',), 'cytoimagenet':('keras','h5py')}
    if int(torch.__version__.split('.')[0])<2 or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Modern CUDA BF16 evaluator environment required')
    if args.host == 'deepcad':
        allowed = manifest.get('deepcad_authorized_gpu_indices', list(range(4)))
        if len(allowed) > manifest.get('deepcad_authorized_max_gpus', 4) or not set(args.gpus) <= set(allowed):
            raise RuntimeError('Deepcad placement exceeds recorded authorization')
    state = args.output/'_state'; state.mkdir(exist_ok=True)
    for name in ('claims','done','workers','running','failed_resource'): (state/name).mkdir(exist_ok=True)
    host_lock = (state/'workers'/f'{args.host}.lock').open('a')
    fcntl.flock(host_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (state/'manifest.lock').open('a') as registration:
        fcntl.flock(registration,fcntl.LOCK_EX)
        registered = json.loads((args.output/'campaign_manifest.json').read_text())
        registered.setdefault('execution_hosts',{})[args.host] = dict(
            hostname=os.uname().nodename,pid=os.getpid(),gpu_indices=args.gpus,git_commit=commit,git_status_porcelain='',
            python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
            verified_input_manifests='_state/inputs/*.json',job_manifests='cells/*/invocation_manifest.json')
        save(args.output/'campaign_manifest.json',registered)
    status_path = state/'workers'/f'{args.host}.json'
    stopping = False
    def stop(signum,frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    active = {}
    deferred_until = {}
    last_snapshot = 0
    while not stopping:
        if manifest.get('online_checkpoints'):
            updated = json.loads((args.output/'campaign_manifest.json').read_text())
            if updated['source_snapshot'] != manifest['source_snapshot']:
                raise RuntimeError('Cannot change source of an online campaign')
            manifest['tasks'] = updated['tasks']
            if family != 'mixed':
                manifest['tasks'] = [task for task in manifest['tasks'] if
                    (task['dataset']['task'] == 'segmentation') == (family == 'segmentation')]
        for key,(process,log,task,claim,invocation) in list(active.items()):
            code = process.poll()
            if code is None: continue
            log.close()
            directory = args.output/'cells'/key
            try:
                if code: raise RuntimeError(f'Evaluator exit {code}')
                report = validate_cell(task,directory,invocation)
                save(state/'done'/f'{key}.json',report)
            except Exception as error:
                resource_failure=bool(code and task['asset'].get('kind')=='external' and
                    any(text in (directory/'run.log').read_text(errors='replace')[-16000:] for text in ('OutOfMemoryError','CUDA out of memory')))
                failure=dict(status='INVALID_PROTOCOL' if code==0 else 'FAILED',error=str(error),
                             failure_kind='FAILED_RESOURCE' if resource_failure else 'PROTOCOL_OR_RUNTIME',
                             task=key,host=args.host,gpu=invocation['gpu'],time=time.time())
                save(directory/'validation_report.json',failure)
                if resource_failure:
                    save(state/'failed_resource'/f'{key}.json',failure)
                elif code and manifest.get('isolate_runtime_failures'):
                    save(state/'failed_resource'/f'{key}.json',failure)
                else:save(state/'PAUSED.json',failure)
            del active[key]
            (state/'running'/f'{key}.json').unlink(missing_ok=True)
        cards = resources(); counts = project_tests()
        ram = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
        snapshot = dict(host=args.host,pid=os.getpid(),git_commit=commit,git_status_porcelain='',
                        python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
                        gpu_memory_mib=cards,actual_test_counts=counts,own_active=list(active),ram_available_gib=ram,
                        memory_target=args.memory_target,under_target_gpus=[g for g in args.gpus if cards[g][0]/cards[g][1] < .70],
                        updated_unix=time.time(),status='PAUSED' if (state/'PAUSED.json').exists() else 'RUNNING')
        save(status_path,snapshot)
        if time.time()-last_snapshot>=1800:
            save(state/'workers'/f'{args.host}.resources.{int(time.time())}.json',snapshot)
            last_snapshot = time.time()
        if (state/'PAUSED.json').exists():
            if not active: break
            time.sleep(5); continue
        if ram<32 or __import__('shutil').disk_usage(args.output).free<64*1024**3:
            time.sleep(10); continue
        launched = False
        for gpu in sorted(args.gpus,key=lambda g:(counts.get(str(g),0),cards[g][0]/cards[g][1],g)):
            used,total = cards[gpu]
            own = sum(v[4]['gpu']==gpu for v in active.values())
            actual = max(counts.get(str(gpu),0),own)
            unmeasured = any(v[4]['gpu']==gpu and time.time()-v[4]['started_unix']<60 for v in active.values())
            if actual >= args.target_per_gpu or len(active) >= args.max_host_jobs or used / total >= args.memory_target: continue
            for task in manifest['tasks']:
                key = task['key']
                if deferred_until.get(task['asset']['path'], 0) > time.time(): continue
                if task['dataset'].get('requires_pyarrow') and not has_pyarrow: continue
                if task['asset'].get('kind') == 'external' and not has_external: continue
                if (state/'done'/f'{key}.json').exists() or (state/'claims'/key).exists(): continue
                guard = getattr(args, 'admission_guard', None)
                if guard is not None and not guard(gpu, actual, task): continue
                if task['asset'].get('kind')=='external' and any(importlib.util.find_spec(module) is None
                        for module in optional_modules.get(task['asset']['model_id'],())):continue
                dense=task['dataset']['task']=='segmentation'
                if dense and sum(v[4]["gpu"] == gpu and v[2]["dataset"]["task"] == "segmentation" for v in active.values()) >= 2: continue
                if task['dataset'].get('requires_modules') and any(importlib.util.find_spec(module) is None
                        for module in task['dataset']['requires_modules']):continue
                reserve = int(task['dataset'].get('reserve_mib') or max(18432 if task['dataset']['image_size']>=384 else 4096,
                              int(task['asset'].get('reserve_mib') or 0)))
                pending = sum(int(v[4].get('reserve_mib',0)) for v in active.values()
                              if v[4]['gpu']==gpu and time.time()-v[4]['started_unix']<60)
                if total-used-pending < reserve + 2048: continue
                claim = state/'claims'/key
                with Path(getattr(args, 'shared_admission_lock', state/'admission.lock')).open('a') as admission:
                    fcntl.flock(admission,fcntl.LOCK_EX)
                    if guard is not None and not guard(gpu, actual, task): continue
                    if len(list((state/'running').glob('*.json'))) >= args.max_global_jobs: break
                    try: claim.mkdir()
                    except FileExistsError: continue
                    save(state/'running'/f'{key}.json',dict(host=args.host,gpu=gpu,time=time.time()))
                save(claim/'owner.json',dict(host=args.host,pid=os.getpid(),gpu=gpu,time=time.time()))
                try:
                    inputs = checkpoint_record(args.output,task['asset'])
                    for path,digest in task['dataset']['source_hashes'].items():
                        verify_source(state,path,digest)
                    directory = args.output/'cells'/key; directory.mkdir(parents=True,exist_ok=True)
                    cmd = command(task,directory,manifest['benchmark_root'])
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),**THREADS)
                    invocation = dict(git_commit=commit,git_status_porcelain='',host=args.host,gpu=gpu,
                        python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
                        command=cmd,environment={key:env[key] for key in ('CUDA_VISIBLE_DEVICES',*THREADS)},
                        checkpoint=inputs,dataset=task['dataset'],batch_size=task['dataset'].get('feature_batch_size',64),autocast_dtype='bf16',
                        n_last_blocks=1,use_avgpool=True,seed=0,num_workers=2,features_persisted=False,started_unix=time.time())
                    invocation['reserve_mib'] = reserve
                    invocation['asset_kind'] = task['asset'].get('kind','dinov3')
                    invocation['external_source_hashes'] = manifest.get('external_source_hashes',{})
                    save(directory/'invocation_manifest.json',invocation)
                    log = (directory/'run.log').open('w')
                    log.write('COMMAND '+__import__('shlex').join(cmd)+'\n'); log.flush()
                    invocation['source_snapshot_sha256'] = manifest.get('source_snapshot', {}).get('sha256')
                    invocation['numerical_environment'] = expected_environment
                    save(directory/'invocation_manifest.json',invocation)
                    process = subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[key] = (process,log,task,claim,invocation)
                    print('START',args.host,gpu,process.pid,key,flush=True)
                    launched = True
                except Exception as error:
                    if str(error) in ('Checkpoint is still being written', 'Model assets still being written'):
                        # Only first registration can report this transient condition.
                        # A changed, already registered fingerprint remains a hard error.
                        deferred_until[task['asset']['path']] = time.time() + 300
                        save(state/'deferred'/f'{key}.json',dict(task=key,host=args.host,error=str(error),time=time.time()))
                        (claim/'owner.json').unlink(missing_ok=True)
                        claim.rmdir()
                    else:
                        save(state/'PAUSED.json',dict(task=key,host=args.host,error=str(error),time=time.time()))
                    (state/'running'/f'{key}.json').unlink(missing_ok=True)
                break
            if launched: break
        if not manifest.get('online_checkpoints') and not active and all(
                (state/'done'/f"{t['key']}.json").exists() or (state/'failed_resource'/f"{t['key']}.json").exists()
                for t in manifest['tasks']): break
        time.sleep(3)
    if stopping:
        for process,log,*_ in active.values():
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            log.close()
    snapshot['status'] = 'STOPPED' if stopping else 'PAUSED' if (state/'PAUSED.json').exists() else 'COMPONENT_QUEUE_FINISHED'
    save(status_path,snapshot)

if __name__ == '__main__':
    os.umask(0o000)  # Shared NFS campaign has different Unix users on each authorized host.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'worker', 'watch'))
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--host', default=os.uname().nodename)
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--target-per-gpu', type=int, default=12)
    p.add_argument('--memory-target', type=float, default=.75)
    p.add_argument('--max-host-jobs', type=int, default=40)
    p.add_argument('--max-global-jobs', type=int, default=120)
    p.add_argument('--task-family', choices=('mixed','frozen','segmentation'), default='mixed')
    args = p.parse_args()
    if not 1 <= args.target_per_gpu <= 12 or not .70 < args.memory_target < .90:
        p.error('Use 1–12 slots and a memory target strictly between .70 and .90')
    if args.mode != 'prepare':
        manifest = json.loads((args.output / 'campaign_manifest.json').read_text())
        for path, digest in manifest['external_source_hashes'].items():
            if sha256(path) != digest:
                raise RuntimeError('Unregistered source change: ' + path)
    {'prepare':prepare, 'worker':worker, 'watch':watch}[args.mode](args)
