#!/usr/bin/env python3
"""Schedule the frozen selected29 evaluator on explicitly assigned GPUs."""
import argparse
import csv
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--family', choices=['frozen', 'segmentation', 'mixed'], required=True)
    p.add_argument('--max-jobs', type=int, default=2)
    p.add_argument('--max-global-jobs', type=int, default=8)
    p.add_argument('--target-per-gpu', type=int, choices=range(1, 65), default=1)
    p.add_argument('--allow-training-share', action='store_true')
    p.add_argument('--memory-headroom-mib', type=int, default=2048)
    p.add_argument('--memory-headroom-fraction', type=float, default=.20)
    p.add_argument('--memory-admission-ceiling', type=float, default=.80)
    p.add_argument('--checkpoint-cache', type=Path)
    p.add_argument('--prefetch-only', action='store_true')
    p.add_argument('--allow-dense-colocation', action='store_true')
    p.add_argument('--minimum-per-gpu', type=int, default=1)
    p.add_argument('--memory-target-fraction', type=float, default=.65)
    a = p.parse_args()
    exclusion_path = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/unified_paper/gpu_exclusions.json')

    def excluded(gpu):
        if a.host.split(':')[0] != 'deepcad' or not exclusion_path.exists():
            return False
        return int(gpu) in json.loads(exclusion_path.read_text()).get('hosts', {}).get('deepcad', {}).get('gpus', [])

    a.gpus = [gpu for gpu in a.gpus if not excluded(gpu)]
    if not a.gpus:
        return

    os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                       'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    sys.path[:0] = [str(a.snapshot / 'scripts'), str(a.snapshot)]
    import run_l20tb_selected29_worker_20261019 as locked
    q = locked.fleet.queue
    if a.allow_dense_colocation:
        # Adapt admission only. The on-disk frozen evaluator, commands, data,
        # seeds, batch sizes and validators retain their locked hashes.
        source = inspect.getsource(q.worker)
        before = 'if dense and (actual or own):continue'
        assert source.count(before) == 1
        source = source.replace(before, 'if dense and (actual or own) and not args.allow_dense_colocation:continue')
        before = '(used/total>=0.6 and (actual>=3 or unmeasured))'
        assert source.count(before) == 1
        source = source.replace(before, '(used/total>=args.memory_target_fraction and actual>=args.minimum_per_gpu)')
        exec(compile(source, str(Path(__file__).resolve()) + ':admission', 'exec'), q.__dict__)
    q.command = locked.command
    q.validate_cell = locked.validate
    activity_path = a.output / '_state/activity' / f'{a.host}.json'
    activity_path.parent.mkdir(parents=True, exist_ok=True)

    def activity(phase, **details):
        q.save(activity_path, dict(host=a.host, pid=os.getpid(), phase=phase,
                                   updated_unix=time.time(), **details))

    def cached_asset(asset):
        if a.checkpoint_cache is None:
            return asset
        source = Path(asset['path'])
        stat = source.stat()
        directory = a.checkpoint_cache
        directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(source).encode()).hexdigest()[:20]
        target = directory / f'{key}.pth'
        metadata = directory / f'{key}.json'
        with (directory / f'{key}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            valid = False
            if target.exists() and metadata.exists():
                old = json.loads(metadata.read_text())
                valid = (old['source'], old['bytes'], old['source_mtime_ns']) == (str(source), stat.st_size, stat.st_mtime_ns)
                valid = valid and target.stat().st_size == stat.st_size and target.stat().st_mtime_ns == old['cache_mtime_ns']
            if not valid:
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                if available < stat.st_size + (64 << 30) or shutil.disk_usage(directory).free < stat.st_size + (32 << 30):
                    raise RuntimeError('Insufficient headroom for local checkpoint cache')
                # Sequential local read avoids repeated random torch.load reads
                # through mergerfs/NTFS. No network transfer or tensor conversion.
                direct = source
                try:
                    candidate = Path(os.getxattr(source, 'user.mergerfs.fullpath').decode())
                    if candidate.stat().st_size == stat.st_size:
                        direct = candidate
                except OSError:
                    pass
                partial = target.with_suffix('.partial')
                copied, last_update = 0, 0
                digest = hashlib.sha256()
                activity('CACHING_LOCAL_CHECKPOINT', checkpoint=asset['checkpoint_id'], source=str(source),
                         cache=str(target), bytes_copied=0, bytes_total=stat.st_size)
                with direct.open('rb') as inp, partial.open('wb') as out:
                    for block in iter(lambda: inp.read(8 << 20), b''):
                        out.write(block); digest.update(block); copied += len(block)
                        if time.time() - last_update > 5:
                            activity('CACHING_LOCAL_CHECKPOINT', checkpoint=asset['checkpoint_id'], source=str(source),
                                     cache=str(target), bytes_copied=copied, bytes_total=stat.st_size)
                            last_update = time.time()
                after = source.stat()
                if copied != stat.st_size or (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError('Checkpoint changed during local cache copy')
                original_key = q.fingerprint(str(source))
                original_record = a.output / '_state/inputs' / f'{original_key}.json'
                if original_record.exists() and json.loads(original_record.read_text())['sha256'] != digest.hexdigest():
                    raise RuntimeError('Local cache checksum differs from registered checkpoint')
                partial.replace(target)
                target.chmod(0o444)
                q.save(metadata, dict(source=str(source), bytes=copied, source_mtime_ns=stat.st_mtime_ns,
                                      cache_mtime_ns=target.stat().st_mtime_ns, sha256=digest.hexdigest()))
        return dict(asset, path=str(target))

    original_checkpoint_record = q.checkpoint_record

    def checkpoint_record(output, asset):
        local = cached_asset(asset)
        activity('VALIDATING_CHECKPOINT', checkpoint=asset['checkpoint_id'], source=asset['path'], path=local['path'])
        record = original_checkpoint_record(output, local)
        activity('CHECKPOINT_READY', checkpoint=asset['checkpoint_id'], sha256=record['sha256'])
        return dict(record, original_checkpoint_path=asset['path'])

    def command(task, output, benchmark):
        local = copy.deepcopy(task)
        local['asset'] = cached_asset(task['asset'])
        return locked.command(local, output, benchmark)

    def validate(task, output, invocation):
        local = copy.deepcopy(task)
        local['asset'] = cached_asset(task['asset'])
        if local['asset']['path'] != invocation['checkpoint']['path']:
            raise RuntimeError('Validation checkpoint differs from evaluated cache')
        return locked.validate(local, output, invocation)

    q.checkpoint_record = checkpoint_record
    q.command = command
    q.validate_cell = validate
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    uuids = {int(i): u.strip() for i, u in csv.reader(cards.splitlines())}
    if not set(a.gpus) <= set(uuids):
        raise RuntimeError('Requested GPU absent')
    original_popen = subprocess.Popen

    def pinned(*args, **kwargs):
        env = kwargs.get('env')
        if env and env.get('CUDA_VISIBLE_DEVICES', '').isdigit():
            index = int(env['CUDA_VISIBLE_DEVICES'])
            if excluded(index):
                raise RuntimeError('GPU reserved by user; refusing evaluation launch')
            if index in a.gpus:
                kwargs['env'] = dict(env, CUDA_VISIBLE_DEVICES=uuids[index])
        process = original_popen(*args, **kwargs)
        if env and 'CUDA_VISIBLE_DEVICES' in env:
            activity('EVALUATING', child_pid=process.pid, gpu=kwargs.get('env', env)['CUDA_VISIBLE_DEVICES'])
        return process

    subprocess.Popen = pinned
    # Count our evaluation slots; all-process memory and training occupancy are
    # independently checked below, allowing safe use of spare evaluation memory.
    def own_tests():
        counts = {}
        for f in (a.output / '_state/running').glob('*.json'):
            try:
                row = json.loads(f.read_text())
            except FileNotFoundError:
                continue
            if row.get('host', '').split(':')[0] == a.host.split(':')[0]:
                k = str(row['gpu'])
                counts[k] = counts.get(k, 0) + 1
        return counts

    q.project_tests = own_tests
    cache = {'time': 0, 'training': set(), 'allocations': {}, 'peaks': {}}
    policy_path = a.output / '_state/admission_policy' / (a.host.split(':')[0] + '.json')

    def guard(gpu, actual, task):
        if excluded(gpu):
            return False
        if a.checkpoint_cache is not None:
            key = hashlib.sha256(task['asset']['path'].encode()).hexdigest()[:20]
            if not (a.checkpoint_cache / f'{key}.json').exists() or not (a.checkpoint_cache / f'{key}.pth').exists():
                return False
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        if policy.get('hold_new_tasks'):
            return False
        if time.time() - cache['time'] > 3:
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True)
            blocked = set()
            allocations = {}
            for line in csv.reader(processes.splitlines()):
                if len(line) != 3:
                    continue
                uid, pid, memory = (v.strip() for v in line)
                try:
                    cmd = Path('/proc', pid, 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                except OSError:
                    continue
                evaluation = any(token in cmd for token in
                    ('dinov3.eval.', '/dinov3/eval/', '/bio_segmentation/', 'linear_probe', '--probe-epochs'))
                training = any(token in cmd for token in ('dinov3/train/', 'train_irflow', '/train.py', ' train '))
                if training and not evaluation:
                    blocked.update(i for i, u in uuids.items() if u == uid)
                prefix = str(a.output / 'cells') + '/'
                if prefix in cmd and memory.isdigit():
                    key = cmd.split(prefix, 1)[1].split('/', 1)[0].split(' ', 1)[0]
                    allocations[key] = allocations.get(key, 0) + int(memory)
            for key, memory in allocations.items():
                cache['peaks'][key] = max(cache['peaks'].get(key, 0), memory)
            cache.update(time=time.time(), training=blocked, allocations=allocations)
        actual = max(actual, own_tests().get(str(gpu), 0))
        maximum = min(a.target_per_gpu, int(policy.get('max_tasks_per_gpu', a.target_per_gpu)))
        if (gpu in cache['training'] and not a.allow_training_share) or actual >= maximum:
            return False
        used, total = q.resources()[gpu]
        if a.allow_dense_colocation:
            target = min(a.memory_target_fraction, float(policy.get('memory_target_fraction', a.memory_target_fraction)))
            if used / total >= target and actual >= a.minimum_per_gpu:
                return False
            # Keep startup reservations until CUDA is observed, however long
            # loading takes. Retain observed peaks through later CPU phases.
            pending = 0
            for f in (a.output / '_state/running').glob('*.json'):
                try:
                    row = json.loads(f.read_text())
                except FileNotFoundError:
                    continue
                if row.get('host', '').split(':')[0] != a.host.split(':')[0] or row['gpu'] != gpu:
                    continue
                current = cache['allocations'].get(f.stem, 0)
                if f.stem in cache['peaks']:
                    pending += max(cache['peaks'][f.stem] - current, 0)
                else:
                    invocation = a.output / 'cells' / f.stem / 'invocation_manifest.json'
                    record = json.loads(invocation.read_text()) if invocation.exists() else {}
                    pending += int(record.get('reserve_mib', 4096))
            used += pending
        spec = task['dataset']
        reserve = int(spec.get('reserve_mib') or max(18432 if spec['image_size'] >= 384 else 4096,
                                                     int(task['asset'].get('reserve_mib') or 0)))
        headroom = max(a.memory_headroom_mib, int(policy.get('memory_headroom_mib', 0)),
                       int(total * max(a.memory_headroom_fraction, float(policy.get('memory_headroom_fraction', 0)))))
        ceiling = min(a.memory_admission_ceiling, float(policy.get('memory_admission_ceiling', 1)))
        return used + reserve <= min(total - headroom, total * ceiling)

    args = argparse.Namespace(output=a.output, host=a.host, gpus=a.gpus,
        task_family=a.family,
        target_per_gpu=a.target_per_gpu, max_host_jobs=a.max_jobs, max_global_jobs=a.max_global_jobs,
        admission_guard=guard, allow_dense_colocation=a.allow_dense_colocation,
        minimum_per_gpu=a.minimum_per_gpu, memory_target_fraction=a.memory_target_fraction)
    if a.prefetch_only:
        if a.checkpoint_cache is None:
            raise ValueError('--prefetch-only requires --checkpoint-cache')
        state = a.output / '_state/workers'
        state.mkdir(parents=True, exist_ok=True)
        with (state / f'{a.host}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                manifest = json.loads((a.output / 'campaign_manifest.json').read_text())
                for asset in sorted(manifest['checkpoint_assets'], key=lambda x: int(x['checkpoint_id'])):
                    q.save(state / f'{a.host}.json', dict(host=a.host, pid=os.getpid(), status='PREFETCHING',
                           own_active=[], updated_unix=time.time(), checkpoint=asset['checkpoint_id']))
                    try:
                        cached_asset(asset)
                    except Exception as exc:
                        activity('CACHE_RETRY_PENDING', checkpoint=asset['checkpoint_id'], error=repr(exc))
                        time.sleep(30)
                        break
                else:
                    activity('ALL_CHECKPOINTS_CACHED')
                time.sleep(15)
        return
    q.worker(args)


if __name__ == '__main__':
    main()
