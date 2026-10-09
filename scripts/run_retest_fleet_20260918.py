"""User-authorized immutable-source independent components and online teachers."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MODEL_ROOT = Path(os.environ.get('BENCHMARK_MODEL_ROOT', '/mnt/huawei_deepcad/benchmark_model'))
if not (BENCHMARK_MODEL_ROOT / 'benchmark_eval/rules_dense_preflight.py').is_file():
    raise RuntimeError(f'External FM source missing: {BENCHMARK_MODEL_ROOT}')
os.environ['DINOV3_CODE_ROOT'] = str(ROOT)
sys.path[:0] = [str(BENCHMARK_MODEL_ROOT), str(ROOT), str(ROOT / 'scripts')]
import run_shared_frozen_stage1_20260917 as queue
from dinov3.eval.bio_frozen_eval.external_fm_protocol import PANNUKE_PROTOCOLS, validate_best_head

NUMERICAL = {'torch': '2.10.0', 'torchvision': '0.25.0', 'numpy': '2.4.3',
             'scipy': '1.17.1', 'scikit-learn': '1.8.0', 'Pillow': '12.1.1'}
BLOCKED = {'monuseg': 'NOT TESTED: official30/test14 sample identity evidence pending;37-pool forbidden',
           'ctc': 'NOT TESTED: native20/five-fold fixed decoder/linker acceptance pending',
           'xray': 'NOT TESTED: common OOD selection/config freeze pending',
           'cryo': 'NOT TESTED: common OOD selection/config freeze pending',
           'bbbc038': 'NOT TESTED in formal queue: separate observation required'}
ORIGINAL_COMMAND = queue.command
ORIGINAL_VALIDATE = queue.validate_cell


def command(task, directory, benchmark):
    spec, asset = task['dataset'], task['asset']
    if spec['task'] != 'segmentation':
        return ORIGINAL_COMMAND(task, directory, benchmark)
    if asset.get('kind') == 'external':
        args = [sys.executable, '-u', '-m', 'dinov3.eval.bio_frozen_eval.run_external_dense_rules',
                '--models', asset['model_id'], '--datasets', spec['dataset'],
                '--comparison-view', 'primary-last', '--benchmark-root', str(benchmark),
                '--out-root', str(directory), '--device', 'cuda:0',
                '--extract-batch-size', '32', '--probe-batch-size', '32']
        if spec['dataset'] == 'pannuke':
            args += ['--split-protocol', spec['split']]
        return args
    original = Path(asset['path'])
    staged = directory / 'checkpoint_links' / str(asset['checkpoint_id'])
    staged.parent.mkdir(parents=True, exist_ok=True)
    if original.is_dir():
        if not staged.exists():
            staged.symlink_to(original, target_is_directory=True)
    else:
        staged.mkdir(exist_ok=True)
        link = staged / 'checkpoint.pth'
        if not link.exists():
            link.symlink_to(original)
    return [sys.executable, '-u', '-m', 'dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline',
            '--datasets', spec['dataset'], '--checkpoints-dir', str(staged.parent),
            '--checkpoint-iters', str(asset['checkpoint_id']), '--train-config', asset['config'],
            '--data-root-base', str(Path(benchmark) / 'segmentation'), '--protocol', 'manual',
            '--dataset-split-protocol', spec['split'], '--feature-img-size', str(spec['image_size']),
            '--resize-mode', spec['resize_mode'], '--layer-preset', 'last1',
            '--feature-batch-size', '32', '--autocast-dtype', 'bf16', '--feature-num-workers', '2',
            '--probe-epoch-grid', '20', '50', '--probe-seeds', '0', '1', '2',
            '--probe-eval-every', '1', '--probe-batch-size', '32', '--probe-num-workers', '2',
            '--probe-lr', '0.001', '--probe-weight-decay', '0.0001',
            '--probe-class-weight-mode', 'sqrt_inverse' if spec['dataset'] == 'conic' else 'none',
            '--channel-policy', 'auto', '--chunked-cache', '--no-compress-cache',
            '--cache-root', str(directory / 'cache'), '--output-root', str(directory / 'results'),
            '--run-name', 'primary_last']


def validate(task, directory, invocation):
    spec = task['dataset']
    if spec['task'] != 'segmentation':
        return ORIGINAL_VALIDATE(task, directory, invocation)
    paths = list(directory.rglob('results.json'))
    if len(paths) != 6:
        raise ValueError('Exactly six E20/E50 x seed0/1/2 fits required per dense split')
    signatures = set()
    for path in paths:
        result = json.loads(path.read_text())
        meta = result['_meta']
        validate_best_head(meta)
        if meta['probe_batch_size'] != 32 or meta['probe_eval_every'] != 1:
            raise ValueError('Dense batch/validation mismatch')
        if meta['full_train_samples'] != spec['counts']['train'] or meta['used_train_samples'] != spec['counts']['train']:
            raise ValueError('Dense training count/subsampling mismatch')
        if not math.isfinite(result['test']['mDice']):
            raise ValueError('Nonfinite dense primary metric')
        signatures.add((meta['probe_epochs'], meta['seed']))
    if signatures != {(e, s) for e in (20, 50) for s in (0, 1, 2)}:
        raise ValueError('Dense budget/seed coverage mismatch')
    report = dict(status='VALID_COMPLETE', scope='INDEPENDENT_COMPONENT_ONLY',
                  full_v3_aggregate_allowed=False, validator_commit=invocation['git_commit'],
                  source_snapshot_sha256=invocation['source_snapshot_sha256'],
                  input_fingerprint=queue.fingerprint(invocation), expected_counts=spec['counts'],
                  result_sha256={str(p): queue.sha256(p) for p in paths})
    queue.save(directory / 'validation_report.json', report)
    return report


def tasks_for(assets, datasets):
    tasks = []
    priorities = {'breastmnist': 0, 'bbbc013': 1, 'nct-crc-he-1k': 2, 'organcmnist': 3,
                  'pneumoniamnist': 4, 'cellpose': 5, 'conic': 6, 'pannuke': 7}
    for spec in sorted(datasets, key=lambda d: priorities.get(d['dataset'], 10)):
        if spec['status'] != 'PASS':
            continue
        for asset in assets:
            key = f"{asset['arm']}_ck{asset['checkpoint_id']}__{spec['task']}__{spec['dataset']}"
            if spec['task'] == 'segmentation':
                key += '__primary-last__' + spec['split']
            tasks.append(dict(key=key, asset=asset, dataset=spec))
    return tasks


def prepare(args):
    if (args.output / 'campaign_manifest.json').exists():
        raise RuntimeError('Do not replace an existing campaign')
    os.environ.update(queue.THREADS)
    from benchmark_eval.rules_dense_preflight import audit_dense
    protocol = json.loads((ROOT / 'Evaluation Rules/protocol_v3.json').read_text())
    snapshot = json.loads((ROOT / 'source_snapshot.json').read_text())
    queue.verify_campaign_source(dict(git_commit=snapshot['git_commit'], source_snapshot=snapshot))
    datasets = []
    for task in ('classification', 'regression', 'retrieval', 'segmentation', 'cell_tracking', 'ood'):
        names = protocol['tier_a'].get(task, []) + protocol['tier_b'].get(task, [])
        for name in names:
            folds = PANNUKE_PROTOCOLS if name == 'pannuke' else (None,)
            for fold in folds:
                try:
                    if name in BLOCKED:
                        raise RuntimeError(BLOCKED[name])
                    if name == 'rxrx3-core':
                        raise RuntimeError('NOT TESTED: dedicated compact3 integration acceptance pending')
                    if task == 'segmentation':
                        spec = audit_dense(name, args.benchmark, protocol=fold)
                        spec.update(comparison_view='primary-last', reserve_mib=16000 if spec['image_size'] >= 512 else 8192)
                    elif task == 'retrieval':
                        spec = queue.retrieval_preflight(name, args.benchmark)
                    else:
                        spec = queue.dataset_preflight(name, task, args.benchmark)
                except Exception as error:
                    spec = dict(task=task, dataset=name, split=fold, status='BLOCKED', reason=str(error))
                datasets.append(spec)
                queue.save(args.output / 'dataset_preflight.json', datasets)
                print(name, fold or '', spec['status'], spec.get('reason', ''), flush=True)
    assets = json.loads(args.assets.read_text())
    for asset in assets:
        if not Path(asset['path']).exists() or (asset.get('config') and not Path(asset['config']).is_file()):
            raise RuntimeError('Resident checkpoint/config missing: ' + str(asset))
    manifest = dict(protocol_id='bio-eval-formal-v3-retest-20260918-components',
                    git_commit=snapshot['git_commit'], source_snapshot=snapshot,
                    authorization='Explicit user launch authorization 20260918',
                    git_status_porcelain='LOCAL_COPY_HASH_VERIFIED', numerical_environment=NUMERICAL,
                    protocol_sha256=queue.sha256(ROOT / 'Evaluation Rules/protocol_v3.json'),
                    plan='Evaluation Rules/plans/retest_fleet_20260918.md',
                    plan_sha256=queue.sha256(ROOT / 'Evaluation Rules/plans/retest_fleet_20260918.md'),
                    expected_full_suite_task_pairs_per_checkpoint=41, checkpoint_assets=assets,
                    datasets=datasets, tasks=tasks_for(assets, datasets), benchmark_root=str(args.benchmark),
                    batch_size=64, autocast_dtype='bf16', n_last_blocks=1, use_avgpool=True, seed=0,
                    num_workers=2, full_v3_aggregate_allowed=False, legacy_reuse=False,
                    no_checkpoint_or_data_transfers=True, online_checkpoints=args.online,
                    isolate_runtime_failures=True, created_unix=time.time(), deadline_unix=time.time()+5*3600)
    if args.external_hashes:
        manifest['external_source_hashes'] = json.loads(args.external_hashes.read_text())
    queue.save(args.output / 'campaign_manifest.json', manifest)
    (args.output / '_state/inputs').mkdir(parents=True, exist_ok=True)
    print('PREPARED', len(assets), 'checkpoints', len(manifest['tasks']), 'ready component groups', flush=True)


def worker(args):
    manifest = json.loads((args.output / 'campaign_manifest.json').read_text())
    for path, digest in manifest.get('external_source_hashes', {}).items():
        if queue.sha256(path) != digest:
            raise RuntimeError('Unregistered external source change: ' + path)
    queue.command, queue.validate_cell = command, validate
    queue.worker(args)


def watch(args):
    lock = (args.output / '_state/online_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    roots = json.loads(args.training_roots.read_text())
    observed = {}
    while True:
        admission = (args.output / '_state/manifest.lock').open('a')
        with admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            manifest = json.loads((args.output / 'campaign_manifest.json').read_text())
            known = {asset['path'] for asset in manifest['checkpoint_assets']}
            additions = []
            for arm, root in roots.items():
                root = Path(root)
                for path in (root / 'eval').glob('training_*/teacher_checkpoint.pth'):
                    if str(path) in known:
                        continue
                    stat = path.stat()
                    identity = (stat.st_size, stat.st_mtime_ns)
                    if observed.get(str(path)) != identity or time.time()-stat.st_mtime < 120:
                        observed[str(path)] = identity
                        continue
                    asset = dict(arm=arm, checkpoint_id=path.parent.name.split('_')[-1],
                                 path=str(path), config=str(root/'config.yaml'), kind='dinov3',
                                 model_id='', reserve_mib=4096)
                    inputs = queue.checkpoint_record(args.output, asset)
                    additions.append(asset)
                    journal = args.output / '_state/checkpoint_admission.jsonl'
                    with journal.open('a') as handle:
                        handle.write(json.dumps(dict(time=time.time(), asset=asset, fingerprint=inputs,
                                                     source_snapshot_sha256=manifest['source_snapshot']['sha256']))+'\n')
            if additions:
                manifest['checkpoint_assets'] += additions
                manifest['tasks'] += tasks_for(additions, manifest['datasets'])
                queue.save(args.output / 'campaign_manifest.json', manifest)
                print('ADMITTED_NEW_TEACHERS', len(additions), flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'worker', 'watch', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--benchmark', type=Path, default=Path('/mnt/huawei_deepcad/benchmark'))
    parser.add_argument('--external-hashes', type=Path)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--training-roots', type=Path)
    parser.add_argument('--host', default=os.uname().nodename)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--target-per-gpu', type=int, choices=range(1,6), default=5)
    parser.add_argument('--max-host-jobs', type=int, default=40)
    parser.add_argument('--max-global-jobs', type=int, default=160)
    parser.add_argument('--task-family', choices=('mixed','frozen','segmentation'), default='mixed')
    args = parser.parse_args()
    if args.mode == 'run':
        if not (args.output / 'campaign_manifest.json').exists():
            prepare(args)
        worker(args)
    else:
        {'prepare': prepare, 'worker': worker, 'watch': watch}[args.mode](args)


if __name__ == '__main__':
    main()
