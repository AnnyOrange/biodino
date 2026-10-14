#!/usr/bin/env python3
"""Prepare isolated paired-component and full-v4 campaigns for the 20TB interventions."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
SOURCE = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918')
REFERENCE = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
PAIRED = REPO / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
FULL = REPO / 'outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012'
TRAIN = REPO / 'outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork38063_20261009'
TEMPLATE = REPO / 'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'


def load_fleet():
    path = SOURCE / 'scripts/run_retest_fleet_20260918.py'
    spec = importlib.util.spec_from_file_location('resampling_cls_fleet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_paired():
    if (PAIRED / 'campaign_manifest.json').exists():
        raise FileExistsError(PAIRED / 'campaign_manifest.json')
    reference = json.loads((REFERENCE / 'campaign_manifest.json').read_text())
    arms = [
        ('resampling_only_20tb', 'resampling_only_38063_8x3090qi', [40015, 41967, 43919]),
        ('fixed_cls_w1_20tb', 'fixed_cls_w1_anchor38063', [40015]),
    ]
    assets = []
    for arm, directory, checkpoints in arms:
        run = TRAIN / directory
        for step in checkpoints:
            checkpoint = run / f'eval/training_{step}/teacher_checkpoint.pth'
            if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
                raise FileNotFoundError(f'Incomplete teacher checkpoint: {checkpoint}')
            assets.append(dict(arm=arm, checkpoint_id=str(step), path=str(checkpoint),
                               config=str(run / 'config.yaml'), kind='dinov3', model_id='',
                               reserve_mib=4096))
    fleet = load_fleet()
    manifest = dict(reference)
    manifest.update(
        campaign_scope='20TB_RESAMPLING_AND_FIXED_CLS_PAIRED_V4_COMPONENTS',
        authorization='2026-10-12 user requested batch testing on single-card 3090 fleet and 3090-qi, over five tests per card, and merge into hs0_hs6_1tb_5tb_fm14_20260924',
        checkpoint_assets=assets,
        training_roots={a: str(Path(x['config']).parent) for a, _, _ in arms
                        for x in assets if x['arm'] == a},
        tasks=fleet.tasks_for(assets, reference['datasets']),
        online_checkpoints=False,
        no_checkpoint_or_data_transfers=True,
        scheduling=dict(target_per_gpu=5, memory_target=.75, minimum_target=.50,
                        max_concurrent_per_gpu=5, settle_seconds=60,
                        note='Five active tasks maximum per GPU; each participating GPU receives more than five tasks cumulatively.'))
    manifest['execution_hosts'] = {}
    manifest['scheduler_revisions'] = {}
    manifest['created_unix'] = time.time()
    fleet.queue.save(PAIRED / 'campaign_manifest.json', manifest)
    (PAIRED / '_state/inputs').mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(campaign=str(PAIRED), assets=len(assets), tasks=len(manifest['tasks']),
                          checkpoints=[(a['arm'], a['checkpoint_id']) for a in assets]), indent=2))


def prepare_full():
    if not (PAIRED / 'campaign_manifest.json').is_file():
        raise FileNotFoundError(PAIRED / 'campaign_manifest.json')
    if (FULL / 'campaign_manifest.json').exists():
        raise FileExistsError(FULL / 'campaign_manifest.json')
    for sub in ('tasks', 'claims', 'logs', 'runtime', 'provenance', 'workers', 'registered'):
        (FULL / sub).mkdir(parents=True, exist_ok=True)
    ref = json.loads((PAIRED / 'campaign_manifest.json').read_text())
    alltemplates = [json.loads(p.read_text()) for p in (TEMPLATE / 'tasks').glob('*.json')]
    tag = 'adaptive_continue_v2_gpu2_ck15615'
    templates = [t for t in alltemplates if t['arm'] == tag and
                 (t['dataset'] in ('rxrx3-core', 'ctc', 'xray', 'cryo', 'conic-cell-count',
                                   'livecell-cell-count', 'lc25000', 'nct-crc-he-100') or
                  t['family'] == 'detection_proxy')]
    if len(templates) != 12:
        raise RuntimeError(f'Expected 12 frozen full-v4 templates, found {len(templates)}')
    ctc = next(t for t in templates if t['dataset'] == 'ctc')
    oldck = ctc['cmd'][ctc['cmd'].index('--checkpoint') + 1]
    oldconfig = ctc['cmd'][ctc['cmd'].index('--config') + 1]
    adapter = (TEMPLATE / 'runtime/run_rxrx3.py').read_text().replace(str(TEMPLATE), str(FULL))
    (FULL / 'runtime/run_rxrx3.py').write_text(adapter)
    shutil.copyfile(TEMPLATE / 'runtime/run_detection_single_rank.py',
                    FULL / 'runtime/run_detection_single_rank.py')
    for path in (TEMPLATE / 'provenance').glob('source_*.json'):
        shutil.copyfile(path, FULL / 'provenance' / path.name)
    for asset in ref['checkpoint_assets']:
        arm = f"{asset['arm']}_ck{asset['checkpoint_id']}"
        tasks = []
        for template in templates:
            def replace(value):
                if isinstance(value, str):
                    return (value.replace(oldck, asset['path']).replace(oldconfig, asset['config'])
                            .replace(str(TEMPLATE), str(FULL)).replace(tag, arm))
                if isinstance(value, list):
                    return [replace(x) for x in value]
                if isinstance(value, dict):
                    return {k: replace(v) for k, v in value.items()}
                return value
            task = replace(template)
            task['asset'] = asset
            cmd = task['cmd']
            for flag in ('--step', '--ckpt-iter', '--checkpoint-id'):
                if flag in cmd:
                    cmd[cmd.index(flag) + 1] = asset['checkpoint_id']
            for stale in ('checkpoint_sha256', 'train_config_sha256', 'runtime_adapter_sha256', 'template_sha256'):
                task.pop(stale, None)
            task['template_sha256'] = sha256(TEMPLATE / 'tasks' / f"{template['id']}.json")
            task['priority'] = {'cell_tracking': 0, 'ood': 1, 'detection_proxy': 3}.get(task['family'], 2)
            task['created_unix'] = time.time()
            if task['family'] == 'ood':
                link = FULL / 'ood_adapters' / arm / asset['checkpoint_id'] / 'checkpoint.pth'
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(asset['path'])
                task['done']['path'] = task['done']['path'].replace('/15615/', f"/{asset['checkpoint_id']}/")
            if task['family'] == 'detection_proxy':
                if '--conic-split-protocol' not in cmd:
                    cmd += ['--conic-split-protocol', 'official-baseline-fold0-nested-v1']
            tasks.append(task)
        if len(tasks) != 12:
            raise RuntimeError(f'Expected 12 non-MoNuSeg full-v4 tasks for {arm}, found {len(tasks)}')
        for task in tasks:
            (FULL / 'tasks' / f"{task['id']}.json").write_text(json.dumps(task, indent=2) + '\n')
        (FULL / 'registered' / f'{arm}.json').write_text(
            json.dumps(dict(asset=asset, task_ids=[t['id'] for t in tasks],
                           monuseg_status='BLOCKED_MISSING_FROZEN_20260929_SOURCE_SNAPSHOT'), indent=2) + '\n')
    manifest = dict(protocol_id='bio-eval-union-v4',
                    authorization='2026-10-12 user requested full v4 batch on 3090 fleets and report merge',
                    source_component_campaign=str(PAIRED), admission_reference=str(TEMPLATE / 'FULL_V4_ADMISSION.json'),
                    assets=len(ref['checkpoint_assets']), extension_executions=len(list((FULL / 'tasks').glob('*.json'))),
                    expected_cells_per_checkpoint=56, monuseg='BLOCKED_MISSING_FROZEN_20260929_SOURCE_SNAPSHOT',
                    strict_aggregate_allowed=False,
                    runtime_hashes={str(p): sha256(p) for p in
                                    [Path(__file__).resolve(), REPO / 'scripts/run_v2_full_v4_20261007.py',
                                     FULL / 'runtime/run_rxrx3.py', FULL / 'runtime/run_detection_single_rank.py']},
                    created_unix=time.time())
    (FULL / 'campaign_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(campaign=str(FULL), tasks=manifest['extension_executions'],
                          assets=manifest['assets'], monuseg=manifest['monuseg']), indent=2))


def run_full_worker(args):
    from argparse import Namespace
    module = runpy.run_path(str(REPO / 'scripts/run_v2_full_v4_20261007.py'))
    module['worker'].__globals__.update(PAIRED=PAIRED, ROOT=FULL, MONUSRC=SOURCE)
    module['worker'](Namespace(host=args.host, gpus=args.gpus, max_jobs=args.max_jobs,
                               ram_reserve=args.ram_reserve, family='all'))


def run_full_inventory():
    module = runpy.run_path(str(REPO / 'scripts/run_v2_full_v4_20261007.py'))
    module['inventory'].__globals__.update(PAIRED=PAIRED, ROOT=FULL)
    module['inventory']()


def run_paired_worker(args):
    from argparse import Namespace
    os.umask(0)
    module = runpy.run_path(str(REPO / 'scripts/run_v2_progress_fleet_20261006.py'))
    module['worker'](Namespace(output=PAIRED, host=args.host, gpus=args.gpus,
        target_per_gpu=args.target_per_gpu, memory_target=args.memory_target,
        max_host_jobs=args.max_jobs, max_global_jobs=args.max_global_jobs,
        task_family='mixed'))


def requeue_interrupted():
    state = PAIRED / '_state'
    stamp = str(int(time.time()))
    attempts = state / 'attempts'
    for sub in ('claims', 'running', 'cells', 'logs', 'workers'):
        (attempts / sub).mkdir(parents=True, exist_ok=True)
    pending = 0
    for marker in list((state / 'running').glob('*.json')):
        key = marker.stem
        if (state / 'done' / marker.name).exists():
            marker.unlink()
            continue
        claim = state / 'claims' / key
        cell = PAIRED / 'cells' / key
        for path, root in ((marker, attempts / 'running'), (claim, attempts / 'claims'),
                           (cell, attempts / 'cells')):
            if path.exists():
                destination = root / f'{key}__{stamp}'
                path.rename(destination)
        pending += 1
    manifest_path = PAIRED / 'campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['scheduling'] = dict(target_per_gpu=6, memory_target=.80,
        minimum_target=.50, max_concurrent_per_gpu=6, settle_seconds=60,
        task_memory_reserve=True,
        note='Six simultaneous tests maximum per GPU, with measured VRAM admission capped at 80%; dense jobs retain per-task reserve.')
    revision = dict(time_unix=time.time(),
        revision='target6_memory80_after_preserving_interrupted_attempts',
        archived_inflight_tasks=pending)
    revisions = manifest.setdefault('scheduler_revisions', {})
    if isinstance(revisions, dict):
        revisions['20261012_target6'] = revision
    else:
        revisions.append(revision)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(archived_inflight_tasks=pending,
                          completed=len(list((state / 'done').glob('*.json'))),
                          target_per_gpu=6, memory_target=.80), indent=2))


def requeue_host(host: str):
    state = PAIRED / '_state'
    stamp = str(int(time.time()))
    attempts = state / 'attempts'
    for sub in ('claims', 'running', 'cells'):
        (attempts / sub).mkdir(parents=True, exist_ok=True)
    archived = []
    for marker in list((state / 'running').glob('*.json')):
        if (state / 'done' / marker.name).exists():
            continue
        record = load_json(marker)
        if record.get('host') != host:
            continue
        key = marker.stem
        claim = state / 'claims' / key
        owner = load_json(claim / 'owner.json') if claim.is_dir() else {}
        if owner and owner.get('host') != host:
            continue
        cell = PAIRED / 'cells' / key
        for path, root in ((marker, attempts / 'running'), (claim, attempts / 'claims'),
                           (cell, attempts / 'cells')):
            if path.exists():
                path.rename(root / f'{key}__{host}_{stamp}')
        archived.append(key)
    manifest_path = PAIRED / 'campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault('scheduler_revisions', {})[f'requeue_{host}_{stamp}'] = dict(
        time_unix=time.time(), archived_inflight_tasks=len(archived))
    manifest['scheduler_revisions'][f'placement_{stamp}'] = dict(
        time_unix=time.time(), destination='5090-zxr', gpus=list(range(8)),
        note='3090-qi worker stopped; its inflight attempts archived and requeued.')
    manifest.setdefault('execution_hosts', {}).pop(host, None)
    manifest['scheduling'] = dict(target_per_gpu=6, memory_target=.80,
        minimum_target=.50, max_concurrent_per_gpu=6, settle_seconds=60,
        task_memory_reserve=True,
        note='Up to six simultaneous tests per GPU; placement moved from 3090-qi to the local 8x5090 host.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(host=host, archived_inflight_tasks=len(archived), archived_keys=archived,
                          completed=len(list((state / 'done').glob('*.json'))),
                          destination='5090-zxr', gpus=list(range(8))), indent=2))


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def repair_source_references():
    old_root = Path('/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918')
    manifest_path = PAIRED / 'campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    replacements = {}

    def repair_dataset(dataset):
        hashes = dataset.get('source_hashes', {})
        repaired = {}
        for source, expected in hashes.items():
            path = Path(source)
            if str(path).startswith(str(old_root) + '/'):
                relative = path.relative_to(old_root)
                target = SOURCE / relative
                if not target.is_file() or sha256(target) != expected:
                    raise RuntimeError(f'No byte-identical frozen source for {source}')
                replacements[source] = str(target)
                repaired[str(target)] = expected
            else:
                if not path.is_file() or sha256(path) != expected:
                    raise RuntimeError(f'Registered evaluation source is missing or changed: {source}')
                repaired[source] = expected
        dataset['source_hashes'] = repaired

    for dataset in manifest['datasets']:
        repair_dataset(dataset)
    for task in manifest['tasks']:
        repair_dataset(task['dataset'])
    manifest['source_reference_remap'] = dict(old_root=str(old_root), new_root=str(SOURCE),
        files=replacements, verified_byte_identical=True, time_unix=time.time())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for task_path in (PAIRED / 'tasks').glob('*.json'):
        task = json.loads(task_path.read_text())
        repair_dataset(task['dataset'])
        task_path.write_text(json.dumps(task, indent=2) + '\n')
    state = PAIRED / '_state'
    paused = state / 'PAUSED.json'
    if paused.exists():
        attempts = state / 'attempts/paused'
        attempts.mkdir(parents=True, exist_ok=True)
        paused.rename(attempts / f'PAUSED__{int(time.time())}.json')
    key = 'resampling_only_20tb_ck40015__classification__cyclops-protein-loc'
    claim = state / 'claims' / key
    if claim.exists():
        archive = state / 'attempts/claims' / f'{key}__source_path_pause_{int(time.time())}'
        archive.parent.mkdir(parents=True, exist_ok=True)
        claim.rename(archive)
    print(json.dumps(dict(remapped_files=len(replacements),
                          byte_hashes_verified=True, archived_pause=True), indent=2))


def refresh_full_runtime_hash():
    path = FULL / 'campaign_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.setdefault('runtime_hashes', {})[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('UPDATED', str(Path(__file__).resolve()), manifest['runtime_hashes'][str(Path(__file__).resolve())])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare-paired', 'prepare-full', 'worker-paired',
                                         'worker-full', 'inventory-full', 'requeue-interrupted',
                                         'repair-source-references', 'refresh-full-runtime-hash',
                                         'requeue-host'))
    parser.add_argument('--host', default='')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--max-jobs', type=int, default=24)
    parser.add_argument('--ram-reserve', type=float, default=96)
    parser.add_argument('--max-global-jobs', type=int, default=120)
    parser.add_argument('--target-per-gpu', type=int, default=6)
    parser.add_argument('--memory-target', type=float, default=.80)
    parser.add_argument('--old-host', default='3090qi')
    args = parser.parse_args()
    if args.mode == 'prepare-paired':
        prepare_paired()
    elif args.mode == 'prepare-full':
        prepare_full()
    elif args.mode == 'worker-full':
        if not args.host:
            parser.error('--host is required for worker-full')
        run_full_worker(args)
    elif args.mode == 'worker-paired':
        if not args.host:
            parser.error('--host is required for worker-paired')
        run_paired_worker(args)
    elif args.mode == 'requeue-interrupted':
        requeue_interrupted()
    elif args.mode == 'repair-source-references':
        repair_source_references()
    elif args.mode == 'refresh-full-runtime-hash':
        refresh_full_runtime_hash()
    elif args.mode == 'requeue-host':
        requeue_host(args.old_host)
    else:
        run_full_inventory()


if __name__ == '__main__':
    main()
