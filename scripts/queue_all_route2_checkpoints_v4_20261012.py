#!/usr/bin/env python3
"""Prepare and run V4 on every saved teacher checkpoint in both 20TB route2 runs."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
OUT = REPO / 'outputs/02_eval_runs/route2_all_checkpoints_v4_20261012'
PAIRED = OUT / 'components'
FULL = OUT / 'extensions'
REFERENCE = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006/campaign_manifest.json'
MONU_REFERENCE = REPO / 'plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999/monuseg/campaign_manifest.json'
MONU_SOURCE = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')
METHOD_SOURCE = Path('/mnt/huawei_deepcad/dinov3_method_full_v4_snapshot_20260928')
TRAINING_RUNS = {
    'noGRAM20tb': REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924',
    'route2_boundary20tb': REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_15tb07_5tbboundary03_ablation_8x3090qi_20260928',
}
SINGLE_GPU_HOSTS = ('cpu1', 'cpu2', 'cpu5', 'cpu8', 'cpu9', 'cpu10', 'cpu11', 'cpu12', 'cpu15', 'cpu19')
REMOTE_PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_assets() -> list[dict]:
    result = []
    for arm, run in TRAINING_RUNS.items():
        config = run / 'config.yaml'
        if not config.is_file():
            raise FileNotFoundError(config)
        checkpoints = sorted((run / 'eval').glob('training_*/teacher_checkpoint.pth'),
                             key=lambda p: int(p.parent.name.removeprefix('training_')))
        for checkpoint in checkpoints:
            step = int(checkpoint.parent.name.removeprefix('training_'))
            if (step + 1) % 488 or checkpoint.stat().st_size < 1_000_000_000:
                raise RuntimeError(f'Unexpected or incomplete teacher checkpoint: {checkpoint}')
            result.append(dict(arm=arm, checkpoint_id=str(step), path=str(checkpoint),
                               config=str(config), kind='dinov3', model_id='', reserve_mib=4096))
    return result


def prepare() -> None:
    if (FULL / 'campaign_manifest.json').exists():
        raise FileExistsError(f'All-checkpoint campaign already exists: {OUT}')
    reference = read(REFERENCE)
    assets = all_assets()
    if len(assets) != 236 or len({a['path'] for a in assets}) != 236:
        raise RuntimeError(f'Expected 236 unique teacher checkpoints, found {len(assets)}')
    if sum(a['arm'] == 'noGRAM20tb' for a in assets) != 153 or \
            sum(a['arm'] == 'route2_boundary20tb' for a in assets) != 83:
        raise RuntimeError('Checkpoint inventory does not match the two requested training branches')

    fleet_root = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918')
    fleet = load_module('route2_all_ckpt_fleet', fleet_root / 'scripts/run_retest_fleet_20260918.py')
    protocol = REPO / 'Evaluation Rules/protocol_v4.json'
    manifest = {k: reference[k] for k in ('datasets', 'benchmark_root', 'numerical_environment',
                                           'source_snapshot', 'git_commit', 'source_snapshot_path')}
    manifest.update(protocol_id='bio-eval-union-v4', campaign_scope='ALL_SAVED_TEACHERS_BOTH_ORIGINAL_20TB_ROUTE2_RUNS',
        v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
        authorization='2026-10-12 user: test every V4 checkpoint for both original 20TB route2 training branches',
        external_source_hashes={str(protocol): sha(protocol), str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
        protocol_sha256=sha(protocol), checkpoint_assets=assets,
        training_roots={arm: str(run) for arm, run in TRAINING_RUNS.items()},
        tasks=fleet.tasks_for(assets, reference['datasets']), online_checkpoints=False,
        no_checkpoint_or_data_transfers=True, isolate_runtime_failures=True,
        batch_size=64, seed=0, n_last_blocks=1, use_avgpool=True, autocast_dtype='bf16', num_workers=2,
        scheduling=dict(target_per_gpu=5, memory_target=.75, max_concurrent_per_gpu=5,
                        workers='10 single-card 3090 hosts + 8-card 3090-qi + local 5090 free cards'),
        created_unix=time.time())
    if len(manifest['tasks']) != 236 * 38:
        raise RuntimeError(f'Component task count is not 236 x 38: {len(manifest["tasks"])}')
    existing_paired = PAIRED / 'campaign_manifest.json'
    if existing_paired.exists():
        old = read(existing_paired)
        if {a['path'] for a in old.get('checkpoint_assets', [])} != {a['path'] for a in assets}:
            raise RuntimeError('Existing partial component queue is for a different checkpoint set')
    else:
        save(existing_paired, manifest)
    (PAIRED / '_state/inputs').mkdir(parents=True, exist_ok=True)

    full_builder = runpy.run_path(str(REPO / 'scripts/prepare_20tb_resampling_cls_v4_20261012.py'))
    full_builder['prepare_full'].__globals__.update(PAIRED=PAIRED, FULL=FULL)
    full_builder['prepare_full']()

    monu_campaign = read(MONU_REFERENCE)
    monu_spec = monu_campaign['datasets'][0]
    monu_source_manifest = Path(next(iter(monu_spec['source_hashes'])))
    authoritative_lock = REPO / 'outputs/02_eval_inputs/monuseg37_v4_20261012/manifest.json'
    if monu_spec['counts'] != dict(train=30, val=7, test=14) or \
            sha(monu_source_manifest) != monu_spec['source_hashes'][str(monu_source_manifest)] or \
            read(authoritative_lock)['source_manifest_sha256'] != sha(monu_source_manifest):
        raise RuntimeError('MoNuSeg source assignment differs from the 2026-10-12 authoritative lock')
    source_snapshot = read(MONU_SOURCE / 'source_snapshot.json')
    if source_snapshot != monu_campaign['source_snapshot']:
        raise RuntimeError('Pinned MoNuSeg evaluator source does not match its admitted campaign')
    full_manifest_path = FULL / 'campaign_manifest.json'
    full_manifest = read(full_manifest_path)
    for asset in assets:
        arm = f"{asset['arm']}_ck{asset['checkpoint_id']}"
        key = f'{arm}__segmentation__monuseg__primary-last__formal-static-v1'
        task = dict(id=key, arm=arm, family='segmentation', dataset='monuseg', priority=4,
                    asset=asset, monuseg_spec=monu_spec, monuseg_source=source_snapshot,
                    expected_memory_mib=16000)
        save(FULL / 'tasks' / f'{key}.json', task)
    full_manifest.update(assets=len(assets), expected_cells_per_checkpoint=56,
        extension_executions=len(list((FULL / 'tasks').glob('*.json'))),
        monuseg_protocol=dict(counts=monu_spec['counts'], split_protocol_id=monu_spec['split_protocol_id'],
            split_identity_sha256=monu_spec['split_identity_sha256'],
            raw_manifest_sha256=sha(monu_source_manifest),
            authoritative_lock_sha256=sha(authoritative_lock),
            evaluator_snapshot_sha256=source_snapshot['sha256']),
        strict_aggregate_allowed=False)
    full_manifest.setdefault('runtime_hashes', {})[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    save(full_manifest_path, full_manifest)
    inventory()
    print(json.dumps(dict(output=str(OUT), checkpoints=len(assets), mix009=153, boundary=83,
                          component_tasks=len(manifest['tasks']), extension_and_monu_tasks=full_manifest['extension_executions'],
                          total_executions=len(manifest['tasks']) + full_manifest['extension_executions'],
                          tasks_per_checkpoint=51, v4_inventory_rows=56 * len(assets),
                          monuseg_split=monu_spec['split_identity_sha256']), indent=2))


def inventory() -> None:
    paired = read(PAIRED / 'campaign_manifest.json')
    full = read(FULL / 'campaign_manifest.json')
    assets = paired['checkpoint_assets']
    task_keys = {t['key'] for t in paired['tasks']}
    full_tasks = [read(p) for p in (FULL / 'tasks').glob('*.json')]
    full_ids = {t['id'] for t in full_tasks}
    if len(task_keys) != 38 * len(assets) or len(full_ids) != 13 * len(assets):
        raise RuntimeError(f'Inventory mismatch: components={len(task_keys)}, extensions={len(full_ids)}')
    if len(assets) != 236:
        raise RuntimeError(f'Checkpoint coverage mismatch: {len(assets)}')
    expected = {a['path'] for a in assets}
    component_paths = {t['asset']['path'] for t in paired['tasks']}
    full_paths = {t['asset']['path'] for t in full_tasks}
    if expected != component_paths or expected != full_paths:
        raise RuntimeError('At least one saved teacher lacks one or more V4 task families')
    done_components = PAIRED / '_state/done'
    done_full = FULL / 'claims'
    summary = dict(updated_unix=time.time(), checkpoints=len(assets), component_tasks=len(task_keys),
                   full_v4_tasks=len(full_ids), expected_total=len(task_keys) + len(full_ids),
                   completed_components=sum((done_components / f'{k}.json').is_file() for k in task_keys),
                   completed_full=sum((done_full / t / 'status.json').is_file() and
                       read(done_full / t / 'status.json').get('state') == 'VALID_COMPLETE' for t in full_ids),
                   per_checkpoint=dict(components=38, full_v4_extensions=12, monuseg=1),
                   tasks_per_checkpoint=51, unique_v4_inventory_cells_per_checkpoint=56)
    save(OUT / 'ALL_CHECKPOINT_V4_INVENTORY.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def repair_full_sources() -> None:
    provenance = FULL / 'provenance'
    old = provenance / 'source_a2efa3c6e6d1ad0b.json'
    replacement = provenance / 'source_6d88ac85930047c0.json'
    archived = provenance / 'source_a2efa3c6e6d1ad0b.json.unavailable'
    if not archived.exists():
        archived.write_bytes(old.read_bytes())
    record = read(replacement)
    if record.get('source') != str(METHOD_SOURCE):
        raise RuntimeError('Unexpected pinned replacement evaluator source')
    record['substitution'] = dict(
        original_source='/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923',
        original_source_status='MISSING_ON_ALL_EVALUATION_HOSTS',
        reason='Use the already admitted byte-equivalent frozen method snapshot.',
        equivalence_checks={
            'dinov3_eval_modules': 'byte-identical to the unavailable source fingerprint',
            'protocol_v4': 'byte-identical to the unavailable source fingerprint',
            'run_external4_fixedbudget_model.py': 'SHA256 b3d74b6610833f9ea6d360d4823cadd213acaa190a2ec3b16a63647d76110c63',
            'rxrx3_split_manifest': 'SHA256 94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5',
        })
    save(old, record)
    adapter = FULL / 'runtime/run_rxrx3.py'
    adapter_text = adapter.read_text()
    old_root = "SRC=Path('/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923')"
    new_root = f'SRC=Path({str(METHOD_SOURCE)!r})'
    if old_root in adapter_text:
        adapter_text = adapter_text.replace(old_root, new_root)
    elif new_root not in adapter_text:
        raise RuntimeError('RxRx3 adapter source path is unexpected')
    adapter_text = adapter_text.replace(
        "str(SRC/'scripts/run_external4_fixedbudget_model.py')",
        f"str(Path({str(REPO / 'scripts/run_external4_fixedbudget_model.py')!r}))")
    adapter.write_text(adapter_text)
    for path in (FULL / 'tasks').glob('*.json'):
        task = read(path)
        if task.get('dataset') == 'monuseg':
            task['cwd'] = str(MONU_SOURCE)
            task['pythonpath'] = str(MONU_SOURCE)
            save(path, task)
        if task.get('dataset') == 'rxrx3-core':
            task['cwd'] = str(METHOD_SOURCE)
            task['pythonpath'] = str(METHOD_SOURCE)
            save(path, task)
        if task.get('id', '').startswith('ctc__native__'):
            wrapper = REPO / 'scripts/run_ctc_native_with_gitref_20261012.py'
            bootstrap = (f"import sys,runpy;sys.path.insert(0,{str(METHOD_SOURCE / 'scripts')!r});"
                         f"runpy.run_path({str(wrapper)!r},run_name='__main__')")
            cmd = task['cmd']
            task['cmd'] = [cmd[0], '-B', '-c', bootstrap, *cmd[4:]]
            save(path, task)
    manifest_path = FULL / 'campaign_manifest.json'
    manifest = read(manifest_path)
    manifest.setdefault('runtime_hashes', {})[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    manifest['runtime_hashes'][str(adapter)] = sha(adapter)
    wrapper = REPO / 'scripts/run_ctc_native_with_gitref_20261012.py'
    manifest['runtime_hashes'][str(wrapper)] = sha(wrapper)
    external4 = REPO / 'scripts/run_external4_fixedbudget_model.py'
    if sha(external4) != 'b3d74b6610833f9ea6d360d4823cadd213acaa190a2ec3b16a63647d76110c63':
        raise RuntimeError('RxRx3 evaluator differs from its previously admitted SHA-256')
    manifest['runtime_hashes'][str(external4)] = sha(external4)
    save(manifest_path, manifest)
    component_manifest_path = PAIRED / 'campaign_manifest.json'
    component_manifest = read(component_manifest_path)
    component_manifest.setdefault('external_source_hashes', {})[
        str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    save(component_manifest_path, component_manifest)
    archive = FULL / 'attempt_archive/rxrx3_missing_source'
    for path in (FULL / 'tasks').glob('*.json'):
        task = read(path)
        claim = FULL / 'claims' / task['id']
        status = read(claim / 'status.json') if (claim / 'status.json').is_file() else {}
        if status.get('state') != 'FAILED' or not (
                task.get('dataset') == 'monuseg' or task.get('dataset') == 'rxrx3-core'):
            continue
        stamp = str(time.time_ns())
        if claim.exists():
            target = archive / 'claims' / f'{task["id"]}__{stamp}'
            target.parent.mkdir(parents=True, exist_ok=True)
            claim.rename(target)
        log = FULL / 'logs' / f'{task["id"]}.log'
        if log.exists():
            target = archive / 'logs' / f'{task["id"]}__{stamp}.log'
            target.parent.mkdir(parents=True, exist_ok=True)
            log.rename(target)
    print(json.dumps(dict(provenance=str(old),replacement=record['source'],
                          rxrx3_tasks=sum(read(p).get('dataset') == 'rxrx3-core' for p in (FULL/'tasks').glob('*.json')),
                          ctc_tasks=sum(read(p).get('id','').startswith('ctc__native__') for p in (FULL/'tasks').glob('*.json'))),indent=2))


def refresh_report() -> dict:
    summary = inventory()
    report_root = REPO / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
    report_out = report_root / '20tb_route2_all_checkpoints_v4_20261012'
    module = runpy.run_path(str(REPO / 'scripts/summarize_20tb_resampling_cls_v4_20261012.py'))
    module['main'].__globals__.update(
        PAIRED=PAIRED, FULL=FULL, OUT=report_out,
        status_title='# 两条原始 20TB Route2 全 checkpoint V4 评测',
        inventory_label='本次全量 V4 清单状态（236 checkpoints × 56 cells）',
        write_index=False)
    module['main']()
    return summary


def report_loop(interval: int) -> None:
    while True:
        summary = refresh_report()
        completed = summary['completed_components'] + summary['completed_full']
        if completed >= summary['expected_total']:
            return
        time.sleep(interval)


def worker_component(args) -> None:
    os.environ.update(THREADS)
    module = runpy.run_path(str(REPO / 'scripts/run_v2_progress_fleet_20261006.py'))
    module['OUTPUT'] = PAIRED
    module['worker'](Namespace(output=PAIRED, host=args.host, gpus=args.gpus,
        target_per_gpu=args.target_per_gpu, memory_target=.75, max_host_jobs=args.max_jobs,
        max_global_jobs=160, task_family='mixed', components=None))


def worker_full(args) -> None:
    os.environ.update(THREADS)
    module = runpy.run_path(str(REPO / 'scripts/run_v2_full_v4_20261007.py'))
    module['worker'].__globals__.update(PAIRED=PAIRED, ROOT=FULL, MONUSRC=MONU_SOURCE)
    module['worker'](Namespace(host=args.host, gpus=args.gpus, max_jobs=args.max_jobs,
                               ram_reserve=48, family='all'))


def launch() -> None:
    inventory()
    launches = [(host, 'component', [0]) for host in SINGLE_GPU_HOSTS]
    launches += [('3090-qi', 'full', list(range(8)))]
    for host, kind, gpus in launches:
        output = PAIRED if kind == 'component' else FULL
        log = OUT / 'logs' / f'{host}.{kind}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        mode = 'worker-component' if kind == 'component' else 'worker-full'
        cli = [REMOTE_PYTHON, '-u', str(Path(__file__).resolve()), mode,
               '--host', host, '--gpus', *map(str, gpus), '--max-jobs', str(max(5, len(gpus) * 5))]
        remote = f'cd {shlex.quote(str(REPO))} && setsid -f {shlex.join(cli)} >> {shlex.quote(str(log))} 2>&1 < /dev/null'
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, remote],
                                capture_output=True, text=True)
        print(json.dumps(dict(host=host, campaign=kind, gpus=gpus, returncode=result.returncode,
                              stderr=result.stderr.strip()[-300:])), flush=True)
        if result.returncode:
            continue
    local_log = OUT / 'logs/local5090.component.log'
    local_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker-component',
           '--host', 'server-local', '--gpus', '0', '3', '4', '5', '6', '7', '--max-jobs', '30']
    with local_log.open('a') as log:
        subprocess.Popen(cmd, cwd=REPO, env=dict(os.environ, **THREADS), stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(host='server-local', campaign='component', gpus=[0, 3, 4, 5, 6, 7],
                          log=str(local_log), state='STARTED')), flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('prepare')
    sub.add_parser('repair-full-sources')
    sub.add_parser('inventory')
    sub.add_parser('launch')
    sub.add_parser('report')
    report_loop_parser = sub.add_parser('report-loop')
    report_loop_parser.add_argument('--interval', type=int, default=900)
    for name in ('worker-component', 'worker-full'):
        worker = sub.add_parser(name)
        worker.add_argument('--host', required=True)
        worker.add_argument('--gpus', nargs='+', type=int, required=True)
        worker.add_argument('--max-jobs', type=int, default=40)
        worker.add_argument('--target-per-gpu', type=int, default=5)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'repair-full-sources':
        repair_full_sources()
    elif args.mode == 'inventory':
        inventory()
    elif args.mode == 'launch':
        launch()
    elif args.mode == 'report':
        refresh_report()
    elif args.mode == 'report-loop':
        report_loop(args.interval)
    elif args.mode == 'worker-component':
        worker_component(args)
    else:
        worker_full(args)


if __name__ == '__main__':
    main()
