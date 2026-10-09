#!/usr/bin/env python3
"""Evaluate only the newly trained repaired 100TB teacher on admitted v4 ID cells."""

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
TRAIN = REPO / 'outputs/01_training_runs/HS6_L_100tb_global_repaired1m_biosafe256_gb1024_e1_single3090_20260928'
ROOT = REPO / 'outputs/02_eval_runs/hs6_l_100tb_global_repaired1m_single3090_v4_id_20260928'
SHARED = ROOT / 'shared'
EXTENSION = ROOT / 'extension'
REFERENCE = REPO / 'outputs/02_eval_inputs/shared_fleet_reference_20260930/corrected_campaign_manifest.json'
SOURCE = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918')
PY = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
ARM = 'hs6_l_100tb_global_repaired1m_e1_ck976'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def await_new_checkpoint(wait):
    exit_path = TRAIN / 'exit.json'
    while wait and not exit_path.is_file():
        if subprocess.run(['tmux', 'has-session', '-t', 'hs6_100tb_repaired_1m_3090_20260928'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError('Training tmux session disappeared before exit record')
        metrics_path = TRAIN / 'raw_loss_metrics.jsonl'
        updates = sum(1 for line in metrics_path.open() if line.strip()) if metrics_path.is_file() else 0
        save(ROOT / 'RUN_STATUS.json', dict(time_unix=time.time(), state='TRAINING',
             current_optimizer_updates=updates, target_optimizer_updates=977,
             effective_batch=1024, training_root=str(TRAIN),
             v4_id_state='WAITING_FOR_NEW_CHECKPOINT',
             shared_planned=38, extension_planned=10))
        time.sleep(60)
    if not exit_path.is_file():
        raise RuntimeError('Training has not finished')
    exit_record = json.loads(exit_path.read_text())
    if exit_record.get('returncode') != 0:
        raise RuntimeError(f'Training failed: {exit_record}')
    checkpoint = TRAIN / 'eval/training_976/teacher_checkpoint.pth'
    config = TRAIN / 'config.yaml'
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000 or not config.is_file():
        raise RuntimeError('Fresh step 976 teacher checkpoint or config missing')
    metrics_path = TRAIN / 'raw_loss_metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if len(metrics) != 977 or [row['optimizer_update'] for row in metrics] != list(range(977)):
        raise RuntimeError(f'Expected exactly 977 fresh optimizer updates, got {len(metrics)}')
    last = metrics[-1]
    if last['effective_global_batch_size'] != 1024 or last['optimizer_updates_completed'] != 977:
        raise RuntimeError('Optimizer batch/update provenance mismatch')
    if '100tb_final_1m_repaired_20260928' not in last['dataset_path']:
        raise RuntimeError('Wrong training data in final metric')
    import yaml
    cfg = yaml.safe_load(config.read_text())
    if cfg['train']['batch_size_per_gpu'] != 8 or cfg['optim']['gradient_accumulation_steps'] != 128:
        raise RuntimeError('Training config batch mismatch')
    if cfg['train']['OFFICIAL_EPOCH_LENGTH'] != 977 or cfg['optim']['epochs'] != 1:
        raise RuntimeError('Training config epoch mismatch')
    return checkpoint, config


def prepare_shared(checkpoint, config):
    if (SHARED / 'campaign_manifest.json').exists():
        raise FileExistsError('Shared v4 campaign already exists')
    ref = json.loads(REFERENCE.read_text())
    spec = importlib.util.spec_from_file_location('v4_completion_repaired', REPO / 'scripts/run_fourmodel_v4_completion_20260924.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asset = dict(arm=ARM, checkpoint_id='976', path=str(checkpoint), config=str(config),
                 kind='dinov3', model_id='', reserve_mib=4096)
    checkpoint_sha = module.sha256(checkpoint)
    config_sha = module.sha256(config)
    ref.pop('execution_hosts', None)
    ref.update(campaign_scope='V4_ID_SHARED_V3_COMPONENTS_ONLY',
               explicit_user_authorization='2026-09-28 repaired 100TB 1M, one RTX 3090, one epoch and v4 evaluation',
               created_unix=time.time(), checkpoint_assets=[asset],
               checkpoint_teacher_sha256={ARM: checkpoint_sha},
               checkpoint_config_sha256={ARM: config_sha},
               old_results_relabelled=False, legacy_reuse=False,
               online_checkpoints=False, v4_aggregate_allowed=False,
               full_v3_aggregate_allowed=False)
    ref['external_source_hashes'] = {
        str(REPO / 'Evaluation Rules/protocol_v4.json'): module.sha256(REPO / 'Evaluation Rules/protocol_v4.json'),
        str(Path(__file__).resolve()): module.sha256(Path(__file__).resolve()),
        str(SOURCE / 'scripts/run_retest_fleet_20260918.py'): module.sha256(SOURCE / 'scripts/run_retest_fleet_20260918.py'),
    }
    specs = [row for row in ref['datasets'] if row['status'] == 'PASS' and row['task'] not in ('ood', 'cell_tracking')]
    ref['tasks'] = [dict(key=f"{ARM}_ck976__{row['task']}__{row['dataset']}" +
                         (f"__primary-last__{row['split']}" if row['task'] == 'segmentation' else ''),
                         asset=asset, dataset=row) for row in specs]
    ref['inventory'] = [row for row in ref['inventory'] if row['task'] not in ('ood', 'cell_tracking')]
    save(SHARED / 'campaign_manifest.json', ref)
    save(ROOT / 'provenance.json', dict(training_manifest=str(TRAIN / 'launch_manifest.json'),
         training_exit=str(TRAIN / 'exit.json'), training_metric_rows=977,
         checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
         config=str(config), config_sha256=config_sha,
         protocol='bio-eval-union-v4', scope='ID only',
         shared_task_count=len(ref['tasks']), source_snapshot=str(SOURCE),
         reference_manifest=str(REFERENCE)))
    log(f'Prepared {len(ref["tasks"])} inherited v4 ID tasks, checkpoint sha256={checkpoint_sha}')
    return module, checkpoint_sha, config_sha


def prepare_extension(module, checkpoint, config, checkpoint_sha, config_sha):
    module.ROOT = EXTENSION
    asset = dict(arm=ARM, checkpoint_id=976, checkpoint=checkpoint,
                 config=config, checkpoint_sha256=checkpoint_sha, shared_evidence=SHARED)
    module.ASSETS = (asset,)
    for path, expected in module.LOCKED_INPUTS.items():
        if module.sha256(path) != expected:
            raise RuntimeError(f'Locked v4 input changed: {path}')
    builders = [
        lambda order: module.retrieval_task(asset, 'nct-crc-he-100', order),
        lambda order: module.frozen_task(asset, 'conic-cell-count', order),
        lambda order: module.monuseg_task(asset, order),
        lambda order: module.rxrx3_task(asset, order),
        lambda order: module.detection_task(asset, 'bbbc038', order),
        lambda order: module.frozen_task(asset, 'lc25000', order),
        lambda order: module.retrieval_task(asset, 'lc25000', order),
        lambda order: module.frozen_task(asset, 'livecell-cell-count', order),
        lambda order: module.detection_task(asset, 'conic', order),
        lambda order: module.detection_task(asset, 'livecell', order),
    ]
    tasks = []
    for order, builder in enumerate(builders, 1):
        task = builder(order)
        task.update(protocol_id=module.PROTOCOL_ID, checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_sha, config=str(config),
                    config_sha256=config_sha, source_entry_sha256=module.source_digest_for(task))
        tasks.append(task)
        module.atomic_json(EXTENSION / 'tasks' / f"{task['id']}.json", task)
        if task['done_kind'] == 'rxrx3_json':
            module.atomic_json(Path(task['output']) / 'campaign_manifest.json', dict(
                protocol_id=module.PROTOCOL_ID, protocol_sha256=module.sha256(module.PROTOCOL),
                fixed_split_protocol_id=module.RXRX3_PROTOCOL,
                fixed_split_sha256=module.LOCKED_INPUTS[str(module.RXRX3_CACHE / 'split_manifest.jsonl')],
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                config=str(config), config_sha256=config_sha,
                teacher_branch=True, batch_size=64, query=734, gallery=734,
                created_utc=module.now()))
    module.atomic_json(EXTENSION / 'campaign_manifest.json', dict(
        protocol_id=module.PROTOCOL_ID, protocol_sha256=module.sha256(module.PROTOCOL),
        scope='admitted v4 ID extensions only', checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha, config=str(config), config_sha256=config_sha,
        task_ids=[task['id'] for task in tasks], source_snapshot=str(module.SOURCE)))
    log(f'Prepared {len(tasks)} v4 ID extension tasks')


def collect_id_results(checkpoint, checkpoint_sha):
    shared_manifest = json.loads((SHARED / 'campaign_manifest.json').read_text())
    shared_rows = []
    for task in shared_manifest['tasks']:
        directory = SHARED / 'cells' / task['key']
        report = directory / 'validation_report.json'
        result_path = directory / 'component_result.json'
        if not report.is_file():
            continue
        validation = json.loads(report.read_text())
        if validation.get('status') != 'VALID_COMPLETE':
            continue
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            sources = ([row.get('checkpoint') for row in result['rows']]
                       if isinstance(result.get('rows'), list) else [result.get('checkpoint')])
            if not sources or any(source != str(checkpoint) for source in sources):
                raise RuntimeError(f'Shared result came from another checkpoint: {result_path}')
            metric_paths = [str(result_path)]
        elif task['dataset']['task'] == 'segmentation' and len(validation.get('result_sha256', {})) == 6:
            result = {}
            metric_paths = sorted(validation['result_sha256'])
        else:
            continue
        shared_rows.append(dict(task=task['dataset']['task'], dataset=task['dataset']['dataset'],
                                split=task['dataset'].get('split'), result=metric_paths,
                                metrics={key: value for key, value in result.items()
                                         if isinstance(value, (int, float)) and key not in
                                         ('batch_size', 'seed', 'image_size', 'resize_size', 'n_train', 'n_test', 'n_samples', 'n_classes')}))
    extension_rows = []
    for task_path in sorted((EXTENSION / 'tasks').glob('*.json')):
        task = json.loads(task_path.read_text())
        report = Path(task['output']) / 'validation_report.json'
        if not report.is_file():
            continue
        data = json.loads(report.read_text())
        if data.get('status') != 'VALID_COMPLETE' or data.get('checkpoint_sha256') != checkpoint_sha:
            raise RuntimeError(f'Invalid extension provenance: {report}')
        extension_rows.append(dict(task=task['family'], dataset=task['dataset'],
                                   output=task['output'], validation=data['validation']))
    expected_shared = len(shared_manifest['tasks'])
    expected_extension = len(list((EXTENSION / 'tasks').glob('*.json')))
    summary = dict(protocol='bio-eval-union-v4', scope='ID only',
                   checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                   shared_completed=len(shared_rows), shared_expected=expected_shared,
                   extension_completed=len(extension_rows), extension_expected=expected_extension,
                   complete=len(shared_rows) == expected_shared and len(extension_rows) == expected_extension,
                   shared_results=shared_rows, extension_results=extension_rows,
                   created_unix=time.time())
    save(ROOT / 'ID_RESULTS.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    try:
        checkpoint, config = await_new_checkpoint(args.wait)
        module, checkpoint_sha, config_sha = prepare_shared(checkpoint, config)
        prepare_extension(module, checkpoint, config, checkpoint_sha, config_sha)
        save(ROOT / 'RUN_STATUS.json', dict(time_unix=time.time(), state='V4_ID_EVALUATION',
             checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
             current_optimizer_updates=977, target_optimizer_updates=977,
             shared_planned=38, extension_planned=10))
        cmd = [PY, '-u', str(SOURCE / 'scripts/run_retest_fleet_20260918.py'), 'worker',
               '--output', str(SHARED), '--host', platform.node(), '--gpus', '0',
               '--target-per-gpu', '1', '--max-host-jobs', '1', '--max-global-jobs', '1']
        with (ROOT / 'shared_worker.log').open('w') as log_file:
            result = subprocess.run(cmd, cwd=SOURCE, stdout=log_file, stderr=subprocess.STDOUT)
        save(ROOT / 'shared_worker_exit.json', dict(returncode=result.returncode, command=cmd))
        log(f'Shared worker exited rc={result.returncode}; starting ID extensions')
        with (ROOT / 'extension_worker.log').open('w') as log_file:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
            result2 = subprocess.run([PY, '-u', '-c',
                "import importlib.util; p='/mnt/huawei_deepcad/dinov3/scripts/run_fourmodel_v4_completion_20260924.py'; "
                "s=importlib.util.spec_from_file_location('v4',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"m.ROOT=__import__('pathlib').Path('{EXTENSION}'); "
                f"m.ASSETS=({{'arm':'{ARM}'}},); m.worker('repaired100tb',14000)"],
                cwd=REPO, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        save(ROOT / 'extension_worker_exit.json', dict(returncode=result2.returncode))
        log(f'Extension worker exited rc={result2.returncode}')
        summary = collect_id_results(checkpoint, checkpoint_sha)
        save(ROOT / 'RUN_STATUS.json', dict(time_unix=time.time(), state='COMPLETE' if summary['complete'] else 'PARTIAL',
             checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
             shared_completed=summary['shared_completed'], shared_planned=summary['shared_expected'],
             extension_completed=summary['extension_completed'], extension_planned=summary['extension_expected']))
        log(f"ID results: shared {summary['shared_completed']}/{summary['shared_expected']}, "
            f"extension {summary['extension_completed']}/{summary['extension_expected']}")
        return 0 if result.returncode == result2.returncode == 0 and summary['complete'] else 1
    except Exception as error:
        save(ROOT / 'pipeline_error.json', dict(type=type(error).__name__, error=str(error), time=time.time()))
        save(ROOT / 'RUN_STATUS.json', dict(time_unix=time.time(), state='ERROR',
             error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
