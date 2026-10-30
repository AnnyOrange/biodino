#!/usr/bin/env python3
"""Queue complete Union-v4 inventories against pinned ViT-L checkpoints."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/suxin/project/dinov3')
sys.path.insert(0, str(REPO))
PROFILE = os.environ.get('DINOV3_1TB_EVAL_PROFILE', 'legacy')
if PROFILE == 'biosafe256':
    RUN = Path('/data_2/suxin/runs/HS6_L_1TB_biosafe256_15374_to80000_2xH100_20261020')
    SOURCE = RUN
    START_CHECKPOINT = 15374
    FULL_CHECKPOINTS = (15374, 19999, 29999, 39999, 49999, 59999, 69999, 79999)
    CHECKPOINTS = (15374, *range(15999, 80000, 1000))
    PARALLEL_SLOTS = {'0': 14, '1': 4}
    EVAL_GPUS = (0, 1)
    MODEL_TAG = 'hs6_l_1tb'
elif PROFILE == 'legacy':
    RUN = Path('/data_2/suxin/runs/HS6_L_1TB_40000_to80000_2xH100_DDP_20261019')
    SOURCE = Path('/data_2/suxin/runs/HS6_L_1TB_ep15_to40000_2xH100_20261015')
    START_CHECKPOINT = 39999
    FULL_CHECKPOINTS = (39999, 49999, 59999, 69999, 79999)
    CHECKPOINTS = (15374, *range(25999, 80000, 1000))
    PARALLEL_SLOTS = {'0': 12, '1': 4}
    EVAL_GPUS = (0, 1)
    MODEL_TAG = 'hs6_l_1tb'
elif PROFILE == '5tb_nogram':
    RUN = Path('/data_2/suxin/runs/HS6_L_5TB_nogram_41479_to80000_4xH100_20261026')
    SOURCE = RUN
    START_CHECKPOINT = 41479
    CHECKPOINTS = (41479, *(step - 1 for step in range(41663, 61471, 683)),
                   *range(61999, 80000, 1000))
    FULL_CHECKPOINTS = (41479, 61469, 79999)
    EVAL_GPUS = (0, 1, 4, 5, 6, 7)
    PARALLEL_SLOTS = {'0': 1, '1': 1, '4': 2, '5': 2, '6': 2, '7': 2}
    MODEL_TAG = 'hs6_l_5tb_nogram'
else:
    raise ValueError(f'Unknown DINOV3_1TB_EVAL_PROFILE: {PROFILE}')
PROFILE_PREFIX = ('--profile-token', PROFILE) if PROFILE == '5tb_nogram' else ()
ROOT = RUN / 'eval/union_v4'
MULTIMODAL_V2_PROTOCOL = 'neurips22-author-public52-v2'
MULTIMODAL_V2_ROOT = ROOT / MULTIMODAL_V2_PROTOCOL
MULTIMODAL_V2_READINESS = REPO / 'outputs/02_eval_runs/unified_paper/multimodal_official_readiness.json'
BENCHMARK = Path('/data_2/suxin/test-dataset')
PYTHON = '/data_2/suxin/envs/dinov3/bin/python'
PROTOCOL = REPO / 'Evaluation Rules/protocol_v4.json'
CURVE_TASK_IDS = {
    'classification__bloodmnist', 'classification__pathmnist',
    'regression__bbbc005', 'segmentation__cellpose', 'segmentation__monuseg',
    'classification__organcmnist', 'classification__dermamnist',
    'classification__pcam', 'classification__chammi-allen-task1',
    'retrieval_clustering__nct-crc-he-1k', 'retrieval_clustering__rxrx3-core',
    'detection_proxy__bbbc038',
}
PANNUKE = (
    'pannuke-fold1-train-fold2-val-fold3-test',
    'pannuke-fold2-train-fold1-val-fold3-test',
    'pannuke-fold3-train-fold2-val-fold1-test',
)
MONUSEG_MANIFEST = 'monuseg37_v4_20261012.json'
MONUSEG_SHA256 = '6b71c7d02c3031558ceeb2b17eb57c1916520d55807249668696c061550688e7'
MONUSEG_COUNTS = {'train': 30, 'val': 7, 'test': 14}
RXRX3_CACHE = Path('/data_2/suxin/eval_inputs/formal_v3/rxrx3-core')
RXRX3_SPLIT_SHA256 = '94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5'
GPU1_CLASSIFICATION = {
    'organcmnist', 'dermamnist', 'pneumoniamnist', 'chestmnist',
    'cyclops-protein-loc', 'pcam', 'chammi-allen-task1', 'chammi-cp-task1',
    'chammi-cp-task3', 'chammi-hpa-task2',
}


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def child_args(mode, *args):
    return (*PROFILE_PREFIX, mode, *args)


def gpu_ready(gpu):
    if PROFILE != '5tb_nogram':
        return True
    result = subprocess.run(
        ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free_mib = int(result.stdout.strip())
    return free_mib >= (16000 if gpu in (0, 1) else 20000)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def inventory(protocol):
    tasks = []
    for family in ('classification', 'regression'):
        names = (protocol['tier_a'].get(family, []) + protocol['tier_b'].get(family, [])
                 + protocol['union_extension'].get(family, []))
        tasks.extend(dict(family=family, dataset=name,
                          gpu=1 if family == 'classification' and name in GPU1_CLASSIFICATION else 0)
                     for name in names)
    names = (protocol['tier_a']['retrieval'] + protocol['tier_b']['retrieval']
             + protocol['union_extension']['retrieval'])
    tasks.extend(dict(family='retrieval_clustering', dataset=name, gpu=1) for name in names)
    names = protocol['tier_a']['segmentation'] + protocol['tier_b']['segmentation']
    for name in names:
        for split in PANNUKE if name == 'pannuke' else (None,):
            tasks.append(dict(family='segmentation', dataset=name, split=split, gpu=0))
    tasks.extend(dict(family='detection_proxy', dataset=name, gpu=1)
                 for name in protocol['union_extension']['detection_proxy'])
    tasks.extend(dict(family='cell_tracking', dataset=name, gpu=1, blocker='Fixed native head/linker pending')
                 for name in protocol['tier_b']['cell_tracking'])
    tasks.extend(dict(family='ood', dataset=name, gpu=1, blocker='Common OOD selection/config freeze pending')
                 for name in protocol['tier_b']['ood'])
    for task in tasks:
        if task['dataset'] == 'rxrx3-core' and not (RXRX3_CACHE / 'validation.json').is_file():
            task['blocker'] = 'Fixed RxRx3 cache and split manifest absent on H100'
        if task['family'] == 'segmentation' and task['dataset'] == 'monuseg' and not (
                BENCHMARK / 'segmentation/monuseg/extracted' / MONUSEG_MANIFEST).is_file():
            task['blocker'] = 'Active 2026-10-12 MoNuSeg 30/7/14 manifest absent on H100'
        task['id'] = task['family'] + '__' + task['dataset'] + ('__' + task['split'] if task.get('split') else '')
    if PROFILE == '5tb_nogram':
        light = [task for task in tasks if task['family'] in ('classification', 'regression')]
        heavy = [task for task in tasks if task['family'] not in ('classification', 'regression')]
        for index, task in enumerate(light):
            task['gpu'] = EVAL_GPUS[index % len(EVAL_GPUS)]
        for index, task in enumerate(heavy):
            task['gpu'] = (4, 5, 6, 7)[index % 4]
            if task['id'] == 'segmentation__multimodal_cellseg':
                task['blocker'] = 'Corrected Multimodal-v2 campaign owns this task'
    counts = protocol['expected_unique_dataset_counts']
    assert len([t for t in tasks if t['family'] == 'classification']) == counts['classification']
    assert len([t for t in tasks if t['family'] == 'regression']) == counts['regression']
    assert len([t for t in tasks if t['family'] == 'retrieval_clustering']) == counts['retrieval']
    assert len([t for t in tasks if t['family'] == 'segmentation']) == counts['segmentation'] + 2
    assert len([t for t in tasks if t['family'] == 'detection_proxy']) == counts['detection_proxy']
    return tasks


def prepare():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['protocol_id'] != 'bio-eval-union-v4':
        raise RuntimeError('Wrong evaluation protocol')
    path = ROOT / 'manifest.json'
    if path.exists():
        raise RuntimeError(f'Campaign already exists: {path}')
    tasks = inventory(protocol)
    source_checkpoint = (RUN / f'ckpt/{START_CHECKPOINT}/checkpoint.pth'
                         if PROFILE != 'legacy' else SOURCE / 'ckpt/39999/checkpoint.pth')
    source_config = (RUN / 'source_config.yaml' if PROFILE != 'legacy'
                     else SOURCE / 'config.yaml')
    manifest = dict(protocol_id=protocol['protocol_id'], protocol_sha256=digest(PROTOCOL),
                    training_profile=PROFILE,
                    original_training_run=(
                        '/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/'
                        'HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818'
                        if PROFILE == 'biosafe256' else None),
                    checkpoints=CHECKPOINTS, full_checkpoints=CHECKPOINTS,
                    milestone_checkpoints=FULL_CHECKPOINTS,
                    evaluation_scope='complete_union_v4_every_available_checkpoint',
                    parallel_slots=PARALLEL_SLOTS,
                    curve_task_ids=sorted(CURVE_TASK_IDS),
                    source_checkpoint_iteration=START_CHECKPOINT,
                    source_checkpoint_sha256=digest(source_checkpoint),
                    source_training_config_sha256=digest(source_config),
                    source=SOURCE.as_posix(), training_run=RUN.as_posix(),
                    benchmark=BENCHMARK.as_posix(), tasks=tasks,
                    frozen_batch=64, dense_feature_batch=32, dense_probe_batch=32,
                    detection_batch=8, seed=0, autocast_dtype='bf16',
                    preliminary_results=(str(RUN / 'eval/ck39999') if PROFILE == 'legacy' else None),
                    no_all_union_aggregate=True,
                    admission_notes={'classification/lc25000': 'PROVISIONAL_LEGACY_ONLY',
                                     'retrieval/nct-crc-he-100': 'LOW_N_AUDIT_REQUIRED'},
                    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(path, manifest)
    pinned = ROOT / f'checkpoints/{START_CHECKPOINT}/checkpoint.pth'
    pinned.parent.mkdir(parents=True, exist_ok=True)
    os.link(source_checkpoint, pinned)
    shutil.copy2(source_config, pinned.parent / 'config.yaml')
    write_json(pinned.parent / 'identity.json', dict(iteration=START_CHECKPOINT,
               bytes=pinned.stat().st_size, sha256=manifest['source_checkpoint_sha256'],
               config_sha256=digest(pinned.parent / 'config.yaml')))
    print(f'Prepared {len(tasks)} tasks per checkpoint across {len(CHECKPOINTS)} checkpoints', flush=True)


def extend():
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    if digest(PROTOCOL) != manifest['protocol_sha256']:
        raise RuntimeError('Protocol changed since campaign preparation')
    tasks = {task['id']: task for task in manifest['tasks']}
    if not CURVE_TASK_IDS <= tasks.keys():
        raise RuntimeError(f'Missing curve tasks: {CURVE_TASK_IDS - tasks.keys()}')
    counts = {gpu: sum(tasks[task_id]['gpu'] == gpu for task_id in CURVE_TASK_IDS) for gpu in (0, 1)}
    if counts != {0: 5, 1: 7}:
        raise RuntimeError(f'Unexpected curve GPU assignment: {counts}')
    manifest['checkpoints'] = CHECKPOINTS
    manifest['full_checkpoints'] = CHECKPOINTS
    manifest['milestone_checkpoints'] = FULL_CHECKPOINTS
    manifest['evaluation_scope'] = 'complete_union_v4_every_available_checkpoint'
    manifest['parallel_slots'] = PARALLEL_SLOTS
    manifest['curve_task_ids'] = sorted(CURVE_TASK_IDS)
    write_json(path, manifest)
    print(f'Full {len(manifest["tasks"])}-task Union-v4 on {len(CHECKPOINTS)} checkpoints; '
          f'GPU 0 has {PARALLEL_SLOTS["0"]} concurrent slots, GPU 1 has {PARALLEL_SLOTS["1"]}', flush=True)


def pin():
    for iteration in CHECKPOINTS:
        if PROFILE == 'biosafe256':
            source = RUN / f'ckpt/{iteration}/checkpoint.pth'
            config = RUN / ('source_config.yaml' if iteration == 15374 else 'config.yaml')
        elif PROFILE == '5tb_nogram':
            source = RUN / f'ckpt/{iteration}/checkpoint.pth'
            config = (RUN / 'source_config.yaml' if iteration == START_CHECKPOINT else
                      ROOT / 'config_original_ddp.yaml' if iteration <= 61469 else
                      RUN / 'config.yaml')
        elif iteration == 15374:
            source = SOURCE / 'source/checkpoint_15374.pth'
            config = SOURCE / 'source/original_config.yaml'
        elif iteration <= 39999:
            source = SOURCE / f'ckpt/{iteration}/checkpoint.pth'
            config = SOURCE / 'config.yaml'
        else:
            source = RUN / f'ckpt/{iteration}/checkpoint.pth'
            config = RUN / 'config.yaml'
        target = ROOT / f'checkpoints/{iteration}/checkpoint.pth'
        if target.exists():
            continue
        while not source.is_file():
            time.sleep(20)
        while True:
            before = source.stat()
            time.sleep(20)
            after = source.stat()
            if after.st_size >= 3_000_000_000 and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                break
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        shutil.copy2(config, target.parent / 'config.yaml')
        write_json(target.parent / 'identity.json', dict(iteration=iteration, bytes=after.st_size,
                   sha256=digest(target), config_sha256=digest(target.parent / 'config.yaml'),
                   source=str(source)))
        print(f'PINNED {iteration} {target}', flush=True)


def command(task, iteration, output):
    checkpoint = ROOT / f'checkpoints/{iteration}/checkpoint.pth'
    config = checkpoint.parent / 'config.yaml'
    common = ['--checkpoint', str(checkpoint), '--train-config', str(config),
              '--benchmark-root', str(BENCHMARK)]
    family, dataset = task['family'], task['dataset']
    if family in ('classification', 'regression'):
        return [PYTHON, '-u', '-m', 'dinov3.eval.bio_frozen_eval.run_classification', *common,
                '--datasets', dataset, '--output-dir', str(output), '--model-name', f'{MODEL_TAG}_ck{iteration}',
                '--batch-size', '64', '--num-workers', '2', '--seed', '0', '--n-last-blocks', '1',
                '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
                '--resolution-protocol', 'best', '--split-protocol', 'current', '--save-paths',
                '--no-save-features']
    if family == 'retrieval_clustering':
        if dataset == 'rxrx3-core':
            return [PYTHON, '-u', str(REPO / 'scripts/run_external4_fixedbudget_model.py'),
                    '--model', f'{MODEL_TAG}_ck{iteration}', '--campaign', str(output),
                    '--cache-root', str(RXRX3_CACHE.parent), '--datasets', 'rxrx3',
                    '--checkpoint', str(checkpoint), '--train-config', str(config),
                    '--device', 'cuda:0', '--batch-size', '64']
        return [PYTHON, '-u', '-m', 'dinov3.eval.bio_frozen_eval.run_retrieval_clustering', *common,
                '--datasets', dataset, '--output-dir', str(output), '--model-name', f'{MODEL_TAG}_ck{iteration}',
                '--batch-size', '64', '--num-workers', '2', '--seed', '0', '--n-last-blocks', '1',
                '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
                '--metric-device', 'cpu', '--no-save-features']
    if family == 'detection_proxy':
        return [PYTHON, '-u', '-m', 'dinov3.eval.bio_detection.center_probe', *common,
                '--dataset', dataset, '--output-dir', str(output), '--batch-size', '8',
                '--num-workers', '2', '--image-size', '224', '--epochs', '5', '--lr', '0.001',
                '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--seed', '0',
                '--max-samples-per-split', '0', '--conic-split-protocol', 'official-baseline-fold0-nested-v1']
    if family == 'segmentation':
        recipe = json.loads(PROTOCOL.read_text())['segmentation'][dataset]
        split = task.get('split_protocol') or task.get('split') or (
            'official-baseline-fold0-nested-v1' if dataset == 'conic' else 'formal-v1')
        seeds = task.get('seeds', (0, 1, 2))
        return [PYTHON, '-u', '-m', 'dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline',
                '--datasets', dataset, '--checkpoint-file', str(checkpoint), '--checkpoint-id', str(iteration),
                '--train-config', str(config), '--data-root-base', str(BENCHMARK / 'segmentation'),
                '--protocol', 'manual', '--dataset-split-protocol', split,
                '--feature-img-size', str(recipe['resolution']), '--resize-mode', recipe['resize'],
                '--layer-preset', 'last1', '--feature-batch-size', '32', '--feature-num-workers', '2',
                '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
                '--probe-epoch-grid', '20', '50', '--probe-seeds', *[str(seed) for seed in seeds],
                '--probe-batch-size', '32', '--probe-lr', '0.001', '--probe-weight-decay', '0.0001',
                '--probe-eval-every', '1', '--probe-num-workers', '0', '--gpu', '0',
                '--probe-class-weight-mode', recipe['class_weight'], '--chunked-cache', '--no-compress-cache',
                '--cache-root', str(output / 'cache'), '--output-root', str(output / 'results'),
                '--run-name', f'{MODEL_TAG}_ck{iteration}_v4']
    raise ValueError(f'No command for {family}')


def validate(task, output):
    family, dataset = task['family'], task['dataset']
    if family == 'retrieval_clustering' and dataset == 'rxrx3-core':
        path = output / 'models' / output.parent.name.replace('ck', f'{MODEL_TAG}_ck') / 'results.json'
        result = json.loads(path.read_text())
        row = result['tests']['rxrx3']
        if result['status'] != 'VALID_COMPLETE' or result['batch_size'] != 64 or result['teacher_branch'] != 'teacher':
            raise ValueError('RxRx3 encoder/recipe mismatch')
        if row['status'] != 'FORMAL' or row['protocol_id'] != 'crispr-query-guide-plate-disjoint-all-eligible-genes-v1':
            raise ValueError('RxRx3 fixed protocol mismatch')
        if (row['n_gallery'], row['n_query']) != (734, 734):
            raise ValueError('RxRx3 fixed query/gallery count mismatch')
        if not all(math.isfinite(float(row[metric])) for metric in ('recall_at_1', 'nmi')):
            raise ValueError('RxRx3 retrieval/clustering metrics are non-finite')
        return {'recall_at_1': row['recall_at_1'], 'nmi': row['nmi']}
    if family in ('classification', 'regression', 'retrieval_clustering'):
        with (output / 'summary.csv').open(newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if row.get('dataset') == dataset and not row.get('error')]
        if family == 'retrieval_clustering':
            retrieval_rows = [row for row in rows if row.get('recall_at_1')]
            clustering_rows = [row for row in rows if row.get('nmi')]
            if not retrieval_rows or not clustering_rows:
                raise ValueError('missing retrieval or clustering result')
            if not all(math.isfinite(float(row['recall_at_1'])) for row in retrieval_rows):
                raise ValueError('non-finite retrieval recall')
            if not all(math.isfinite(float(row['nmi'])) for row in clustering_rows):
                raise ValueError('non-finite clustering NMI')
            return {'dataset': dataset, 'retrieval_rows': len(retrieval_rows),
                    'clustering_rows': len(clustering_rows),
                    'recall_at_1': retrieval_rows[0]['recall_at_1'], 'nmi': clustering_rows[0]['nmi']}
        if len(rows) != 1:
            raise ValueError(f'expected one successful summary row, got {len(rows)}')
        row = rows[0]
        metric = ('r2' if family == 'regression'
                  else 'macro_average_precision' if row['task'] == 'multilabel_classification'
                  else 'balanced_accuracy')
        if not math.isfinite(float(row[metric])):
            raise ValueError(f'non-finite {metric}')
        if row['batch_size'] != '64':
            raise ValueError('frozen batch differs from 64')
        return {key: row[key] for key in (metric, 'dataset')}
    if family == 'detection_proxy':
        row = json.loads((output / 'results_bio_detection.json').read_text())
        if row['dataset'] != dataset or row['batch_size'] != 8 or row['epochs'] != 5:
            raise ValueError('detection recipe mismatch')
        if not math.isfinite(float(row['test_patch_f1'])):
            raise ValueError('non-finite detection F1')
        return {'test_patch_f1': row['test_patch_f1']}
    paths = list((output / 'results').rglob('results.json'))
    signatures = set()
    dice = []
    for path in paths:
        row = json.loads(path.read_text())
        meta = row['_meta']
        if meta['probe_batch_size'] != 32 or meta['probe_eval_every'] != 1 or meta['test_evaluations'] != 1:
            raise ValueError('segmentation recipe mismatch')
        if not math.isfinite(float(row['test']['mDice'])):
            raise ValueError('non-finite segmentation mDice')
        dice.append(float(row['test']['mDice']))
        if dataset == 'monuseg' and (meta['full_train_samples'], meta['used_train_samples']) != (30, 30):
            raise ValueError('MoNuSeg must use all 30 official training samples')
        signatures.add((meta['probe_epochs'], meta['seed']))
    if signatures != {(epoch, seed) for epoch in (20, 50) for seed in (0, 1, 2)}:
        raise ValueError(f'incomplete segmentation fits: {sorted(signatures)}')
    return {'fits': 6, 'mDice': sum(dice) / len(dice)}


def refresh_curves(manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (ROOT / 'curves.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tasks = {task['id']: task for task in manifest['tasks']}
        rows = []
        for iteration in CHECKPOINTS:
            for task_id in sorted(tasks):
                state_path = ROOT / f'states/ck{iteration}/{task_id}.json'
                if not state_path.is_file():
                    continue
                state = json.loads(state_path.read_text())
                if state['status'] != 'VALID_COMPLETE':
                    continue
                task = tasks[task_id]
                metrics = state['metrics']
                if task['family'] == 'segmentation' and 'mDice' not in metrics:
                    metrics = validate(task, ROOT / f'ck{iteration}/{task_id}')
                metric_keys = ([key for key in ('recall_at_1', 'nmi') if key in metrics]
                               if task['family'] == 'retrieval_clustering' else
                               [key for key in ('balanced_accuracy', 'macro_average_precision',
                                                'r2', 'mDice', 'test_patch_f1') if key in metrics])
                if not metric_keys:
                    raise ValueError(f'No curve metric for {iteration}/{task_id}')
                for metric in metric_keys:
                    family = 'clustering' if metric == 'nmi' else task['family']
                    rows.append(dict(step=iteration + 1, checkpoint=iteration, gpu=task['gpu'],
                                     task_id=task_id, family=family, dataset=task['dataset'],
                                     metric=metric, value=float(metrics[metric])))
        csv_path = ROOT / 'curves.csv'
        temp_csv = csv_path.with_suffix('.csv.tmp')
        with temp_csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('step', 'checkpoint', 'gpu', 'task_id',
                                                         'family', 'dataset', 'metric', 'value'))
            writer.writeheader()
            writer.writerows(rows)
        temp_csv.replace(csv_path)

        families = ('classification', 'regression', 'retrieval_clustering',
                    'clustering', 'segmentation', 'detection_proxy')
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for family, ax in zip(families, axes.flat):
            series = {}
            for row in rows:
                if row['family'] == family and row['task_id'] in CURVE_TASK_IDS:
                    series.setdefault((row['dataset'], row['metric']), []).append(row)
            for (dataset, metric), points in series.items():
                ax.plot([point['step'] / 1000 for point in points],
                        [point['value'] for point in points], marker='o', markersize=3,
                        linewidth=1.3, label=f'{dataset} ({metric})')
            ax.set_title(family.replace('_', ' ').title())
            ax.set_xlabel('Training step (thousands)')
            ax.grid(alpha=0.25)
            if series:
                ax.legend(fontsize=7)
        fig.suptitle(f'{MODEL_TAG} representative validation curves', fontsize=14)
        temp_png = ROOT / 'curves.tmp.png'
        fig.savefig(temp_png, dpi=140)
        plt.close(fig)
        temp_png.replace(ROOT / 'curves.png')


def evaluate_task(gpu, task, iteration, manifest, checkpoint, state_path):
    output = ROOT / f'ck{iteration}/{task["id"]}'
    if output.exists():
        try:
            metrics = validate(task, output)
            write_json(state_path, dict(status='VALID_COMPLETE', gpu=gpu,
                       reason='adopted verified existing output', metrics=metrics, output=str(output)))
            return
        except (FileNotFoundError, KeyError, ValueError):
            pass
    if task.get('blocker'):
        write_json(state_path, dict(status='BLOCKED_NOT_TESTED', reason=task['blocker']))
        return
    output.mkdir(parents=True, exist_ok=True)
    args = command(task, iteration, output)
    identity = json.loads((checkpoint.parent / 'identity.json').read_text())
    write_json(output / 'invocation.json', dict(protocol_id=manifest['protocol_id'],
               protocol_sha256=manifest['protocol_sha256'], checkpoint=str(checkpoint),
               checkpoint_sha256=identity['sha256'], config_sha256=identity['config_sha256'],
               command=args, gpu=gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    write_json(state_path, dict(status='RUNNING', gpu=gpu, output=str(output)))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if task['family'] == 'segmentation' and task['dataset'] == 'monuseg':
        env.update(DINOV3_MONUSEG_MANIFEST_FILE=MONUSEG_MANIFEST,
                   DINOV3_MONUSEG_MANIFEST_SHA256=MONUSEG_SHA256,
                   DINOV3_MONUSEG_COUNTS=json.dumps(MONUSEG_COUNTS))
    print(f'START ck{iteration} gpu{gpu} {task["id"]}', flush=True)
    with (output / 'task.log').open('a') as log:
        result = subprocess.run(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        metrics = validate(task, output)
        status = 'VALID_COMPLETE' if result.returncode == 0 else 'FAILED'
        reason = '' if result.returncode == 0 else f'exit code {result.returncode}'
    except Exception as error:
        metrics, status, reason = {}, 'FAILED', f'{type(error).__name__}: {error}'
    write_json(state_path, dict(status=status, gpu=gpu, returncode=result.returncode,
               reason=reason, metrics=metrics, output=str(output)))
    print(f'{status} ck{iteration} gpu{gpu} {task["id"]} {reason}', flush=True)
    if task['family'] == 'segmentation':
        shutil.rmtree(output / 'cache', ignore_errors=True)


def multimodal_v2_manifest():
    readiness = json.loads(MULTIMODAL_V2_READINESS.read_text())
    if readiness['protocol'] != MULTIMODAL_V2_PROTOCOL or readiness['counts'] != {
            'train': 1000, 'val': 101, 'test': 52}:
        raise ValueError('Corrected Multimodal author split identity mismatch')
    manifest_dir = MULTIMODAL_V2_READINESS.parent / 'manifests' / MULTIMODAL_V2_PROTOCOL
    for split in ('train', 'val', 'test'):
        if digest(manifest_dir / f'{split}.csv') != readiness['manifest_sha256'][split]:
            raise ValueError(f'Corrected Multimodal {split} manifest hash mismatch')
    return dict(protocol_id='bio-eval-union-v4-multimodal-' + MULTIMODAL_V2_PROTOCOL,
                split_protocol=MULTIMODAL_V2_PROTOCOL, source_protocol_sha256=digest(PROTOCOL),
                readiness_sha256=digest(MULTIMODAL_V2_READINESS),
                manifest_sha256=readiness['manifest_sha256'], counts=readiness['counts'],
                budgets=[20, 50], seeds=list(range(5)),
                parallel_slots=({'4': 1, '5': 1, '6': 1, '7': 1} if PROFILE == '5tb_nogram'
                                else {'0': 1, '1': 2}),
                legacy_results=str(ROOT / 'ck{checkpoint}/segmentation__multimodal_cellseg'))


def prepare_multimodal_v2():
    from dinov3.eval.bio_segmentation.datasets.multimodal_cellseg import get_multimodal_cellseg_paths
    root = BENCHMARK / 'segmentation/Multimodal_CellSeg/neurips22_cellseg'
    manifest = multimodal_v2_manifest()
    for split, count in manifest['counts'].items():
        images, masks = get_multimodal_cellseg_paths(
            str(root), split=split, split_protocol=MULTIMODAL_V2_PROTOCOL)
        if (len(images), len(masks)) != (count, count):
            raise ValueError(f'Corrected Multimodal {split} count mismatch')
    MULTIMODAL_V2_ROOT.mkdir(parents=True, exist_ok=True)
    path = MULTIMODAL_V2_ROOT / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        slots = old.pop('parallel_slots')
        expected = dict(manifest)
        expected.pop('parallel_slots')
        if old != expected:
            raise ValueError('Existing corrected campaign manifest differs from locked protocol')
        manifest['parallel_slots'] = slots
    write_json(path, manifest)
    original_path = ROOT / 'manifest.json'
    original = json.loads(original_path.read_text())
    overrides = original.setdefault('task_protocol_overrides', {})
    overrides['segmentation__multimodal_cellseg'] = {
        'split_protocol': MULTIMODAL_V2_PROTOCOL,
        'counts': manifest['counts'],
        'corrected_campaign': str(MULTIMODAL_V2_ROOT),
    }
    write_json(original_path, original)
    print(f'Corrected Multimodal ready: {manifest["counts"]}; slots={manifest["parallel_slots"]}', flush=True)


def validate_multimodal_v2(output):
    paths = list((output / 'results').rglob('results.json'))
    signatures = set()
    scores = {}
    for path in paths:
        if '_spneurips22_author_public52_v2/' not in str(path):
            raise ValueError(f'Unexpected Multimodal split in {path}')
        row = json.loads(path.read_text())
        meta = row['_meta']
        signature = (meta['probe_epochs'], meta['seed'])
        if signature in signatures:
            raise ValueError(f'Duplicate Multimodal fit {signature}')
        if (meta['full_train_samples'], meta['used_train_samples']) != (1000, 1000):
            raise ValueError('Corrected Multimodal must use all 1000 author training images')
        if meta['probe_batch_size'] != 32 or meta['probe_eval_every'] != 1 or meta['test_evaluations'] != 1:
            raise ValueError('Corrected Multimodal probe recipe mismatch')
        if len(meta['validation_history']) != meta['probe_epochs']:
            raise ValueError('Corrected Multimodal validation history incomplete')
        score = float(row['test']['mDice'])
        if not math.isfinite(score):
            raise ValueError('Corrected Multimodal test mDice is non-finite')
        signatures.add(signature)
        scores[f'E{signature[0]}_seed{signature[1]}'] = score
    expected = {(budget, seed) for budget in (20, 50) for seed in range(5)}
    if signatures != expected:
        raise ValueError(f'Corrected Multimodal fits incomplete: {sorted(signatures)}')
    return dict(fits=10, test_samples=52, scores=scores,
                E20_mean=sum(scores[f'E20_seed{s}'] for s in range(5)) / 5,
                E50_mean=sum(scores[f'E50_seed{s}'] for s in range(5)) / 5,
                mDice=sum(scores.values()) / 10)


def promote_multimodal_v2(iteration):
    corrected_path = MULTIMODAL_V2_ROOT / f'states/ck{iteration}.json'
    corrected = json.loads(corrected_path.read_text())
    if corrected['status'] != 'VALID_COMPLETE':
        return
    task_id = 'segmentation__multimodal_cellseg'
    lock_path = ROOT / f'task_locks/ck{iteration}/{task_id}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        original_path = ROOT / f'states/ck{iteration}/{task_id}.json'
        original_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(original_path.read_text()) if original_path.exists() else None
        if previous and previous.get('split_protocol') == MULTIMODAL_V2_PROTOCOL:
            return
        backup = MULTIMODAL_V2_ROOT / f'ck{iteration}/superseded_legacy_state.json'
        if previous and not backup.exists():
            write_json(backup, previous)
        metrics = dict(corrected['metrics'])
        metrics['mDice'] = (metrics['E20_mean'] + metrics['E50_mean']) / 2
        write_json(original_path, dict(status='VALID_COMPLETE', gpu=corrected['gpu'],
                                       split_protocol=MULTIMODAL_V2_PROTOCOL,
                                       metrics=metrics, output=corrected['output'],
                                       superseded_legacy_state=str(backup) if previous else None))


def run_multimodal_v2(gpu, iteration, checkpoint, state_path, manifest):
    output = MULTIMODAL_V2_ROOT / f'ck{iteration}'
    if output.exists():
        try:
            metrics = validate_multimodal_v2(output)
            write_json(state_path, dict(status='VALID_COMPLETE', gpu=gpu, metrics=metrics,
                                        output=str(output), reason='adopted verified corrected output'))
            return
        except (FileNotFoundError, KeyError, ValueError):
            pass
    output.mkdir(parents=True, exist_ok=True)
    task = dict(family='segmentation', dataset='multimodal_cellseg',
                split_protocol=MULTIMODAL_V2_PROTOCOL, seeds=list(range(5)))
    args = command(task, iteration, output)
    identity = json.loads((checkpoint.parent / 'identity.json').read_text())
    write_json(output / 'invocation.json', dict(protocol_id=manifest['protocol_id'],
               readiness_sha256=manifest['readiness_sha256'],
               split_manifest_sha256=manifest['manifest_sha256'], checkpoint=str(checkpoint),
               checkpoint_sha256=identity['sha256'], config_sha256=identity['config_sha256'],
               command=args, gpu=gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    write_json(state_path, dict(status='RUNNING', gpu=gpu, output=str(output)))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    print(f'START corrected Multimodal ck{iteration} gpu{gpu}', flush=True)
    with (output / 'task.log').open('a') as log:
        result = subprocess.run(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        metrics = validate_multimodal_v2(output)
        status = 'VALID_COMPLETE' if result.returncode == 0 else 'FAILED'
        reason = '' if result.returncode == 0 else f'exit code {result.returncode}'
    except Exception as error:
        metrics, status, reason = {}, 'FAILED', f'{type(error).__name__}: {error}'
    write_json(state_path, dict(status=status, gpu=gpu, returncode=result.returncode,
                                reason=reason, metrics=metrics, output=str(output)))
    print(f'{status} corrected Multimodal ck{iteration}: {reason}', flush=True)
    if status == 'VALID_COMPLETE':
        shutil.rmtree(output / 'cache', ignore_errors=True)


def multimodal_v2_worker(gpu, slot, once=False):
    path = MULTIMODAL_V2_ROOT / f'gpu{gpu}_slot{slot}.lock'
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((MULTIMODAL_V2_ROOT / 'manifest.json').read_text())
        if manifest['readiness_sha256'] != digest(MULTIMODAL_V2_READINESS):
            raise ValueError('Corrected Multimodal readiness changed after campaign preparation')
        while True:
            if not gpu_ready(gpu):
                time.sleep(10)
                continue
            claimed = False
            for iteration in reversed(CHECKPOINTS):
                checkpoint = ROOT / f'checkpoints/{iteration}/checkpoint.pth'
                if not checkpoint.is_file() or not (checkpoint.parent / 'identity.json').is_file():
                    continue
                state_path = MULTIMODAL_V2_ROOT / f'states/ck{iteration}.json'
                state = json.loads(state_path.read_text())['status'] if state_path.exists() else 'PENDING'
                if state in ('VALID_COMPLETE', 'FAILED'):
                    continue
                task_lock_path = MULTIMODAL_V2_ROOT / f'task_locks/ck{iteration}.lock'
                task_lock_path.parent.mkdir(parents=True, exist_ok=True)
                with task_lock_path.open('a') as task_lock:
                    try:
                        fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    state = json.loads(state_path.read_text())['status'] if state_path.exists() else 'PENDING'
                    if state in ('VALID_COMPLETE', 'FAILED'):
                        continue
                    claimed = True
                    run_multimodal_v2(gpu, iteration, checkpoint, state_path, manifest)
                    break
            if claimed and once:
                return
            if not claimed:
                time.sleep(10)


def supervise_multimodal_v2():
    MULTIMODAL_V2_ROOT.mkdir(parents=True, exist_ok=True)
    with (MULTIMODAL_V2_ROOT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = str(Path(__file__).resolve())
        while True:
            manifest = json.loads((MULTIMODAL_V2_ROOT / 'manifest.json').read_text())
            running = set()
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    argv = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
                except (OSError, UnicodeDecodeError):
                    continue
                if len(argv) > 2 and argv[1] == '-u' and Path(argv[2]).name == Path(script).name:
                    running.add(tuple(argv[3:]))
            started = []
            for gpu in sorted(map(int, manifest['parallel_slots'])):
                for slot in range(manifest['parallel_slots'][str(gpu)]):
                    args = child_args('multimodal_v2_worker', '--gpu', str(gpu), '--slot', str(slot))
                    if any(command_line[:len(args)] == args for command_line in running):
                        continue
                    name = f'gpu{gpu}_slot{slot}'
                    env = os.environ.copy()
                    env['DINOV3_1TB_EVAL_PROFILE'] = PROFILE
                    with (MULTIMODAL_V2_ROOT / f'{name}.log').open('a') as log:
                        subprocess.Popen([PYTHON, '-u', script, *args], cwd=REPO, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                    started.append(name)
            gpu_rows = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
            gpu_status = {}
            for row in gpu_rows.splitlines():
                index, used, total, utilization = (value.strip() for value in row.split(','))
                if int(index) in EVAL_GPUS:
                    gpu_status[index] = dict(used_mib=int(used), total_mib=int(total),
                                             utilization_percent=int(utilization))
            states = {}
            for iteration in CHECKPOINTS:
                if not (ROOT / f'checkpoints/{iteration}/checkpoint.pth').is_file():
                    continue
                path = MULTIMODAL_V2_ROOT / f'states/ck{iteration}.json'
                states[str(iteration)] = json.loads(path.read_text())['status'] if path.exists() else 'PENDING'
            health = dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          started=started, slots=manifest['parallel_slots'], gpu=gpu_status,
                          checkpoint_states=states)
            write_json(MULTIMODAL_V2_ROOT / 'supervisor_status.json', health)
            for step, state in states.items():
                if state == 'VALID_COMPLETE':
                    promote_multimodal_v2(int(step))
            refresh_multimodal_v2_curves(states)
            print(json.dumps(health), flush=True)
            time.sleep(60)


def refresh_multimodal_v2_curves(states):
    rows = []
    for step, status in states.items():
        if status != 'VALID_COMPLETE':
            continue
        state = json.loads((MULTIMODAL_V2_ROOT / f'states/ck{step}.json').read_text())
        for budget in (20, 50):
            for seed in range(5):
                rows.append(dict(step=int(step) + 1, checkpoint=int(step), budget=budget,
                                 seed=seed, mDice=state['metrics']['scores'][f'E{budget}_seed{seed}']))
    rows.sort(key=lambda row: (row['checkpoint'], row['budget'], row['seed']))
    path = MULTIMODAL_V2_ROOT / 'curves.csv'
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('step', 'checkpoint', 'budget', 'seed', 'mDice'))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for budget in (20, 50):
        points = sorted((int(step), json.loads((MULTIMODAL_V2_ROOT / f'states/ck{step}.json').read_text())
                         ['metrics'][f'E{budget}_mean']) for step, status in states.items()
                        if status == 'VALID_COMPLETE')
        ax.plot([step + 1 for step, _ in points], [value for _, value in points],
                marker='o', markersize=3, label=f'E{budget}, five-seed mean')
    ax.set(xlabel='Training step', ylabel='Test mDice',
           title='Multimodal CellSeg: author public 52 test images')
    ax.grid(alpha=0.25)
    ax.legend()
    temp_png = MULTIMODAL_V2_ROOT / 'curves.tmp.png'
    fig.savefig(temp_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    temp_png.replace(MULTIMODAL_V2_ROOT / 'curves.png')


def scale_multimodal_v2(slots0):
    if not 1 <= slots0 <= 18:
        raise ValueError('GPU 0 corrected worker slots must be between 1 and 18')
    path = MULTIMODAL_V2_ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['parallel_slots'] = {'0': slots0, '1': 2}
    write_json(path, manifest)
    print(f'Corrected Multimodal slots: GPU0={slots0}, GPU1=2', flush=True)


def scale_union_slots(slots0, slots1):
    if not 1 <= slots0 <= 14 or not 1 <= slots1 <= 4:
        raise ValueError('Union-v4 worker slots exceed the original 14/4 limits')
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['parallel_slots'] = {'0': slots0, '1': slots1}
    write_json(path, manifest)
    print(f'Union-v4 slots: GPU0={slots0}, GPU1={slots1}', flush=True)


def retry_multimodal_v2_split_rejection():
    count = 0
    for path in (MULTIMODAL_V2_ROOT / 'states').glob('ck*.json'):
        state = json.loads(path.read_text())
        if state['status'] != 'FAILED' or state.get('returncode') != 2:
            continue
        output = Path(state['output'])
        if "invalid choice: 'neurips22-author-public52-v2'" not in (output / 'task.log').read_text():
            continue
        write_json(path, dict(status='PENDING', previous_failure=state,
                              reason='split choice registered; retry corrected protocol'))
        count += 1
    print(f'Requeued {count} corrected tasks rejected by the missing CLI split choice', flush=True)


def worker(gpu, slot, slots):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if digest(PROTOCOL) != manifest['protocol_sha256']:
        raise RuntimeError('Protocol changed since campaign preparation')
    expected_slots = manifest['parallel_slots'][str(gpu)]
    if slots != expected_slots or not 0 <= slot < slots:
        raise ValueError(f'GPU {gpu} requires {expected_slots} slots; got slot {slot}/{slots}')
    lock_path = ROOT / f'gpu{gpu}_slot{slot}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = [task for task in manifest['tasks'] if task['gpu'] == gpu]
        if PROFILE in ('biosafe256', '5tb_nogram') and slot == 0:
            tasks.sort(key=lambda task: task['id'] not in CURVE_TASK_IDS)
            iterations = tuple(reversed(CHECKPOINTS))
        else:
            iterations = CHECKPOINTS
        print(f'GPU {gpu} slot {slot}/{slots}: shared queue with {len(tasks)} tasks per checkpoint', flush=True)
        while True:
            if not gpu_ready(gpu):
                time.sleep(10)
                continue
            claimed = False
            for iteration in iterations:
                checkpoint = ROOT / f'checkpoints/{iteration}/checkpoint.pth'
                if not checkpoint.is_file() or not (checkpoint.parent / 'identity.json').is_file():
                    continue
                for task in tasks:
                    state_path = ROOT / f'states/ck{iteration}/{task["id"]}.json'
                    state = json.loads(state_path.read_text())['status'] if state_path.exists() else 'PENDING'
                    if state in ('VALID_COMPLETE', 'FAILED') or (state == 'BLOCKED_NOT_TESTED' and task.get('blocker')):
                        continue
                    task_lock_path = ROOT / f'task_locks/ck{iteration}/{task["id"]}.lock'
                    task_lock_path.parent.mkdir(parents=True, exist_ok=True)
                    with task_lock_path.open('a') as task_lock:
                        try:
                            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        state = json.loads(state_path.read_text())['status'] if state_path.exists() else 'PENDING'
                        if state in ('VALID_COMPLETE', 'FAILED') or (state == 'BLOCKED_NOT_TESTED' and task.get('blocker')):
                            continue
                        claimed = True
                        evaluate_task(gpu, task, iteration, manifest, checkpoint, state_path)
                        break
                if claimed:
                    break
            if not claimed:
                time.sleep(10)


def status():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for iteration in CHECKPOINTS:
        summary = {}
        for task in manifest['tasks']:
            path = ROOT / f'states/ck{iteration}/{task["id"]}.json'
            state = json.loads(path.read_text())['status'] if path.exists() else 'PENDING'
            summary[state] = summary.get(state, 0) + 1
        print(f'ck{iteration}: {summary}')


def supervise():
    lock_path = ROOT / 'supervisor.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = str(Path(__file__).resolve())
        while True:
            manifest = json.loads((ROOT / 'manifest.json').read_text())
            running = set()
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    argv = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
                except (OSError, UnicodeDecodeError):
                    continue
                if len(argv) > 2 and argv[1] == '-u' and argv[2] == script:
                    running.add(tuple(argv[3:]))
            started = []
            expected = [child_args('pin')] if any(not (ROOT / f'checkpoints/{step}/checkpoint.pth').is_file()
                                                   for step in CHECKPOINTS) else []
            expected.append(child_args('follow_curves'))
            for gpu in EVAL_GPUS:
                slots = manifest['parallel_slots'][str(gpu)]
                expected.extend(child_args('worker', '--gpu', str(gpu), '--slot', str(slot), '--slots', str(slots))
                                for slot in range(slots))
            for args in expected:
                mode = args[len(PROFILE_PREFIX)]
                if args in running or (mode == 'worker' and any(
                        command_line[:len(PROFILE_PREFIX) + 5] == args[:len(PROFILE_PREFIX) + 5]
                        for command_line in running)):
                    continue
                offset = len(PROFILE_PREFIX)
                name = mode if mode != 'worker' else f'gpu{args[offset + 2]}_slot{args[offset + 4]}'
                with (ROOT / f'{name}.log').open('a') as log:
                    env = os.environ.copy()
                    env['DINOV3_1TB_EVAL_PROFILE'] = PROFILE
                    subprocess.Popen([PYTHON, '-u', script, *args], cwd=REPO, env=env,
                                     stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started.append(name)
            gpu_rows = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
            gpu_status = {}
            for row in gpu_rows.splitlines():
                index, used, total, utilization = (value.strip() for value in row.split(','))
                if int(index) in EVAL_GPUS:
                    gpu_status[index] = dict(used_mib=int(used), total_mib=int(total),
                                             utilization_percent=int(utilization))
            states = {}
            for step in CHECKPOINTS:
                if not (ROOT / f'checkpoints/{step}/checkpoint.pth').is_file():
                    continue
                counts = {}
                for task in manifest['tasks']:
                    path = ROOT / f'states/ck{step}/{task["id"]}.json'
                    state = json.loads(path.read_text())['status'] if path.is_file() else 'PENDING'
                    counts[state] = counts.get(state, 0) + 1
                states[str(step)] = counts
            health = dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          started=started, gpu=gpu_status, checkpoint_states=states)
            write_json(ROOT / 'supervisor_status.json', health)
            print(json.dumps(health), flush=True)
            time.sleep(60)


def unblock_monuseg():
    from dinov3.eval.bio_segmentation.datasets.monuseg_official import official_paths
    data_root = BENCHMARK / 'segmentation/monuseg/extracted'
    for split, count in MONUSEG_COUNTS.items():
        images, _ = official_paths(data_root, split, MONUSEG_SHA256, MONUSEG_MANIFEST, MONUSEG_COUNTS)
        if len(images) != count:
            raise ValueError(f'MoNuSeg {split}: expected {count}, got {len(images)}')
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    for task in manifest['tasks']:
        if task['family'] == 'segmentation' and task['dataset'] == 'monuseg':
            task.pop('blocker', None)
    manifest['monuseg_active_lock_sha256'] = MONUSEG_SHA256
    write_json(path, manifest)
    print('MoNuSeg active 30/7/14 lock admitted', flush=True)


def unblock_rxrx3():
    if digest(RXRX3_CACHE / 'split_manifest.jsonl') != RXRX3_SPLIT_SHA256:
        raise ValueError('RxRx3 split manifest hash mismatch')
    validation = json.loads((RXRX3_CACHE / 'validation.json').read_text())
    if (validation.get('status'), validation.get('eligible_genes'), validation.get('gallery_wells'),
            validation.get('query_wells')) != ('VALID_COMPLETE', 734, 734, 734):
        raise ValueError('RxRx3 cache identity mismatch')
    metadata = json.loads((RXRX3_CACHE / 'metadata.json').read_text())
    if metadata.get('protocol_id') != 'crispr-query-guide-plate-disjoint-all-eligible-genes-v1':
        raise ValueError('RxRx3 metadata protocol mismatch')
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    for task in manifest['tasks']:
        if task['family'] == 'retrieval_clustering' and task['dataset'] == 'rxrx3-core':
            task.pop('blocker', None)
    manifest['rxrx3_split_sha256'] = RXRX3_SPLIT_SHA256
    write_json(path, manifest)
    print('RxRx3 fixed 734-gene cache admitted', flush=True)


def rebalance():
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    for task in manifest['tasks']:
        if task['family'] == 'classification':
            task['gpu'] = 1 if task['dataset'] in GPU1_CLASSIFICATION else 0
    manifest['gpu_assignment'] = '28 tasks on GPU0; 23 tasks on GPU1; ten classification cells moved to GPU1'
    write_json(path, manifest)
    print(manifest['gpu_assignment'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'extend', 'pin', 'worker', 'status',
                                         'curves', 'follow_curves', 'supervise', 'unblock_monuseg',
                                         'unblock_rxrx3', 'rebalance', 'multimodal_v2_prepare',
                                         'multimodal_v2_worker', 'multimodal_v2_supervise',
                                         'multimodal_v2_scale', 'multimodal_v2_retry_split_rejection',
                                         'scale_union_slots'))
    parser.add_argument('--profile-token')
    parser.add_argument('--gpu', type=int, choices=EVAL_GPUS)
    parser.add_argument('--slot', type=int)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--slots0', type=int)
    parser.add_argument('--slots1', type=int)
    parser.add_argument('--once', action='store_true')
    opts = parser.parse_args()
    if opts.profile_token and opts.profile_token != PROFILE:
        parser.error('profile token and evaluation profile differ')
    if opts.mode == 'prepare':
        prepare()
    elif opts.mode == 'extend':
        extend()
    elif opts.mode == 'pin':
        pin()
    elif opts.mode == 'worker':
        if opts.gpu is None or opts.slot is None or opts.slots is None:
            parser.error('worker requires --gpu, --slot, and --slots')
        worker(opts.gpu, opts.slot, opts.slots)
    elif opts.mode == 'unblock_monuseg':
        unblock_monuseg()
    elif opts.mode == 'unblock_rxrx3':
        unblock_rxrx3()
    elif opts.mode == 'rebalance':
        rebalance()
    elif opts.mode == 'curves':
        refresh_curves(json.loads((ROOT / 'manifest.json').read_text()))
    elif opts.mode == 'follow_curves':
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        while True:
            refresh_curves(manifest)
            time.sleep(60)
    elif opts.mode == 'supervise':
        supervise()
    elif opts.mode == 'multimodal_v2_prepare':
        prepare_multimodal_v2()
    elif opts.mode == 'multimodal_v2_worker':
        if opts.gpu is None or opts.slot is None:
            parser.error('multimodal_v2_worker requires --gpu and --slot')
        multimodal_v2_worker(opts.gpu, opts.slot, once=opts.once)
    elif opts.mode == 'multimodal_v2_supervise':
        supervise_multimodal_v2()
    elif opts.mode == 'multimodal_v2_scale':
        if opts.slots0 is None:
            parser.error('multimodal_v2_scale requires --slots0')
        scale_multimodal_v2(opts.slots0)
    elif opts.mode == 'multimodal_v2_retry_split_rejection':
        retry_multimodal_v2_split_rejection()
    elif opts.mode == 'scale_union_slots':
        if opts.slots0 is None or opts.slots1 is None:
            parser.error('scale_union_slots requires --slots0 and --slots1')
        scale_union_slots(opts.slots0, opts.slots1)
    else:
        status()
