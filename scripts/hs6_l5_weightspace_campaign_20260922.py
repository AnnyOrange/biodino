#!/usr/bin/env python3
"""Shared definitions + task inventory for the weight-space baseline campaign.

Usage: python scripts/hs6_l5_weightspace_campaign_20260922.py write-tasks
       python scripts/hs6_l5_weightspace_campaign_20260922.py write-manifest

The extraction commands mirror, argument for argument, the pinned coexistence
launchers (frozen pilot script, retrieval v3 launcher, dense_v4 detection
launcher, spatial single-teacher launcher) and execute from the SAME immutable
source snapshots, so weight-space arms are directly comparable to E/M/L.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
COEX = REPO / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
ROOT = REPO / 'outputs/02_eval_runs/hs6_l5_weightspace_baseline_v4_20260922'
RUN = REPO / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'
PYTHON = Path('/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
BENCH = '/mnt/huawei_deepcad/benchmark'

SNAP_CLS = COEX / 'source_snapshot'              # launch_manifest.json
SNAP_RET = COEX / 'source_snapshot_retrieval_v3'  # retrieval_v3_manifest.json
SNAP_DET = COEX / 'source_snapshot_dense_v4'      # dense_v4_manifest.json
SNAP_SEG = COEX / 'source_snapshot'              # same as spatial single-teacher launcher
SNAP_PAIR = ROOT / 'source_snapshot_weightspace_v1'  # lc_v7 copy + pairing modules (CPU only)

WEIGHT_ARMS = ['WA050', 'AVG3', 'WA025', 'WA075']   # claim order inside a priority block
BASE_ROLES = {'E': 12687, 'M': 20007, 'L': 29279}
SEG_CKPT_ID = {'WA025': 10025, 'WA050': 10050, 'WA075': 10075, 'AVG3': 10333}
SEG_RUN_LABEL = {'E': 'early', 'M': 'middle', 'L': 'late',
                 'WA025': 'wa025', 'WA050': 'wa050', 'WA075': 'wa075', 'AVG3': 'avg3'}

# Classification/regression ordered by approximate extraction cost (n_train + n_test, resolution).
CLS_SMALL = ['breastmnist', 'retinamnist', 'bbbc013', 'conic-cell-count', 'livecell-cell-count',
             'pneumoniamnist', 'dermamnist', 'bbbc005', 'midog25-atypical', 'bloodmnist',
             'organcmnist', 'organsmnist', 'lc25000', 'chammi-hpa-task2', 'chammi-allen-task1',
             'chammi-cp-task3', 'chammi-hpa-task1', 'chammi-cp-task1', 'organamnist',
             'chammi-cp-task2', 'chammi-allen-task2', 'cyclops-protein-loc', 'bbbc048-cellcycle']
CLS_LARGE = ['octmnist', 'pathmnist', 'nct-crc-he', 'chestmnist', 'tissuemnist', 'pcam']
RET_WITHIN = ['nct-crc-he-1k', 'nct-crc-he-100', 'crc-val-he-7k', 'lc25000']
RET_HPA = ['hpa-subcellular']
DET = ['bbbc038', 'conic', 'livecell']
SEG_FIRST = ['cellpose']
SEG_REST = ['conic', 'livecell', 'multimodal_cellseg', 'tissuenet', 'pannuke']
# monuseg: BLOCKED_NOT_TESTED in protocol_v4 (official identity evidence pending) -> not scheduled.
# retrieval rxrx1-cross / rxrx3-core: no E/M/L baseline exists in the coexistence campaign -> not scheduled.

THREADS = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_for(arm: str) -> Path:
    if arm in BASE_ROLES:
        return RUN / f'eval/training_{BASE_ROLES[arm]}/teacher_checkpoint.pth'
    return ROOT / 'checkpoints' / arm / 'teacher_checkpoint.pth'


def model_name(arm: str) -> str:
    return f'hs6_l5_{arm}_{BASE_ROLES[arm]}' if arm in BASE_ROLES else f'hs6_l5_{arm}'


def cls_task(dataset: str, arm: str, order: int) -> dict:
    out = ROOT / 'frozen' / dataset / arm
    cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_classification',
           '--checkpoint', str(checkpoint_for(arm)), '--train-config', str(RUN / 'config.yaml'),
           '--benchmark-root', BENCH, '--datasets', dataset, '--output-dir', str(out),
           '--model-name', model_name(arm), '--batch-size', '64', '--num-workers', '2',
           '--n-last-blocks', '1', '--autocast-dtype', 'bf16', '--channel-policy', 'auto',
           '--channel-tta-samples', '8', '--resolution-protocol', 'best',
           '--split-protocol', 'current', '--save-paths']
    return {'id': f'cls__{dataset}__{arm}', 'family': 'classification', 'dataset': dataset, 'arm': arm,
            'order': order, 'cwd': str(SNAP_CLS), 'pythonpath': str(SNAP_CLS), 'cmd': cmd,
            'done': {'type': 'summary_csv', 'path': str(out / 'summary.csv'), 'dataset': dataset, 'rows': 1},
            'pinned_manifest': str(COEX / 'launch_manifest.json')}


def ret_task(dataset: str, arm: str, order: int) -> dict:
    out = ROOT / 'retrieval' / dataset / arm
    cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_retrieval_clustering',
           '--checkpoint', str(checkpoint_for(arm)), '--train-config', str(RUN / 'config.yaml'),
           '--benchmark-root', BENCH, '--datasets', dataset, '--output-dir', str(out),
           '--model-name', model_name(arm), '--batch-size', '64', '--num-workers', '2',
           '--autocast-dtype', 'bf16', '--n-last-blocks', '1', '--channel-policy', 'auto',
           '--channel-tta-samples', '8', '--metric-device', 'cpu', '--seed', '0']
    return {'id': f'ret__{dataset}__{arm}', 'family': 'retrieval', 'dataset': dataset, 'arm': arm,
            'order': order, 'cwd': str(SNAP_RET), 'pythonpath': str(SNAP_RET), 'cmd': cmd,
            'done': {'type': 'summary_csv', 'path': str(out / 'summary.csv'), 'dataset': dataset,
                     'rows': 3 if dataset == 'hpa-subcellular' else 1},
            'pinned_manifest': str(COEX / 'retrieval_v3_manifest.json')}


def det_task(dataset: str, arm: str, order: int) -> dict:
    out = ROOT / 'detection' / dataset / arm
    cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_detection.center_probe',
           '--checkpoint', str(checkpoint_for(arm)), '--train-config', str(RUN / 'config.yaml'),
           '--benchmark-root', BENCH, '--dataset', dataset, '--output-dir', str(out),
           '--batch-size', '8', '--num-workers', '2', '--image-size', '224', '--epochs', '5',
           '--lr', '0.001', '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--seed', '0',
           '--max-samples-per-split', '0', '--conic-split-protocol', 'official-baseline-fold0-nested-v1']
    return {'id': f'det__{dataset}__{arm}', 'family': 'detection_proxy', 'dataset': dataset, 'arm': arm,
            'order': order, 'cwd': str(SNAP_DET), 'pythonpath': str(SNAP_DET), 'cmd': cmd,
            'done': {'type': 'detection_json', 'path': str(out / 'results_bio_detection.json'), 'dataset': dataset},
            'pinned_manifest': str(COEX / 'dense_v4_manifest.json')}


def seg_task(dataset: str, arm: str, order: int) -> dict:
    cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline',
           '--datasets', dataset, '--checkpoint-file', str(checkpoint_for(arm)),
           '--checkpoint-id', str(SEG_CKPT_ID[arm]), '--train-config', str(RUN / 'config.yaml'),
           '--protocol', 'best', '--dataset-split-protocol', 'formal-v1',
           '--feature-batch-size', '32', '--feature-num-workers', '2',
           '--autocast-dtype', 'bf16', '--channel-policy', 'auto', '--channel-tta-samples', '8',
           '--probe-epoch-grid', '20', '50', '--probe-seeds', '0', '1', '2',
           '--probe-batch-size', '32', '--probe-lr', '0.001', '--probe-weight-decay', '0.0001',
           '--probe-eval-every', '1', '--probe-num-workers', '2', '--gpu', '{GPU}',
           '--cache-root', str(ROOT / 'segmentation/cache'),
           '--output-root', str(ROOT / 'segmentation/results'),
           '--run-name', f'hs6_l5_{SEG_RUN_LABEL[arm]}']
    return {'id': f'seg__{dataset}__{arm}', 'family': 'segmentation', 'dataset': dataset, 'arm': arm,
            'order': order, 'cwd': str(SNAP_SEG), 'pythonpath': str(SNAP_SEG), 'cmd': cmd,
            'done': {'type': 'seg_results', 'root': str(ROOT / 'segmentation/results'),
                     'run_prefix': f'hs6_l5_{SEG_RUN_LABEL[arm]}_', 'ckpt_id': SEG_CKPT_ID[arm],
                     'dataset': dataset, 'expect': 18 if dataset == 'pannuke' else 6},
            'heavy': True, 'pinned_manifest': str(COEX / 'launch_manifest.json')}


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    order = 0

    def add(t):
        nonlocal order
        order += 1
        t['order'] = order
        tasks.append(t)

    for ds in RET_WITHIN:
        for arm in WEIGHT_ARMS:
            add(ret_task(ds, arm, 0))
    for ds in DET:
        arms = WEIGHT_ARMS if ds == 'bbbc038' else ['E', 'M', 'L'] + WEIGHT_ARMS
        for arm in arms:
            add(det_task(ds, arm, 0))
    for ds in SEG_FIRST:
        for arm in WEIGHT_ARMS:
            add(seg_task(ds, arm, 0))
    for ds in CLS_SMALL:
        for arm in WEIGHT_ARMS:
            add(cls_task(ds, arm, 0))
    for ds in SEG_REST:
        for arm in WEIGHT_ARMS:
            add(seg_task(ds, arm, 0))
    for ds in CLS_LARGE:
        for arm in WEIGHT_ARMS:
            add(cls_task(ds, arm, 0))
    for ds in RET_HPA:
        for arm in WEIGHT_ARMS:
            add(ret_task(ds, arm, 0))
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task ids')
    return tasks


def task_done(done: dict) -> tuple[bool, str]:
    """Rule04-style completion: output exists, parses, no error, identity fields match."""
    kind = done['type']
    if kind == 'summary_csv':
        path = Path(done['path'])
        if not path.exists():
            return False, 'missing'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != done['rows']:
            return False, f'rows={len(rows)} expected {done["rows"]}'
        for row in rows:
            if row.get('dataset') != done['dataset'] or row.get('error'):
                return False, f'row mismatch/error: {row.get("dataset")} {row.get("error")}'
        return True, 'ok'
    if kind == 'detection_json':
        path = Path(done['path'])
        if not path.exists():
            return False, 'missing'
        data = json.loads(path.read_text())
        ok = (data.get('dataset') == done['dataset'] and data.get('batch_size') == 8 and data.get('epochs') == 5
              and math.isfinite(float(data.get('test_patch_f1', float('nan')))))
        return ok, 'ok' if ok else 'invalid detection json'
    if kind == 'seg_results':
        root = Path(done['root'])
        found = []
        for run_dir in root.glob(done['run_prefix'] + '*'):
            if done['dataset'] == 'pannuke' and 'pannuke' not in run_dir.name:
                continue
            for res in run_dir.glob(f'budget*/seed*/*/{done["ckpt_id"]}/results.json'):
                if not res.parts[-3].startswith(done['dataset']):
                    continue  # run-name prefixes are shared across datasets
                try:
                    parsed = json.loads(res.read_text())
                except Exception:
                    return False, f'unparseable {res}'
                meta = parsed.get('_meta', {})
                if meta.get('test_evaluations') == 1 and 'test' in parsed and 'mIoU' in parsed['test']:
                    found.append(res)
        if len(found) >= done['expect']:
            return True, f'{len(found)} results'
        return False, f'{len(found)}/{done["expect"]} results'
    raise ValueError(kind)


def write_tasks() -> None:
    tasks = build_tasks()
    path = ROOT / 'tasks.json'
    if path.exists():
        raise FileExistsError(path)
    ck_manifest = json.loads((ROOT / 'checkpoints/checkpoint_manifest.json').read_text())
    for t in tasks:
        ck = Path(t['cmd'][t['cmd'].index('--checkpoint') + 1] if '--checkpoint' in t['cmd']
                  else t['cmd'][t['cmd'].index('--checkpoint-file') + 1])
        if not ck.is_file():
            raise FileNotFoundError(ck)
        t['checkpoint_sha256'] = (ck_manifest['outputs'][t['arm']]['sha256'] if t['arm'] in ck_manifest['outputs']
                                  else ck_manifest['sources'][t['arm']]['sha256'])
    path.write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
                                'n_tasks': len(tasks), 'tasks': tasks}, indent=1) + '\n')
    from collections import Counter
    print(f'{len(tasks)} tasks -> {path}; by family: {Counter(t["family"] for t in tasks)}')


def write_manifest() -> None:
    path = ROOT / 'campaign_manifest.json'
    if path.exists():
        raise FileExistsError(path)
    import numpy, sklearn, torch
    pinned = {}
    for name in ('launch_manifest.json', 'retrieval_v3_manifest.json', 'dense_v4_manifest.json', 'lc_v7_manifest.json'):
        p = COEX / name
        pinned[name] = {'path': str(p), 'sha256': sha256(p), 'snapshot': json.loads(p.read_text())['source_snapshot']}
    new_sources = {}
    for rel in ('scripts/build_hs6_l5_weightspace_checkpoints_20260922.py',
                'scripts/hs6_l5_weightspace_campaign_20260922.py',
                'scripts/run_hs6_l5_weightspace_worker_20260922.py',
                'scripts/launch_hs6_l5_weightspace_workers_20260922.sh',
                'dinov3/eval/bio_frozen_eval/run_weightspace_classification.py',
                'dinov3/eval/bio_frozen_eval/run_weightspace_retrieval.py',
                'scripts/pair_hs6_l5_weightspace_20260922.py',
                'scripts/summarize_hs6_l5_weightspace_20260922.py',
                'Evaluation Rules/plans/hs6_l5_weightspace_baseline_v4_20260922.md'):
        p = REPO / rel
        new_sources[rel] = sha256(p) if p.exists() else 'NOT_YET_WRITTEN'
    pair_snapshot = {str(p.relative_to(SNAP_PAIR)): sha256(p) for p in sorted(SNAP_PAIR.rglob('*'))
                     if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh', '.yaml'}} if SNAP_PAIR.exists() else {}
    manifest = {
        'protocol': 'bio-eval-union-v4',
        'campaign': 'hs6_l5_weightspace_baseline_v4_20260922',
        'purpose': ('Cheap weight-space baseline: can asynchronous E/M/L capabilities be merged inside one '
                    'parameter basin by linear interpolation/averaging? Diagnostic only; not a method claim.'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'git_status_porcelain': subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
        'host': platform.node(),
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'numpy': numpy.__version__, 'sklearn': sklearn.__version__, 'interpreter': str(PYTHON)},
        'checkpoint_manifest': str(ROOT / 'checkpoints/checkpoint_manifest.json'),
        'checkpoint_manifest_sha256': sha256(ROOT / 'checkpoints/checkpoint_manifest.json'),
        'train_config': {'path': str(RUN / 'config.yaml'), 'sha256': sha256(RUN / 'config.yaml')},
        'pinned_extraction_sources': pinned,
        'baseline_campaign': str(COEX),
        'baseline_reuse_rule': ('E/M/L, E+L, M+L, PCA(E+L) scores are reused from the coexistence campaign only after '
                                'ordered sample-identity + label equality with the weight-space banks; alpha=0 is M '
                                'and alpha=1 is L by checkpoint SHA256.'),
        'new_source_sha256': new_sources,
        'pairing_snapshot': str(SNAP_PAIR), 'pairing_snapshot_sha256': pair_snapshot,
        'tasks': str(ROOT / 'tasks.json'), 'tasks_sha256': sha256(ROOT / 'tasks.json'),
        'not_scheduled': {'segmentation/monuseg': 'BLOCKED_NOT_TESTED (protocol_v4 official identity pending)',
                          'retrieval/rxrx1-cross': 'no E/M/L baseline in the coexistence campaign',
                          'retrieval/rxrx3-core': 'no E/M/L baseline in the coexistence campaign',
                          'cell_tracking/ctc': 'APPROVED_PENDING_FIXED_HEAD_AND_LINKER',
                          'ood/xray, ood/cryo': 'no E/M/L baseline evaluator run in the coexistence campaign'},
        'no_transfer': 'checkpoints and datasets are read in place on Huawei shared storage; nothing is copied to nodes',
        'full_v4_status': 'INCOMPLETE_UNTIL_EVERY_ADMITTED_CELL_MATCHED',
    }
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    {'write-tasks': write_tasks, 'write-manifest': write_manifest}[sys.argv[1]]()
