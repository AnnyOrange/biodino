#!/usr/bin/env python3
"""Prepare and continuously admit sparse L checkpoints to a locked evaluation."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3')
OUT = ROOT / 'outputs/02_eval_runs/l20tb_selected29_20261019'
SNAPSHOT = Path('/mnt/huawei_deepcad/dinov3_l20tb_selected29_snapshot_20261019')
SELECTED = ROOT / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv'
REFERENCE = ROOT / 'outputs/02_eval_inputs/shared_fleet_reference_20260930/corrected_campaign_manifest.json'
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
CONTROL = ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018'
RUNS = ROOT / 'outputs/01_training_runs'
FAMILIES = dict(classification=16, regression=2, retrieval=4, clustering=2, segmentation=5)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ssh(argv, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           '5090-lyx-xr', shlex.join(argv)], check=True, **kwargs)


def references():
    return [dict(arm='5tb_global_cls_w3', checkpoint_id='32695',
                 remote='/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls_w3/eval/training_32695/teacher_checkpoint.pth',
                 remote_config='/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/bin/config.yaml'),
            dict(arm='20tb_distill_lw5', checkpoint_id='79543',
                 remote='/data/xuzijing/cooldown_20261013_eval/20tb_distill_lw5/eval/training_79543/teacher_checkpoint.pth',
                 remote_config='/data/xuzijing/cooldown_20261013_eval/20tb_distill_lw5/source/config.yaml')]


def prepare():
    if (OUT / 'campaign_manifest.json').exists():
        raise RuntimeError('Existing campaign must not be replaced')
    OUT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.mkdir(exist_ok=False)
    for folder in ('dinov3', 'Evaluation Rules'):
        shutil.copytree(ROOT / folder, SNAPSHOT / folder,
                        ignore=shutil.ignore_patterns('__pycache__', 'dataset_webdataset', 'outputs', '*.pyc'))
    (SNAPSHOT / 'scripts').mkdir()
    for name in ('run_shared_frozen_stage1_20260917.py', 'run_retest_fleet_20260918.py',
                 'run_l20tb_selected29_worker_20261019.py'):
        shutil.copy2(ROOT / 'scripts' / name, SNAPSHOT / 'scripts' / name)
    sys.path.insert(0, str(SNAPSHOT / 'scripts'))
    import run_shared_frozen_stage1_20260917 as queue
    template = json.loads(REFERENCE.read_text())
    selected = {(r['family'], r['dataset']) for r in csv.DictReader(SELECTED.open())}
    datasets = [copy.deepcopy(d) for d in template['datasets']
                if (d['task'], d['dataset']) in selected and d['status'] == 'PASS']
    assert len(datasets) == 29  # 27 datasets plus two extra PanNuke rotations.
    for spec in datasets:
        for path, digest in spec['source_hashes'].items():
            if not Path(path).is_file():
                raise RuntimeError('Missing audited split source: ' + path)
    manifest = dict(protocol_id='SELECTED29_LOCKED_E20_3SEEDS_20261019',
        campaign_scope='fixed_selected29_five_family_mean',
        explicit_user_authorization='User delegated L20TB improvement and simultaneous train/test/curves on 2026-10-19',
        full_v3_aggregate_allowed=False, v4_aggregate_allowed=False,
        selected29_aggregate_allowed=True, legacy_reuse=False,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_snapshot=queue.current_source_snapshot(), source_snapshot_path=str(SNAPSHOT),
        selected_inventory_sha256=sha(SELECTED), selected_inventory=sorted(selected),
        expected_unique_dataset_counts=FAMILIES,
        primary_metrics=dict(classification='macro_f1', regression='spearman', retrieval='map_at_5',
                             clustering='nmi', segmentation='mDice_E20'),
        aggregation='mean within each family; equal mean of five complete families; all three PanNuke rotations and seeds required',
        segmentation=dict(epochs=20, seeds=[0, 1, 2], feature_batch=32, probe_batch=32,
                          head_selection='best validation mDice', comparison_view='primary-last'),
        benchmark_root=template['benchmark_root'], numerical_environment=template['numerical_environment'],
        external_source_hashes={}, checkpoint_assets=[], tasks=[], datasets=datasets,
        online_checkpoints=True, isolate_runtime_failures=False,
        deepcad_authorized_gpu_indices=[1, 2, 6], deepcad_authorized_max_gpus=3,
        created_unix=time.time())
    save(OUT / 'campaign_manifest.json', manifest)
    points = []
    for ref in references():
        base = OUT / 'assets' / ref['arm']
        points.append(dict(arm=ref['arm'], checkpoint_id=ref['checkpoint_id'],
                           path=str(base / 'teacher_checkpoint.pth'), config=str(base / 'config.yaml'),
                           readiness_marker=str(base / 'verified_copy.json')))
    for ck in (50263, 52703, 57583, 67343):
        for arm, folder in (
            ('l20tb_domain', 'l20tb_domain_recoverability_from47823_len19520_20261018'),
            ('l20tb_complement_bio5tb', 'l20tb_complement_bio5tb_from47823_len19520_qi_20261018')):
            points.append(dict(arm=arm, checkpoint_id=str(ck),
                path=str(RUNS / folder / 'eval' / f'training_{ck}' / 'teacher_checkpoint.pth'),
                config=str(RUNS / folder / 'candidate.yaml')))
    save(OUT / 'admission_plan.json', dict(points=points, sparse_points=[50263, 52703, 57583, 67343],
        rationale='One strongest historical selected29 5TB reference, one balanced old20TB reference, two innovations; no weight sweeps',
        automatic_7b_promotion=False, promotion='Require completed plain7B baseline and validated L improvement first'))
    print(json.dumps(dict(prepared=str(OUT), snapshot=str(SNAPSHOT), datasets=len(datasets))))


def fetch():
    lock = (OUT / 'reference_transfer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for ref in references():
        base = OUT / 'assets' / ref['arm']
        base.mkdir(parents=True, exist_ok=True)
        if (base / 'verified_copy.json').exists():
            continue
        metadata = {}
        for src, filename in ((ref['remote'], 'teacher_checkpoint.pth'), (ref['remote_config'], 'config.yaml')):
            digest = ssh(['sha256sum', src], text=True, capture_output=True, timeout=300).stdout.split()[0]
            dest = base / filename
            subprocess.run(['rsync', '-aL', '--partial', '--delay-updates',
                            '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                            '5090-lyx-xr:' + src, str(dest)], check=True, timeout=3600)
            if sha(dest) != digest:
                raise RuntimeError('Reference transfer hash mismatch: ' + src)
            metadata[filename] = dict(source=src, sha256=digest, bytes=dest.stat().st_size)
        save(base / 'verified_copy.json', dict(host='5090-lyx-xr', files=metadata, verified_unix=time.time()))
        print('REFERENCE_READY', ref['arm'], flush=True)


def maintain_reference_transfer():
    missing = [r['arm'] for r in references()
               if not (OUT / 'assets' / r['arm'] / 'verified_copy.json').exists()]
    path = OUT / 'reference_transfer_state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(attempts=0)
    if not missing:
        state.update(status='COMPLETE', updated_unix=time.time())
    elif Path('/proc/' + str(state.get('pid', -1)) + '/cmdline').exists() and \
            b'manage_l20tb_selected29_20261019.py' in Path('/proc/' + str(state['pid']) + '/cmdline').read_bytes():
        state.update(status='TRANSFERRING', missing=missing, updated_unix=time.time())
    elif state['attempts'] < 3:
        with (OUT / 'reference_transfer.log').open('a') as log:
            p = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'fetch'],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='TRANSFERRING', pid=p.pid, attempts=state['attempts'] + 1,
                     missing=missing, updated_unix=time.time())
    else:
        state.update(status='FAILED_THREE_ATTEMPTS', missing=missing, updated_unix=time.time())
    save(path, state)
    return state


def refresh_remote_screening():
    path = CONTROL / 'curves_20261019/lyx_snapshot.json'
    previous = json.loads(path.read_text())
    if time.time() - previous['captured'] < 300:
        return previous['captured']
    roots = {arm: data['root'] for arm, data in previous['arms'].items()}
    source = (ROOT / 'scripts/score_cooldown_arms_20261013.py').read_text().split('def parse_20tb_csv():')[0]
    code = source + '\nimport time\nroots=' + repr(roots) + '''
arms={}
for arm,path in roots.items():
 root=Path(path);scores=parse_fleet(root);state={}
 for ck in scores:
  done=[p.stem.split('__',1)[1] for p in (root/'_state/done').glob(f'ckpt_{ck}__*.json')]
  state[str(ck)]=dict(done=len(done),done_lanes=done,terminal=len(list((root/'_state/terminal').glob(f'ckpt_{ck}__*.json'))))
 arms[arm]=dict(root=str(root),state=state,scores={str(ck):{'/'.join(k):v for k,v in cells.items()} for ck,cells in scores.items()})
print(json.dumps(dict(captured=time.time(),arms=arms)))
'''
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        '5090-lyx-xr', 'python3', '-'], input=code, text=True,
                       capture_output=True, timeout=45, check=True)
    snapshot = json.loads(p.stdout)
    save(path, snapshot)
    return snapshot['captured']


def admit():
    sys.path.insert(0, str(SNAPSHOT / 'scripts'))
    import run_shared_frozen_stage1_20260917 as queue
    state = OUT / '_state'
    state.mkdir(exist_ok=True)
    with (state / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((OUT / 'campaign_manifest.json').read_text())
        queue.verify_campaign_source(manifest)
        points = json.loads((OUT / 'admission_plan.json').read_text())['points']
        present = {(a['arm'], a['checkpoint_id']) for a in manifest['checkpoint_assets']}
        added = []
        for point in points:
            key = (point['arm'], point['checkpoint_id'])
            if key in present:
                continue
            path = Path(point['path'])
            if not path.is_file() or path.stat().st_size < 1000000 or time.time() - path.stat().st_mtime < 120:
                continue
            if point.get('readiness_marker') and not Path(point['readiness_marker']).exists():
                continue
            config = OUT / 'frozen_configs' / (point['arm'] + '.yaml')
            config.parent.mkdir(exist_ok=True)
            if not config.exists():
                shutil.copy2(point['config'], config)
            elif sha(config) != sha(point['config']):
                raise RuntimeError('Training architecture config changed after freeze: ' + point['arm'])
            asset = dict(arm=point['arm'], checkpoint_id=point['checkpoint_id'], path=str(path),
                         config=str(config), kind='dinov3', model_id='', reserve_mib=4096)
            queue.checkpoint_record(OUT, asset)
            manifest['checkpoint_assets'].append(asset)
            added.append(key)
        # Interleave reference/candidate cells and complete the small families
        # before the longest classification fits. This changes ordering only.
        rank = {(p['arm'], p['checkpoint_id']): i for i, p in enumerate(points)}
        assets = sorted(manifest['checkpoint_assets'], key=lambda a: rank[a['arm'], a['checkpoint_id']])
        priority = {'breastmnist': 0, 'bbbc013': 1, 'nct-crc-he-1k': 2, 'cellpose': 3,
                    'hpa-subcellular': 4, 'rxrx1-cross': 5, 'bbbc005': 6, 'crc-val-he-7k': 7}
        tasks = []
        for spec in sorted(manifest['datasets'], key=lambda d: priority.get(d['dataset'], 10)):
            for asset in assets:
                key = f"{asset['arm']}_ck{asset['checkpoint_id']}__{spec['task']}__{spec['dataset']}"
                if spec['task'] == 'segmentation':
                    key += '__primary-last__' + spec['split']
                tasks.append(dict(key=key, asset=asset, dataset=spec))
        if added or tasks != manifest['tasks']:
            manifest['tasks'] = tasks
            save(OUT / 'campaign_manifest.json', manifest)
        return added


def watch():
    lock = (OUT / 'manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        status = dict(pid=os.getpid(), updated_unix=time.time())
        try:
            status['reference_transfer'] = maintain_reference_transfer()
            status['admitted'] = admit()
            status['state'] = 'WATCHING_SPARSE_CHECKPOINTS'
            try:
                status['remote_screening_captured'] = refresh_remote_screening()
            except Exception as exc:
                status['remote_screening_error'] = repr(exc)
            plot = ROOT / 'scripts/plot_l20tb_selected29_20261019.py'
            if plot.exists():
                p = subprocess.run([PYTHON, str(plot)], text=True, capture_output=True, timeout=90)
                status['plot_returncode'] = p.returncode
                if p.returncode:
                    status['plot_error'] = p.stderr[-3000:]
            screening = subprocess.run([PYTHON, str(ROOT / 'scripts/plot_20tb_progress_20261019.py')],
                                       text=True, capture_output=True, timeout=90)
            status['screening_plot_returncode'] = screening.returncode
            if screening.returncode:
                status['screening_plot_error'] = screening.stderr[-3000:]
            state_path = OUT / '_state/workers/deepcad.json'
            status['worker'] = json.loads(state_path.read_text()) if state_path.exists() else None
            pause = OUT / '_state/PAUSED.json'
            if pause.exists():
                status.update(state='EVALUATOR_PAUSED', pause=json.loads(pause.read_text()))
        except Exception as exc:
            status.update(state='ERROR', error=repr(exc))
        save(OUT / 'manager_status.json', status)
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'fetch', 'admit', 'watch'])
    args = parser.parse_args()
    result = globals()[args.mode]()
    if result:
        print(result)
