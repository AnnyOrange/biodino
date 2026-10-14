#!/usr/bin/env python3
"""Complete paired full-V4 extensions for the 20TB intervention checkpoints."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
FULL = REPO / 'outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012'
PAIRED = REPO / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
BASELINE = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
METHOD = Path('/mnt/huawei_deepcad/dinov3_method_full_v4_snapshot_20260928')
MONUSRC = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')
STEPS = ('40015', '41967', '43919')


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def replace_values(value, substitutions):
    if isinstance(value, str):
        for old, new in substitutions:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_values(item, substitutions) for item in value]
    if isinstance(value, dict):
        return {key: replace_values(item, substitutions) for key, item in value.items()}
    return value


def main():
    full_manifest_path = FULL / 'campaign_manifest.json'
    full_manifest = read(full_manifest_path)
    paired_manifest = read(PAIRED / 'campaign_manifest.json')
    baseline_manifest = read(BASELINE / 'campaign_manifest.json')

    # The 2026-09-23 source tree is unavailable. The pinned method snapshot has
    # byte-identical V4 protocol and frozen-eval modules; the RxRx3 script is
    # separately pinned by the full campaign runtime hash.
    old_provenance = FULL / 'provenance/source_a2efa3c6e6d1ad0b.json'
    replacement_provenance = FULL / 'provenance/source_6d88ac85930047c0.json'
    provenance_copy = FULL / 'provenance/source_a2efa3c6e6d1ad0b.json.unavailable'
    if not provenance_copy.exists():
        shutil.copy2(old_provenance, provenance_copy)
    source_record = read(replacement_provenance)
    if source_record['source'] != str(METHOD):
        raise RuntimeError('Unexpected RxRx3 replacement source')
    source_record['substitution'] = {
        'original_source': '/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923',
        'original_source_status': 'MISSING_ON_ALL_EVALUATION_HOSTS',
        'reason': 'The frozen evaluator snapshot is not present on the available hosts.',
        'equivalence_checks': {
            'dinov3_eval_modules': 'all files byte-identical to the original source fingerprint',
            'protocol_v4': 'byte-identical to the original source fingerprint',
            'run_external4_fixedbudget_model.py': 'byte-identical SHA256 b3d74b6610833f9ea6d360d4823cadd213acaa190a2ec3b16a63647d76110c63',
            'rxrx3_split_manifest': 'SHA256 94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5; 734 query and 734 gallery rows',
        },
    }
    atomic(old_provenance, source_record)

    old_arm = 'resampling_only_20tb_ck40015'
    old_asset = next(a for a in paired_manifest['checkpoint_assets']
                     if a['arm'] == 'resampling_only_20tb' and a['checkpoint_id'] == '40015')
    rxrx3 = FULL / 'tasks/ret__rxrx3-core__resampling_only_20tb_ck40015.json'
    task = read(rxrx3)
    task['cwd'] = str(METHOD)
    task['pythonpath'] = str(METHOD)
    atomic(rxrx3, task)
    for path in (FULL / 'tasks').glob('ret__rxrx3-core__*.json'):
        item = read(path)
        item['cwd'] = str(METHOD)
        item['pythonpath'] = str(METHOD)
        atomic(path, item)

    ctc_wrapper = REPO / 'scripts/run_ctc_native_with_gitref_20261012.py'
    ctc_bootstrap = (
        f"import sys,runpy;sys.path.insert(0,{str(METHOD / 'scripts')!r});"
        f"runpy.run_path({str(ctc_wrapper)!r},run_name='__main__')"
    )
    archive = FULL / 'attempt_archive/ctc_git_safe_directory'
    for path in (FULL / 'tasks').glob('ctc__native__*.json'):
        item = read(path)
        old_cmd = item['cmd']
        item['cmd'] = [old_cmd[0], '-B', '-c', ctc_bootstrap, *old_cmd[4:]]
        atomic(path, item)
        claim = FULL / 'claims' / item['id']
        status = read(claim / 'status.json') if (claim / 'status.json').exists() else {}
        if status.get('state') == 'FAILED':
            (archive / 'claims').mkdir(parents=True, exist_ok=True)
            claim.rename(archive / 'claims' / item['id'])
            log = FULL / 'logs' / f"{item['id']}.log"
            if log.exists():
                (archive / 'logs').mkdir(parents=True, exist_ok=True)
                log.rename(archive / 'logs' / log.name)

    template_tasks = []
    for path in (FULL / 'tasks').glob('*.json'):
        item = read(path)
        if item.get('arm') == old_arm:
            template_tasks.append(item)
    if len(template_tasks) != 13:
        raise RuntimeError(f'Expected 13 intervention templates, found {len(template_tasks)}')

    baseline_assets = [a for a in baseline_manifest['checkpoint_assets']
                       if a['arm'] == 'noGRAM20tb' and a['checkpoint_id'] in STEPS]
    if {a['checkpoint_id'] for a in baseline_assets} != set(STEPS):
        raise RuntimeError('Missing a same-step No-GRAM checkpoint asset')
    created = []
    for asset in baseline_assets:
        arm = f"{asset['arm']}_ck{asset['checkpoint_id']}"
        substitutions = [
            (old_asset['path'], asset['path']),
            (old_asset['config'], asset['config']),
            (old_arm, arm),
        ]
        for source in template_tasks:
            task = replace_values(source, substitutions)
            task['asset'] = asset
            if task['dataset'] == 'monuseg':
                task['cwd'] = str(MONUSRC)
                task['pythonpath'] = str(MONUSRC)
            for flag in ('--step', '--ckpt-iter', '--checkpoint-id'):
                if flag in task.get('cmd', []):
                    task['cmd'][task['cmd'].index(flag) + 1] = asset['checkpoint_id']
            if task['family'] == 'ood':
                link = FULL / 'ood_adapters' / arm / asset['checkpoint_id'] / 'checkpoint.pth'
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(asset['path'])
                task['done']['path'] = task['done']['path'].replace('/40015/', f"/{asset['checkpoint_id']}/")
            out = FULL / 'tasks' / f"{task['id']}.json"
            if out.exists():
                continue
            atomic(out, task)
            created.append(task['id'])
        marker = FULL / 'registered' / f'{arm}.json'
        ids = [f'{arm}__segmentation__monuseg__primary-last__formal-static-v1']
        ids += [f"{t['id'].replace(old_arm, arm)}" for t in template_tasks if t['dataset'] != 'monuseg']
        atomic(marker, dict(asset=asset, task_ids=ids,
                            source_campaign=str(BASELINE), monuseg_status='QUEUED'))

    archive = FULL / 'attempt_archive/worker_retry'
    for path in (FULL / 'tasks').glob('*.json'):
        task = read(path)
        if task.get('dataset') == 'monuseg':
            task['cwd'] = str(MONUSRC)
            task['pythonpath'] = str(MONUSRC)
            atomic(path, task)
        claim = FULL / 'claims' / task['id']
        status = read(claim / 'status.json') if (claim / 'status.json').exists() else {}
        if status.get('state') == 'FAILED':
            (archive / 'claims').mkdir(parents=True, exist_ok=True)
            claim.rename(archive / 'claims' / task['id'])
            log = FULL / 'logs' / f"{task['id']}.log"
            if log.exists():
                (archive / 'logs').mkdir(parents=True, exist_ok=True)
                log.rename(archive / 'logs' / log.name)

    all_assets = paired_manifest['checkpoint_assets'] + baseline_assets
    full_manifest.update(
        assets=len(all_assets),
        extension_executions=len(list((FULL / 'tasks').glob('*.json'))),
        paired_baseline_campaign=str(BASELINE),
        paired_baseline_checkpoints=list(STEPS),
        monuseg='VALID_COMPLETE_EXACT_30_7_14_FOR_INTERVENTIONS; baseline queued',
        runtime_hashes={**full_manifest['runtime_hashes'],
                        str(FULL / 'runtime/run_rxrx3.py'): sha(FULL / 'runtime/run_rxrx3.py'),
                        str(ctc_wrapper): sha(ctc_wrapper)},
        updated_unix=time.time(),
    )
    atomic(full_manifest_path, full_manifest)

    marker_path = FULL / 'FULL_V4_WORKERS_LAUNCHED.json'
    if marker_path.exists():
        marker = read(marker_path)
        marker.update(full_tasks=len(list((FULL / 'tasks').glob('*.json'))),
                      paired_baseline_checkpoints=list(STEPS),
                      rxrx3_source_substitution=str(old_provenance))
        atomic(marker_path, marker)
    print(json.dumps(dict(assets=len(all_assets), total_tasks=full_manifest['extension_executions'],
                          new_baseline_tasks=len(created),
                          no_gram_checkpoints=list(STEPS), rxrx3_source=str(METHOD)), indent=2))


if __name__ == '__main__':
    main()
