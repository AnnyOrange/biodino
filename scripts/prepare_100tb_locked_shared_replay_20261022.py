#!/usr/bin/env python3
"""Build an isolated 100TB shared-task replay from the recorded source archive."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONTROL = REPO / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
ARCHIVE = REPO / 'outputs/03_source_archives/100tb_fullpool_v4_20261013/replay_root'
UNRELATED = {
    'dinov3/eval/eval_ood/metrics.py',
    'dinov3/eval/bio_detection/center_probe.py',
    'scripts/run_unified_paper_global.py',
    'scripts/unified_paper_queue.py',
    'scripts/accelerate_20tb_selected29_20261021.py',
    'scripts/build_unified_paper_report.py',
    'scripts/unified_paper_global_statistics.py',
    'scripts/snapshot_unified_paper_resources.py',
    'scripts/unified_paper_dense_statistics.py',
    'scripts/unified_paper_statistics.py',
    'scripts/run_unified_paper_detection.py',
    'scripts/run_ctc_native_full_hs6.py',
    'scripts/plot_hs6_five_scales_selected29_20261020.py',
    'scripts/run_20tb_selected29_fill_worker_20261021.py',
    'scripts/run_unified_paper_ood.py',
    'scripts/watch_unified_paper.py',
    'scripts/run_unified_paper_segmentation.py',
    'scripts/run_new_fm_v4_segmentation_20261016.py',
    'scripts/manage_20tb_selected29_fill_20261021.py',
    'scripts/unified_paper_dense_reporting.py',
    'scripts/run_unified_paper_ctc.py',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=(41499, 41999), required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    shared = CONTROL / f'campaigns/100tb_ck{args.step}/shared'
    manifest_path = shared / 'campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if {task['dataset']['task'] for task in manifest['tasks']} - {
            'classification', 'regression', 'retrieval', 'segmentation'}:
        raise RuntimeError('Unexpected task family in shared V4 campaign')
    original = manifest['source_snapshot']
    root = ARCHIVE.parent / f'replay_ck{args.step}'
    selected, revisions = {}, {}
    for name, expected in original['files'].items():
        archived, current = ARCHIVE / name, REPO / name
        if digest(archived) == expected:
            selected[name] = archived
        elif digest(current) == expected:
            selected[name] = current
        elif name in UNRELATED:
            source = archived if archived.is_file() else current
            if not source.is_file():
                raise FileNotFoundError(name)
            selected[name] = source
            revisions[name] = {'recorded': expected, 'replay': digest(source)}
        else:
            raise RuntimeError(f'Required V4 shared source unavailable: {name}')
    if not revisions or set(revisions) - UNRELATED:
        raise RuntimeError('Unexpected source revision set')
    if args.apply:
        if root.exists():
            raise FileExistsError(root)
        shutil.copytree(ARCHIVE, root)
        for name, source in selected.items():
            destination = root / name
            if digest(destination) != digest(source):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for name, source in selected.items():
            if digest(root / name) != digest(source):
                raise RuntimeError(f'Copy verification failed: {name}')
        updated = dict(manifest)
        snapshot = dict(original)
        snapshot['files'] = {name: digest(root / name) for name in original['files']}
        snapshot['sha256'] = hashlib.sha256(json.dumps(snapshot['files'], sort_keys=True).encode()).hexdigest()
        updated['source_snapshot'] = snapshot
        updated['source_snapshot_path'] = str(root)
        updated.setdefault('source_revisions', []).append({
            'reason': 'Isolated archived replay for shared V4 tasks; unrelated OOD, detection, and reporting files differ',
            'original_snapshot_sha256': original['sha256'],
            'replay_snapshot_sha256': snapshot['sha256'],
            'changed_files': revisions,
            'replay_root': str(root),
        })
        temporary = manifest_path.with_name(f'.{manifest_path.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(updated, indent=2) + '\n')
        os.replace(temporary, manifest_path)
    print(json.dumps({'step': args.step, 'apply': args.apply, 'replay_root': str(root),
                      'unchanged_evaluator_files': len(selected) - len(revisions),
                      'unrelated_revisions': revisions}, indent=2))


if __name__ == '__main__':
    main()
