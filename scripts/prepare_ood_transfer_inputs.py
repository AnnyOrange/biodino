#!/usr/bin/env python3
"""Export portable, byte-locked development/test manifests from canonical tasks."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import read_json, save_json, sha256


EXPECTED = {
    'bone_age': {'train': 12611, 'val': 1425, 'test': 200},
    'pneumonia': {'train': 24078, 'val': 2606, 'test': 3000},
    'bbbc041': {'train': 1087, 'val': 121, 'test': 120},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=EXPECTED, required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = read_json(args.source_manifest)
    root, output = Path(args.source_root).resolve(), Path(args.output)
    source_hashes = {args.source_manifest: sha256(args.source_manifest)}
    sex = {}
    if args.task == 'bone_age':
        for path in original['label_paths']:
            source_hashes[path] = sha256(path)
            with open(path, newline='') as stream:
                for row in csv.DictReader(stream):
                    sid = Path(row.get('id', row.get('Image ID', row.get('image_ID', '')))).stem
                    value = row.get('male', row.get('sex', '')).strip().lower()
                    if value not in ('true', 'false', 'm', 'f') or sid in sex:
                        raise ValueError('Invalid or duplicate sex metadata')
                    sex[sid] = int(value in ('true', 'm'))
    rows, ids, files = [], set(), []
    for index, original_row in enumerate(original['records']):
        row = dict(original_row)
        sid = str(row['sample_id'])
        if sid in ids:
            raise ValueError('Duplicate canonical ID')
        ids.add(sid)
        path = Path(row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source image outside source root')
        row.update(sample_id=sid, path=str(path.relative_to(root)), input_sha256=sha256(path), bytes=path.stat().st_size)
        row['split'] = {'official_validation': 'val', 'official_test': 'test'}.get(row['split'], row['split'])
        row['group_id'] = str(row.get('group_id', row.get('group', sid)))
        if args.task == 'bone_age':
            if sid not in sex:
                raise ValueError('Missing author sex value: ' + sid)
            row['sex'] = sex[sid]
        rows.append(row)
        files.append(row['path'])
        if index % 1000 == 0:
            print('HASHED', args.task, index, len(original['records']), flush=True)
    if dict(Counter(row['split'] for row in rows)) != EXPECTED[args.task]:
        raise ValueError('Canonical population changed')
    output.mkdir(parents=True, exist_ok=True)
    for role, splits in [('development', {'train', 'val'}), ('test', {'test'})]:
        selected = [row for row in rows if row['split'] in splits]
        value = dict(schema='ood-transfer-inputs-v1', task=args.task, role=role,
                     records=selected, classes=original.get('classes', []),
                     source_manifest_sha256=sha256(args.source_manifest), source_hashes=source_hashes,
                     split_counts=dict(Counter(row['split'] for row in selected)),
                     grouping=original.get('grouping', original.get('limitation', 'Source image identity')),
                     limitation=original.get('limitation'), source_member_identity_reaudit='canonical source membership preserved; not new server certification')
        target = output / f'{args.task}_{role}.json'
        if target.exists() and read_json(target) != value:
            raise ValueError('Refusing to overwrite changed manifest: ' + str(target))
        save_json(target, value)
    (output / f'{args.task}_files.txt').write_text('\n'.join(files) + '\n')
    save_json(output / f'{args.task}_source_summary.json', dict(task=args.task, files=len(rows),
              bytes=sum(row['bytes'] for row in rows), source_root=str(root), source_hashes=source_hashes))


if __name__ == '__main__':
    main()
