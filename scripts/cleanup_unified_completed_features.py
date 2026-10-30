#!/usr/bin/env python3
"""Remove reconstructable global feature arrays after validated predictions exist."""
import argparse
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
AUDIT = OUT / 'audits/completed_global_feature_cleanup.jsonl'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sweep(delete):
    found = removed = bytes_found = bytes_removed = 0
    for result in (OUT / 'results').glob('*/global/*/component_result.json'):
        try:
            row = json.loads(result.read_text())
            prediction = Path(row['prediction_path'])
            manifest = Path(row['sample_manifest'])
            if (row.get('protocol') != 'bio-eval-union-v4' or not prediction.is_file()
                    or not manifest.is_file() or prediction.stat().st_size == 0):
                continue
            sample_rows = json.loads(manifest.read_text())
            if not sample_rows.get('test') or not sample_rows.get('train'):
                continue
            key = f"{row['model']}__global__{row['dataset']}"
            if (OUT / 'locks' / f'{key}.claimdir').exists():
                continue
            caches = [result.parent / f'{part}_features.npz' for part in ('train', 'test')]
            for cache in caches:
                if not cache.is_file():
                    continue
                size = cache.stat().st_size
                found += 1
                bytes_found += size
                if delete:
                    evidence = {'task': key, 'result': str(result), 'result_sha256': sha256(result),
                                'prediction': str(prediction), 'manifest': str(manifest),
                                'removed': str(cache), 'bytes': size, 'time': time.time()}
                    cache.unlink()
                    AUDIT.parent.mkdir(parents=True, exist_ok=True)
                    with AUDIT.open('a') as stream:
                        stream.write(json.dumps(evidence) + '\n')
                    removed += 1
                    bytes_removed += size
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    print(json.dumps({'deletion_enabled': delete, 'eligible_files': found,
                      'eligible_gib': round(bytes_found / 2**30, 3),
                      'removed_files': removed, 'removed_gib': round(bytes_removed / 2**30, 3),
                      'audit': str(AUDIT)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        sweep(args.delete)
        if not args.watch:
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
