#!/usr/bin/env python3
"""Prune segmentation feature arrays only after every final fit is retained."""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
CACHE_ROOT = OUT / 'segmentation_cache_recovery'
AUDIT = OUT / 'audits/completed_segmentation_feature_cleanup.jsonl'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def completed(model, family, dataset, split):
    for result_family in ((family,) if family in ('segmentation', 'segmentation_refit')
                          else ('segmentation', 'segmentation_refit')):
        base = OUT / 'results' / model / result_family / dataset / split
        paths = [base / f'E{budget}' / f'seed{seed}' / 'component_result.json'
                 for budget in (20, 50) for seed in range(5)]
        if all(path.is_file() and (record := read(path)).get('split') == split
               and Path(record.get('prediction_path', '')).is_file()
               for path in paths):
            return True
    return False


def candidates():
    live = read(OUT / 'resource_snapshot.json')
    if time.time() - live.get('observed_at_controller', 0) >= 180:
        raise RuntimeError('Resource snapshot stale; cannot verify active tasks')
    queue_identity = {job['key']: (job['model'], job['dataset'])
                      for queue in OUT.glob('*queue.json')
                      for job in read(queue).get('tasks', [])}
    active = {queue_identity[key] for key in live.get('live_tasks', {})
              if key in queue_identity}
    for model_dir in CACHE_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        for path in model_dir.rglob('*.npz'):
            parts = path.relative_to(CACHE_ROOT).parts
            # FM: model/segmentation/dataset/split/cache/dataset/model/{part}.npz.
            if len(parts) == 8 and parts[1] in ('segmentation', 'segmentation_refit'):
                model, family, dataset, split = parts[:4]
            # BioDINO: model/dataset/split/cache/{part}.npz.
            elif len(parts) == 5 and parts[3] == 'cache':
                model, dataset, split = parts[:3]
                family = 'segmentation'
            else:
                continue
            if (model, dataset) in active or not completed(model, family, dataset, split):
                continue
            yield path, dict(model=model, dataset=dataset, split=split, family=family)


def run(delete):
    count = size = 0
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    for path, identity in candidates():
        try:
            nbytes = path.stat().st_size
        except FileNotFoundError:
            continue
        count += 1
        size += nbytes
        if delete:
            with AUDIT.open('a') as audit:
                audit.write(json.dumps(dict(path=str(path), bytes=nbytes, checked=identity,
                                            action='delete', time=time.time())) + '\n')
                audit.flush()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    print(json.dumps(dict(delete=delete, files=count, gib=round(size / 2**30, 3),
                          audit=str(AUDIT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            run(args.delete)
        except Exception as error:
            print(json.dumps(dict(error=repr(error), time=time.time())), flush=True)
        if not args.watch:
            break
        time.sleep(300)
