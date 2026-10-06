#!/usr/bin/env python3
"""Summarize only validated, same-step pairs in the resident 20TB campaign."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import time


def summarize(root, output):
    manifest = json.loads((root / 'campaign_manifest.json').read_text())
    values = {}
    done = list((root / '_state/done').glob('*.json'))
    for path in done:
        report = json.loads(path.read_text())
        if report.get('status') != 'VALID_COMPLETE':
            continue
        cell = root / 'cells' / path.stem
        inv = json.loads((cell / 'invocation_manifest.json').read_text())
        if inv['source_snapshot_sha256'] != manifest['source_snapshot']['sha256']:
            raise ValueError('Evaluator source mismatch: ' + path.stem)
        arm_ck = path.stem.split('__')[0]
        arm, ck = arm_ck.rsplit('_ck', 1)
        spec = inv['dataset']
        ds, family = spec['dataset'], spec['task']
        if family == 'segmentation':
            fits = collections.defaultdict(list)
            for res in (cell / 'results').rglob('results.json'):
                result = json.loads(res.read_text())
                fits[int(result['_meta']['probe_epochs'])].append(float(result['test']['mDice']))
            for budget, metrics in fits.items():
                if len(metrics) != 3:
                    raise ValueError('Incomplete segmentation seeds')
                values[(arm, int(ck), family, ds, 'mDice', spec['split'], f'E{budget}')] = statistics.mean(metrics)
            continue
        result = json.loads((cell / 'component_result.json').read_text())
        for row in result.get('rows', [result]):
            metrics = ('recall_at_1', 'nmi') if family == 'retrieval' else ('r2',) if family == 'regression' else ('macro_auc',) if ds == 'chestmnist' else ('balanced_accuracy',)
            for metric in metrics:
                if metric not in row:
                    continue
                key = (arm, int(ck), family, ds, metric, row.get('split', row.get('protocol', '')), row.get('aggregation', ''))
                values[key] = float(row[metric])
    rows = []
    for key, method in sorted(values.items()):
        arm, ck, family, ds, metric, split, variant = key
        if arm != 'cls_slow2_20tb':
            continue
        baseline = values.get(('noGRAM20tb', *key[1:]))
        if baseline is not None:
            rows.append(dict(checkpoint=ck, family=family, dataset=ds, metric=metric, split=split,
                             variant=variant, method=method, baseline=baseline, delta=method-baseline,
                             delta_times_100=100*(method-baseline)))
    output.mkdir(parents=True, exist_ok=True)
    fields = ['checkpoint','family','dataset','metric','split','variant','method','baseline','delta','delta_times_100']
    with (output / 'paired_deltas.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['family'], row['dataset'], row['metric'], row['variant'])].append(row)
    lines = ['# 20TB 同步数评测：阶段结果', '',
             f'UTC {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}；已校验 {len(done)}/{len(manifest["tasks"])} 个任务。', '',
             '仅包含双方均完成且通过校验的相同步数结果。队列仍在运行；以下不是完整 v4 结论，也不能用于挑选最佳 checkpoint。', '',
             '| 任务 | 数据集 | 指标 | 设置 | 配对数 | 均值 Δ×100 | 胜/负 |', '|---|---|---|---|---:|---:|---:|']
    for (fam,ds,metric,variant), vals in sorted(groups.items()):
        diffs = [v['delta_times_100'] for v in vals]
        lines.append(f'| {fam} | {ds} | {metric} | {variant} | {len(vals)} | {statistics.mean(diffs):+.3f} | {sum(x>0 for x in diffs)}/{sum(x<0 for x in diffs)} |')
    lines += ['', '分类/检索/分割的 Δ×100 为百分点；R² 的该列只是差值乘100，不是准确率百分点。跨 checkpoint 的结果相关；本表未进行显著性检验。']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/v2_20tb_paired_20261006'))
    p.add_argument('--output', type=Path, default=Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/deepcad_method_20260927/progress_20261006/20tb'))
    a=p.parse_args();summarize(a.root,a.output)
