#!/usr/bin/env python3
"""Merge live and completed 20TB intervention V4 results into the historical report."""
from __future__ import annotations

import csv
import json
import runpy
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
PAIRED = REPO / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
FULL = REPO / 'outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012'
BASELINE = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
REPORT = REPO / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
OUT = REPORT / '20tb_resampling_cls_v4_20261012'
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc', 'micro_auc',
           'macro_average_precision', 'recall_at_1', 'nmi', 'mae', 'r2', 'spearman',
           'auroc', 'average_precision', 'test_patch_f1', 'dice', 'iou', 'TRA', 'SEG')


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def task_state(root: Path, key: str):
    claim = read_json(root / '_state/claims' / key / 'status.json')
    done = read_json(root / '_state/done' / f'{key}.json')
    if done:
        return 'VALID_COMPLETE', {**claim, **done}
    if (root / '_state/failed_resource' / f'{key}.json').exists():
        return 'FAILED', {**claim, **read_json(root / '_state/failed_resource' / f'{key}.json')}
    if (root / '_state/running' / f'{key}.json').exists():
        return 'RUNNING', {**claim, **read_json(root / '_state/running' / f'{key}.json')}
    return 'QUEUED', {}


def read_summary(cell: Path):
    path = cell / 'summary.csv'
    if not path.is_file():
        return {}
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows[0] if len(rows) == 1 and not rows[0].get('error') else {}


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float('inf') else None


def extension_metrics(task, root: Path):
    done = task.get('done', {})
    kind = done.get('type')
    path = Path(done.get('path', ''))
    rows = []

    def add(metric, value, stratum=''):
        value = numeric(value)
        if value is not None:
            rows.append(dict(metric=metric, value=value, stratum=stratum))

    if kind == 'summary_csv' and path.is_file():
        with path.open(newline='') as stream:
            values = list(csv.DictReader(stream))
        if len(values) == 1 and not values[0].get('error'):
            for metric in METRICS:
                add(metric, values[0].get(metric))
        if task.get('dataset') == 'rxrx3-core':
            result_path = (root / 'retrieval/rxrx3-core' / task['arm'] /
                           'models' / task['arm'] / 'results.json')
            result = read_json(result_path)
            for metric, value in result.get('tests', {}).get('rxrx3', {}).items():
                if metric not in ('protocol_id', 'clustering_method', 'status', 'proxy'):
                    add(metric, value)
    elif kind == 'detection_json' and path.is_file():
        for metric, value in read_json(path).items():
            if metric.startswith('test_'):
                add(metric, value)
    elif kind == 'ood_json' and path.is_file():
        for metric, value in read_json(path).items():
            key = metric.lower()
            if isinstance(value, (int, float)) and any(token in key for token in
                    ('accuracy', 'f1', 'auroc', 'average_precision', 'spearman', '_r2', 'mae')):
                add(metric, value)
    elif kind == 'ctc_native' and path.is_file():
        result = read_json(path)
        for metric, value in result.get('macro', {}).items():
            add(metric, value, 'macro')
        for domain in result.get('domain_rows', []):
            name = domain.get('domain', '')
            for metric, value in domain.get('ctc_metrics', {}).items():
                if metric != 'Valid':
                    add(metric, value, name)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = runpy.run_path(str(REPO / 'scripts/run_v2_full_v4_20261007.py'))
    full['inventory'].__globals__.update(PAIRED=PAIRED, ROOT=FULL)
    full['inventory']()

    paired_manifest = read_json(PAIRED / 'campaign_manifest.json')
    task_rows = []
    score_rows = []
    paired_tasks = {task['key']: task for task in paired_manifest.get('tasks', [])}
    for key, task in paired_tasks.items():
        state, report = task_state(PAIRED, key)
        dataset = task['dataset']
        cell = PAIRED / 'cells' / key
        invocation = read_json(cell / 'invocation_manifest.json')
        summary = read_summary(cell)
        row = dict(campaign='paired-components', task=key, arm=task['asset']['arm'],
                   checkpoint=int(task['asset']['checkpoint_id']), family=dataset['task'],
                   dataset=dataset['dataset'], status=state,
                   host=report.get('host', invocation.get('host', '')),
                   gpu=report.get('gpu', invocation.get('gpu', '')))
        task_rows.append(row)
        if summary:
            for metric in METRICS:
                value = summary.get(metric)
                if value not in (None, ''):
                    try:
                        score_rows.append(dict(task=key, arm=task['asset']['arm'],
                            checkpoint=int(task['asset']['checkpoint_id']), family=dataset['task'],
                            dataset=dataset['dataset'], metric=metric, value=float(value),
                            status=state, source=str(cell / 'summary.csv')))
                    except ValueError:
                        pass

    full_tasks = [read_json(p) for p in sorted((FULL / 'tasks').glob('*.json'))]
    monuseg_score_rows = []
    extension_score_rows = []
    for task in full_tasks:
        key = task['id']
        claim = read_json(FULL / 'claims' / key / 'status.json')
        state = claim.get('state', 'QUEUED')
        task_rows.append(dict(campaign='full-v4-extension', task=key, arm=task['asset']['arm'],
            checkpoint=int(task['asset']['checkpoint_id']), family=task['family'],
            dataset=task['dataset'], status='VALID_COMPLETE' if state == 'DONE' else state,
            host=claim.get('host', ''), gpu=claim.get('gpu', '')))
        if task['dataset'] == 'monuseg':
            cell = FULL / 'cells' / key
            for result_path in sorted(cell.rglob('results.json')):
                result = read_json(result_path)
                relative_parts = result_path.relative_to(cell).parts
                fit = '/'.join(relative_parts[2:4]) if len(relative_parts) >= 4 else result_path.parent.name
                for split, metrics in result.items():
                    if not isinstance(metrics, dict) or split.startswith('_'):
                        continue
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            monuseg_score_rows.append(dict(task=key, arm=task['asset']['arm'],
                                checkpoint=int(task['asset']['checkpoint_id']), fit=fit,
                                split=split, metric=metric, value=float(value), status=state,
                                source=str(result_path)))
                            if split == 'test' and state in ('DONE', 'VALID_COMPLETE'):
                                extension_score_rows.append(dict(task=key, arm=task['asset']['arm'],
                                    checkpoint=int(task['asset']['checkpoint_id']),
                                    family='segmentation', dataset='monuseg', metric=metric,
                                    stratum=fit, value=float(value), status='VALID_COMPLETE',
                                    source=str(result_path)))
        if state in ('DONE', 'VALID_COMPLETE') and task['dataset'] != 'monuseg':
            for measurement in extension_metrics(task, FULL):
                extension_score_rows.append(dict(task=key, arm=task['asset']['arm'],
                    checkpoint=int(task['asset']['checkpoint_id']), family=task['family'],
                    dataset=task['dataset'], metric=measurement['metric'],
                    stratum=measurement['stratum'], value=measurement['value'],
                    status='VALID_COMPLETE', source=str(task.get('done', {}).get('path', ''))))

    write_csv(OUT / 'task_status.csv', task_rows,
              ['campaign', 'task', 'arm', 'checkpoint', 'family', 'dataset', 'status', 'host', 'gpu'])
    write_csv(OUT / 'paired_component_scores.csv', score_rows,
              ['task', 'arm', 'checkpoint', 'family', 'dataset', 'metric', 'value', 'status', 'source'])
    write_csv(OUT / 'monuseg_scores.csv', monuseg_score_rows,
              ['task', 'arm', 'checkpoint', 'fit', 'split', 'metric', 'value', 'status', 'source'])
    write_csv(OUT / 'full_v4_extension_scores.csv', extension_score_rows,
              ['task', 'arm', 'checkpoint', 'family', 'dataset', 'metric', 'stratum',
               'value', 'status', 'source'])

    baseline_manifest = read_json(BASELINE / 'campaign_manifest.json')
    baseline_tasks = {}
    for task in baseline_manifest.get('tasks', []):
        a = task['asset']
        if a['arm'] == 'noGRAM20tb':
            baseline_tasks[(int(a['checkpoint_id']), task['dataset']['task'], task['dataset']['dataset'])] = task
    baseline_rows = []
    for row in score_rows:
        pair = (row['checkpoint'], row['family'], row['dataset'])
        control = baseline_tasks.get(pair)
        if not control:
            continue
        state, _ = task_state(BASELINE, control['key'])
        if state != 'VALID_COMPLETE':
            continue
        summary = read_summary(BASELINE / 'cells' / control['key'])
        value = summary.get(row['metric'])
        try:
            baseline_value = float(value)
        except (TypeError, ValueError):
            continue
        baseline_rows.append(dict(checkpoint=row['checkpoint'], family=row['family'],
            dataset=row['dataset'], metric=row['metric'], baseline_arm='noGRAM20tb',
            baseline=baseline_value, method_arm=row['arm'], method=row['value'],
            delta=row['value'] - baseline_value, status=row['status']))
    write_csv(OUT / 'paired_deltas_vs_nogram.csv', baseline_rows,
              ['checkpoint', 'family', 'dataset', 'metric', 'baseline_arm', 'baseline',
               'method_arm', 'method', 'delta', 'status'])

    baseline_extension = {(row['checkpoint'], row['family'], row['dataset'],
                           row['metric'], row['stratum']): row['value']
                          for row in extension_score_rows if row['arm'] == 'noGRAM20tb'}
    extension_deltas = []
    for row in extension_score_rows:
        if row['arm'] == 'noGRAM20tb':
            continue
        key = (row['checkpoint'], row['family'], row['dataset'], row['metric'], row['stratum'])
        control = baseline_extension.get(key)
        if control is not None:
            extension_deltas.append(dict(checkpoint=row['checkpoint'], family=row['family'],
                dataset=row['dataset'], metric=row['metric'], stratum=row['stratum'],
                baseline_arm='noGRAM20tb', baseline=control, method_arm=row['arm'],
                method=row['value'], delta=row['value'] - control, status=row['status']))
    write_csv(OUT / 'full_v4_extension_deltas_vs_nogram.csv', extension_deltas,
              ['checkpoint', 'family', 'dataset', 'metric', 'stratum', 'baseline_arm',
               'baseline', 'method_arm', 'method', 'delta', 'status'])

    full_inventory = read_json(FULL / 'FULL_V4_INVENTORY_20261007.json')
    states = Counter(row['state'] for row in full_inventory.get('cells', []))
    task_states = Counter(row['status'] for row in task_rows)
    extension_task_states = Counter(row['status'] for row in task_rows
                                    if row['campaign'] == 'full-v4-extension')
    paired_task_states = Counter(row['status'] for row in task_rows
                                 if row['campaign'] == 'paired-components')
    monuseg_task_states = Counter(row['status'] for row in task_rows
                                  if row['campaign'] == 'full-v4-extension' and row['dataset'] == 'monuseg')
    scheduling = paired_manifest.get('scheduling', {})
    card_counts = Counter((row['host'], str(row['gpu'])) for row in task_rows
                          if row['status'] == 'VALID_COMPLETE' and row['host'])
    card_started = Counter((row['host'], str(row['gpu'])) for row in task_rows
                           if row['status'] in ('VALID_COMPLETE', 'RUNNING') and row['host'])
    worker_mem = {}
    for hostdir in (FULL / 'fleet').glob('*/status.json'):
        status = read_json(hostdir)
        for gpu, info in status.get('gpus', {}).items():
            worker_mem[(status.get('worker_host', hostdir.parent.name), str(gpu))] = dict(
                used=int(info.get('used', 0)), total=int(info.get('total', 0)))
    for path in (FULL / 'workers').glob('*.json'):
        status = read_json(path)
        for gpu, info in status.get('gpu_memory', {}).items():
            worker_mem[(path.stem, str(gpu))] = dict(used=int(info.get('used', 0)),
                total=int(info.get('total', 0)), util=int(info.get('util', 0)))
    all_cards = sorted(set(card_counts) | set(worker_mem), key=str)
    card_rows = []
    for host, gpu in all_cards:
        mem = worker_mem.get((host, gpu), {})
        total = mem.get('total', 0)
        used = mem.get('used', 0)
        card_rows.append(dict(host=host, gpu=gpu, tests_started=card_started[(host, gpu)],
                              valid_tests_completed=card_counts[(host, gpu)],
                              memory_used_mib=used if total else 'N/A',
                              memory_total_mib=total if total else 'N/A',
                              memory_percent=round(100 * used / total, 2) if total else 'N/A',
                              gpu_util_percent=mem.get('util', 'N/A')))
    write_csv(OUT / 'per_gpu_status.csv', card_rows,
              ['host', 'gpu', 'tests_started', 'valid_tests_completed', 'memory_used_mib',
               'memory_total_mib', 'memory_percent', 'gpu_util_percent'])

    status_title = globals().get('status_title', '# 20TB 重采样与固定 CLS V4 批测')
    inventory_label = globals().get('inventory_label', '原 intervention 矩阵库存状态（4 个 intervention checkpoint）')
    write_index = globals().get('write_index', True)
    status_text = [status_title, '',
        f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}', '',
        f'训练 checkpoint 数：{len(paired_manifest.get("checkpoint_assets", []))}；V4 清单评测单元：{len(full_inventory.get("cells", []))}。',
        f'共享组件任务状态：{dict(paired_task_states)}。',
        f'全部批测任务状态：{dict(task_states)}。',
        f"调度上限：每卡 {scheduling.get('target_per_gpu', 'unknown')} 个并发任务，显存 admission 上限 {100 * scheduling.get('memory_target', 0):.0f}%。表中显存为最近快照。",
        f'{inventory_label}：{dict(states)}。',
        f'本次 full-V4 扩展执行状态（含同点 No-GRAM 对照）：{dict(extension_task_states)}。',
        f'共享组件有效指标行：{len(score_rows)}；共享组件配对差值：{len(baseline_rows)}。',
        f'full-V4 扩展有效指标行：{len(extension_score_rows)}；同 checkpoint No-GRAM 配对差值：{len(extension_deltas)}。',
        f'MoNuSeg 30/7/14 扩展任务状态：{dict(monuseg_task_states)}；'
        f'有效指标行 {len(monuseg_score_rows)}，见 `monuseg_scores.csv`。', '',
        '|机器/GPU|已启动/已验证|显存 MiB|显存使用率|GPU 利用率|', '|---|---:|---:|---:|---:|']
    for row in card_rows:
        status_text.append(f"|{row['host']} / {row['gpu']}|{row['tests_started']} / {row['valid_tests_completed']}|"
                           f"{row['memory_used_mib']}/{row['memory_total_mib']}|{row['memory_percent']}%|{row['gpu_util_percent']}%|")
    status_text += ['', '测试指标和配对差值见同目录 CSV；MoNuSeg 指标按每个 checkpoint 的六组拟合单独列出。']
    (OUT / 'RESULTS.md').write_text('\n'.join(status_text) + '\n')
    if write_index:
        index = REPORT / '20TB_RESAMPLING_CLS_V4_INDEX.md'
        index.write_text('# 20TB 新方法批测\n\n'
            '重采样与固定 CLS 的 V4 批测状态和配对结果见 '
            '[RESULTS.md](20tb_resampling_cls_v4_20261012/RESULTS.md)。\n')
    print(json.dumps(dict(out=str(OUT), task_states=dict(task_states),
                          v4_inventory_states=dict(states), metric_rows=len(score_rows),
                          paired_deltas=len(baseline_rows), extension_metric_rows=len(extension_score_rows),
                          extension_deltas=len(extension_deltas),
                          monuseg_metric_rows=len(monuseg_score_rows)), indent=2))


if __name__ == '__main__':
    main()
