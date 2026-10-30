#!/usr/bin/env python3
"""Refresh existing paper tables from validated current native and global outputs."""
import csv
import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook
from report_ood_scope import refresh as refresh_ood_scope


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
NATIVE = {'ctc': 'cell_tracking', 'nucls_qc': 'nuclei_detection_segmentation',
          'bbbc038-official': 'instance_segmentation',
          'bccd': 'box_detection', 'bbbc041': 'box_detection',
          'xray_chest': 'box_detection'}


def write_csv(path, rows, columns):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metric_from_result(row, payload):
    dataset, metric = row['dataset'], row['metric']
    if dataset == 'ctc':
        if (payload.get('n_domains') != 4 or
                any(item.get('status') != 'VALID_COMPLETE' for item in payload.get('domain_rows', []))):
            return None
        return payload.get('macro', {}).get(metric)
    if dataset == 'nucls_qc':
        if not Path(payload.get('prediction_path', '')).is_file():
            return None
        return payload.get('metrics', {}).get(metric)
    if dataset == 'bbbc038-official':
        if not Path(payload.get('prediction_path', '')).is_file():
            return None
        return payload.get('stages', {}).get(row['fold'], {}).get('metrics', {}).get(metric)
    if dataset in ('bccd', 'bbbc041', 'xray_chest'):
        protocol = ('rsna-native-fixed-patient-purged-v1' if dataset == 'xray_chest'
                    else 'unified-native-box-primary-last-v1')
        if (payload.get('model') != row['model'] or payload.get('dataset') != dataset or
                payload.get('protocol') != protocol or
                not Path(payload.get('prediction_path', '')).is_file() or
                not Path(payload.get('manifest', '')).is_file() or
                hashlib.sha256(Path(payload['manifest']).read_bytes()).hexdigest() !=
                payload.get('manifest_sha256')):
            return None
        return payload.get('metrics', {}).get(metric)
    if (payload.get('protocol') == 'bio-eval-union-v4' and
            Path(payload.get('prediction_path', '')).is_file() and
            Path(payload.get('sample_manifest', '')).is_file()):
        return payload.get(metric)
    return None


def native_results(model, dataset, fold):
    base = OUT / 'results' / model
    if dataset == 'ctc':
        pattern = base / 'ctc' / f'fold{fold}'
    elif dataset == 'nucls_qc':
        pattern = base / 'nucls_qc' / f'fold{fold}'
    elif dataset in ('bccd', 'bbbc041'):
        pattern = base / 'native_box' / dataset
    elif dataset == 'xray_chest':
        pattern = base / 'rsna_pneumonia'
    else:
        pattern = base / 'bbbc038_official'
    results = []
    for path in sorted(pattern.glob('seed*/component_result.json')):
        try:
            payload = json.loads(path.read_text())
            seed = int(path.parent.name.removeprefix('seed'))
            if seed in range(5) and payload.get('seed') == seed:
                results.append((seed, path, payload))
        except (ValueError, json.JSONDecodeError, OSError):
            continue
    return results


def supplemental_measure(row):
    model, dataset, metric = row['model'], row['dataset'], row['metric']
    if row['task'] == 'bone_age_regression':
        path = OUT / 'results' / model / 'bone_age/component_result.json'
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if (payload.get('model') != model or payload.get('dataset') != dataset or
                payload.get('protocol') != row['protocol'] or
                payload.get('evaluation_cohort') != 'official_validation' or
                payload.get('n_test') != 1425 or payload.get('seed_applicable') is not False or
                payload.get('manifest_sha256') != row['split_sha256'] or
                not Path(payload.get('prediction_path', '')).is_file() or
                not Path(payload.get('manifest', '')).is_file()):
            return None
        value = payload.get('metrics', {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return [(None, float(value), path, payload['prediction_path'])], payload
    if row['task'] not in ('retrieval', 'clustering'):
        return None
    path = OUT / 'results' / model / 'retrieval_clustering' / dataset / 'component_result.json'
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    manifest = Path(payload.get('manifest', ''))
    if (payload.get('model') != model or payload.get('dataset') != dataset or
            not payload.get('protocol') or not manifest.is_file() or
            hashlib.sha256(manifest.read_bytes()).hexdigest() != payload.get('manifest_sha256')):
        return None
    paths = ([payload['retrieval_result']] if row['task'] == 'retrieval' else
             payload.get('clustering_results', []))
    measures = []
    for item in paths:
        child_path = Path(item)
        if not child_path.is_file():
            return None
        child = json.loads(child_path.read_text())
        prediction = child.get('prediction_path', '')
        value = child.get('metrics', {}).get(metric)
        if (child.get('model') != model or child.get('dataset') != dataset or
                child.get('manifest_sha256') != payload['manifest_sha256'] or
                not prediction or not Path(prediction).is_file() or
                not isinstance(value, (int, float)) or not math.isfinite(value)):
            return None
        measures.append((child.get('seed'), float(value), child_path, prediction))
    if row['task'] == 'retrieval' and len(measures) != 1:
        return None
    if row['task'] == 'clustering' and sorted(item[0] for item in measures) != list(range(5)):
        return None
    return measures, payload


def main():
    table = REPORT / 'paper_results.csv'
    with table.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    archive = REPORT / 'audits/paper_results_before_live_refresh.csv'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(table, archive)
    cache = {}
    changed = []
    corrections = []
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        dataset, model, fold = row['dataset'], row['model'], row['fold']
        if dataset in NATIVE and row['task'] == NATIVE[dataset]:
            key = (model, dataset, fold)
            if key not in cache:
                cache[key] = native_results(*key)
            measures = []
            for seed, path, payload in cache[key]:
                value = metric_from_result(row, payload)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    measures.append((seed, float(value), path, payload))
            if not measures:
                continue
            scores = [item[1] for item in measures]
            mean = statistics.mean(scores)
            old_value = row.get('value', '')
            old_count = row.get('completed_seeds', '')
            if old_count == str(len(measures)) and old_value and math.isclose(float(old_value), mean, abs_tol=1e-12):
                continue
            row.update(value=str(mean), sd=str(statistics.stdev(scores)) if len(scores) > 1 else '',
                       completed_seeds=str(len(measures)), required_seeds='5',
                       seed_ids=json.dumps([item[0] for item in measures]),
                       seed_values=json.dumps(scores),
                       raw_result_path=json.dumps([str(item[2]) for item in measures]),
                       prediction_path=json.dumps([item[3].get('prediction_path',
                                                   str(item[2].parent / 'domains')) for item in measures]),
                       display=(f'{mean:.6f} ± {statistics.stdev(scores):.6f}' if len(scores) > 1
                                else f'{mean:.6f}') + ('' if len(scores) == 5 else f' ({len(scores)}/5 seed)'),
                       status='部分完成', metric_status='partial' if len(scores) < 5 else 'complete',
                       display_status=f'{len(scores)}/5 seed',
                       reason=f'{len(scores)}/5 downstream fits complete; statistics must match these predictions',
                       ci_low='', ci_high='', statistics_status='recompute grouped CI and paired comparisons',
                       snapshot_utc=now, protocol_verified='True')
            changed.append((model, dataset, fold, row['metric']))
        elif row['task'] in ('bone_age_regression', 'retrieval', 'clustering') and (
                not row.get('value') or row['task'] == 'retrieval'):
            try:
                evidence = supplemental_measure(row)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                evidence = None
            if evidence is None:
                continue
            measures, payload = evidence
            values = [item[1] for item in measures]
            mean = statistics.mean(values)
            if row.get('value'):
                if math.isclose(float(row['value']), mean, abs_tol=1e-12):
                    continue
                corrections.append({'model': model, 'dataset': dataset, 'metric': row['metric'],
                                    'old_value': row['value'], 'old_result': row.get('result', ''),
                                    'new_value': mean, 'new_result': str(measures[0][2]),
                                    'new_prediction': measures[0][3],
                                    'reason': 'Old score-only retrieval result disagrees with current fixed-manifest sample predictions'})
            sd = statistics.stdev(values) if len(values) > 1 else None
            trained = row['task'] == 'clustering'
            row.update(value=str(mean), sd=str(sd) if sd is not None else '',
                       display=f'{mean:.6f} ± {sd:.6f}' if sd is not None else f'{mean:.6f} (CI pending)',
                       required_seeds='5' if trained else '0',
                       completed_seeds='5' if trained else '0',
                       seed_ids=json.dumps([item[0] for item in measures]) if trained else '[]',
                       seed_values=json.dumps(values) if trained else '[]',
                       seed_semantics='MiniBatchKMeans downstream randomness' if trained else 'not_applicable',
                       raw_result_path=(json.dumps([str(item[2]) for item in measures]) if trained
                                        else str(measures[0][2])),
                       result=str(measures[0][2]) if row['task'] == 'retrieval' else row.get('result', ''),
                       predictions=measures[0][3] if row['task'] == 'retrieval' else row.get('predictions', ''),
                       prediction_path=(json.dumps([item[3] for item in measures]) if row['task'] != 'bone_age_regression'
                                        else measures[0][3]),
                       sample_manifest=payload['manifest'], split_id=payload['protocol'],
                       split_sha256=payload['manifest_sha256'], protocol_verified='True',
                       status='部分完成', metric_status='complete',
                       reason='Metric and predictions complete; grouped CI and paired comparison pending',
                       ci_low='', ci_high='', statistics_status='grouped CI and paired comparison pending',
                       snapshot_utc=now)
            changed.append((model, dataset, fold, row['metric']))
        elif row['task'] in ('classification', 'regression'):
            path = OUT / 'results' / model / 'global' / dataset / 'component_result.json'
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text())
                value = metric_from_result(row, payload)
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            if not (Path(payload['prediction_path']).is_file() and
                    Path(payload['sample_manifest']).is_file()):
                continue
            if (row.get('value') and math.isclose(float(row['value']), value, abs_tol=1e-12)
                    and row.get('metric_status') == 'complete'):
                continue
            row.update(value=str(value), display=f'{value:.6f} (CI pending)',
                       completed_seeds='0', required_seeds='0', seed_semantics='not_applicable',
                       raw_result_path=str(path), prediction_path=payload['prediction_path'],
                       sample_manifest=payload['sample_manifest'],
                       status='部分完成', metric_status='complete',
                       reason='Deterministic score and predictions complete; grouped CI pending',
                       ci_low='', ci_high='', statistics_status='grouped CI and paired comparison pending',
                       snapshot_utc=now, protocol_verified='True')
            changed.append((model, dataset, fold, row['metric']))
    for row in rows:
        if row['task'] not in ('classification', 'regression', 'retrieval') or not row.get('value'):
            continue
        model, dataset, metric = row['model'], row['dataset'], row['metric']
        folder = OUT / 'statistics'
        if row['task'] == 'retrieval':
            folder /= 'retrieval'
        path = folder / f'{model}__{dataset}__ci.json'
        if not path.is_file():
            continue
        try:
            stat = json.loads(path.read_text())
            measure = stat['metrics'][metric]
            point = float(measure['value'])
            if (stat.get('model') != model or stat.get('dataset') != dataset or
                    not math.isclose(point, float(row['value']), abs_tol=1e-7) or
                    not all(math.isfinite(float(measure[key])) for key in ('ci_low', 'ci_high'))):
                continue
            if row['task'] == 'retrieval':
                result = json.loads(Path(row['raw_result_path']).read_text())
                prediction = Path(result['prediction_path'])
                valid_source = hashlib.sha256(prediction.read_bytes()).hexdigest() == stat['prediction_sha256']
            else:
                source = OUT / 'results' / model / 'global' / dataset / 'component_result.json'
                signature = stat.get('provenance', {}).get('component_sha256', [])
                valid_source = (source.is_file() and len(signature) == 1 and
                                hashlib.sha256(source.read_bytes()).hexdigest() == signature[0])
            if not valid_source:
                continue
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        low, high = str(measure['ci_low']), str(measure['ci_high'])
        if row.get('ci_low') == low and row.get('ci_high') == high:
            continue
        row.update(ci_low=low, ci_high=high,
                   display=f'{point:.6f} [{float(low):.6f}, {float(high):.6f}]; training seed N/A',
                   statistics_path=str(path),
                   statistics_status='Grouped test-sample CI complete; paired comparisons indexed separately',
                   snapshot_utc=now)
        if row['task'] in ('classification', 'regression'):
            row['reason'] = ('Deterministic score and predictions complete; grouped CI verified; '
                             'paired comparisons indexed separately')
        changed.append((model, dataset, row['fold'], metric))
    if corrections:
        correction_path = REPORT / 'audits/paper_score_corrections.csv'
        previous = list(csv.DictReader(correction_path.open(newline=''))) if correction_path.is_file() else []
        write_csv(correction_path, previous + corrections,
                  ['model', 'dataset', 'metric', 'old_value', 'old_result',
                   'new_value', 'new_result', 'new_prediction', 'reason'])
    if changed:
        seed_rows = []
        for row in rows:
            if row['dataset'] not in NATIVE or not row.get('seed_ids'):
                continue
            try:
                seeds = json.loads(row['seed_ids'])
                values = json.loads(row['seed_values'])
                paths = json.loads(row['raw_result_path'])
            except (ValueError, TypeError):
                continue
            for seed, value, path in zip(seeds, values, paths):
                seed_rows.append({'model': row['model'], 'dataset': row['dataset'],
                                  'task': row['task'], 'fold': row['fold'], 'seed': seed,
                                  'metric': row['metric'], 'value': value, 'result': path})
        write_csv(REPORT / 'native_seed_metrics_current.csv', seed_rows,
                  ['model', 'dataset', 'task', 'fold', 'seed', 'metric', 'value', 'result'])
        write_csv(table, rows, columns)
        wide_path = REPORT / 'paper_results_wide.csv'
        with wide_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            wide_columns = reader.fieldnames
            wide_rows = list(reader)
        changed_set = set(changed)
        displays = {(r['model'], r['task'], r['dataset'], r['budget'], r['fold'], r['metric']): r['display']
                    for r in rows if (r['model'], r['dataset'], r['fold'], r['metric']) in changed_set}
        for wide in wide_rows:
            for model in wide.keys() - {'task', 'dataset', 'budget', 'fold', 'metric', 'unit'}:
                key = (model, wide['task'], wide['dataset'], wide['budget'], wide['fold'], wide['metric'])
                if key in displays:
                    wide[model] = displays[key]
        write_csv(wide_path, wide_rows, wide_columns)
        book = Workbook(write_only=True)
        for title, records, header in (
                ('Results', rows, columns),
                ('Execution', list(csv.DictReader((REPORT / 'execution_jobs.csv').open(newline=''))), None),
                ('Native seeds', seed_rows, ['model', 'dataset', 'task', 'fold', 'seed', 'metric', 'value', 'result']),
                ('Seed0', list(csv.DictReader((REPORT / 'seed0_first_pass.csv').open(newline=''))), None)):
            sheet = book.create_sheet(title)
            header = header or list(records[0])
            sheet.append(header)
            for record in records:
                sheet.append([record.get(column, '') for column in header])
        temporary = REPORT / 'paper_results.tmp.xlsx'
        book.save(temporary)
        temporary.replace(REPORT / 'paper_results.xlsx')
    refresh_ood_scope(REPORT)
    audit = REPORT / 'audits/live_paper_refresh.json'
    audit.write_text(json.dumps({'updated': now, 'changed_cells': len(changed),
                                 'changed': changed, 'original_table': str(archive),
                                 'live_result_root': str(OUT / 'results')}, indent=2) + '\n')
    print(json.dumps({'changed_cells': len(changed), 'paper_csv': str(table),
                      'paper_xlsx': str(REPORT / 'paper_results.xlsx')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        main()
        if not args.watch:
            break
        time.sleep(300)
