"""Read existing results only; never dispatch training or evaluation."""
import json
import statistics
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = REPO / 'outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923'
OUT = REPO / 'outputs/00_reports/gram_replacement_design_20260924'


def main():
    report = {'checkpoint': 14151,
              'selection': 'Fixed common scheduled snapshot; audit of already exposed results, not method selection.',
              'arms': {}, 'sources': []}
    for arm in ['vanilla', 'gram', 'fixed', 'adaptive']:
        name = f'{arm}_formal_ck14151'
        cls = {}
        for path in (ROOT / 'frozen').glob(f'*/{name}/last_result.json'):
            row = json.loads(path.read_text())
            if row.get('task') not in ('classification', 'multilabel_classification') or row['dataset'] == 'lc25000':
                continue
            metric = 'macro_auc' if row['dataset'] == 'chestmnist' else 'balanced_accuracy'
            cls[row['dataset']] = {'value': row[metric], 'metric': metric}
            report['sources'].append(str(path))
        assert len(cls) == 24, (name, len(cls))
        item = {'classification_mean_24': statistics.mean(v['value'] for v in cls.values()),
                'classification_by_dataset': cls, 'segmentation_E50': {}, 'detection_patch_f1': {}}
        for ds in ['cellpose', 'tissuenet']:
            paths = list((ROOT / 'segmentation/results').glob(f'hs6_l5_{name}_*/budget50/seed*/{ds}/14151/results.json'))
            assert len(paths) == 3, (name, ds, len(paths))
            rows = [json.loads(p.read_text()) for p in paths]
            item['segmentation_E50'][ds] = {s: statistics.mean(row[s]['mIoU'] for row in rows) for s in ['val', 'test']}
            report['sources'].extend(map(str, paths))
        for ds in ['bbbc038', 'conic', 'livecell']:
            path = ROOT / 'detection' / ds / name / 'results_bio_detection.json'
            row = json.loads(path.read_text())
            item['detection_patch_f1'][ds] = {s: row[f'{s}_patch_f1'] for s in ['val', 'test']}
            report['sources'].append(str(path))
        path = ROOT / 'retrieval/rxrx3-core' / name / 'models' / name / 'results.json'
        item['rxrx3_recall_at_1'] = json.loads(path.read_text())['tests']['rxrx3']['recall_at_1']
        report['sources'].append(str(path))
        if arm in ('fixed', 'adaptive'):
            path = REPO / 'outputs/01_training_runs/hs6_l5_selective_retention_20260923' / f'{arm}_formal/raw_loss_metrics.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()[-100:]]
            item['final_100_record_mean'] = {k: statistics.mean(row[k] for row in rows) for k in
                                            ['recovery_global_gate', 'recovery_local_gate', 'recovery_global_error', 'recovery_local_error']}
            report['sources'].append(str(path))
        report['arms'][arm] = item
    report['regression_train_only_CV'] = {}
    for ds in ['conic-cell-count', 'livecell-cell-count']:
        path = ROOT / 'regression_tuning' / f'{ds}.json'
        rows = json.loads(path.read_text())['results']
        report['regression_train_only_CV'][ds] = {k: {'fixed_R2': v['fixed']['r2'], 'tuned_R2': v['tuned']['r2']}
                                                for k, v in rows.items()}
        report['sources'].append(str(path))
    OUT.mkdir(exist_ok=True)
    (OUT / 'existing_evidence.json').write_text(json.dumps(report, indent=2))
    for arm, row in report['arms'].items():
        print(arm, row['classification_mean_24'], row['segmentation_E50'],
              row['detection_patch_f1']['bbbc038'], row['rxrx3_recall_at_1'])


if __name__ == '__main__':
    main()
