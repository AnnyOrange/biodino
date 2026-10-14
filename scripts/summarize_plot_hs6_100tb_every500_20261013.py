#!/usr/bin/env python3
"""Refresh the selected-29 route2 overlay from fully validated 100TB checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path('/mnt/huawei_deepcad/dinov3')
DATA = Path('/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009/v4_every500_20261013')
FIGURE = ROOT / 'plot/fig2/route2_20tb_v4_20261008'
CONTROL = ROOT / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
STEM = 'route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_with_100tb_ck7999'
SELECTED = ROOT / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv'
BASE_CURVE = FIGURE / 'route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_diagnostic_curve.csv'
OLD_CAMPAIGN = ROOT / 'plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999'
OLD_REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_ck7999_partial_20261012'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')
EXPECTED = {'classification': 16, 'regression': 2, 'retrieval': 4,
            'clustering': 2, 'segmentation': 5}
METRICS = {'classification': 'Macro-F1', 'regression': 'Spearman', 'retrieval': 'mAP@5',
           'clustering': 'NMI', 'segmentation': 'mDice (E20)', 'overall': 'Equal-family mean'}
LABELS = {'classification': 'Classification', 'regression': 'Regression', 'retrieval': 'Retrieval',
          'clustering': 'Clustering', 'segmentation': 'Segmentation', 'overall': 'Equal-family mean'}
MODEL_STYLE = {
    '1tb_hs6': ('#D58166', '^', '-', '1TB HS6-L | validated v3'),
    '5tb_no_gram': ('#58636D', 'o', '--', '5TB no-GRAM | history + continuation'),
    '20tb_route2': ('#16807A', 's', '-', '20TB route2 | v4 shared components'),
    '100tb_fullpool_every500': ('#B23A48', 'D', '-', '100TB full-pool | every 500 updates'),
}


def read(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def selected_sets() -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(SELECTED):
        out[row['family']].add(row['dataset'])
    if {key: len(out[key]) for key in FAMILIES} != EXPECTED:
        raise ValueError('Selected-29 inventory changed')
    return out


def is_complete(step: int, campaign: Path) -> bool:
    if step == 7999:
        status = read(OLD_REPORT / 'status.json')
        return status.get('shared_valid_complete') == status.get('shared_expected') == 38 and \
            status.get('extension_valid_complete') == status.get('extension_expected') == 9 and \
            status.get('monuseg') == 'VALID_COMPLETE'
    path = campaign / 'driver_status.json'
    return path.is_file() and read(path).get('state') == 'COMPLETE'


def metric_from_component(family: str, result: dict) -> float:
    if family == 'classification':
        return float(result['macro_f1'])
    if family == 'regression':
        return float(result['spearman'])
    rows = result.get('rows')
    if not isinstance(rows, list):
        return float(result['map_at_5' if family == 'retrieval' else 'nmi'])
    if family == 'retrieval':
        matches = [row for row in rows if row.get('task') == 'retrieval'
                   and row.get('aggregation') == 'global']
        field = 'map_at_5'
    else:
        matches = [row for row in rows if row.get('task') == 'clustering'
                   and 'ge10' not in row.get('protocol', '')]
        field = 'nmi'
    if len(matches) != 1:
        raise ValueError(f'Expected one selected {family} metric, found {len(matches)}')
    return float(matches[0][field])


def selected_values(step: int, campaign: Path) -> dict[str, dict[str, float]]:
    chosen = selected_sets()
    shared = campaign / 'shared'
    manifest = read(shared / 'campaign_manifest.json')
    done_dir = shared / '_state/done'
    done = {path.stem for path in done_dir.glob('*.json')
            if read(path).get('status') == 'VALID_COMPLETE'}
    task_by_family_dataset = {}
    for task in manifest['tasks']:
        family = task['dataset']['task']
        dataset = task['dataset']['dataset']
        task_by_family_dataset[(family, dataset)] = task
    values: dict[str, dict[str, float]] = {}
    for family in FAMILIES[:-1]:
        task_family = 'retrieval' if family == 'clustering' else family
        family_values = {}
        for dataset in sorted(chosen[family]):
            task = task_by_family_dataset[(task_family, dataset)]
            if task['key'] not in done:
                raise ValueError(f'Missing validated shared task: {task["key"]}')
            cell = shared / 'cells' / task['key']
            family_values[dataset] = metric_from_component(
                family, read(cell / 'component_result.json'))
        values[family] = family_values

    family_values = {}
    for dataset in sorted(chosen['segmentation']):
        task = task_by_family_dataset[('segmentation', dataset)]
        if task['key'] not in done:
            raise ValueError(f'Missing validated shared task: {task["key"]}')
        cell = shared / 'cells' / task['key']
        by_seed = {}
        for path in (cell / 'results').rglob('results.json'):
            parts = path.parts
            if 'budget20' not in parts:
                continue
            index = parts.index('budget20')
            seed = int(parts[index + 1].removeprefix('seed'))
            by_seed[seed] = float(read(path)['test']['mDice'])
        if set(by_seed) != {0, 1, 2}:
            raise ValueError(f'Expected E20 seeds 0/1/2 for {dataset}: {sorted(by_seed)}')
        family_values[dataset] = statistics.mean(by_seed.values())
    values['segmentation'] = family_values
    return values


def validated_campaign(step: int, campaign: Path) -> bool:
    if not is_complete(step, campaign):
        return False
    if step == 7999:
        return True
    manifest = read(campaign / 'shared/campaign_manifest.json')
    done = {path.stem for path in (campaign / 'shared/_state/done').glob('*.json')
            if read(path).get('status') == 'VALID_COMPLETE'}
    ext_manifest = read(campaign / 'extension/campaign_manifest.json')
    ext_done = 0
    for task_path in (campaign / 'extension/tasks').glob('*.json'):
        task = read(task_path)
        validation = Path(task['output']) / 'validation_report.json'
        ext_done += validation.is_file() and read(validation).get('status') == 'VALID_COMPLETE'
    monu = read(campaign / 'monuseg/campaign_manifest.json')
    monu_spec = monu['datasets'][0]
    task = monu['tasks'][0]
    validation = campaign / 'monuseg/cells' / task['key'] / 'validation_report.json'
    return len(done) == 38 and len(manifest['tasks']) == 38 and \
        ext_done == len(ext_manifest['task_ids']) == 9 and validation.is_file() and \
        read(validation).get('status') == 'VALID_COMPLETE' and \
        monu_spec.get('counts') == {'train': 30, 'val': 7, 'test': 14}


def point_rows(step: int, campaign: Path, complete: bool) -> list[dict]:
    template = dict(model='100tb_fullpool_every500', checkpoint=step, complete=complete,
                    source_protocol='v4_id_38+extensions_9+monuseg_30_7_14')
    if not complete:
        return [dict(template, family=family, value='') for family in (*FAMILIES, 'overall')]
    values = selected_values(step, campaign)
    means = {family: statistics.mean(values[family].values()) for family in FAMILIES}
    means['overall'] = statistics.mean(means[family] for family in FAMILIES)
    source = str(OLD_REPORT / 'v4_id_validated_partial_cell_scores.csv') if step == 7999 else str(campaign)
    return [dict(template, family=family, value=mean, source=source) for family, mean in means.items()]


def draw(rows: list[dict], complete_steps: list[int]) -> list[dict]:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, layout='constrained')
    fig.get_layout_engine().set(rect=(.045, .16, .99, .82))
    peaks = []
    for ax, family in zip(axes.flat, (*FAMILIES, 'overall')):
        for model, (color, marker, style, label) in MODEL_STYLE.items():
            points = sorted((row for row in rows if row['model'] == model and row['family'] == family
                             and str(row['complete']).lower() == 'true' and row['value'] != ''),
                            key=lambda row: int(row['checkpoint']))
            if not points:
                continue
            if model == '100tb_fullpool_every500':
                run: list[dict] = []
                label_used = False
                for point in points:
                    if run and int(point['checkpoint']) - int(run[-1]['checkpoint']) != 500:
                        ax.plot([int(p['checkpoint']) for p in run], [float(p['value']) for p in run],
                                color=color, linestyle=style, linewidth=1.65, marker=marker,
                                markersize=3.1, alpha=.94, label=label if not label_used else None)
                        label_used = True
                        run = []
                    run.append(point)
                if run:
                    ax.plot([int(p['checkpoint']) for p in run], [float(p['value']) for p in run],
                            color=color, linestyle=style, linewidth=1.65, marker=marker,
                            markersize=3.1, alpha=.94, label=label if not label_used else None)
            else:
                ax.plot([int(p['checkpoint']) for p in points], [float(p['value']) for p in points],
                        color=color, linestyle=style, linewidth=1.55, marker=marker,
                        markersize=2.8 if model == '5tb_no_gram' else 3.2, alpha=.92, label=label)
            peak = max(points, key=lambda row: float(row['value']))
            peaks.append({'model': model, 'family': family, 'peak_checkpoint': int(peak['checkpoint']),
                          'peak_value': float(peak['value']), 'complete_points': len(points)})
        n = 16 if family == 'classification' else 2 if family in ('regression', 'clustering') else \
            4 if family == 'retrieval' else 5 if family == 'segmentation' else 29
        ax.set_title(f'{LABELS[family]} | {n} datasets', fontsize=10.5)
        ax.set_ylabel(METRICS[family])
        ax.grid(axis='y', color='#D9DEE3', linewidth=.75)
        ax.grid(axis='x', color='#EDF0F2', linewidth=.55)
    for ax in axes[1]:
        ax.set_xlabel('Checkpoint (optimizer updates)')
    max_step = max([int(row['checkpoint']) for row in rows] or [40_000])
    for ax in axes.flat:
        ax.set_xlim(0, max(40_000, max_step) * 1.02)
    fig.suptitle('HS6-L | 1TB, 5TB, 20TB and 100TB', fontsize=16, fontweight='bold')
    fig.text(.5, .957, 'Fixed post hoc selected-29 datasets; equal dataset mean per family, then equal-family mean',
             ha='center', fontsize=9.8, color='#4B5862')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8.2,
               bbox_to_anchor=(.5, .055))
    fig.text(.5, .015,
             f'Exploratory cross-protocol diagnostic, not a controlled comparison or full v4 aggregate. '
             f'100TB: {len(complete_steps)}/80 fully validated checkpoints. Missing checkpoint results remain gaps. '
             'MoNuSeg is evaluated on registered train30/val7/test14 and is not part of selected-29.',
             ha='center', fontsize=8.4, color='#68747D', wrap=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(FIGURE / f'{STEM}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    return peaks


def refresh() -> dict:
    source_rows = read_csv(BASE_CURVE)
    rows = list(source_rows)
    inventory = []
    complete_steps = []
    for step in range(499, 40_000, 500):
        campaign = OLD_CAMPAIGN if step == 7999 else DATA / 'campaigns' / f'100tb_ck{step}'
        complete = validated_campaign(step, campaign) if campaign.exists() else False
        inventory.append(dict(checkpoint=step, complete=complete,
                              campaign=str(campaign), status='VALID_COMPLETE' if complete else 'PENDING'))
        if complete:
            complete_steps.append(step)
        rows.extend(point_rows(step, campaign, complete))
    peaks = draw(rows, complete_steps)
    curve_path = FIGURE / f'{STEM}_curve.csv'
    with curve_path.open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status_path = CONTROL / 'checkpoint_inventory.csv'
    CONTROL.mkdir(parents=True, exist_ok=True)
    with status_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    manifest = dict(
        generated_unix=time.time(), checkpoints_expected=80, checkpoints_fully_validated=len(complete_steps),
        completed_steps=complete_steps, incomplete_steps=[row['checkpoint'] for row in inventory if not row['complete']],
        status_file=str(status_path), curve_csv=str(curve_path), campaign_root=str(DATA / 'campaigns'),
        protocol='38 shared ID tasks + 9 v4 extensions + registered MoNuSeg train30/val7/test14',
        selected29_aggregation='Per family, equal mean over selected datasets; segmentation mDice E20 seeds 0/1/2; overall equal mean over five family means',
        missing_data='Incomplete checkpoints are retained in inventory and omitted from plotted values; lines break across missing 500-update checkpoints.',
        diagnostic_only=True)
    (FIGURE / f'{STEM}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    readme = [
        '# Route2 selected-29 diagnostic with 100TB checkpoints', '',
        f'Fully validated 100TB checkpoints: {len(complete_steps)}/80.',
        '100TB points enter the curve only after all 38 shared ID tasks, 9 v4 extensions, and registered MoNuSeg train30/val7/test14 validate.',
        'Classification uses macro-F1, regression Spearman, retrieval mAP@5, clustering NMI, and segmentation mDice at E20 averaged over seeds 0/1/2.',
        'Each family is an unweighted mean over selected datasets; overall is the equal mean of five family means.',
        'Incomplete checkpoints remain listed as pending in the inventory and are not interpolated or connected across gaps.',
        'This remains an exploratory cross-protocol diagnostic, not a controlled comparison or full v4 aggregate.',
        '', 'Files:', f'- `{STEM}.png/.svg/.pdf`: selected-29 overlay.',
        f'- `{STEM}_curve.csv`: original comparator rows and 100TB checkpoint family rows.',
        f'- `{STEM}_manifest.json`: protocol, aggregation, completeness, and missing-data policy.',
        f'- `{status_path}`: one status row for each of the 80 expected checkpoints.',
    ]
    (FIGURE / f'{STEM}_README.md').write_text('\n'.join(readme) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--interval', type=int, default=900)
    args = parser.parse_args()
    while True:
        result = refresh()
        print(json.dumps(result), flush=True)
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
