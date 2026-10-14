#!/usr/bin/env python3
"""Add the validated 100TB ck7999 point to the existing selected-29 diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path('/mnt/huawei_deepcad/dinov3')
FIGURE = ROOT / 'plot/fig2/route2_20tb_v4_20261008'
STEM = 'route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_with_100tb_ck7999'
SELECTED = ROOT / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv'
BASE_CURVE = FIGURE / 'route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_diagnostic_curve.csv'
SCORES = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_ck7999_partial_20261012/v4_id_validated_partial_cell_scores.csv'
REPORT_STATUS = SCORES.parent / 'status.json'
CAMPAIGN = ROOT / 'plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999'
CELLS = CAMPAIGN / 'shared/cells'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')
COLORS = {'classification': '#2563A6', 'regression': '#9A4EAE', 'retrieval': '#16807A',
          'clustering': '#D17A22', 'segmentation': '#3A8F5C', 'overall': '#C64B4B'}
LABELS = {'classification': 'Classification', 'regression': 'Regression', 'retrieval': 'Retrieval',
          'clustering': 'Clustering', 'segmentation': 'Segmentation', 'overall': 'Equal-family mean'}
METRICS = {'classification': 'Macro-F1', 'regression': 'Spearman', 'retrieval': 'mAP@5',
           'clustering': 'NMI', 'segmentation': 'mDice (E20)', 'overall': 'Equal-family mean'}
MODEL_STYLE = {
    '1tb_hs6': ('#D58166', '^', '-', '1TB HS6-L | validated v3'),
    '5tb_no_gram': ('#58636D', 'o', '--', '5TB no-GRAM | history + continuation'),
    '20tb_route2': ('#16807A', 's', '-', '20TB route2 | v4 shared components'),
}
UPDATED_100TB = ('100tb_fullpool_ck7999', '#B23A48', 'D',
                 '100TB full-pool | v4 ID ck7999 (single point)')


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_sets() -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(SELECTED):
        result[row['family']].add(row['dataset'])
    expected = {'classification': 16, 'regression': 2, 'retrieval': 4,
                'clustering': 2, 'segmentation': 5}
    if {family: len(result[family]) for family in FAMILIES} != expected:
        raise ValueError('Selected-29 inventory changed')
    return result


def family_rows() -> tuple[list[dict], dict[str, float], dict[str, dict[str, float]]]:
    selected = selected_sets()
    score_rows = read_csv(SCORES)
    per_dataset: dict[str, dict[str, float]] = {}
    for family in FAMILIES[:-1]:
        values = {}
        for dataset in sorted(selected[family]):
            matches = [row for row in score_rows
                       if row['family'] == family and row['dataset'] == dataset
                       and row['status'] == 'VALID_COMPLETE']
            if len(matches) != 1:
                raise ValueError(f'Expected one validated 100TB score for {family}/{dataset}: {len(matches)}')
            values[dataset] = float(matches[0]['value'])
        per_dataset[family] = values

    segmentation: dict[str, dict[int, float]] = defaultdict(dict)
    for path in CELLS.rglob('results.json'):
        parts = path.parts
        if 'budget20' not in parts:
            continue
        index = parts.index('budget20')
        seed = int(parts[index + 1].removeprefix('seed'))
        dataset = parts[index + 2]
        if dataset not in selected['segmentation']:
            continue
        result = json.loads(path.read_text())
        segmentation[dataset][seed] = float(result['test']['mDice'])
    per_dataset['segmentation'] = {}
    for dataset in sorted(selected['segmentation']):
        if set(segmentation[dataset]) != {0, 1, 2}:
            raise ValueError(f'Expected E20 segmentation seeds 0/1/2 for {dataset}')
        per_dataset['segmentation'][dataset] = statistics.mean(segmentation[dataset][s]
                                                                for s in (0, 1, 2))

    family_means = {family: statistics.mean(per_dataset[family].values()) for family in FAMILIES}
    family_means['overall'] = statistics.mean(family_means[family] for family in FAMILIES)
    rows = [{'model': UPDATED_100TB[0], 'checkpoint': 7999, 'family': family,
             'value': value, 'complete': True, 'source_protocol': 'v4_id_ck7999_selected29',
             'source': str(SCORES)} for family, value in family_means.items()]
    return rows, family_means, per_dataset


def plot(rows: list[dict]) -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, layout='constrained')
    fig.get_layout_engine().set(rect=(.045, .16, .99, .82))
    peaks = []
    for ax, family in zip(axes.flat, (*FAMILIES, 'overall')):
        for model, (color, marker, linestyle, _label) in MODEL_STYLE.items():
            points = sorted((row for row in rows if row['model'] == model and row['family'] == family
                             and row['complete'] in (True, 'True')),
                            key=lambda row: int(row['checkpoint']))
            if not points:
                continue
            xs = [int(row['checkpoint']) for row in points]
            ys = [float(row['value']) for row in points]
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.55,
                    marker=marker, markersize=2.8 if model == '5tb_no_gram' else 3.2, alpha=.92)
            peak = max(points, key=lambda row: (float(row['value']), -int(row['checkpoint'])))
            ax.scatter([int(peak['checkpoint'])], [float(peak['value'])], marker='*', s=105,
                       color=color, edgecolor='white', linewidth=.7, zorder=5)
            peaks.append({'model': model, 'family': family, 'peak_checkpoint': int(peak['checkpoint']),
                          'peak_value': float(peak['value']), 'complete_points': len(points)})

        single = [row for row in rows if row['model'] == UPDATED_100TB[0] and row['family'] == family]
        if len(single) != 1:
            raise ValueError(f'Missing single 100TB point for {family}')
        point = single[0]
        ax.scatter([int(point['checkpoint'])], [float(point['value'])], marker=UPDATED_100TB[2],
                   s=74, color=UPDATED_100TB[1], edgecolor='white', linewidth=.9, zorder=7)
        peaks.append({'model': UPDATED_100TB[0], 'family': family, 'peak_checkpoint': 7999,
                      'peak_value': float(point['value']), 'complete_points': 1})

        ax.set_title(f"{LABELS[family]} | {16 if family == 'classification' else 2 if family in ('regression','clustering') else 4 if family == 'retrieval' else 5 if family == 'segmentation' else 29} datasets",
                     fontsize=10.5)
        ax.set_ylabel(METRICS[family])
        ax.grid(axis='y', color='#D9DEE3', linewidth=.75)
        ax.grid(axis='x', color='#EDF0F2', linewidth=.55)
    for ax in axes[1]:
        ax.set_xlabel('Checkpoint (optimizer updates)')
    max_step = max(int(row['checkpoint']) for row in rows)
    for ax in axes.flat:
        ax.set_xlim(0, max_step * 1.035)
    fig.suptitle('HS6-L | 1TB, 5TB, 20TB and 100TB', fontsize=16, fontweight='bold')
    fig.text(.5, .957, 'Fixed post hoc selected-29 datasets; equal dataset mean within each family, then equal-family mean',
             ha='center', fontsize=9.8, color='#4B5862')
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, label=label)
               for color, marker, linestyle, label in MODEL_STYLE.values()]
    handles.append(Line2D([0], [0], color=UPDATED_100TB[1], marker=UPDATED_100TB[2], linestyle='None',
                          label=UPDATED_100TB[3]))
    handles += [Line2D([0], [0], color='#D58166', linestyle=':', label='1TB endpoint ck15374'),
                Line2D([0], [0], color='#AAB1B7', linestyle=':', label='5TB endpoint ck41479')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.2,
               bbox_to_anchor=(.5, .055))
    fig.text(.5, .015,
             'Exploratory cross-protocol diagnostic, not a controlled comparison or full v4 aggregate. '
             '100TB is one validated full-pool point at ck7999; other curves and evaluator protocols are unchanged. '
             'Missing 20TB values remain gaps. Updated MoNuSeg split results are reported separately.',
             ha='center', fontsize=8.4, color='#68747D', wrap=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(FIGURE / f'{STEM}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    with (FIGURE / f'{STEM}_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (FIGURE / f'{STEM}_peaks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(peaks[0]))
        writer.writeheader()
        writer.writerows(peaks)


def main() -> None:
    status = json.loads(REPORT_STATUS.read_text())
    if status['shared_valid_complete'] != status['shared_expected'] or status['shared_expected'] != 38:
        raise ValueError('100TB v4 shared task set is incomplete')
    if status['checkpoint_step'] != 7999:
        raise ValueError('Unexpected 100TB checkpoint in report')
    source_rows = read_csv(BASE_CURVE)
    point_rows, family_means, per_dataset = family_rows()
    rows = source_rows + point_rows
    plot(rows)
    sources = (SELECTED, BASE_CURVE, SCORES, REPORT_STATUS,
               CAMPAIGN / 'shared/campaign_manifest.json')
    manifest = {
        'diagnostic_only': True, 'full_v4_aggregate_allowed': False,
        'selected_dataset_counts': {family: len(ds) for family, ds in selected_sets().items()},
        'one_tb_five_tb_twenty_tb': 'Existing validated selected-29 diagnostic curve, unchanged',
        '100tb': {
            'checkpoint_step': 7999, 'optimizer_updates_completed': 8000,
            'checkpoint_sha256': status['checkpoint_sha256'],
            'protocol': 'bio-eval-union-v4 ID components; segmentation E20, seeds 0/1/2',
            'aggregation': 'Mean over selected datasets per family; overall is equal mean of five family means',
            'family_means': family_means, 'dataset_values': per_dataset,
            'source_scores_csv': str(SCORES),
        },
        'known_limitations': [
            'Cross-protocol and post hoc selected-29 diagnostic; not a controlled comparison.',
            '100TB is a single intermediate checkpoint, not the current final training checkpoint.',
            'MoNuSeg is outside selected-29 and its updated-manifest retest is recorded separately.',
        ],
        'source_sha256': {str(path): sha(path) for path in sources},
    }
    (FIGURE / f'{STEM}_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    readme = [
        '# Route2 selected-29 diagnostic with 100TB', '',
        'This is a separate overlay; existing 1TB/5TB/20TB figures are not replaced.', '',
        '- The 100TB point is checkpoint 7999 (8,000 optimizer updates), full-pool 100TB training, v4 ID scope.',
        '- Classification, regression, retrieval and clustering use the validated v4 ID per-dataset primary metrics.',
        '- Segmentation uses mDice at E20, averaged over seeds 0/1/2, matching the selected-29 E20 segmentation field.',
        '- Aggregation is an unweighted mean over selected datasets within each family, then an equal mean over five families.',
        '- This is cross-protocol diagnostic evidence, not a controlled comparison or full v4 score. The 100TB run is still training.',
        '', '100TB selected-29 family means:',
        *[f'- {family}: `{family_means[family]:.6f}`' for family in (*FAMILIES, 'overall')],
        '', 'MoNuSeg is not in selected-29; its separate updated-manifest E20/E50 result is in the 100TB report.',
        'The source manifest is explicitly named superseded; it is not substituted for the registered formal 30/7/14 result.',
        '', 'Files:', f'- `{STEM}.png/.svg/.pdf`: overlay figure.',
        f'- `{STEM}_curve.csv`: original comparison rows plus 100TB point.',
        f'- `{STEM}_manifest.json`: aggregation rules, source hashes and limitations.',
    ]
    (FIGURE / f'{STEM}_README.md').write_text('\n'.join(readme) + '\n')
    print(json.dumps({'figure': str(FIGURE / f'{STEM}.png'), 'family_means': family_means,
                      'points': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
