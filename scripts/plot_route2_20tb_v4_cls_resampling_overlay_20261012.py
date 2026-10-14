#!/usr/bin/env python3
"""Overlay CLS and resampling checkpoints on the existing route2 V4 component figure."""
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


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / 'plot/fig2/route2_20tb_v4_20261008'
SELECTION = ROOT / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv'
BASELINE = ROOT / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
INTERVENTIONS = ROOT / 'outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012'
HISTORIC = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/family_curve.csv'
OUT = FIGURE / 'route2_20tb_v4_cls_resampling_overlay_20261012'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')
PANELS = (*FAMILIES, 'overall')
METRICS = {'classification': 'macro_f1', 'regression': 'spearman', 'retrieval': 'map_at_5',
           'clustering': 'nmi', 'segmentation': 'mDice', 'overall': 'equal_five_family_mean'}
LABELS = {'classification': 'Classification', 'regression': 'Regression',
          'retrieval': 'Retrieval', 'clustering': 'Clustering',
          'segmentation': 'Segmentation', 'overall': 'Equal-family mean'}
FAMILY_COLORS = {'classification': '#2563A6', 'regression': '#9A4EAE',
                 'retrieval': '#16807A', 'clustering': '#D17A22',
                 'segmentation': '#3A8F5C', 'overall': '#C64B4B'}
MODEL_COLORS = {'historic_route2': '#48545E', 'paired_noGRAM': '#15191D',
                'fixed_cls_w1': '#D17B13', 'resampling_only': '#087E8B'}
STEPS = (40015, 41967, 43919)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def selection() -> dict[str, set[str]]:
    chosen: dict[str, set[str]] = defaultdict(set)
    with SELECTION.open(newline='') as stream:
        for row in csv.DictReader(stream):
            chosen[row['family']].add(row['dataset'])
    expected = {'classification': 16, 'regression': 2, 'retrieval': 4,
                'clustering': 2, 'segmentation': 5}
    if {family: len(chosen[family]) for family in FAMILIES} != expected:
        raise ValueError(f'Unexpected selected-29 inventory: {dict((f, len(chosen[f])) for f in FAMILIES)}')
    return chosen


def manifest_tasks(root: Path, arm: str, checkpoint: int) -> list[dict]:
    manifest = read_json(root / 'campaign_manifest.json')
    tasks = [task for task in manifest['tasks']
             if task['asset']['arm'] == arm and int(task['asset']['checkpoint_id']) == checkpoint]
    if not tasks:
        raise ValueError(f'No tasks for {arm} checkpoint {checkpoint} in {root}')
    return tasks


def assert_valid(root: Path, task: dict) -> None:
    path = root / '_state/done' / f"{task['key']}.json"
    done = read_json(path)
    if done.get('status') != 'VALID_COMPLETE':
        raise ValueError(f'Cell lacks VALID_COMPLETE evidence: {path}')


def component_path(root: Path, task: dict) -> Path:
    path = root / 'cells' / task['key'] / 'component_result.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_value(root: Path, task: dict, family: str) -> tuple[float, list[str]]:
    assert_valid(root, task)
    if family != 'segmentation':
        path = component_path(root, task)
        result = read_json(path)
        if family == 'classification':
            value = result['macro_f1']
        elif family == 'regression':
            value = result['spearman']
        elif family == 'clustering':
            value = result['nmi']
        elif family == 'retrieval':
            if 'rows' in result:
                rows = [row for row in result['rows']
                        if row.get('task') == 'retrieval' and row.get('aggregation') == 'global']
                if len(rows) != 1:
                    raise ValueError(f'Expected one global retrieval result in {path}')
                value = rows[0]['map_at_5']
            else:
                value = result['map_at_5']
        else:
            raise KeyError(family)
        return float(value), [str(path)]

    cell = root / 'cells' / task['key']
    paths = sorted(cell.glob('results/**/budget20/seed*/**/results.json'))
    values = []
    for seed in range(3):
        matches = [path for path in paths if f'/seed{seed}/' in str(path)]
        if len(matches) != 1:
            raise ValueError(f'Expected one E20 seed{seed} result in {cell}, found {len(matches)}')
        result = read_json(matches[0])
        if result['_meta']['probe_epochs'] != 20 or result['_meta']['seed'] != seed:
            raise ValueError(f'Unexpected segmentation fit metadata: {matches[0]}')
        values.append(float(result['test']['mDice']))
    return statistics.mean(values), [str(path) for path in paths]


def scores_for(root: Path, arm: str, checkpoint: int, chosen: dict[str, set[str]]) -> tuple[dict, list[dict]]:
    tasks = manifest_tasks(root, arm, checkpoint)
    observed: dict[tuple[str, str], list[tuple[float, list[str]]]] = defaultdict(list)
    for task in tasks:
        ds = task['dataset']['dataset']
        task_family = task['dataset']['task']
        if task_family == 'retrieval':
            for family in ('clustering', 'retrieval'):
                if ds in chosen[family]:
                    value, sources = read_value(root, task, family)
                    observed[(family, ds)].append((value, sources))
        elif task_family in chosen and ds in chosen[task_family]:
            value, sources = read_value(root, task, task_family)
            observed[(task_family, ds)].append((value, sources))

    dataset_rows = []
    family_values = {}
    for family in FAMILIES:
        per_dataset = {}
        for ds in sorted(chosen[family]):
            values = observed.get((family, ds), [])
            expected_count = 3 if family == 'segmentation' and ds == 'pannuke' else 1
            if len(values) != expected_count:
                raise ValueError(f'{arm} ck{checkpoint}: {family}/{ds} has {len(values)} cells, expected {expected_count}')
            value = statistics.mean(item[0] for item in values)
            per_dataset[ds] = value
            source_list = [source for _, sources in values for source in sources]
            dataset_rows.append(dict(checkpoint=checkpoint, model=arm, family=family,
                                     dataset=ds, metric=METRICS[family], value=value,
                                     cells=len(values), source='|'.join(source_list)))
        family_values[family] = statistics.mean(per_dataset.values())
    family_values['overall'] = statistics.mean(family_values.values())
    return family_values, dataset_rows


def build() -> None:
    FIGURE.mkdir(parents=True, exist_ok=True)
    chosen = selection()
    with HISTORIC.open(newline='') as stream:
        historic = [row for row in csv.DictReader(stream) if row['family'] in PANELS]
    if not historic or any(row['complete'] != 'True' for row in historic):
        raise ValueError('Historic route2 component curve contains incomplete points')
    historic_steps = {int(row['checkpoint']) for row in historic}
    if not set(STEPS).issubset(historic_steps):
        raise ValueError('Historic route2 curve lacks a matched intervention checkpoint')

    curves = []
    for row in historic:
        curves.append(dict(model='historic_route2', checkpoint=int(row['checkpoint']),
                           family=row['family'], metric=row['metric'], value=float(row['value']),
                           complete=True, source=str(HISTORIC)))

    checkpoint_models = [('paired_noGRAM', BASELINE, 'noGRAM20tb', STEPS),
                         ('resampling_only', INTERVENTIONS, 'resampling_only_20tb', STEPS),
                         ('fixed_cls_w1', INTERVENTIONS, 'fixed_cls_w1_20tb', (40015,))]
    dataset_rows = []
    family_lookup = {}
    for model, root, arm, checkpoints in checkpoint_models:
        for checkpoint in checkpoints:
            values, source_rows = scores_for(root, arm, checkpoint, chosen)
            family_lookup[(model, checkpoint)] = values
            dataset_rows.extend(source_rows)
            for family, value in values.items():
                curves.append(dict(model=model, checkpoint=checkpoint, family=family,
                                   metric=METRICS[family], value=value, complete=True,
                                   source=str(root / 'campaign_manifest.json')))

    deltas = []
    for method, method_arm, steps in [('resampling_only', 'resampling_only_20tb', STEPS),
                                      ('fixed_cls_w1', 'fixed_cls_w1_20tb', (40015,))]:
        for checkpoint in steps:
            for family in PANELS:
                baseline_value = family_lookup[('paired_noGRAM', checkpoint)][family]
                method_value = family_lookup[(method, checkpoint)][family]
                deltas.append(dict(checkpoint=checkpoint, model=method, family=family,
                                   metric=METRICS[family], paired_noGRAM=baseline_value,
                                   method=method_value, delta=method_value - baseline_value,
                                   datasets=len(chosen[family]) if family != 'overall' else 29,
                                   interpretation='method minus same-checkpoint paired noGRAM'))
    write_csv(OUT.with_name(OUT.name + '_curve.csv'), curves,
              ['model', 'checkpoint', 'family', 'metric', 'value', 'complete', 'source'])
    write_csv(OUT.with_name(OUT.name + '_dataset_deltas.csv'), dataset_rows,
              ['checkpoint', 'model', 'family', 'dataset', 'metric', 'value', 'cells', 'source'])
    write_csv(OUT.with_name(OUT.name + '_family_deltas.csv'), deltas,
              ['checkpoint', 'model', 'family', 'metric', 'paired_noGRAM', 'method', 'delta', 'datasets', 'interpretation'])
    make_plot(curves, chosen, deltas)

    metadata = dict(
        generator=str(Path(__file__).resolve()), generator_sha256=sha256(Path(__file__).resolve()),
        selection=str(SELECTION), selection_sha256=sha256(SELECTION),
        historic_route2_curve=str(HISTORIC), historic_route2_curve_sha256=sha256(HISTORIC),
        paired_noGRAM_campaign=str(BASELINE / 'campaign_manifest.json'),
        paired_noGRAM_manifest_sha256=sha256(BASELINE / 'campaign_manifest.json'),
        intervention_campaign=str(INTERVENTIONS / 'campaign_manifest.json'),
        intervention_manifest_sha256=sha256(INTERVENTIONS / 'campaign_manifest.json'),
        checkpoints={'fixed_cls_w1': [40015], 'resampling_only': list(STEPS)},
        selected_dataset_counts={family: len(chosen[family]) for family in FAMILIES},
        selected_dataset_count=29,
        overall_definition='equal mean of the five selected-29 family means',
        segmentation_rule='mDice at E20, mean of seed 0/1/2; PanNuke mean of three rotations',
        delta_definition='intervention metric minus its VALID_COMPLETE same-checkpoint paired noGRAM metric',
        historical_overlay_note='Historical route2 line is preserved from its original component curve. New symbols are direct paired-campaign retests; paired noGRAM points expose any rerun offset from the historical line.',
        protocol_scope='selected-29 V4 shared components only; not a full V4 aggregate',
        outputs=[str(OUT.with_suffix('.png')), str(OUT.with_suffix('.svg')), str(OUT.with_suffix('.pdf')),
                 str(OUT.with_name(OUT.name + '_curve.csv')),
                 str(OUT.with_name(OUT.name + '_dataset_deltas.csv')),
                 str(OUT.with_name(OUT.name + '_family_deltas.csv'))])
    OUT.with_name(OUT.name + '_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_readme(chosen, deltas)
    print(json.dumps(dict(figure=str(OUT), selected_29=True,
                          overlay_models={x[0]: list(x[3]) for x in checkpoint_models},
                          family_delta_rows=len(deltas), dataset_rows=len(dataset_rows)), indent=2))


def make_plot(curves: list[dict], chosen: dict[str, set[str]], deltas: list[dict]) -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'semibold'})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.7), sharex=True)
    for ax, family in zip(axes.flat, PANELS):
        base = sorted((r for r in curves if r['model'] == 'historic_route2' and r['family'] == family),
                      key=lambda r: r['checkpoint'])
        ax.plot([r['checkpoint'] for r in base], [r['value'] for r in base],
                color=MODEL_COLORS['historic_route2'], linewidth=1.6, marker='o',
                markersize=2.8, alpha=.75, zorder=1)
        pair = sorted((r for r in curves if r['model'] == 'paired_noGRAM' and r['family'] == family),
                      key=lambda r: r['checkpoint'])
        ax.scatter([r['checkpoint'] for r in pair], [r['value'] for r in pair],
                   marker='D', s=45, facecolor='white', edgecolor=MODEL_COLORS['paired_noGRAM'],
                   linewidth=1.5, zorder=4)
        resampling = sorted((r for r in curves if r['model'] == 'resampling_only' and r['family'] == family),
                            key=lambda r: r['checkpoint'])
        ax.plot([r['checkpoint'] for r in resampling], [r['value'] for r in resampling],
                color=MODEL_COLORS['resampling_only'], linewidth=2.0, linestyle='--',
                marker='s', markersize=5.2, zorder=5)
        fixed = [r for r in curves if r['model'] == 'fixed_cls_w1' and r['family'] == family]
        ax.scatter([r['checkpoint'] for r in fixed], [r['value'] for r in fixed],
                   marker='^', s=100, color=MODEL_COLORS['fixed_cls_w1'],
                   edgecolor='white', linewidth=.8, zorder=6)
        ax.axvline(38063, color='#9BA4AB', linestyle=':', linewidth=1.0, zorder=0)
        ax.set_title(f"{LABELS[family]} | {29 if family == 'overall' else len(chosen[family])} datasets",
                     fontsize=11)
        ax.set_ylabel('Equal-family mean' if family == 'overall' else METRICS[family])
        ax.grid(axis='y', color='#D9DEE3', linewidth=.8)
        ax.grid(axis='x', color='#EDF0F2', linewidth=.6)
        ax.set_xlim(0, 68000)
        if family == 'overall':
            lookup = {(r['model'], r['checkpoint']): r['delta'] for r in deltas if r['family'] == 'overall'}
            annotations = [('fixed_cls_w1', 40015, (-45, 13)),
                           ('resampling_only', 40015, (7, 12)),
                           ('resampling_only', 41967, (8, -13)),
                           ('resampling_only', 43919, (8, -26))]
            for model, step, offset in annotations:
                value = lookup[(model, step)]
                ax.annotate(f'{value * 100:+.2f} pp', (step, next(r['value'] for r in curves
                             if r['model'] == model and r['checkpoint'] == step and r['family'] == family)),
                            xytext=offset, textcoords='offset points',
                            fontsize=8, color=MODEL_COLORS[model])
    for ax in axes[1]:
        ax.set_xlabel('Checkpoint (optimizer steps)')
    fig.suptitle('HS6-L 20TB route2 | CLS and resampling intervention overlay',
                 fontsize=17, fontweight='bold', y=.984)
    fig.text(.5, .945, 'Selected-29 V4 components: original route2 trajectory plus direct same-checkpoint retests',
             ha='center', fontsize=10.5, color='#4B5862')
    handles = [
        Line2D([0], [0], color=MODEL_COLORS['historic_route2'], marker='o', markersize=4,
               label='Original route2 component curve'),
        Line2D([0], [0], color='none', marker='D', markerfacecolor='white',
               markeredgecolor=MODEL_COLORS['paired_noGRAM'], label='Paired no-GRAM retest'),
        Line2D([0], [0], color=MODEL_COLORS['resampling_only'], linestyle='--', marker='s',
               label='Resampling only'),
        Line2D([0], [0], color='none', marker='^', markerfacecolor=MODEL_COLORS['fixed_cls_w1'],
               label='Fixed CLS, w=1')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .012))
    fig.text(.5, .055,
             'Open diamonds are matched reruns; overall labels show paired change in percentage points. '
             'Dotted line marks fork checkpoint 38063. Descriptive selected-29 comparison, not a full V4 score.',
             ha='center', fontsize=9, color='#68747D')
    fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.89, hspace=.30, wspace=.25)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT.with_suffix(f'.{extension}'), dpi=220, facecolor='white')
    plt.close(fig)


def write_readme(chosen: dict[str, set[str]], deltas: list[dict]) -> None:
    lines = [
        '# 20TB route2 V4 with CLS and resampling', '',
        'This is a new overlay. The original `route2_20tb_v4_components` files are unchanged.', '',
        'The historical line comes from the existing route2 V4 selected-29 family curve. Open diamonds '
        'are direct no-GRAM retests from the same-checkpoint paired campaign; intervention markers are '
        'direct fixed-CLS or resampling-only retests. Showing paired no-GRAM points makes small rerun '
        'offsets from the older historical line visible.', '',
        'Family means use the existing selected-29 membership (16 classification, 2 regression, '
        '4 retrieval, 2 clustering, 5 segmentation). The overall panel is a descriptive equal mean '
        'of the five family means. Segmentation is E20 test mDice averaged over seeds 0/1/2, with '
        'PanNuke averaged over its three rotations. Intervention deltas compare each method to its '
        'VALID_COMPLETE no-GRAM retest at the same checkpoint.', '',
        'Only fixed CLS checkpoint 40015 and resampling checkpoints 40015, 41967, 43919 are shown. '
        'This is a selected-29 component diagnostic, not a complete V4 aggregate or a full training trajectory.', '',
        'Files: PNG/SVG/PDF figure; `_curve.csv` has historical and retest points; `_dataset_deltas.csv` '
        'has the selected dataset measurements; `_family_deltas.csv` has matched family and overall deltas; '
        '`_manifest.json` records source hashes and aggregation rules.', '',
        f'Family panel counts: {json.dumps({family: len(chosen[family]) for family in FAMILIES}, sort_keys=True)}.',
        f'Paired intervention delta records: {len(deltas)}.'
    ]
    OUT.with_name(OUT.name + '_README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    build()
