#!/usr/bin/env python3
"""Plot original baselines and Adaptive from the consolidated evidence, by fixed lists."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
OUT = ROOT / 'outputs/00_reports/deepcad_method_20260927/adaptive_original_union_20260929'
FRESH = OUT / 'fresh_collection'
# 2026-09-30: MoNuSeg only from the official train30 / extra7 val / test14 re-test; CoNIC detection
# only on the official source-disjoint split (the builder marks everything else as excluded).
MONUSEG_CAMPAIGN = ROOT / 'outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929'
MONUSEG_SPLIT = 'monuseg2018-train30-extra7val-test14-v1'
FAMILIES = ['classification', 'regression', 'retrieval', 'clustering', 'segmentation', 'detection_proxy']
N, G, A = 'no-GRAM', 'GRAM', 'Adaptive'
COLORS = {N: '#64748b', G: '#dc8b19', A: '#008c88'}
NAMES = ['Classification', 'Regression', 'Retrieval', 'Clustering', 'Segmentation E20', 'Detection proxy']
LABELS = ['Mean BA / macro AUC (%)', 'Mean R-squared', 'Mean Recall@1 (%)', 'Mean NMI (%)', 'Mean mDice (%)', 'Mean patch F1 (%)']


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    union_path = BASE / 'v4_all_models.json'
    union_bytes = union_path.read_bytes()
    union = json.loads(union_bytes)
    expected = json.loads((BASE / 'v4_manifest.json').read_text())['expected']
    expected = {f: expected[f] for f in FAMILIES}
    assert [len(expected[f]) for f in FAMILIES] == [25, 4, 7, 7, 7, 3]
    common = defaultdict(list)
    for row in csv_rows(BASE / 'common_cells.csv'):
        common[row['family']].append(row['dataset'])
    assert sum(map(len, common.values())) == 40
    # The sixth panel is the same separately reported detection proxy as in the README.
    common['detection_proxy'] = expected['detection_proxy']
    candidates = defaultdict(list)
    steps = defaultdict(set)

    def add(method, step, family, dataset, observation):
        if family not in expected or dataset not in expected[family]:
            return
        value = observation.get('value')
        if value is None or not math.isfinite(float(value)):
            return
        if family == 'segmentation' and observation.get('budget_epochs') != 20:
            return
        if observation.get('superseded_by_split') or observation.get('excluded_reason'):
            return
        if dataset == 'monuseg' and observation.get('provenance') not in ('monuseg_30_7_14_validated',):
            return
        key = method, int(step), family, dataset
        item = dict(observation)
        item['value'] = float(value)
        item['legacy'] = item['provenance'] == 'legacy_accepted'
        candidates[key].append(item)

    for name, method in [('hs6_l_5tb_no_gram', N), ('hs6_l_5tb_gram12687', G)]:
        for step, checkpoint in union[name]['checkpoints'].items():
            # This is a shared no-GRAM anchor, not an independent GRAM observation.
            if method == G and int(step) == 12687:
                continue
            steps[method].add(int(step))
            for family, datasets in checkpoint['cells'].items():
                for dataset, cell in datasets.items():
                    for observation in cell.get('observations', []):
                        add(method, step, family, dataset, observation)
    fresh_cells = csv_rows(FRESH / 'CELLS.csv')
    fresh_map = {'Vanilla (5TB no-GRAM)': N, 'GRAM (5TB)': G, 'Adaptive': A}
    for row in fresh_cells:
        method = fresh_map[row['method']]
        if method == G and int(row['checkpoint']) == 12687:
            continue
        steps[method].add(int(row['checkpoint']))
        add(method, row['checkpoint'], row['family'], row['dataset'], dict(
            value=float(row['value']), metric=row['metric'], provenance='completed_task_collector',
            priority=85, source=row['source'], budget_epochs=int(row['budget']) or None))
    manifest = json.loads((MONUSEG_CAMPAIGN / 'campaign_manifest.json').read_text())
    for task in manifest['tasks']:
        done = MONUSEG_CAMPAIGN / '_state/done' / f"{task['key']}.json"
        if task['asset']['arm'] != 'adaptive' or not done.is_file():
            continue
        if json.loads(done.read_text()).get('status') != 'VALID_COMPLETE':
            continue
        seeds = {}
        for path in (MONUSEG_CAMPAIGN / 'cells' / task['key']).rglob('results.json'):
            obj = json.loads(path.read_text())
            meta = obj.get('_meta', {})
            if meta.get('probe_epochs') == 20 and meta.get('full_train_samples') == 30:
                seeds[meta.get('seed')] = (float(obj['test']['mDice']), str(path))
        if set(seeds) == {0, 1, 2}:
            step = int(task['asset']['checkpoint_id'])  # plotted only where the step is registered
            add(A, step, 'segmentation', 'monuseg', dict(
                value=statistics.mean(v[0] for v in seeds.values()), metric='mDice_E20',
                provenance='monuseg_30_7_14_validated', priority=110, budget_epochs=20,
                source=';'.join(seeds[k][1] for k in (0, 1, 2)), note=MONUSEG_SPLIT))
    for row in csv_rows(FRESH / 'TASK6_MEANS_AND_COVERAGE.csv'):
        if row['method'] == 'Adaptive':
            steps[A].add(int(row['checkpoint']))
    latest = max(steps[A])
    selections = {}
    alternatives = []
    for key, observations in candidates.items():
        ranked = sorted(observations, key=lambda o: -o.get('priority', 0))
        for scope in ('historical', 'matched'):
            allowed = [o for o in ranked if scope == 'historical' or not o['legacy']]
            if allowed:
                selections[(scope,) + key] = allowed[0]
        for observation in ranked[1:]:
            delta = observation['value'] - ranked[0]['value']
            if abs(delta) > 1e-6:
                alternatives.append(dict(method=key[0], checkpoint=key[1], family=key[2], dataset=key[3],
                    retained=ranked[0]['value'], alternative=observation['value'], delta=delta,
                    retained_source=ranked[0]['source'], alternative_source=observation['source']))

    cells = []
    for (scope, method, step, family, dataset), obs in sorted(selections.items()):
        cells.append(dict(scope=scope, method=method, checkpoint=step, family=family, dataset=dataset,
                          value=obs['value'], metric=obs['metric'], budget=20 if family == 'segmentation' else 0,
                          legacy=obs['legacy'], provenance=obs['provenance'], source=obs['source'], note=obs.get('note', '')))
    save_csv(OUT / 'SELECTED_CELLS.csv', cells)
    save_csv(OUT / 'ALTERNATIVE_OBSERVATIONS.csv', alternatives)
    summaries = []
    lookup = {}
    for view, scope, targets in [('historical', 'historical', expected), ('matched', 'matched', expected),
                                 ('common40', 'matched', common)]:
        for method in COLORS:
            for step in sorted(steps[method]):
                for family in FAMILIES:
                    observations = [selections[scope, method, step, family, ds] for ds in targets[family]
                                    if (scope, method, step, family, ds) in selections]
                    missing = [ds for ds in targets[family] if (scope, method, step, family, ds) not in selections]
                    legacy = sum(o['legacy'] for o in observations)
                    row = dict(view=view, method=method, checkpoint=step, family=family,
                               n=len(observations), expected=len(targets[family]),
                               mean=statistics.mean(o['value'] for o in observations) if not missing else None,
                               legacy_cells=legacy, missing=';'.join(missing))
                    assert row['mean'] is None or row['n'] == row['expected']
                    summaries.append(row)
                    lookup[view, method, step, family] = row
    save_csv(OUT / 'TASK_MEANS_AND_COVERAGE.csv', summaries)
    gaps = [{k: row[k] for k in ('view', 'method', 'checkpoint', 'family', 'n', 'expected', 'missing')}
            for row in summaries if row['missing']]
    save_csv(OUT / 'MISSING_CELLS.csv', gaps)
    deltas = []
    for view in ('historical', 'matched', 'common40'):
        for step in sorted(steps[A]):
            for family in FAMILIES:
                a = lookup[view, A, step, family]
                for baseline in (N, G):
                    b = lookup.get((view, baseline, step, family))
                    if a['mean'] is not None and b and b['mean'] is not None:
                        scale = 1 if family == 'regression' else 100
                        deltas.append(dict(view=view, checkpoint=step, family=family, baseline=baseline,
                                           adaptive=a['mean'], baseline_mean=b['mean'],
                                           delta=(a['mean'] - b['mean']) * scale,
                                           unit='R2' if scale == 1 else 'percentage_points',
                                           baseline_legacy_cells=b['legacy_cells']))
    save_csv(OUT / 'SAME_STEP_DELTAS.csv', deltas)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    pdf = PdfPages(OUT / 'ADAPTIVE_NOGRAM_GRAM.pdf')
    for view in ('historical', 'matched', 'common40'):
        for zoom in (False, True):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
            targets = common if view == 'common40' else expected
            for ax, family, name, ylabel in zip(axes.flat, FAMILIES, NAMES, LABELS):
                scale = 1 if family == 'regression' else 100
                for method, color in COLORS.items():
                    grid = sorted(s for s in steps[method] if not zoom or 12687 <= s <= latest)
                    segments = [([s for s in grid if s <= 29279], '-'),
                                ([s for s in grid if s >= 29767], '--')] if method == N else [(grid, '-')]
                    for segment, style in segments:
                        # HXW resumed ck23911; do not join it to the original ck29279 endpoint.
                        if method == N and style == '--' and segment and 23911 in grid:
                            segment = [23911] + segment
                        # A protocol switch must not look like a training-induced cliff.
                        x, y, previous_legacy = [], [], None
                        for s in segment:
                            row = lookup[view, method, s, family]
                            legacy = bool(row['legacy_cells'])
                            if view == 'historical' and previous_legacy is not None and legacy != previous_legacy:
                                x.append(math.nan)
                                y.append(math.nan)
                            x.append(s)
                            y.append(row['mean'] * scale if row['mean'] is not None else math.nan)
                            previous_legacy = legacy
                        ax.plot(x, y, color=color, ls=style, lw=2 if method == A else 1.4,
                                marker='o', ms=4 if method == A else 2.3, zorder=5 if method == A else 2)
                    legacy = [s for s in grid if lookup[view, method, s, family]['mean'] is not None
                              and lookup[view, method, s, family]['legacy_cells']]
                    if legacy:
                        ax.scatter(legacy, [lookup[view, method, s, family]['mean'] * scale for s in legacy],
                                   s=22, facecolors='white', edgecolors=color, linewidths=1, zorder=4)
                ax.set_title(f'{name} | {len(targets[family])} datasets', loc='left', weight='bold', fontsize=11)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.18)
                ax.axvline(12687, color='#c2c5ca', ls=':', lw=1)
                ax.set_xlim((12500, latest + 280) if zoom else (0, 42000))
                ax.ticklabel_format(axis='x', style='plain', useOffset=False)
                visible = {m: sum(lookup[view, m, s, family]['mean'] is not None for s in steps[m]
                                   if not zoom or 12687 <= s <= latest) for m in COLORS}
                ax.text(.02, .025, 'Complete points: ' + ' / '.join(f'{m} {visible[m]}' for m in COLORS),
                        transform=ax.transAxes, fontsize=8, color='#4b5563')
                if zoom:
                    at = lookup[view, A, latest, family]
                    if at['mean'] is None:
                        ax.text(.98, .97, f'Adaptive ck{latest}: {at["n"]}/{at["expected"]} evaluated',
                                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=COLORS[A])
            for ax in axes[1]:
                ax.set_xlabel('Optimizer updates')
            handles = [Line2D([0], [0], color=c, lw=2, marker='o', ms=4, label=m) for m, c in COLORS.items()]
            if not zoom:
                handles.append(Line2D([0], [0], color=COLORS[N], ls='--', label='no-GRAM continuation from ck23911'))
            if view == 'historical':
                handles.append(Line2D([0], [0], color='#777', marker='o', mfc='white', ls='', label='Includes legacy setting'))
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .952), ncol=len(handles), frameon=False, fontsize=9)
            title = {'historical': 'Six task trajectories | all indexed evidence',
                     'matched': 'Six task trajectories | legacy B4, old MoNuSeg split, legacy CoNIC split excluded',
                     'common40': 'README shared-40 task lists | detection shown separately'}[view]
            fig.suptitle(title + (' | Adaptive interval' if zoom else ''), fontsize=17, weight='bold', y=.995)
            foot = ('Fixed 25 / 4 / 7 / 7 / 7 / 3 dataset lists; complete family means only. Gaps are not zero. '
                    'Segmentation: E20, three seeds (PanNuke: three folds). MoNuSeg: official 30/7/14 only; '
                    'CoNIC detection: official source-disjoint split only.')
            if view == 'historical':
                foot += '\nOpen circles include historical B4 detection (BBBC038 / LIVECell). Lines break at setting changes; mixed-setting points are not directly comparable.'
            elif view == 'common40':
                foot = ('Fixed README lists: 24 / 2 / 4 / 4 / 6 datasets (40 cells); detection 3 is separate. '
                        'This supplementary view is not the full v4 list.\nComplete family means only; no overall score or checkpoint selection.')
            else:
                foot += '\nExcluding legacy settings does not certify all v4 admission checks; LC25000 / NCT100 caveats remain.'
            if view != 'common40':
                foot += '\nLC25000 / NCT100 remain provisional. CTC / OOD are separate from these six task families. No test-selected checkpoints.'
            fig.text(.5, .025, foot, ha='center', fontsize=8.2, color='#4b5563', linespacing=1.5)
            fig.subplots_adjust(left=.075, right=.985, top=.865, bottom=.14, wspace=.27, hspace=.35)
            stem = view.upper() + ('_ZOOM' if zoom else '_FULL')
            for extension in ('png', 'svg'):
                fig.savefig(OUT / f'{stem}.{extension}', dpi=185)
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()

    complete_counts = {view: {m: {f: sum(r['mean'] is not None for r in summaries
                                        if r['view'] == view and r['method'] == m and r['family'] == f)
                                  for f in FAMILIES} for m in COLORS}
                       for view in ('historical', 'matched', 'common40')}
    fully = [s for s in sorted(steps[A]) if all(lookup['matched', A, s, f]['mean'] is not None for f in FAMILIES)]
    meta = dict(generated_utc=datetime.now(timezone.utc).isoformat(), union_source=str(union_path),
                union_sha256=hashlib.sha256(union_bytes).hexdigest(),
                fresh_cells_sha256=hashlib.sha256((FRESH / 'CELLS.csv').read_bytes()).hexdigest(),
                latest_registered_adaptive=latest, latest_adaptive_six_families_observed=max(fully) if fully else None,
                checkpoints={m: sorted(s) for m, s in steps.items()}, complete_family_counts=complete_counts,
                protocol='v4 fixed task lists; sources retain their individual protocol labels',
                expected=expected, alternatives_over_1e_6=len(alternatives))
    (OUT / 'SUMMARY.json').write_text(json.dumps(meta, indent=2) + '\n')
    lines = ['# Adaptive / 原始 5TB no-GRAM / 原始 GRAM 合并曲线', '',
             '## 数据核对', '',
             '用户指出的 README 原版采用共享40项：分类24、回归2、检索4、聚类4、分割6；检测另外展示。'
             '这些记录确实存在。v4 六任务目标则是25/4/7/7/7/3，共53项，不能由40项完整直接推断53项全部完成。', '',
             f'本次读取 `{union_path}`，并从当前 DONE 任务重新收集 Adaptive 与原始基线补测。'
             f'采集时间 {meta["generated_utc"]}。Adaptive 最新登记评测点 {latest}，六任务数值齐全的最新点 {meta["latest_adaptive_six_families_observed"]}。', '',
             '## 图', '',
             '- HISTORICAL_FULL / HISTORICAL_ZOOM：固定 v4 六任务数据集清单，纳入旧结果。空心点明确表示含旧 B4 检测或旧 MoNuSeg 划分。',
             '- MATCHED_FULL / MATCHED_ZOOM：去掉上述两类旧设置；仍有 LC25000/NCT100 的数据准入限制，因此不声称严格完整 v4 全通过。',
             '- COMMON40_FULL / COMMON40_ZOOM：沿用 README 的40项固定清单，附独立检测面板，方便与旧汇总对应；不替代53项主图。',
             '- ADAPTIVE_NOGRAM_GRAM.pdf：上述六张图；PNG/SVG 为独立可插入汇报的图片。', '',
             '## 统计规则', '',
             '原始 no-GRAM 从0开始训练，首个已评测点487，原始轨迹至29279。29767–41479是从23911恢复的 HXW 续训，灰色虚线单独连接恢复点，'
             '不伪装成29279之后直接连续续训。GRAM 从12687分支至30255；12687仅为共享锚点，不增加一份独立 GRAM 评测。'
             'Adaptive 从12687接入，不使用短程 vanilla_formal/gram_formal 充当原始基线。', '',
             '每个任务固定数据集列表并等权平均；缺任一数据集就不画该任务均值，不用部分平均补齐。'
             '分割只取 E20，先按种子/折平均。分类 ChestMNIST 用macro AUC，其余BA；回归R²；检索Recall@1；聚类NMI；检测patch F1。'
             '同一单元按预定来源优先级选取，不按分数大小选择。CTC/OOD不属于此六任务图。', '',
             'SELECTED_CELLS.csv 包含每个单元的数值、来源与旧设置标记；TASK_MEANS_AND_COVERAGE.csv 包含固定分母、完整均值和缺项；'
             'MISSING_CELLS.csv 是确切缺项表；SAME_STEP_DELTAS.csv 仅比较共同 checkpoint 的完整任务均值；'
             'ALTERNATIVE_OBSERVATIONS.csv 保留重复来源数值差异。', '',
             '## Adaptive 当前评测覆盖', '',
             '|checkpoint|分类|回归|检索|聚类|分割E20|检测|', '|---|---|---|---|---|---|---|']
    for step in sorted(steps[A]):
        lines.append('|' + str(step) + '|' + '|'.join(f'{lookup["matched", A, step, f]["n"]}/{len(expected[f])}' for f in FAMILIES) + '|')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
