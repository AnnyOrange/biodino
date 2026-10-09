#!/usr/bin/env python3
"""Quantify MoNuSeg setting shifts at identical pretrained checkpoints."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/00_reports/deepcad_method_20260927/adaptive_original_union_20260929/segmentation_setting_audit'
PARENT = OUT.parent


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    union = json.loads((ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_all_models.json').read_text())
    pairs = []
    meta_rows = []
    fields = ['used_train_samples', 'probe_batch_size', 'probe_epochs', 'learning_rate',
              'weight_decay', 'dropout', 'optimizer', 'global_optimizer_steps', 'selection_metric']
    for model, name in [('hs6_l_5tb_no_gram', 'no-GRAM'), ('hs6_l_5tb_gram12687', 'GRAM')]:
        for step, ck in sorted(union[model]['checkpoints'].items(), key=lambda x: int(x[0])):
            observations = [o for o in ck['cells'].get('segmentation', {}).get('monuseg', {}).get('observations', [])
                            if o.get('budget_epochs') == 20]
            old = [o for o in observations if o['provenance'] == 'legacy_accepted']
            new = [o for o in observations if o['provenance'] != 'legacy_accepted']
            if not old or not new:
                continue
            legacy, formal = old[0], max(new, key=lambda o: o['priority'])
            pairs.append(dict(model=name, checkpoint=int(step), old_mdice=legacy['value'],
                              new_mdice=formal['value'], difference_pp=100*(legacy['value']-formal['value']),
                              seven_dataset_mean_difference_pp=100*(legacy['value']-formal['value'])/7,
                              old_source=legacy['source'], new_source=formal['source']))
            for setting, o in [('old', legacy), ('new', formal)]:
                paths = [Path(p) for p in o['source'].split(';')]
                assert len(paths) == 3, (step, setting, paths)
                scores = []
                seeds = set()
                for path in paths:
                    obj = json.loads(path.read_text())
                    meta = obj['_meta']
                    seeds.add(meta['seed'])
                    scores.append(obj['test']['mDice'])
                    meta_rows.append(dict(model=name, checkpoint=int(step), setting=setting, seed=meta['seed'],
                                          **{f: meta.get(f) for f in fields}, source=str(path)))
                assert seeds == {0, 1, 2}
                assert abs(statistics.mean(scores)-o['value']) < 1e-12
    assert len(pairs) == 14
    assert all(p['difference_pp'] > 0 for p in pairs)
    write_csv(OUT/'PAIRED_MONUSEG_E20.csv', pairs)
    write_csv(OUT/'PROBE_METADATA.csv', meta_rows)

    # Fixed six-dataset diagnostic, explicitly distinct from full seven-dataset v4.
    cells = list(csv.DictReader((PARENT/'SELECTED_CELLS.csv').open()))
    groups = defaultdict(list)
    for r in cells:
        if r['scope'] == 'matched' and r['family'] == 'segmentation' and r['dataset'] != 'monuseg':
            groups[r['method'], int(r['checkpoint'])].append(r)
    six = []
    expected = {'cellpose', 'conic', 'livecell', 'pannuke', 'tissuenet', 'multimodal_cellseg'}
    for (model, step), records in sorted(groups.items()):
        if {r['dataset'] for r in records} == expected:
            six.append(dict(model=model, checkpoint=step, mean=statistics.mean(float(r['value']) for r in records)))
    write_csv(OUT/'SIX_DATASETS_DIAGNOSTIC.csv', six)
    colors = {'no-GRAM': '#64748b', 'GRAM': '#dc8b19', 'Adaptive': '#008c88'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for i, p in enumerate(pairs):
        c = colors[p['model']]
        axes[0].plot([i, i], [p['new_mdice']*100, p['old_mdice']*100], color=c, alpha=.6)
        axes[0].scatter(i, p['old_mdice']*100, facecolors='white', edgecolors=c, s=45, zorder=3)
        axes[0].scatter(i, p['new_mdice']*100, color=c, s=25, zorder=3)
    axes[0].set_xticks(range(len(pairs)), [('N ' if p['model']=='no-GRAM' else 'G ')+str(p['checkpoint']) for p in pairs], rotation=65, ha='right', fontsize=8)
    axes[0].set_title('Same checkpoint: legacy vs current MoNuSeg', loc='left', weight='bold')
    axes[0].set_ylabel('MoNuSeg E20 mDice (%)')
    axes[0].plot([], [], 'o', mfc='white', mec='#555', label='Legacy: train 30 / val 7')
    axes[0].plot([], [], 'o', color='#555', ms=5, label='Current: train 24 / val 6')
    axes[0].legend(fontsize=8, loc='upper right')
    for model, color in colors.items():
        selected = [r for r in six if r['model'] == model and 12687 <= r['checkpoint'] <= 17567]
        axes[1].plot([r['checkpoint'] for r in selected], [r['mean']*100 for r in selected],
                     color=color, marker='o', ms=4, label=model, lw=1.8)
    axes[1].set_title('Other six segmentation datasets: fixed-list diagnostic', loc='left', weight='bold', fontsize=10)
    axes[1].set_xlabel('Optimizer updates')
    axes[1].set_ylabel('Mean E20 mDice (%)')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('The apparent segmentation gap includes a MoNuSeg evaluation-setting shift', fontsize=14, weight='bold')
    fig.text(.5, .025, 'Left: identical pretrained checkpoints; E20 / B32 / seeds 0,1,2 in both settings. All 14 paired legacy scores are higher.\n'
             'Right: excludes MoNuSeg for diagnosis only; it is not the full seven-dataset v4 segmentation score.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT/f'SEGMENTATION_SETTING_SHIFT.{ext}', dpi=190)
    plt.close(fig)
    summary = {}
    for model in ('no-GRAM', 'GRAM'):
        pp = [p for p in pairs if p['model'] == model]
        summary[model] = dict(paired_checkpoints=len(pp), min_difference_pp=min(p['difference_pp'] for p in pp),
                             max_difference_pp=max(p['difference_pp'] for p in pp),
                             mean_difference_pp=statistics.mean(p['difference_pp'] for p in pp))
    lookup = {(r['model'],r['checkpoint']):r['mean'] for r in six}
    summary['six_dataset_adaptive_minus_no_gram_pp'] = {
        str(s):100*(v-lookup['no-GRAM',s]) for (m,s),v in lookup.items()
        if m=='Adaptive' and ('no-GRAM',s) in lookup}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'README.md').write_text(
        '# 分割空心点与实心点落差审计\n\n'
        '同一预训练 checkpoint 的 MoNuSeg 新旧 E20 结果有14对：no-GRAM 8对、GRAM 6对，旧设置全部更高。'
        '逐一读取每对两侧的3个seed原始结果，核验均值与种子，记录 probe 参数。\n\n'
        '旧设置训练/验证/测试计数30/7/14，来自旧37图训练池；当前设置为官方30图训练池拆24/6，测试14。'
        '两侧均E20、B32、三个seed。旧设置有更多训练样本、不同验证样本与选择过程；本审计没有单独隔离各因素，'
        '也未据测试数量相等推断样本身份相同，因此不将差距全部归因于训练数量，不据此声称数据泄漏。\n\n'
        'PAIR表中的7数据集均值影响严格为MoNuSeg差值/7，其余6项固定。右图提供剔除MoNuSeg的固定6项诊断，'
        '不能替代v4的7项主结果。原主图已修改为设置切换时断线，不再把口径切换连成模型退化。\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
