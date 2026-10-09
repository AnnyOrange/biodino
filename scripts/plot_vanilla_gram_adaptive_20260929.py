"""Plot frozen completed observations; gaps remain gaps, no checkpoint selection."""
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/00_reports/deepcad_method_20260927/curves_20260929'
COLORS = {'vanilla': '#6b7280', 'Gram': '#d88920', 'Adaptive': '#087f8c'}
EARLY = [13175, 13663, 14151, 14639, 15127]


def main():
    data = json.loads((OUT / 'CURVE_DATA.json').read_text())
    models = data['models']
    arms = {m: {} for m in COLORS}
    for arm in sorted(models):
        method = 'Adaptive' if arm.startswith('adaptive_') else 'Gram' if arm.startswith('gram_') else 'vanilla'
        step = int(arm.rsplit('_ck', 1)[1])
        if step in arms[method]:
            raise ValueError(f'Duplicate trajectory point: {method} {step}')
        arms[method][step] = arm
    common = sorted(set.intersection(*(set(models[arms[m][s]]['classification'])
                                      for m in COLORS for s in EARLY if s in arms[m])))
    assert len(common) == 23, common
    for model in models.values():
        if all(k in model['classification'] for k in common):
            model['metrics']['Classification: fixed 23-task mean BA (%)'] = statistics.mean(model['classification'][k] for k in common)
    allsteps = sorted(set().union(*(set(v) for v in arms.values())))
    records = []
    for method, mapping in arms.items():
        for step, arm in sorted(mapping.items()):
            for group in ('classification', 'metrics'):
                for metric, value in sorted(models[arm][group].items()):
                    records.append(dict(method=method, checkpoint=step, arm=arm, group=group,
                                        metric=metric, value=value, snapshot=data['time']))
    with (OUT / 'CURVE_VALUES.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})
    pdf = PdfPages(OUT / 'VANILLA_GRAM_ADAPTIVE_ALL.pdf')

    def curve(ax, group, key, steps=EARLY, title=None, methods=None):
        for method in methods or COLORS:
            values = [models[arms[method][s]][group].get(key, math.nan)
                      if s in arms[method] else math.nan for s in steps]
            ax.plot(steps, values, marker='o', ms=4, lw=1.8, color=COLORS[method], label=method)
        ax.set_title(title or key, fontsize=10)
        ax.set_ylabel('R-squared' if key.endswith(' R2') else 'Balanced accuracy (%)' if group == 'classification' else 'Score (%)')
        ax.set_xlabel('Checkpoint / optimizer update')
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        if steps == EARLY:
            ax.set_xticks(EARLY); ax.tick_params(axis='x', rotation=25, labelsize=8)

    def finish(fig, filename, title, note):
        fig.suptitle(title, fontsize=16, y=.995)
        fig.text(.5, .008, note, ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .045, 1, .95])
        fig.savefig(OUT / filename, dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

    overview = [
        ('Classification: fixed 23-task mean BA (%)', 'Classification: same 23 tasks'),
        ('bbbc038 patch F1 (%)', 'BBBC038: detection proxy F1'),
        ('RxRx3 mAP (%)', 'RxRx3: retrieval mAP'),
        ('cellpose E50 mIoU (%)', 'Cellpose: E50 segmentation mIoU'),
        ('hpa-subcellular R@1 (%)', 'HPA: retrieval Recall@1'),
        ('bbbc013 R2', 'BBBC013: regression R-squared'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    for ax, (key, title) in zip(axes.flat, overview):
        curve(ax, 'metrics', key, title=title)
    axes.flat[0].legend(loc='best')
    finish(fig, 'OVERVIEW.png', 'Vanilla vs Gram vs Adaptive: continuation from checkpoint 12687',
           'Six displayed measurements, not a full-v4 score. Gram @15127 unavailable. One FM seed; observed differences are not significance tests.')

    # All early available single-label tasks, preserving each task's metric scale.
    for page in range(math.ceil(len(common) / 12)):
        fig, axes = plt.subplots(4, 3, figsize=(15, 15))
        subset = common[page*12:(page+1)*12]
        for ax, key in zip(axes.flat, subset):
            curve(ax, 'classification', key)
        for ax in list(axes.flat)[len(subset):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        finish(fig, f'CLASSIFICATION_{page+1}.png', f'Individual classification curves ({page+1}/2)',
               'Fixed 23 single-label tasks. LC25000 and ChestMNIST excluded from BA panels. Missing results are not imputed.')

    metric_keys = sorted(set.union(*(set(models[arm]['metrics']) for m in arms.values()
                                    for s, arm in m.items() if s in EARLY)))
    metric_keys.remove('Classification: fixed 23-task mean BA (%)')
    for page in range(math.ceil(len(metric_keys) / 9)):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        subset = metric_keys[page*9:(page+1)*9]
        for ax, key in zip(axes.flat, subset): curve(ax, 'metrics', key)
        for ax in list(axes.flat)[len(subset):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        finish(fig, f'OTHER_METRICS_{page+1}.png', f'Detection, retrieval, regression and segmentation ({page+1}/2)',
               'Segmentation uses E50; patch F1 is a detection proxy. CP accuracy overlaps its classification BA task. Full v4 is still incomplete.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True)
    for ax, method in zip(axes, ['Gram', 'Adaptive']):
        delta = np.full((len(common), len(EARLY)), np.nan)
        for j, step in enumerate(EARLY):
            if step not in arms[method]: continue
            for i, key in enumerate(common):
                delta[i, j] = models[arms[method][step]]['classification'][key] - models[arms['vanilla'][step]]['classification'][key]
        cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('#dedede')
        im = ax.imshow(delta, aspect='auto', cmap=cmap, vmin=-2, vmax=2)
        ax.set_xticks(range(len(EARLY)), EARLY, rotation=30)
        ax.set_yticks(range(len(common)), common, fontsize=9)
        ax.set_title(method + ' minus vanilla')
        for i in range(len(common)):
            for j in range(len(EARLY)):
                value = delta[i, j]
                ax.text(j, i, f'{value:+.2f}' if np.isfinite(value) else 'N/A', ha='center', va='center',
                        fontsize=8, color='white' if abs(value) > 1.2 else '#202020')
    fig.subplots_adjust(left=.20, right=.89, bottom=.09, top=.90, wspace=.12)
    cax = fig.add_axes([.915, .2, .016, .6]); fig.colorbar(im, cax=cax, label='BA difference (percentage points); colors clipped at +/-2')
    fig.suptitle('Every classification task vs vanilla: blue = improvement, red = regression', fontsize=13, y=.96)
    fig.text(.5, .015, 'Raw numerical differences, not statistical significance. Gray = unavailable. No task is dropped because it regressed.', ha='center', fontsize=9)
    fig.savefig(OUT / 'CLASSIFICATION_VS_VANILLA.png', dpi=180); pdf.savefig(fig); plt.close(fig)

    longsteps = [s for s in allsteps if s >= 15127]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    longkeys = ['bbbc038 patch F1 (%)', 'RxRx3 mAP (%)', 'hpa-subcellular R@1 (%)',
                'bbbc005 R2', 'bbbc013 R2', 'livecell-cell-count R2']
    for ax, key in zip(axes.flat, longkeys): curve(ax, 'metrics', key, steps=longsteps, methods=['Adaptive'])
    finish(fig, 'ADAPTIVE_LONG.png', 'Adaptive continuation: available later observations',
           'No matched later vanilla/Gram results in this campaign. These are trajectory changes, not estimated method gains. Missing results remain gaps.')
    pdf.close()
    lines = ['# Vanilla / Gram / Adaptive 曲线', '', f"结果采集时间：{data['time']}。", '',
             f"采集时Adaptive训练日志到{data['latest_training']['optimizer_update']}；目标20007。训练与下游评测进度不同。", '',
             '- OVERVIEW.png：三方法6项指标概览。',
             '- CLASSIFICATION_VS_VANILLA.png：23项分类逐项相对vanilla差值，红色为下降。',
             '- CLASSIFICATION_1/2.png：23项分类的完整曲线。',
             '- OTHER_METRICS_1/2.png：已采集检测代理、检索、回归、E50分割指标。',
             '- ADAPTIVE_LONG.png：后期Adaptive已有结果；缺少对应vanilla/Gram，不能估计方法收益。',
             '- VANILLA_GRAM_ADAPTIVE_ALL.pdf：以上全部图页。',
             '- CURVE_VALUES.csv / CURVE_DATA.json：绘图数值、任务源文件与状态快照。', '',
             '口径：从12687接入的续训vanilla/Gram/Adaptive，不是原始5TB no-Gram/Gram整条轨迹。'
             '分类固定23项单标签BA，排除LC25000和ChestMNIST。共有对照是13175/13663/14151/14639；'
             '15127有vanilla与Adaptive，未取得Gram。全量v4尚未完成；原生tracking、OOD及其他预算不在这些图中。'
             '分割为E50预算、按现有评测汇总probe seeds/folds；一个FM训练seed，图中不宣称显著性。', '',
             '判定目标依然是逐任务优于vanilla，平均分提高不能替代该要求。没有根据这些test曲线选择新的超参或训练终点。', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(f'{OUT}: {len(records)} plotted source values, {len(common)} classification tasks')


if __name__ == '__main__':
    main()
