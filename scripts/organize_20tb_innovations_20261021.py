#!/usr/bin/env python3
"""Organize captured 20TB method evidence by parent checkpoint and evaluation protocol."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018/overview_20261021'
PLOT = ROOT / 'plot/fig2/20tb_innovations_20261021'
HISTORY = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/family_curve.csv'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')
STYLE = {
    'l20tb_domain': ('A 领域路由', '#AC4377', 'v'),
    'l20tb_complement_bio5tb': ('B 路由＋生物残差', '#008d80', 'D'),
    '20tb_from47823_lw5': ('旧普通lw5（47,823起点）', '#4d6c9d', 's'),
    '20tb_distill_lw5': ('普通lw5（74,663起点）', '#C18819', 'P'),
    '20tb_distill_lw2': ('普通lw2', '#627fb5', 'o'),
    'l20tb_rebal_resumed20261018': ('lw2＋重平衡', '#7e365f', '^'),
    '20tb_cooldown': ('仅cooldown对照', '#899295', 'x'),
}


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    PLOT.mkdir(parents=True, exist_ok=True)
    training = json.loads((REPORT / 'training.json').read_text())
    selected = json.loads((REPORT / 'selected29.json').read_text())['points']
    screening = json.loads((REPORT / 'screening45.json').read_text())
    complete29 = [p for p in selected if p['complete']]
    complete45 = [p for p in screening['points'] if p['complete']]
    historical = list(csv.DictReader(HISTORY.open()))
    reference = screening['arms']['5tb_peak19519']['scores']['19519']
    base45 = 100 * statistics.mean(statistics.mean(v for k, v in reference.items()
                                   if k.startswith(family + '/')) for family in FAMILIES)
    early29 = 100 * float(next(r['value'] for r in historical
                               if r['checkpoint'] == '47823' and r['family'] == 'overall'))
    late29 = 100 * next(p['overall'] for p in complete29 if p['arm'] == '20tb_nogram_late')
    strong29 = 100 * next(p['overall'] for p in complete29 if p['arm'] == '5tb_global_cls_w3')
    anchors45 = {p['checkpoint']: base45 + p['five_family'] for p in complete45
                 if p['arm'] == 'nogram_main'}
    points = []
    for p in complete29:
        points.append(dict(protocol='selected29_E20_3seeds', arm=p['arm'],
                           checkpoint=p['checkpoint'], score=100*p['overall']))
    for p in complete45:
        points.append(dict(protocol='screening45_E50_seed0', arm=p['arm'],
                           checkpoint=p['checkpoint'], score=base45+p['five_family']))
    csv_write(PLOT / 'complete_points.csv', points)
    font = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
        'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))

    def axis(ax, title, xlabel, baseline, xmax):
        ax.set_title(title, loc='left', fontsize=12, pad=12)
        ax.axhline(baseline, color='#687480', linestyle=':', lw=1, label='各自起点 no-GRAM')
        ax.scatter([0], [baseline], facecolors='white', edgecolors='#687480', s=45, zorder=5)
        ax.set_xlim(-150, xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('五任务族等权分数 × 100')
        ax.grid(axis='y', alpha=.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    def curve(ax, arm, protocol, fork, anchor):
        label, color, marker = STYLE[arm]
        source = selected if protocol == '29' else screening['points']
        data = sorted((p for p in source if p['arm'] == arm), key=lambda p: p['checkpoint'])
        valid = [p for p in data if p['complete']]
        if not valid:
            return
        value = (lambda p: 100*p['overall']) if protocol == '29' else (lambda p: base45+p['five_family'])
        ax.plot([0, valid[0]['checkpoint'] - fork], [anchor, value(valid[0])],
                ls='--', lw=1.3, color=color)
        xs = [p['checkpoint'] - fork for p in data]
        ys = [value(p) if p['complete'] else float('nan') for p in data]
        ax.plot(xs, ys, color=color, marker=marker, ms=6, lw=1.7, label=label)
        last = valid[-1]
        offset = -15 if arm == 'l20tb_rebal_resumed20261018' else 6
        ax.annotate(f'{value(last):.3f}', (last['checkpoint']-fork, value(last)),
                    xytext=(6, offset), textcoords='offset points', color=color, fontsize=9)

    a, b, c, d = axes.flat
    axis(a, '① 固定29项｜47,823起点：hxw当前两条新方法',
         '从47,823起新增的训练步数', early29, 7100)
    a.axhline(strong29, color='#72576C', ls='-.', lw=1, label='强5TB CLS复测')
    curve(a, 'l20tb_domain', '29', 47823, early29)
    a.text(.03, .06, 'B 生物残差：尚无完整29项点\n起点为历史评测；A为锁定复测，虚线仅标继承',
           transform=a.transAxes, fontsize=9, color='#526174', va='bottom')
    a.set_ylim(75.05, 75.95)
    axis(b, '② 固定29项｜74,663起点：已完成的普通lw5',
         '从74,663起新增的训练步数', late29, 5700)
    b.axhline(strong29, color='#72576C', ls='-.', lw=1, label='强5TB CLS复测')
    curve(b, '20tb_distill_lw5', '29', 74663, late29)
    b.text(.03, .06, '起点与终点均已29/29；虚线不是中间实测曲线',
           transform=b.transAxes, fontsize=9, color='#526174')
    axis(c, '③ 45项筛选｜47,823起点：同起点的三种候选',
         '从47,823起新增的训练步数', anchors45[47823], 7100)
    for arm in ('l20tb_domain', 'l20tb_complement_bio5tb', '20tb_from47823_lw5'):
        curve(c, arm, '45', 47823, anchors45[47823])
    axis(d, '④ 45项筛选｜74,663起点：蒸馏与仅cooldown',
         '从74,663起新增的训练步数', anchors45[74663], 5700)
    for arm in ('20tb_distill_lw5', '20tb_distill_lw2', 'l20tb_rebal_resumed20261018', '20tb_cooldown'):
        curve(d, arm, '45', 74663, anchors45[74663])
    for ax in axes.flat:
        ax.legend(loc='upper left', fontsize=8, frameon=False)
    a.legend(loc='upper right', fontsize=8, frameon=False)
    b.legend(loc='lower right', bbox_to_anchor=(1, .12), fontsize=8, frameon=False)
    d.legend(loc='lower right', fontsize=8, frameon=False)
    stamp = dt.datetime.fromtimestamp(training['captured'], dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle('20TB 方法与曲线：先区分起点，再区分评测口径', fontsize=18, y=.98)
    progress = '；'.join(f"{name}训练 {training['arms'][key]['latest']:,} / 67,343"
                        for key, name in [('domain', 'A领域路由'), ('complement', 'B生物残差')])
    fig.text(.07, .93, f'{stamp}｜hxw-xzj：{progress}', fontsize=10, color='#526174')
    fig.text(.07, .035, '上排：固定29项，分割E20×3种子。下排：45项筛选，分割E50单种子；分类/回归/检索指标也不同，分数不可跨排比较。', fontsize=9)
    fig.text(.07, .014, '实心点均为完整评测；虚线标继承，缺失点保留断线。两条hxw新方法计划新增19,520步，旧普通蒸馏仅新增4,880步。', fontsize=9)
    fig.subplots_adjust(left=.07, right=.97, bottom=.12, top=.85, hspace=.48, wspace=.28)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(PLOT / f'20tb_methods_overview_20261021.{ext}', dpi=200)
    plt.close(fig)

    # Detail by family for the two current candidates; protocols remain separate rows.
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    titles = ('分类', '回归', '检索', '聚类', '分割', '五族均值')
    for ax, fam, title in zip(axes.flat, (*FAMILIES, 'overall'), titles):
        for arm in ('l20tb_domain', 'l20tb_complement_bio5tb', '20tb_from47823_lw5'):
            label, color, marker = STYLE[arm]
            data = sorted((p for p in screening['points'] if p['arm'] == arm), key=lambda p: p['checkpoint'])
            key = 'five_family' if fam == 'overall' else fam
            parent = next(p for p in complete45 if p['arm'] == 'nogram_main' and p['checkpoint'] == 47823)
            valid = [p for p in data if p['complete']]
            if not valid:
                continue
            ax.plot([0, valid[0]['checkpoint'] - 47823], [0, valid[0][key]-parent[key]], color=color, ls='--')
            ax.plot([p['checkpoint']-47823 for p in data],
                    [p[key]-parent[key] if p['complete'] else float('nan') for p in data],
                    color=color, marker=marker, ms=5, label=label)
        ax.axhline(0, color='#7d8992', lw=.8)
        ax.set_title(title)
        ax.set_xlabel('从47,823起新增步数')
        ax.set_ylabel('相对自身no-GRAM起点提升（分）')
        ax.grid(axis='y', alpha=.2)
        ax.set_xlim(-100, 5300)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('hxw两条新方法的各任务族效果｜45项筛选口径', fontsize=16)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(PLOT / f'hxw_two_methods_families45_20261021.{ext}', dpi=200)
    plt.close(fig)

    lookup29 = {(p['arm'], p['checkpoint']): 100*p['overall'] for p in complete29}
    lookup45 = {(p['arm'], p['checkpoint']): base45+p['five_family'] for p in complete45}
    rows = []
    for arm in STYLE:
        ps = [p for p in complete45 if p['arm'] == arm]
        if not ps:
            continue
        last = max(ps, key=lambda p: p['checkpoint'])
        fork = 47823 if arm in ('l20tb_domain', 'l20tb_complement_bio5tb', '20tb_from47823_lw5') else 74663
        rows.append(dict(arm=arm, checkpoint=last['checkpoint'], score=base45+last['five_family'],
                         fork=fork, delta_vs_parent=base45+last['five_family']-anchors45[fork],
                         delta_vs_5tb=last['five_family'], complete_points=len(ps)))
    csv_write(PLOT / 'screening45_latest.csv', rows)
    dom=training['arms']['domain']; bio=training['arms']['complement']
    late_gain=lookup29['20tb_distill_lw5',79543]-late29
    same_gain=lookup45['l20tb_complement_bio5tb',50263]-lookup45['l20tb_domain',50263]
    old_gain=lookup45['l20tb_complement_bio5tb',50263]-lookup45['20tb_from47823_lw5',50263]
    report = f'''# 20TB 创新、分支关系与曲线

核查时间：{stamp}。训练直接查5090-hxw-xzj；固定29项读取原始验证结果；45项读取本地与5090-lyx-xr结果。

## 先分清四件事

1. 20TB no-GRAM是所有这些L实验的主线；domain routing不是另一条从零开始的20TB训练。
2. hxw现在训练的是A领域路由、B领域路由＋生物残差。B包含A的路由设计，两者从同一个47,823 checkpoint独立分叉。
3. 旧普通lw5有两条：47,823→52,703和74,663→79,543。名称相似，初始化不同。它们都不等于hxw当前的A/B。
4. 固定29项与45项筛选的任务集、指标、分割预算都不同，必须分别画图。任何一套都不称为完整V4。

```mermaid
flowchart LR
    P[20TB no-GRAM 47,823] --> A[A 领域路由：目标67,343]
    P --> B[B 路由＋生物残差：目标67,343]
    P --> C[旧普通lw5：终点52,703]
    P --> Q[no-GRAM主线继续到74,663]
    Q --> D[普通lw5 / lw2 / 重平衡：终点79,543]
    Q --> E[仅cooldown对照：终点79,543]
```

## hxw-xzj上的两个新方法

| | A：领域路由 | B：领域路由＋生物残差 |
|---|---|---|
| 假设 | 病理专家不适合约束所有显微图像 | 通用教师不能覆盖强5TB模型的全部生物表征 |
| 所有图像 | 通用DINOv3-L教师＋原HS6自监督 | 同A |
| 病理图像 | 额外H-optimus-0监督，映射统计也仅用病理样本 | 同A |
| 非病理图像 | 不加病理教师监督 | 加强5TB/19519教师中不能被通用教师线性预测的残差信息 |
| 原始起点 | 47,823（完整模型＋AdamW） | 同一47,823，独立分叉 |
| 计划 | 新增19,520步，到67,343 | 同A |
| 目前实际训练 | {dom['latest']:,}（新增{dom['latest']-47823:,}，{(dom['latest']-47823)/19520:.1%}） | {bio['latest']:,}（新增{bio['latest']-47823:,}，{(bio['latest']-47823)/19520:.1%}） |
| 完整29项测试 | 50,263、52,703 | 尚无完整点 |
| 完整45项测试 | 48,311、50,263 | 50,263 |
| 迁移历史 | deepcad训练到52,703后迁到hxw | 3090-qi训练到49,775后迁到hxw |

B的残差可写成：`强5TB教师特征 − 根据通用教师特征得到的在线线性预测`。它让学生恢复互补信息，而不是重复拟合两个教师共有的部分。病理标签来自训练元数据，不来自下游测试分数；教师冻结，CLS与patch都使用该设计。这是已实现的创新候选，尚未证明论文级首创或最终胜出。

两条hxw配置当前均为lw5、4卡×microbatch64×累积4，有效batch1024。迁移前曾用microbatch32×累积8；有效batch相同不代表微批统计完全等价。两条计划评测点为50,263/52,703/57,583/67,343；导出每488步一次并不意味着每个导出都要完成29项测试。当前训练进程均存在，下一阶段的效果还不能用训练进度替代。

## 旧实验在整个研究中的位置

| 类别 | 方法 | 检验的问题 |
|---|---|---|
| 数据策略 | boundary配比、resampling | 20TB采样曝光是否不合适 |
| 表征约束 | fixed CLS、slow CLS EMA | 是否需要保持全局语义表征 |
| 收尾对照 | 仅cooldown | 学习率冷却本身能改善多少 |
| 普通外部蒸馏 | lw2、lw5 | 外部通用/病理教师能否恢复能力；2/5是蒸馏权重 |
| 外部蒸馏变体 | lw2＋重平衡 | 改变采样与蒸馏的组合是否有帮助 |
| 其他历史尝试 | self-readout、SCV | 内部可恢复性/一致性约束；沿用历史独立评测记录 |
| 当前A/B | 领域选择、互补残差 | 给适用图像正确的教师目标，并减少重复知识 |

数据策略/CLS旧结果见 `scripts/plot_20tb_methods_20261015.py` 对应的配对比较。其对照、任务覆盖与这次总分不同，不拼成统一排行榜。7B朴素基线是另一条模型规模实验，不是这里的L创新分支。来源内语义覆盖仍属后备设想，不列为已训练方法。

## 当前效果：固定29项

| 方法 | checkpoint | 分数 | 说明 |
|---|---:|---:|---|
| 强5TB CLS复测 | 32,695 | {strong29:.3f} | 此处参照 |
| A领域路由 | 50,263 | {lookup29['l20tb_domain',50263]:.3f} | 完整29项 |
| A领域路由 | 52,703 | {lookup29['l20tb_domain',52703]:.3f} | 比强5TB +{lookup29['l20tb_domain',52703]-strong29:.3f}分 |
| B路由＋生物残差 | — | — | 不能用45项成绩代填 |
| no-GRAM晚期起点 | 74,663 | {late29:.3f} | 现已29/29，上一轮缺失已补齐 |
| 普通lw5 | 79,543 | {lookup29['20tb_distill_lw5',79543]:.3f} | 比自身起点 +{late_gain:.3f}分；比强5TB +{lookup29['20tb_distill_lw5',79543]-strong29:.3f}分 |

普通lw5的+{late_gain:.3f}分包含新增训练和cooldown影响；固定29项没有匹配的cooldown完整对照，因此不能全部归因为蒸馏。A的历史起点75.285与其新锁定复测来源不同，只用于谱系背景。

## 当前效果：45项筛选，同一50,263步对比

| 从47,823开始的分支 | 50,263的分数 | 相对共同no-GRAM起点 |
|---|---:|---:|
| A领域路由 | {lookup45['l20tb_domain',50263]:.3f} | {lookup45['l20tb_domain',50263]-anchors45[47823]:+.3f} |
| B路由＋生物残差 | {lookup45['l20tb_complement_bio5tb',50263]:.3f} | {lookup45['l20tb_complement_bio5tb',50263]-anchors45[47823]:+.3f} |
| 旧普通lw5 | {lookup45['20tb_from47823_lw5',50263]:.3f} | {lookup45['20tb_from47823_lw5',50263]-anchors45[47823]:+.3f} |

B在此已观测点比A高{same_gain:.3f}分，比旧普通lw5高{old_gain:.3f}分。A目前没有在这个筛选点超过旧普通lw5；B有小幅领先，但只有一个完整筛选点、没有完整29项点。旧普通lw5计划4,880步，A/B计划19,520步，冷却日程不同；这是候选表现，不是严格隔离方法贡献的消融结论。

## 45项筛选：各分支最新完整点

| 方法 | checkpoint | 分数 | 相对自身起点 | 相对强5TB |
|---|---:|---:|---:|---:|
'''
    for r in rows:
        report += f"| {STYLE[r['arm']][0]} | {r['checkpoint']} | {r['score']:.3f} | {r['delta_vs_parent']:+.3f} | {r['delta_vs_5tb']:+.3f} |\n"
    report += f'''
同为79,543步，普通lw5相对仅cooldown提高{lookup45['20tb_distill_lw5',79543]-lookup45['20tb_cooldown',79543]:.3f}分。分数只来自完整点；缺失中间点不插值。聚类对总分的贡献需结合任务族图查看。

## 曲线入口与复现

- [四格总览]({PLOT / '20tb_methods_overview_20261021.png'})：左边47,823分支，右边74,663分支；上排29项、下排45项。
- [hxw两个新方法的任务族曲线]({PLOT / 'hxw_two_methods_families45_20261021.png'})：45项口径，包含旧普通lw5作参照。
- 同目录有PDF、SVG、逐点CSV；完整五规模图也已刷新，lw5的74,663起点现可连接。
- `python scripts/organize_20tb_innovations_20261021.py` 从本报告目录的固定快照重建图表。

原始快照：training.json保留hxw实际配置及训练步；selected29.json保留逐任务族/逐数据集结果；lyx_screening.json和screening45.json保留45项结果及完成状态。manifest记录源哈希。旧主线65,391到74,663之间仍有缺失；74,663新复测只补这个端点，不代表中间全部测完。
'''
    (REPORT / 'README.md').write_text(report)
    (PLOT / 'README.md').write_text(report)
    sources = [REPORT / name for name in ('training.json', 'selected29.json', 'lyx_screening.json', 'screening45.json')]
    sources += [HISTORY, Path(__file__).resolve()]
    manifest = dict(captured_utc=stamp, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    selected29_complete=len(complete29), screening45_complete=len(complete45),
                    no_interpolation=True, units='100 * equal mean of five family means')
    (PLOT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(REPORT / 'README.md')
    print(json.dumps(dict(lw5_selected29_gain=late_gain, bio_vs_domain_same_step45=same_gain,
                          bio_vs_old_lw5_same_step45=old_gain)))


if __name__ == '__main__':
    main()
