#!/usr/bin/env python3
"""Image2-style H+ 1TB vs 5TB task-family comparison.

BBBC013 is excluded. Segmentation uses the common available subset between H+ 1TB
and H+ 5TB in per_dataset_best.csv.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures_vertical_white')
BENCH = Path('outputs/00_reports/benchmark_results_summary.csv')
PER_BEST = Path('outputs/03_comparisons/2026-7-1-test-overall/per_dataset_best.csv')
OUT.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = {
    '1TB': 'bio_continue_rgb3_vith16plus',
    '5TB': '5tb_hplus_packwds_ep15_b1024',
}
SCALES = ['1TB', '5TB']
X = np.array([math.log10(1), math.log10(5)])
BUBBLE = {'1TB': 230, '5TB': 360}
BLUE = '#4B87B9'
RED = '#EF443B'
GRID = '#BEBEBE'
TEXT = '#111111'

PANELS = [
    ('Classification', 'Balanced accuracy', 'Classification', '{:.4f}'),
    ('Regression R2', 'R2', 'Regression R2', '{:.4f}'),
    ('Regression MAE', 'MAE', 'Regression MAE', '{:.3f}'),
    ('Retrieval', 'Recall@1', 'Retrieval', '{:.4f}'),
    ('Clustering', 'NMI', 'Clustering', '{:.4f}'),
    ('Segmentation', 'mDice', 'Segmentation', '{:.4f}'),
]

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.family': 'DejaVu Serif', 'axes.labelsize': 12, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
})


def fnum(x):
    try:
        if x == '' or x is None:
            return None
        return float(x)
    except Exception:
        return None


def mean_best(rows, model_key, task, metric, datasets=None, higher=True):
    best = {}
    for r in rows:
        if r.get('model_key') != model_key or r.get('task') != task:
            continue
        ds = r.get('dataset')
        if ds == 'bbbc013':
            continue
        if datasets is not None and ds not in datasets:
            continue
        v = fnum(r.get(metric))
        if v is None and r.get('primary_metric') == metric:
            v = fnum(r.get('metric_value'))
        if v is None:
            continue
        if ds not in best or (v > best[ds] if higher else v < best[ds]):
            best[ds] = v
    return sum(best.values()) / len(best) if best else None


def collect_values():
    rows = list(csv.DictReader(BENCH.open(newline='')))
    values = {name: {} for name, *_ in PANELS}
    for scale, mk in MODEL_KEYS.items():
        values['Classification'][scale] = mean_best(rows, mk, 'classification', 'balanced_accuracy')
        values['Regression R2'][scale] = mean_best(rows, mk, 'regression', 'r2', datasets={'bbbc005'})
        values['Regression MAE'][scale] = mean_best(rows, mk, 'regression', 'mae', datasets={'bbbc005'}, higher=False)
        values['Retrieval'][scale] = mean_best(
            rows, mk, 'retrieval', 'recall_at_1',
            datasets={'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'},
        )
        values['Clustering'][scale] = mean_best(
            rows, mk, 'retrieval', 'nmi',
            datasets={'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'},
        )

    per = list(csv.DictReader(PER_BEST.open(newline='')))
    seg = defaultdict(dict)
    for r in per:
        if r.get('category') != 'segmentation' or r.get('metric_key') != 'segmentation_mDice':
            continue
        if r.get('dataset') == 'multimodal_cellseg':
            continue
        for scale, mk in MODEL_KEYS.items():
            if r.get('model') == mk:
                seg[scale][r['dataset']] = fnum(r['value'])
    common = set(seg['1TB']) & set(seg['5TB'])
    for scale in SCALES:
        vals = [seg[scale][d] for d in common]
        values['Segmentation'][scale] = sum(vals) / len(vals) if vals else None
    return values


def add_axis_arrows(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)
    ax.annotate('', xy=(xmax, ymin), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)


def plot_panel(ax, panel, vals):
    title, ylabel, caption, fmt = panel
    ys = [vals['1TB'], vals['5TB']]
    colors = [BLUE, RED]
    sizes = [BUBBLE['1TB'], BUBBLE['5TB']]
    ax.plot(X, ys, linestyle=(0, (5, 5)), color='black', linewidth=0.8, zorder=1)
    ax.scatter(X, ys, s=sizes, color=colors, edgecolor='black', linewidth=0.7, alpha=0.97, zorder=2)
    for xx, yy, lab in zip(X, ys, SCALES):
        ax.text(xx, yy, lab.replace('TB', ''), ha='center', va='center', fontsize=8.2,
                color='white', fontweight='bold', zorder=3)

    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.50 if hi != lo else abs(hi) * 0.015 + 0.01
    y0 = max(0, lo - pad)
    y1 = hi + pad
    ax.set_xlim(-0.06, math.log10(5) + 0.08)
    ax.set_ylim(y0, y1)
    yr = y1 - y0
    for xx, yy in zip(X, ys):
        ax.text(xx, yy + 0.045 * yr, fmt.format(yy), ha='center', va='bottom', fontsize=8)

    ax.text(X[-1] - 0.01, ys[-1] + 0.18 * yr, 'H+ 5TB', ha='right', va='center',
            fontsize=10, color='white',
            bbox=dict(boxstyle='square,pad=0.22', facecolor=RED, edgecolor='none', alpha=0.95))
    ax.text(X[0] + 0.01, ys[0] - 0.18 * yr, 'H+ 1TB', ha='left', va='center',
            fontsize=10, color='white',
            bbox=dict(boxstyle='square,pad=0.22', facecolor=BLUE, edgecolor='none', alpha=0.95))

    ax.set_title(title, fontsize=11, pad=6, fontweight='normal')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('log-Data')
    ax.set_xticks(X)
    ax.set_xticklabels(SCALES)
    ax.grid(True, color=GRID, linewidth=0.75, alpha=0.75)
    add_axis_arrows(ax)


def main():
    values = collect_values()
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2))
    for ax, panel in zip(axes.ravel(), PANELS):
        plot_panel(ax, panel, values[panel[0]])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.08, wspace=0.36, hspace=0.42)
    fig.suptitle('ViT-H+ Data Scaling: 1TB vs 5TB', fontsize=17, fontweight='bold', y=1.02)
    base = OUT / 'hplus_1tb_vs_5tb_task_family_image2_style'
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(base.with_suffix(f'.{ext}'), dpi=260, bbox_inches='tight')
    plt.close(fig)

    with base.with_suffix('.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task_metric', 'scale', 'score'])
        w.writeheader()
        for task_metric, by_scale in values.items():
            for scale in SCALES:
                w.writerow({'task_metric': task_metric, 'scale': scale, 'score': by_scale[scale]})
    print('wrote', base.with_suffix('.png'))


if __name__ == '__main__':
    main()
