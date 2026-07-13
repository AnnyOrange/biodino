#!/usr/bin/env python3
"""Image2-style data scaling comparison: ViT-L 1/5/10TB vs ViT-H+ 1/5TB.

BBBC013 is excluded. Metrics are pulled from benchmark_results_summary.csv.
Segmentation is not included because current comparable L 5TB/10TB segmentation
rows are not present in the unified benchmark summary.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures_vertical_white')
BENCH = Path('outputs/00_reports/benchmark_results_summary.csv')
OUT.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = {
    'ViT-L': {
        '1TB': 'bio_continue_vitL16_OEP1025_ep15_b1024_1025',
        '5TB': 'bio_5tb_mixed_ori_slfm_rgb1024_vitl16_ep30_clean_trainstats',
        '10TB': '10tb_lossless_uint16_vitl16_b1024_ep30_20260610_074015',
    },
    'ViT-H+': {
        '1TB': 'bio_continue_rgb3_vith16plus',
        '5TB': '5tb_hplus_packwds_ep15_b1024',
    },
}
SCALES = ['1TB', '5TB', '10TB']
X_BY_SCALE = {'1TB': math.log10(1), '5TB': math.log10(5), '10TB': math.log10(10)}

PANEL_SPECS = [
    ('Classification', 'Balanced accuracy', 'classification', 'balanced_accuracy', None, True, '(a) Classification'),
    ('Regression', 'R2', 'regression', 'r2', {'bbbc005'}, True, '(b) Regression R2'),
    ('Regression', 'MAE', 'regression', 'mae', {'bbbc005'}, False, '(c) Regression MAE'),
    ('Retrieval', 'Recall@1', 'retrieval', 'recall_at_1', {'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'}, True, '(d) Retrieval'),
    ('Clustering', 'NMI', 'retrieval', 'nmi', {'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'}, True, '(e) Clustering'),
]

STYLE = {
    'ViT-L': {'color': '#4B87B9', 'tag': '#4B87B9', 'sizes': {'1TB': 110, '5TB': 230, '10TB': 360}},
    'ViT-H+': {'color': '#CFA7D7', 'tag': '#CFA7D7', 'sizes': {'1TB': 170, '5TB': 320}},
}
GRID = '#BEBEBE'
TEXT = '#111111'

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


def best_mean(rows, model_key, task, metric, datasets=None, higher=True):
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


def collect():
    rows = list(csv.DictReader(BENCH.open(newline='')))
    values = {}
    for title, ylabel, task, metric, datasets, higher, caption in PANEL_SPECS:
        key = f'{title} {ylabel}'
        values[key] = {}
        for arch, by_scale in MODEL_KEYS.items():
            values[key][arch] = {}
            for scale, model_key in by_scale.items():
                values[key][arch][scale] = best_mean(rows, model_key, task, metric, datasets, higher)
    return values


def add_axis_arrows(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    ax.annotate('', xy=(xmax, ymin), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)


def plot_one(ax, spec, vals):
    title, ylabel, _task, _metric, _datasets, higher, caption = spec
    all_y = []
    for arch in ['ViT-L', 'ViT-H+']:
        xs, ys, labels, sizes = [], [], [], []
        for scale in SCALES:
            v = vals.get(arch, {}).get(scale)
            if v is None:
                continue
            xs.append(X_BY_SCALE[scale]); ys.append(v); labels.append(scale); sizes.append(STYLE[arch]['sizes'][scale])
        all_y.extend(ys)
        ax.plot(xs, ys, linestyle=(0, (5, 5)), color='black', linewidth=0.75, zorder=1)
        ax.scatter(xs, ys, s=sizes, color=STYLE[arch]['color'], edgecolor='black', linewidth=0.7, alpha=0.96, zorder=2)
        for xx, yy, lab in zip(xs, ys, labels):
            ax.text(xx, yy, lab.replace('TB', ''), ha='center', va='center', color='white', fontsize=7.8,
                    fontweight='bold', zorder=3)
    lo, hi = min(all_y), max(all_y)
    pad = (hi - lo) * 0.35 if hi != lo else abs(hi) * 0.015 + 0.01
    y0 = max(0, lo - pad) if ylabel != 'MAE' else max(0, lo - pad)
    y1 = hi + pad
    ax.set_ylim(y0, y1)
    ax.set_xlim(-0.08, 1.08)

    # value annotations after y-limits are fixed
    yr = y1 - y0
    for arch in ['ViT-L', 'ViT-H+']:
        for scale in SCALES:
            v = vals.get(arch, {}).get(scale)
            if v is None:
                continue
            ax.text(X_BY_SCALE[scale], v + 0.035 * yr, f'{v:.4f}' if ylabel != 'MAE' else f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7.2)

    # model tags near the rightmost available point.
    l_y = vals['ViT-L'].get('10TB')
    h_y = vals['ViT-H+'].get('5TB')
    if l_y is not None:
        ax.text(X_BY_SCALE['10TB'] - 0.02, l_y + 0.12 * yr, 'ViT-L', ha='right', va='center',
                fontsize=9.5, color='white', bbox=dict(boxstyle='square,pad=0.22', facecolor=STYLE['ViT-L']['tag'], edgecolor='none', alpha=0.96))
    if h_y is not None:
        ax.text(X_BY_SCALE['5TB'] + 0.02, h_y - 0.12 * yr, 'ViT-H+', ha='left', va='center',
                fontsize=9.5, color='white', bbox=dict(boxstyle='square,pad=0.22', facecolor=STYLE['ViT-H+']['tag'], edgecolor='none', alpha=0.96))

    ax.set_title(title, fontsize=11, pad=6, fontweight='normal')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('log-Data')
    ax.set_xticks([X_BY_SCALE[s] for s in SCALES])
    ax.set_xticklabels(['1TB', '5TB', '10TB'])
    ax.grid(True, color=GRID, linewidth=0.75, alpha=0.75)
    add_axis_arrows(ax)
    ax.text(0.5, -0.32, caption, transform=ax.transAxes, ha='center', va='top', fontsize=11)


def main():
    values = collect()
    fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.25))
    for ax, spec in zip(axes, PANEL_SPECS):
        key = f'{spec[0]} {spec[1]}'
        plot_one(ax, spec, values[key])
    fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.30, wspace=0.44)
    base = OUT / 'data_scaling_vitl_1_5_10_vs_vithplus_1_5_image2_style'
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(base.with_suffix(f'.{ext}'), dpi=260, bbox_inches='tight')
    plt.close(fig)

    with base.with_suffix('.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'arch', 'scale', 'score'])
        w.writeheader()
        for key, by_arch in values.items():
            for arch, by_scale in by_arch.items():
                for scale in SCALES:
                    v = by_scale.get(scale)
                    w.writerow({'metric': key, 'arch': arch, 'scale': scale, 'score': '' if v is None else v})
    print('wrote', base.with_suffix('.png'))


if __name__ == '__main__':
    main()
