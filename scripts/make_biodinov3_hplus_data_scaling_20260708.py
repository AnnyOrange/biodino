#!/usr/bin/env python3
"""H+ data-scaling plots from available downstream results.

BBBC013 is excluded. H+ 10TB is intentionally shown as N/A because the
current manifest only contains H+ 1TB and H+ 5TB runs.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.')
OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures_vertical_white')
BENCH = Path('outputs/00_reports/benchmark_results_summary.csv')
PER_BEST = Path('outputs/03_comparisons/2026-7-1-test-overall/per_dataset_best.csv')
MANIFEST = Path('outputs/03_comparisons/scaling_manifest_20260707/checkpoint_manifest.csv')

MODELS = {'1TB': 'bio_continue_rgb3_vith16plus', '5TB': '5tb_hplus_packwds_ep15_b1024'}
SCALES = ['1TB', '5TB', '10TB']
X = np.arange(len(SCALES))
COLORS = {'1TB': '#8EA9CE', '5TB': '#EF443B', '10TB': '#D8D8D8'}
TEXT = '#1E2522'
GRID = '#D2D2D2'


def fnum(x):
    try:
        if x == '' or x is None:
            return None
        return float(x)
    except Exception:
        return None


def best_by_dataset(rows, model_key, task, metric, datasets=None, higher=True):
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
    return best


def mean_or_none(vals):
    return sum(vals) / len(vals) if vals else None


def collect_values():
    bench = list(csv.DictReader(BENCH.open(newline='')))
    out = defaultdict(dict)

    for scale, model_key in MODELS.items():
        out['Classification BA'][scale] = mean_or_none(best_by_dataset(
            bench, model_key, 'classification', 'balanced_accuracy', higher=True).values())
        out['Regression R2'][scale] = mean_or_none(best_by_dataset(
            bench, model_key, 'regression', 'r2', datasets={'bbbc005'}, higher=True).values())
        out['Regression MAE'][scale] = mean_or_none(best_by_dataset(
            bench, model_key, 'regression', 'mae', datasets={'bbbc005'}, higher=False).values())
        out['Retrieval R@1'][scale] = mean_or_none(best_by_dataset(
            bench, model_key, 'retrieval', 'recall_at_1',
            datasets={'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'}, higher=True).values())
        out['Clustering NMI'][scale] = mean_or_none(best_by_dataset(
            bench, model_key, 'retrieval', 'nmi',
            datasets={'lc25000', 'nct-crc-he-1k', 'crc-val-he-7k'}, higher=True).values())

    # Use the common segmentation subset available for both H+ 1TB and H+ 5TB.
    per = list(csv.DictReader(PER_BEST.open(newline='')))
    seg_by_model = defaultdict(dict)
    for r in per:
        if r.get('category') != 'segmentation' or r.get('metric_key') != 'segmentation_mDice':
            continue
        if r.get('dataset') == 'multimodal_cellseg':
            continue
        for scale, model_key in MODELS.items():
            if r.get('model') == model_key:
                seg_by_model[scale][r['dataset']] = fnum(r['value'])
    common = set(seg_by_model['1TB']) & set(seg_by_model['5TB'])
    for scale in MODELS:
        out['Segmentation mDice'][scale] = mean_or_none([seg_by_model[scale][d] for d in common])

    # Make 10TB explicit as missing for H+.
    for metric in list(out):
        out[metric]['10TB'] = None
    return out


def hplus_scales_in_manifest():
    scales = set()
    if not MANIFEST.exists():
        return scales
    for r in csv.DictReader(MANIFEST.open(newline='')):
        if r.get('arch') == 'vit_huge2':
            scales.add(r.get('D_label'))
    return scales


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.family': 'DejaVu Sans', 'font.size': 10.5, 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
        'axes.edgecolor': TEXT, 'axes.grid': True, 'grid.color': GRID,
        'grid.alpha': 0.75, 'grid.linewidth': 0.8,
    })


def add_missing(ax, ymid):
    ax.scatter([2], [ymid], s=210, color=COLORS['10TB'], edgecolor='#666666', linewidth=0.8, zorder=3)
    ax.text(2, ymid, 'N/A', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#555555', zorder=4)


def plot_panel(values):
    specs = [
        ('Classification BA', 'Balanced accuracy', True, '{:.4f}'),
        ('Regression R2', 'R2', True, '{:.4f}'),
        ('Regression MAE', 'MAE', False, '{:.3f}'),
        ('Retrieval R@1', 'Recall@1', True, '{:.4f}'),
        ('Clustering NMI', 'NMI', True, '{:.4f}'),
        ('Segmentation mDice', 'mDice', True, '{:.4f}'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.5))
    for ax, (metric, ylabel, higher, fmt) in zip(axes.ravel(), specs):
        vals = [values[metric].get(s) for s in SCALES]
        present_x = [i for i, v in enumerate(vals) if v is not None]
        present_y = [v for v in vals if v is not None]
        ax.plot(present_x, present_y, color='#333333', linestyle=(0, (5, 5)), linewidth=1.0, zorder=1)
        ax.scatter(present_x, present_y, s=[230, 330][:len(present_x)],
                   color=[COLORS[SCALES[i]] for i in present_x], edgecolor='#333333', linewidth=0.8, zorder=2)
        lo, hi = min(present_y), max(present_y)
        pad = (hi - lo) * 0.45 if hi != lo else (abs(hi) * 0.015 + 0.01)
        y0, y1 = lo - pad, hi + pad
        if metric == 'Regression MAE':
            y0, y1 = max(0, y0), y1
        else:
            y0 = max(0, y0)
        ax.set_ylim(y0, y1)
        add_missing(ax, y0 + 0.5 * (y1 - y0))
        for i, v in zip(present_x, present_y):
            dy = 0.035 * (y1 - y0)
            ax.text(i, v + dy, fmt.format(v), ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if (v == max(present_y) if higher else v == min(present_y)) else 'normal')
        ax.set_title(metric, fontsize=12.5, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(X, SCALES)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', visible=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('H+ Data Scaling: 1TB vs 5TB vs 10TB', fontsize=17, fontweight='bold', y=1.02)
    fig.text(0.5, 0.01, 'H+ 10TB downstream run is not available in the current manifest/results.',
             ha='center', va='bottom', fontsize=10.5, color='#555555')
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'hplus_data_scaling_1tb_5tb_10tb_available.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def write_csv(values, available_scales):
    with (OUT / 'hplus_data_scaling_1tb_5tb_10tb_available.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'scale', 'score', 'available_hplus_run'])
        w.writeheader()
        for metric, by_scale in values.items():
            for scale in SCALES:
                w.writerow({
                    'metric': metric,
                    'scale': scale,
                    'score': '' if by_scale.get(scale) is None else by_scale.get(scale),
                    'available_hplus_run': int(scale in available_scales),
                })


def main():
    setup()
    values = collect_values()
    available = hplus_scales_in_manifest()
    plot_panel(values)
    write_csv(values, available)
    print('H+ scales in manifest:', sorted(available))
    print('wrote', OUT / 'hplus_data_scaling_1tb_5tb_10tb_available.png')


if __name__ == '__main__':
    main()
