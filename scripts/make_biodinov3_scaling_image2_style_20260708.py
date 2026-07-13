#!/usr/bin/env python3
"""Image2-style bubble scaling plot for BioDINOv3 model-size comparison."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures_vertical_white')
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ['S+', 'B', 'L', 'H+', '7B']
# Approximate parameter counts; used only for a monotonic log-FLOPs-style x-axis.
PARAM_M = {'S+': 22, 'B': 86, 'L': 300, 'H+': 840, '7B': 7000}
X = np.array([math.log10(PARAM_M[m]) for m in MODELS])
BUBBLE = np.array([90, 150, 230, 330, 470])

DATA = [
    ('Classification', 'Balanced accuracy', [0.7000, 0.7044, 0.7162, 0.7189, 0.7222], (0.690, 0.730), '(a) Classification'),
    ('Retrieval', 'Recall@1', [0.9766, 0.9789, 0.9830, 0.9836, 0.9867], (0.974, 0.989), '(b) Retrieval'),
    ('Clustering', 'NMI', [0.8292, 0.8269, 0.8396, 0.8700, 0.8531], (0.815, 0.878), '(c) Clustering'),
    ('Segmentation', 'mDice', [0.6714, 0.6943, 0.6924, 0.7121, 0.7153], (0.660, 0.724), '(d) Segmentation'),
]

BLUE = '#4B87B9'
TAG_BLUE = '#4B87B9'
GRID = '#BEBEBE'
TEXT = '#111111'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'DejaVu Serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
})


def add_axis_arrows(ax):
    """Mimic the arrowed axes in the provided reference image."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.annotate('', xy=(xmax, ymin), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)


def plot_panel(ax, title, ylabel, values, ylim, caption):
    y = np.array(values)
    ax.plot(X, y, linestyle=(0, (5, 5)), color='black', linewidth=0.8, zorder=1)
    ax.scatter(X, y, s=BUBBLE, color=BLUE, edgecolor='black', linewidth=0.7, alpha=0.96, zorder=2)

    # Put compact model-size labels near points so the talk audience can read the scaling order.
    offsets = [(0.00, -0.0028), (0.00, 0.0026), (0.00, -0.0034), (0.00, 0.0028), (0.00, -0.0030)]
    scale = ylim[1] - ylim[0]
    for m, xx, yy, (dx, dy_frac) in zip(MODELS, X, y, offsets):
        ax.text(xx + dx, yy + dy_frac * scale, m, ha='center', va='center', fontsize=8.2,
                color='white', fontweight='bold', zorder=3)

    ax.text(X[-1] - 0.02, ylim[1] - 0.10 * (ylim[1] - ylim[0]), 'BioDINOv3',
            ha='right', va='center', fontsize=10.5, color='white',
            bbox=dict(boxstyle='square,pad=0.25', facecolor=TAG_BLUE, edgecolor='none', alpha=0.96))

    ax.set_xlim(X[0] - 0.18, X[-1] + 0.22)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('log-FLOPs')
    ax.set_xticks(X)
    ax.set_xticklabels([''] * len(MODELS))
    ax.grid(True, color=GRID, linewidth=0.75, alpha=0.75)
    add_axis_arrows(ax)
    ax.set_title(title, fontsize=11, pad=6, fontweight='normal')
    ax.text(0.5, -0.27, caption, transform=ax.transAxes, ha='center', va='top', fontsize=12)

    for xx, yy, val in zip(X, y, values):
        ax.text(xx, yy + 0.035 * (ylim[1] - ylim[0]), f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.2))
    for ax, item in zip(axes, DATA):
        plot_panel(ax, *item)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.28, wspace=0.42)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'scaling_size_s_b_l_h_7b_image2_style.{ext}', dpi=260, bbox_inches='tight')
    plt.close(fig)

    with (OUT / 'scaling_size_s_b_l_h_7b_image2_style.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'metric', 'model_size', 'log_params_proxy', 'score'])
        w.writeheader()
        for task, metric, values, _, _ in DATA:
            for model, xx, score in zip(MODELS, X, values):
                w.writerow({'task': task, 'metric': metric, 'model_size': model, 'log_params_proxy': xx, 'score': score})
    print('wrote', OUT / 'scaling_size_s_b_l_h_7b_image2_style.png')


if __name__ == '__main__':
    main()
