#!/usr/bin/env python3
"""Figures for the weight-space baseline: alpha sweep per task family + fusion comparison.

Reads only paired/*.json. Fig 1: mean delta vs L along alpha (M->L), with E and
AVG3 reference levels, one panel per family over that family's full paired
inventory. Fig 2: weight averaging vs feature fusion on the common-support cells.
Also writes the underlying numbers as CSV (table view / relief rule).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import ROOT  # noqa: E402
from write_hs6_l5_weightspace_final_report_20260922 import cells  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, SURFACE, GRID = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb', '#e6e5e1'
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_ARMS = ['M', 'WA025', 'WA050', 'WA075', 'L']
FAMILIES = [('classification', 'Classification', 'balanced accuracy'),
            ('regression', 'Regression', 'R²'),
            ('retrieval', 'Retrieval', 'mAP@10'),
            ('clustering', 'Clustering', 'ARI'),
            ('detection_proxy', 'Detection proxy*', 'patch F1 (points)'),
            ('segmentation', 'Segmentation', 'test mIoU')]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def mean_delta(rows, arm, ref='L'):
    vals = [v[arm] - v[ref] for _, _, v, _ in rows if v.get(arm) is not None and v.get(ref) is not None]
    return statistics.mean(vals) if vals else None, len(vals)


def figure1(by_family, out_stem, table_rows):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), facecolor=SURFACE)
    for ax, (key, title, unit) in zip(axes.ravel(), FAMILIES):
        style(ax)
        rows = by_family.get(key, [])
        support = [r for r in rows if all(r[2].get(a) is not None for a in ALPHA_ARMS + ['E', 'AVG3'])]
        ys, n = [], 0
        for arm in ALPHA_ARMS:
            m, n = mean_delta(support, arm)
            ys.append(m)
        e_val, _ = mean_delta(support, 'E')
        avg_val, _ = mean_delta(support, 'AVG3')
        ax.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate('L (α=1)', xy=(0.0, 0), xytext=(11, 3), textcoords='offset points',
                    color=MUTED, fontsize=7.5, ha='left', va='bottom')
        if e_val is not None:
            ax.axhline(e_val, color=ORANGE, linewidth=2.0, zorder=2)
            ax.annotate('E', xy=(0.0, e_val), xytext=(11, -3), textcoords='offset points',
                        color=ORANGE, fontsize=8, fontweight='bold', ha='left', va='top')
        if avg_val is not None:
            ax.axhline(avg_val, color=AQUA, linewidth=2.0, linestyle=(0, (1, 2)), zorder=2)
            ax.annotate('AVG3', xy=(1.0, avg_val), xytext=(-2, -3), textcoords='offset points',
                        color=AQUA, fontsize=8, fontweight='bold', ha='right', va='top')
        ax.plot(ALPHAS, ys, color=BLUE, linewidth=2.0, marker='o', markersize=7,
                markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.set_title(f'{title}   ({n} cells)', color=INK, fontsize=10.5, loc='left', pad=8)
        ax.set_ylabel(f'mean Δ vs L  [{unit}]', color=INK2, fontsize=8.5)
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels(['0\n(M)', '.25', '.5', '.75', '1\n(L)'])
        for arm, a, y in zip(ALPHA_ARMS, ALPHAS, ys):
            table_rows.append({'figure': 'fig1', 'family': key, 'arm': arm, 'alpha': a,
                               'mean_delta_vs_L': y, 'n_cells': n})
        for arm, y in (('E', e_val), ('AVG3', avg_val)):
            table_rows.append({'figure': 'fig1', 'family': key, 'arm': arm, 'alpha': '',
                               'mean_delta_vs_L': y, 'n_cells': n})
    handles = [Line2D([], [], color=BLUE, lw=2, marker='o', markersize=7, markeredgecolor=SURFACE,
                      markeredgewidth=2, label='θ = (1−α)·θ_M + α·θ_L'),
               Line2D([], [], color=ORANGE, lw=2, label='E (early checkpoint)'),
               Line2D([], [], color=AQUA, lw=2, linestyle=(0, (1, 2)), label='AVG3 = (E+M+L)/3'),
               Line2D([], [], color=MUTED, lw=1.2, linestyle=(0, (4, 3)), label='L (late endpoint) = 0')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('Weight-space interpolation of one 5TB trajectory: does the merged model keep each capability?',
                 color=INK, fontsize=12.5, x=0.012, ha='left', y=0.985)
    fig.text(0.012, 0.935, 'Mean change vs the late endpoint L, averaged over the cells where every arm exists. '
             'Above 0 beats L.  *frozen center-patch proxy, not native detection.',
             color=INK2, fontsize=8.8, ha='left')
    fig.tight_layout(rect=(0, 0.045, 1, 0.915))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def figure2(by_family, out_stem, table_rows):
    """Common-support comparison: weight arms vs feature fusion, delta vs best(E,M,L)."""
    groups, labels = [], []
    for key, title, _ in FAMILIES:
        rows = [r for r in by_family.get(key, [])
                if all(r[2].get(a) is not None for a in ['E', 'M', 'L', 'WA075', 'AVG3', 'E+L', 'M+L'])]
        if not rows:
            continue
        best = lambda v: max(v['E'], v['M'], v['L'])
        scale = 100.0 if key == 'detection_proxy' else 1.0
        vals = {arm: statistics.mean((v[arm] - best(v)) / scale for _, _, v, _ in rows)
                for arm in ('WA075', 'AVG3', 'E+L', 'M+L')}
        groups.append(vals)
        labels.append(f'{title}\n({len(rows)} cell{"s" if len(rows) != 1 else ""})')
        for arm, v in vals.items():
            table_rows.append({'figure': 'fig2', 'family': key, 'arm': arm, 'alpha': '',
                               'mean_delta_vs_best_EML': v, 'n_cells': len(rows)})
    fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor=SURFACE)
    style(ax)
    series = [('α=.75  (weight space)', 'WA075', BLUE), ('AVG3  (weight space)', 'AVG3', AQUA),
              ('E+L  (feature fusion)', 'E+L', ORANGE), ('M+L  (feature fusion)', 'M+L', YELLOW)]
    width, gap = 0.19, 0.02
    worst = min(min(g.values()) for g in groups)
    for i, (label, arm, color) in enumerate(series):
        xs = [j + (i - 1.5) * (width + gap) for j in range(len(groups))]
        ax.bar(xs, [g[arm] for g in groups], width=width, color=color, label=label, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    # Selective labels only: the two cells that carry the argument.
    for j, lab in enumerate(labels):
        if lab.startswith('Segmentation'):
            x = j + (2 - 1.5) * (width + gap)
            ax.annotate('E+L recovers the\nearly-checkpoint level', xy=(x, groups[j]['E+L']),
                        xytext=(x - 0.15, worst * 0.42), textcoords='data', fontsize=8.5, color=INK2,
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.9, shrinkA=2, shrinkB=3))
        if lab.startswith('Detection'):
            x = j + (2 - 1.5) * (width + gap)
            ax.annotate('feature fusion collapses\non the detection proxy', xy=(x - 0.09, groups[j]['E+L'] * 0.72),
                        xytext=(x - 0.72, worst * 0.72), textcoords='data', fontsize=8.5, color=INK2,
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.9, shrinkA=2, shrinkB=3))
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=9.5, color=INK2)
    ax.set_ylabel('mean Δ vs best single checkpoint', color=INK2, fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.12))
    fig.suptitle('Weight averaging vs feature-space fusion, on the cells where both exist',
                 color=INK, fontsize=12.5, x=0.012, ha='left', y=0.985)
    fig.text(0.012, 0.925, '0 = the best of E, M and L on that cell. Detection proxy is rescaled from F1 points '
             'to fractions so every family shares one axis.', color=INK2, fontsize=8.8, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    rows = cells()
    by_family = defaultdict(list)
    for r in rows:
        by_family[r[0]].append(r)
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    table: list[dict] = []
    figure1(by_family, str(out / 'fig1_alpha_sweep_by_family'), table)
    figure2(by_family, str(out / 'fig2_weight_vs_feature_fusion'), table)
    keys = ['figure', 'family', 'arm', 'alpha', 'mean_delta_vs_L', 'mean_delta_vs_best_EML', 'n_cells']
    with (out / 'figure_data.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'wrote {out}/fig1_alpha_sweep_by_family.(png|pdf), fig2_weight_vs_feature_fusion.(png|pdf), figure_data.csv')


if __name__ == '__main__':
    main()
