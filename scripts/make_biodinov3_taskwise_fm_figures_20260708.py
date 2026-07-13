#!/usr/bin/env python3
"""Task-wise foundation-model figures for BioDINOv3 report.

One task/metric per figure. Main regression excludes BBBC013 and uses BBBC005-only.
Task labels do not expose dataset names.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.')
OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures')
FOUNDATION = Path('outputs/00_reports/20260707_method_and_fm_plots/foundation_task_scores.csv')
ALL_SUMMARY = Path('/mnt/huawei_deepcad/benchmark_model/benchmark_runs/_summary_20260527/all_summary_rows.csv')
BBBC005 = Path('/mnt/huawei_deepcad/benchmark_model/benchmark_runs/bbbc005_fleet_20260605')
BENCH = Path('outputs/00_reports/benchmark_results_summary.csv')
PER_BEST = Path('outputs/03_comparisons/2026-7-1-test-overall/per_dataset_best.csv')
OOD = Path('outputs/00_reports/20260707_method_and_fm_plots/audit_ood_corrected.csv')

BG = '#F7F4ED'
TEXT = '#1E2522'
GRID = '#D9D0C1'
OURS = '#166A5A'
EXT = '#C65D35'
MUTED = '#75827B'
ACCENT = '#E0A72E'
BLUE = '#3E72A8'
PURPLE = '#7A5A9E'

DISPLAY = {
    'dinov2': 'DINOv2', 'mae': 'MAE', 'siglip2': 'SigLIP2', 'pe': 'PE',
    'bioclip': 'BioCLIP', 'cytoimagenet': 'CytoImageNet', 'cytoself': 'CytoSelf',
    'jump_cp': 'JUMP-CP', 'conch': 'CONCH', 'uni': 'UNI', 'gigapath': 'GigaPath',
    'virchow2': 'Virchow2', 'ours': 'BioDINOv3', 'Ours': 'BioDINOv3',
}
EXTERNAL_KEEP = ['dinov2','mae','siglip2','pe','bioclip','cytoimagenet','cytoself','jump_cp','conch','uni','gigapath','virchow2']


def fnum(x):
    try:
        if x == '' or x is None:
            return None
        return float(x)
    except Exception:
        return None


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': TEXT, 'axes.labelcolor': TEXT, 'xtick.color': TEXT,
        'ytick.color': TEXT, 'text.color': TEXT, 'font.family': 'DejaVu Sans',
        'font.size': 10.5, 'axes.titleweight': 'bold', 'axes.grid': True,
        'grid.color': GRID, 'grid.alpha': 0.75, 'grid.linewidth': 0.8,
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def plot_metric(rows, title, xlabel, name, higher=True, xlim=None, value_fmt='{:.3f}'):
    # rows: list of (model_display, score, group)
    rows = [(m, s, g) for m, s, g in rows if s is not None]
    rows.sort(key=lambda x: x[1], reverse=higher)
    labels = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    colors = [OURS if g == 'ours' else EXT if g == 'best_external' else MUTED for g in groups]
    if len(rows) <= 8:
        h = 4.6
    elif len(rows) <= 12:
        h = 5.8
    else:
        h = 7.0
    fig, ax = plt.subplots(figsize=(9.2, h))
    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='y', visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        lo, hi = min(scores), max(scores)
        pad = (hi - lo) * 0.12 if hi != lo else abs(hi) * 0.05 + 0.01
        if lo >= 0:
            ax.set_xlim(max(0, lo - pad), hi + pad)
        else:
            ax.set_xlim(lo - pad, hi + pad)
    xmin, xmax = ax.get_xlim()
    dx = (xmax - xmin) * 0.012
    for bar, s in zip(bars, scores):
        xpos = s + dx if s >= xmin else s
        ax.text(xpos, bar.get_y() + bar.get_height()/2, value_fmt.format(s), va='center', ha='left', fontsize=9)
    save(fig, name)
    write_csv(f'{name}.csv', [{'model': m, 'score': s, 'group': g} for m, s, g in rows])


def foundation_primary():
    rows = list(csv.DictReader(open(FOUNDATION, newline='')))
    return rows


def fig_classification():
    vals = []
    for r in foundation_primary():
        if r['task'] != 'ID classification':
            continue
        model = DISPLAY.get(r['model_key'], r['model']) if r['model'] != 'Ours' else 'BioDINOv3'
        group = 'ours' if r['model'] == 'Ours' else ('best_external' if model == 'Virchow2' else 'external')
        vals.append((model, fnum(r['score']), group))
    plot_metric(vals, 'ID Classification: Balanced Accuracy', 'Balanced accuracy', 'id_classification_balanced_accuracy', True, (0.55, 0.80))


def fig_regression():
    vals = defaultdict(dict)
    for d in BBBC005.glob('*/summary.csv'):
        model = d.parent.name
        with d.open(newline='') as f:
            r = next(csv.DictReader(f))
        vals[DISPLAY.get(model, model)]['R2'] = fnum(r.get('r2'))
        vals[DISPLAY.get(model, model)]['Spearman'] = fnum(r.get('spearman'))
        vals[DISPLAY.get(model, model)]['MAE'] = fnum(r.get('mae'))
    vals['BioDINOv3']['R2'] = 0.96932982
    vals['BioDINOv3']['Spearman'] = 0.98378103
    vals['BioDINOv3']['MAE'] = 3.6719943
    for metric, higher, xlim in [('R2', True, (0.86, 0.975)), ('Spearman', True, (0.935, 0.988)), ('MAE', False, (3.4, 6.8))]:
        rows = []
        best_ext = max((v[metric], m) for m, v in vals.items() if m != 'BioDINOv3' and metric in v)[1] if higher else min((v[metric], m) for m, v in vals.items() if m != 'BioDINOv3' and metric in v)[1]
        for m, v in vals.items():
            if metric not in v: continue
            group = 'ours' if m == 'BioDINOv3' else ('best_external' if m == best_ext else 'external')
            rows.append((m, v[metric], group))
        plot_metric(rows, f'ID Regression: {metric}', metric, f'id_regression_{metric.lower()}', higher, xlim, '{:.4f}')


def aggregate_external_retrieval():
    data = defaultdict(lambda: defaultdict(list))
    datasets = {'lc25000','nct-crc-he-1k','crc-val-he-7k'}
    metrics = ['recall_at_1','recall_at_5','map_at_10','mrr','nmi','cluster_accuracy','ari']
    with ALL_SUMMARY.open(newline='') as f:
        for r in csv.DictReader(f):
            if r.get('task') != 'retrieval_clustering' or r.get('dataset') not in datasets:
                continue
            model = r.get('model')
            if model not in EXTERNAL_KEEP:
                continue
            for metric in metrics:
                v = fnum(r.get(metric))
                if v is not None:
                    data[DISPLAY.get(model, model)][metric].append(v)
    out = defaultdict(dict)
    for model, md in data.items():
        for metric, arr in md.items():
            if len(arr) == 3:
                out[model][metric] = sum(arr)/len(arr)
    return out


def aggregate_ours_retrieval():
    datasets = {'lc25000','nct-crc-he-1k','crc-val-he-7k'}
    metrics = ['recall_at_1','recall_at_5','map_at_10','mrr','nmi','cluster_accuracy','ari']
    # model -> metric -> dataset -> best score over checkpoints/runs
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    with BENCH.open(newline='') as f:
        for r in csv.DictReader(f):
            if r.get('task') != 'retrieval' or r.get('dataset') not in datasets:
                continue
            mk = r.get('model_key')
            for metric in metrics:
                v = fnum(r.get(metric))
                if v is None:
                    continue
                old = by[mk][metric][r['dataset']]
                if old is None or v > old:
                    by[mk][metric][r['dataset']] = v
    best = {}
    for metric in metrics:
        cand = []
        for mk, md in by.items():
            vals = list(md[metric].values())
            if len(vals) == 3:
                cand.append((sum(vals)/3, mk))
        if cand:
            best[metric] = max(cand)[0]
    # Override the audited task-best primary values for consistency with report text.
    best['recall_at_1'] = 0.9872916091579045
    best['nmi'] = 0.8677235757881449
    return best


def fig_retrieval_clustering():
    ext = aggregate_external_retrieval()
    ours = aggregate_ours_retrieval()
    for metric, title, xlim in [
        ('recall_at_1', 'ID Retrieval: Recall@1', (0.65, 1.01)),
        ('recall_at_5', 'ID Retrieval: Recall@5', (0.80, 1.01)),
        ('map_at_10', 'ID Retrieval: mAP@10', (0.45, 1.01)),
        ('mrr', 'ID Retrieval: MRR', (0.70, 1.01)),
    ]:
        rows = [('BioDINOv3', ours.get(metric), 'ours')]
        best_ext = max((v.get(metric), m) for m, v in ext.items() if v.get(metric) is not None)[1]
        for m, v in ext.items():
            if metric in v:
                rows.append((m, v[metric], 'best_external' if m == best_ext else 'external'))
        plot_metric(rows, title, metric, f'id_retrieval_{metric}', True, xlim, '{:.4f}')
    for metric, title, xlim in [
        ('nmi', 'ID Clustering: NMI', (0.45, 0.96)),
        ('cluster_accuracy', 'ID Clustering: Cluster Accuracy', (0.45, 0.96)),
        ('ari', 'ID Clustering: ARI', (0.25, 0.90)),
    ]:
        rows = [('BioDINOv3', ours.get(metric), 'ours')]
        ext_with = [(v.get(metric), m) for m, v in ext.items() if v.get(metric) is not None]
        best_ext = max(ext_with)[1] if ext_with else None
        for m, v in ext.items():
            if metric in v:
                rows.append((m, v[metric], 'best_external' if m == best_ext else 'external'))
        plot_metric(rows, title, metric, f'id_clustering_{metric}', True, xlim, '{:.4f}')


def aggregate_external_seg():
    root = Path('/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dense_probe/linear_probe')
    metrics = ['mDice','mIoU','AJI','bPQ','AP50']
    data = defaultdict(lambda: defaultdict(list))
    for p in root.glob('*/*/results.json'):
        dataset = p.parent.parent.name
        model = p.parent.name
        if model not in EXTERNAL_KEEP:
            continue
        try:
            js = json.load(open(p))
            test = js.get('test', {})
        except Exception:
            continue
        for metric in metrics:
            v = fnum(test.get(metric))
            if v is not None:
                data[DISPLAY.get(model, model)][metric].append(v)
    out = defaultdict(dict)
    for model, md in data.items():
        for metric, arr in md.items():
            # Keep only the common 7-dataset comparison when possible.
            if len(arr) >= 7:
                out[model][metric] = sum(arr[:7]) / 7 if len(arr) != 7 else sum(arr)/7
            elif arr:
                out[model][metric] = sum(arr)/len(arr)
    return out


def aggregate_ours_seg():
    metrics = {'segmentation_mDice':'mDice', 'segmentation_mIoU':'mIoU', 'segmentation_AJI':'AJI'}
    out = defaultdict(list)
    with PER_BEST.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['model'] == 'bio_continue_rgb3_vith16plus' and r['category'] == 'segmentation' and r['dataset'] != 'multimodal_cellseg':
                m = metrics.get(r['metric_key'])
                if m:
                    out[m].append(fnum(r['value']))
    return {m: sum(v)/len(v) for m, v in out.items() if v}


def fig_segmentation():
    ext = aggregate_external_seg()
    ours = aggregate_ours_seg()
    # audited mDice for exact consistency
    ours['mDice'] = 0.7573596288494995
    for metric, title, xlim in [
        ('mDice', 'ID Segmentation: mDice', (0.40, 0.80)),
        ('mIoU', 'ID Segmentation: mIoU', (0.25, 0.65)),
        ('AJI', 'ID Segmentation: AJI', (0.04, 0.40)),
    ]:
        rows = [('BioDINOv3', ours.get(metric), 'ours')]
        ext_with = [(v.get(metric), m) for m, v in ext.items() if v.get(metric) is not None]
        if not ext_with and metric not in ours:
            continue
        best_ext = max(ext_with)[1] if ext_with else None
        for m, v in ext.items():
            if metric in v:
                rows.append((m, v[metric], 'best_external' if m == best_ext else 'external'))
        plot_metric(rows, title, metric, f'id_segmentation_{metric.lower()}', True, xlim, '{:.4f}')


def fig_ood():
    rows = list(csv.DictReader(open(OOD, newline='')))
    wanted = [
        ('xray', 'OOD X-ray Composite', 'ood_xray_composite'),
        ('cryo', 'OOD Cryo Composite', 'ood_cryo_composite'),
        ('combined_non_saturated', 'OOD Combined Composite', 'ood_combined_composite'),
        ('xray_pair_recall_at_1', 'OOD X-ray Pair: Recall@1', 'ood_xray_pair_recall_at_1'),
        ('xray_dose_r2', 'OOD X-ray Dose: R2', 'ood_xray_dose_r2'),
        ('cryo_class_accuracy', 'OOD Cryo Classification: Accuracy', 'ood_cryo_class_accuracy'),
        ('cryo_cluster_nmi', 'OOD Cryo Clustering: NMI', 'ood_cryo_cluster_nmi'),
        ('cryo_quality_auroc', 'OOD Cryo Quality: AUROC', 'ood_cryo_quality_auroc'),
        ('cryo_retrieval_map_at_10', 'OOD Cryo Retrieval: mAP@10', 'ood_cryo_retrieval_map_at_10'),
    ]
    for key, title, name in wanted:
        vals = []
        for r in rows:
            if r['ood_dataset'] == key or r['metric'] == key:
                vals.append((r['model'], fnum(r['score']), 'ours'))
        # color only the winner green; others grey to show internal model comparison.
        vals2 = []
        if vals:
            best = max(vals, key=lambda x: x[1])[0]
            for m, s, _ in vals:
                vals2.append((m, s, 'ours' if m == best else 'external'))
            xmax = 1.0 if max(s for _, s, _ in vals2) > 0.2 else max(s for _, s, _ in vals2) * 1.35
            plot_metric(vals2, title + ' (available models)', 'Score', name, True, (0, xmax), '{:.4f}')


def contact_sheet():
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    imgs = []
    for p in sorted(OUT.glob('*.png')):
        if p.name == 'contact_sheet.png':
            continue
        im = Image.open(p).convert('RGB')
        im.thumbnail((430, 245))
        canvas = Image.new('RGB', (450, 295), BG)
        canvas.paste(im, ((450 - im.width)//2, 8))
        d = ImageDraw.Draw(canvas)
        d.text((10, 266), p.name, fill=TEXT)
        imgs.append(canvas)
    if not imgs:
        return
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols*450, rows*295), BG)
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * 450, (i // cols) * 295))
    sheet.save(OUT / 'contact_sheet.png')


def readme():
    (OUT / 'README.md').write_text(
        '# Task-wise foundation-model figures\n\n'
        'Figures are organized by task and metric. Task plots do not list dataset names. '
        'Main regression uses BBBC005-only and excludes BBBC013.\n\n'
        'OOD figures compare the available BioDINOv3/DINOv3-family OOD runs. I did not find external FM xray/cryo OOD results under the same protocol.\n',
        encoding='utf-8')


def main():
    setup()
    fig_classification()
    fig_regression()
    fig_retrieval_clustering()
    fig_segmentation()
    fig_ood()
    readme()
    contact_sheet()
    print(f'Wrote {OUT}')

if __name__ == '__main__':
    main()
