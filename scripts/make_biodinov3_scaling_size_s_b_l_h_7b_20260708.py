from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('outputs/00_reports/20260708_taskwise_fm_figures_vertical_white')
OUT.mkdir(parents=True, exist_ok=True)
BG='#FFFFFF'; TEXT='#1E2522'; GRID='#E5E5E5'
COLORS = {'S+':'#EDB28E','B':'#D594C5','L':'#EF443B','H+':'#AEB6BC','7B':'#79CDBF'}
ORDER = ['S+','B','L','H+','7B']
DATA = {
    'classification_balanced_accuracy': {
        'title':'Scaling: Classification Balanced Accuracy', 'ylabel':'Balanced accuracy', 'ylim':(0.68,0.735), 'fmt':'{:.4f}',
        'values': {'S+':0.7000,'B':0.7044,'L':0.7162,'H+':0.7189,'7B':0.7222},
    },
    'retrieval_recall_at_1': {
        'title':'Scaling: Retrieval Recall@1', 'ylabel':'Recall@1', 'ylim':(0.970,0.990), 'fmt':'{:.4f}',
        'values': {'S+':0.9766,'B':0.9789,'L':0.9830,'H+':0.9836,'7B':0.9867},
    },
    'clustering_nmi': {
        'title':'Scaling: Clustering NMI', 'ylabel':'NMI', 'ylim':(0.810,0.885), 'fmt':'{:.4f}',
        'values': {'S+':0.8292,'B':0.8269,'L':0.8396,'H+':0.8700,'7B':0.8531},
    },
    'segmentation_mdice': {
        'title':'Scaling: Segmentation mDice', 'ylabel':'mDice', 'ylim':(0.650,0.735), 'fmt':'{:.4f}',
        'values': {'S+':0.6714,'B':0.6943,'L':0.6924,'H+':0.7121,'7B':0.7153},
    },
}
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'axes.edgecolor':TEXT,'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT,'text.color':TEXT,'font.family':'DejaVu Sans','font.size':10.5,'axes.titleweight':'bold','axes.grid':True,'grid.color':GRID,'grid.alpha':0.75,'grid.linewidth':0.8})

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight'); fig.savefig(OUT/f'{name}.svg',bbox_inches='tight'); plt.close(fig)

def plot_one(key, spec):
    labels = ORDER
    vals = [spec['values'][m] for m in labels]
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    x = np.arange(len(labels))
    bars=ax.bar(x, vals, color=[COLORS[m] for m in labels], edgecolor='#4A4A4A', linewidth=0.7)
    ax.plot(x, vals, color='#333333', linewidth=1.15, marker='o', markersize=3.5, alpha=0.75)
    ax.set_xticks(x, labels)
    ax.set_ylabel(spec['ylabel'])
    ax.set_title(spec['title'])
    ax.grid(axis='x', visible=False); ax.grid(axis='y', visible=True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylim(*spec['ylim'])
    dy=(spec['ylim'][1]-spec['ylim'][0])*0.012
    best=max(vals); best_idx=vals.index(best)
    for i,(bar,v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x()+bar.get_width()/2, v+dy, spec['fmt'].format(v), ha='center', va='bottom', fontsize=9, fontweight='bold' if i==best_idx else 'normal')
    for tick, lab in zip(ax.get_xticklabels(), labels):
        if lab == 'L':
            tick.set_color('#EF443B'); tick.set_fontweight('bold')
    name=f'scaling_size_s_b_l_h_7b_{key}'
    save(fig,name)
    with (OUT/f'{name}.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['model_size','score'])
        w.writeheader(); w.writerows([{'model_size':m,'score':spec['values'][m]} for m in labels])

def plot_panel():
    fig, axes = plt.subplots(2,2,figsize=(12.5,8.2))
    for ax,(key,spec) in zip(axes.ravel(), DATA.items()):
        labels=ORDER; vals=[spec['values'][m] for m in labels]; x=np.arange(len(labels))
        bars=ax.bar(x, vals, color=[COLORS[m] for m in labels], edgecolor='#4A4A4A', linewidth=0.65)
        ax.plot(x, vals, color='#333333', linewidth=1.0, marker='o', markersize=3, alpha=0.7)
        ax.set_xticks(x, labels); ax.set_ylabel(spec['ylabel']); ax.set_title(spec['title'].replace('Scaling: ',''), fontsize=12)
        ax.set_ylim(*spec['ylim']); ax.grid(axis='x', visible=False); ax.grid(axis='y', visible=True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        dy=(spec['ylim'][1]-spec['ylim'][0])*0.012
        best=max(vals); bi=vals.index(best)
        for i,(bar,v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x()+bar.get_width()/2, v+dy, spec['fmt'].format(v), ha='center', va='bottom', fontsize=8.5, fontweight='bold' if i==bi else 'normal')
        for tick, lab in zip(ax.get_xticklabels(), labels):
            if lab == 'L': tick.set_color('#EF443B'); tick.set_fontweight('bold')
    fig.suptitle('Model-Size Scaling: S+ / B / L / H+ / 7B', fontsize=17, fontweight='bold', y=1.01, color=TEXT)
    save(fig,'scaling_size_s_b_l_h_7b_panel')

for k,s in DATA.items(): plot_one(k,s)
plot_panel()
print('wrote scaling size figures to', OUT)
