#!/usr/bin/env python3
"""Reproduce dataset-level diagnosis of the frozen Oct 8 comparison figure.

No training, new evaluation, source-data edits, or changes to the original plots.
Run with /home/lxy/miniconda3/envs/dinov3/bin/python.
"""
import csv
import hashlib
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'plot/fig2/route2_20tb_v4_20261008'
OUT = FIG / 'dataset_diagnosis_20261008'
FIVE = ROOT / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924'
COMPONENTS = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/components.csv'
CAMPAIGN = ROOT / 'outputs/02_eval_runs/20tb_route2_union_v4_online_20260928'
DATA = Path('/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923')
METRICS = {'classification': 'macro_f1', 'regression': 'spearman',
           'retrieval': 'map_at_5', 'clustering': 'nmi', 'segmentation': 'mDice'}
MATCHED = list(range(20495, 40016, 1952))
LATE = list(range(55631, 65392, 1952))
EARLY = [10735, 12687, 14639]


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = defaultdict(set)
    for r in read_csv(FIVE / 'selected_29_datasets.csv'):
        selection[r['family']].add(r['dataset'])
    five, sources = {}, []
    evidence = FIVE / 'v2_update_20261008/all_source_evidence.csv'
    for r in read_csv(evidence):
        # One source can expose several metrics. Never overwrite F1 with AUC or
        # balanced accuracy, Spearman with R2, or mAP@5 with recall@1.
        if r['model'] != 'noGRAM' or r['metric'] != METRICS.get(r['family']):
            continue
        k = int(r['checkpoint']), r['family'], r['dataset']
        assert k not in five, ('Duplicate metric', k)
        five[k] = float(r['value'])
        sources.append(dict(model='5tb', checkpoint=k[0], family=k[1], dataset=k[2],
                            metric=r['metric'], value=five[k], source=r['source'], protocol=r['protocol']))
    grouped = defaultdict(list)
    for r in read_csv(COMPONENTS):
        assert r['status'] == 'VALID_COMPLETE'
        assert r['metric'] == METRICS[r['family']]
        k = int(r['checkpoint']), r['family'], r['dataset']
        grouped[k].append(float(r['value']))
        sources.append(dict(model='20tb', checkpoint=k[0], family=k[1], dataset=k[2],
                            metric=r['metric'], value=float(r['value']), source=r['source'],
                            protocol='v4_shared_components'))
    for k, vs in grouped.items():
        assert len(vs) == (3 if k[2] == 'pannuke' else 1), k
    twenty = {k: st.mean(v) for k, v in grouped.items()}
    # Independent reconstruction must reproduce every existing figure point.
    checks = 0
    for r in read_csv(FIG / 'comparison_curve.csv'):
        if r['complete'] != 'True':
            continue
        values = five if r['model'] == '5tb_no_gram' else twenty
        step = int(r['checkpoint'])
        means = {f: st.mean(values[step, f, d] for d in ds) for f, ds in selection.items()}
        reconstructed = st.mean(means.values()) if r['family'] == 'overall' else means[r['family']]
        assert abs(reconstructed - float(r['value'])) < 1e-10, r
        checks += 1
    rows, pairs = [], []
    for f, d in sorted({(f, d) for _, f, d in twenty}):
        steps = [s for s in MATCHED if (s, f, d) in five]
        assert steps
        selected = d in selection[f]
        if selected:
            assert steps == MATCHED
        a = st.mean(five[s, f, d] for s in steps)
        b = st.mean(twenty[s, f, d] for s in steps)
        deltas = [100 * (twenty[s, f, d] - five[s, f, d]) for s in steps]
        # Late drift always uses the same 20TB reference window, including the
        # nine supplemental datasets whose 5TB history is shorter.
        reference = st.mean(twenty[s, f, d] for s in MATCHED)
        late = st.mean(twenty[s, f, d] for s in LATE)
        rows.append(dict(family=f, dataset=d, metric=METRICS[f], selected29=selected,
                         matched_n=len(steps), matched_first=steps[0], matched_last=steps[-1],
                         five_mean=a, twenty_mean=b, delta_x100=st.mean(deltas),
                         delta_median_x100=st.median(deltas), lower_points=sum(x < 0 for x in deltas),
                         delta_min_x100=min(deltas), delta_max_x100=max(deltas),
                         twenty_reference_mean=reference, twenty_late_mean=late,
                         twenty_late_drift_x100=100 * (late - reference),
                         five_endpoint=five[41479, f, d], twenty_endpoint=twenty[65391, f, d],
                         unmatched_endpoint_delta_x100=100 * (twenty[65391, f, d] - five[41479, f, d])))
        for s in steps:
            pairs.append(dict(checkpoint=s, family=f, dataset=d, metric=METRICS[f],
                              five=five[s, f, d], twenty=twenty[s, f, d],
                              delta_x100=100 * (twenty[s, f, d] - five[s, f, d])))
    rows.sort(key=lambda r: r['delta_x100'])
    save_csv('dataset_comparison.csv', rows)
    save_csv('matched_pairs.csv', pairs)
    save_csv('metric_source_evidence.csv', sources)
    families = []
    for f in METRICS:
        rr = [r for r in rows if r['family'] == f and r['selected29']]
        families.append(dict(family=f, datasets=len(rr), matched_n=len(MATCHED),
                             five_mean=st.mean(r['five_mean'] for r in rr),
                             twenty_mean=st.mean(r['twenty_mean'] for r in rr),
                             delta_x100=st.mean(r['delta_x100'] for r in rr),
                             late_drift_x100=st.mean(r['twenty_late_drift_x100'] for r in rr),
                             unmatched_endpoint_delta_x100=st.mean(r['unmatched_endpoint_delta_x100'] for r in rr)))
    save_csv('family_comparison.csv', families)
    boundary = load_boundary(selection)
    ablation = []
    for f, ds in selection.items():
        for d in sorted(ds):
            a = st.mean(twenty[s, f, d] for s in EARLY)
            b = st.mean(boundary[s, f, d] for s in EARLY)
            ablation.append(dict(family=f, dataset=d, matched_n=3, main_mean=a,
                                 boundary_mean=b, boundary_minus_main_x100=100 * (b - a)))
    save_csv('boundary_early_comparison.csv', ablation)
    composition = source_composition()
    audit_training_and_protocol(sources)
    make_plots(rows, five, twenty, boundary)
    metadata = dict(
        analysis='Exploratory cross-protocol diagnostic, not a causal data-scaling estimate or full-v4 score',
        baseline='5TB noGRAM', metric_policy=METRICS, matched_steps=MATCHED,
        late_steps=LATE, boundary_steps=EARLY, matched_checkpoints_are_not_independent_seeds=True,
        reconstructed_plot_points=checks, dataset_task_pairs=len(rows), selected29=29,
        matched_overall_delta_x100=st.mean(r['delta_x100'] for r in families),
        matched_mean_without_clustering_delta_x100=st.mean(r['delta_x100'] for r in families if r['family'] != 'clustering'),
        inputs={str(p): digest(p) for p in [evidence, COMPONENTS, FIG / 'comparison_curve.csv',
                FIVE / 'selected_29_datasets.csv', DATA / 'final_selection/selected_100tb_sources.parquet',
                DATA / 'phase_repartition_no_old5_20260923/phase5_sources.parquet']},
        source_composition=composition)
    (OUT / 'analysis_manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k not in ['inputs', 'source_composition']}, indent=2))


def load_boundary(selection):
    manifest = json.loads((CAMPAIGN / 'campaign_manifest.json').read_text())
    grouped, evidence = defaultdict(list), []
    for t in manifest['tasks']:
        if t['asset']['arm'] != '3090qiablation':
            continue
        s, f, d = int(t['asset']['checkpoint_id']), t['dataset']['task'], t['dataset']['dataset']
        audit = CAMPAIGN / '_state/done' / (t['key'] + '.json')
        assert json.loads(audit.read_text())['status'] == 'VALID_COMPLETE'
        cell = CAMPAIGN / 'cells' / t['key']
        if f == 'segmentation':
            paths = []
            for seed in range(3):
                found = list(cell.glob(f'results/**/budget20/seed{seed}/**/results.json'))
                assert len(found) == 1, (cell, seed)
                paths.extend(found)
            value = st.mean(json.loads(p.read_text())['test']['mDice'] for p in paths)
            source = '|'.join(map(str, paths))
        else:
            p = cell / 'component_result.json'
            obj = json.loads(p.read_text())
            if f == 'retrieval' and d in selection['clustering']:
                grouped[s, 'clustering', d].append(float(obj['nmi']))
                evidence.append(dict(checkpoint=s, family='clustering', dataset=d, value=obj['nmi'], source=str(p)))
            if 'rows' in obj:
                obj = next(r for r in obj['rows'] if r['task'] == 'retrieval' and r['aggregation'] == 'global')
            value, source = float(obj[METRICS[f]]), str(p)
        grouped[s, f, d].append(value)
        evidence.append(dict(checkpoint=s, family=f, dataset=d, value=value, source=source))
    save_csv('boundary_source_evidence.csv', evidence)
    for k, v in grouped.items():
        assert len(v) == (3 if k[2] == 'pannuke' else 1)
    return {k: st.mean(v) for k, v in grouped.items()}


def source_composition():
    cols = ['source_key', 'source_dataset', 'imaging_family', 'qualified_patches', 'qualified_bytes']
    df = pq.read_table(DATA / 'final_selection/selected_100tb_sources.parquet', columns=cols).to_pandas()
    keys = set(pq.read_table(DATA / 'phase_repartition_no_old5_20260923/phase5_sources.parquet').column(0).to_pylist())
    df['phase'] = df.source_key.isin(keys).map({True: 'boundary5', False: 'phase15_micro'})
    rows, summary = [], {}
    for phase, q in df.groupby('phase'):
        total = int(q.qualified_patches.sum())
        summary[phase] = dict(sources=len(q), patches=total, bytes=int(q.qualified_bytes.sum()),
                             top_one_percent_source_patch_fraction=float(q.qualified_patches.nlargest(max(1,len(q)//100)).sum()/total))
        for col in ['imaging_family', 'source_dataset']:
            g = q.groupby(col).agg(sources=('source_key', 'size'), patches=('qualified_patches', 'sum'), byte_count=('qualified_bytes', 'sum'))
            for name, r in g.iterrows():
                rows.append(dict(phase=phase, grouping=col, name=name, sources=int(r.sources),
                                 patches=int(r.patches), byte_count=int(r.byte_count),
                                 patch_percent=100 * float(r.patches) / total))
    save_csv('planned_source_composition.csv', rows)
    return summary


def audit_training_and_protocol(sources):
    runs = {
        '5tb': 'HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907',
        '20tb': 'HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924',
    }
    schedule = []
    for model, run in runs.items():
        path = ROOT / 'outputs/01_training_runs' / run / 'raw_loss_metrics.jsonl'
        found = None
        for line in path.open():
            r = json.loads(line)
            if r.get('optimizer_update') == 20495:
                found = r
                break
        assert found is not None
        schedule.append(dict(model=model, optimizer_update=20495,
                             **{k: found[k] for k in ['lr', 'wd', 'mom', 'effective_global_batch_size',
                                                     'local_batch_size', 'accum_steps']}, source=str(path)))
    save_csv('training_schedule_observation.csv', schedule)
    observations = defaultdict(dict)
    for r in sources:
        if r['checkpoint'] != 28303 or r['family'] not in ['classification', 'regression', 'retrieval']:
            continue
        observations[r['family'], r['dataset']][r['model']] = r['source']
    rows = []
    for (f, d), paths in sorted(observations.items()):
        if set(paths) != {'5tb', '20tb'}:
            continue
        a, b = [json.loads(Path(paths[m]).read_text())['_component_provenance'] for m in ['5tb','20tb']]
        fields = ['batch_size','autocast_dtype','n_last_blocks','use_avgpool','seed','numerical_environment']
        equal = {k: a.get(k) == b.get(k) for k in fields}
        for k in ['split_identity_sha256','dataset_inventory_sha256']:
            equal[k] = a.get('dataset',{}).get(k) == b.get('dataset',{}).get(k)
        rows.append(dict(checkpoint=28303, family=f, dataset=d, **equal,
                         five_source=paths['5tb'], twenty_source=paths['20tb']))
    assert len(rows) == 22
    save_csv('protocol_spotcheck_28303.csv', rows)


def make_plots(rows, five, twenty, boundary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 10.8), gridspec_kw={'width_ratios': [1.3, 1]})
    labels = {'classification': 'CLS', 'regression': 'REG', 'retrieval': 'RET', 'clustering': 'CLU', 'segmentation': 'SEG'}
    for ax, selected, title in [(axes[0], True, 'Fixed selected29: 11 matched checkpoints'),
                               (axes[1], False, '9 additional tasks: 6 matched checkpoints')]:
        rr = [r for r in rows if r['selected29'] == selected]
        vals = [r['delta_x100'] for r in rr]
        ax.barh(range(len(rr)), vals, color=['#c35b55' if v < 0 else '#238678' for v in vals])
        ax.set_yticks(range(len(rr)), [labels[r['family']] + ' | ' + r['dataset'] for r in rr], fontsize=9)
        ax.invert_yaxis()
        for i, r in enumerate(rr):
            v = r['delta_x100']
            ax.text(v + (.10 if v >= 0 else -.10), i, f'{v:+.2f}', ha='left' if v >= 0 else 'right', va='center', fontsize=9)
        ax.axvline(0, color='#444', lw=.7)
        ax.set_xlim(min(vals) - 1.2, max(vals) + 1.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('(20TB - 5TB) metric difference x 100')
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('20TB vs 5TB no-GRAM | dataset-level changes', fontsize=17)
    fig.text(.5, .02, 'Main window: ck20495-40015; supplemental baseline: ck30255-40015. Same metric per task.\nExploratory cross-protocol comparison; checkpoint repeats are not independent training seeds.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .96))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'dataset_matched_deltas.{ext}', dpi=180)
    plt.close(fig)
    panels = [('classification','chammi-allen-task2'), ('classification','pcam'), ('classification','breastmnist'),
              ('clustering','nct-crc-he-1k'), ('clustering','crc-val-he-7k'), ('retrieval','nct-crc-he-1k'),
              ('classification','nct-crc-he'), ('segmentation','pannuke'), ('segmentation','cellpose')]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for ax, (f, d) in zip(axes.flat, panels):
        for values, label, color in [(five,'5TB no-GRAM','#59616b'), (twenty,'20TB main','#16807a'),
                                      (boundary,'20TB boundary arm','#ce8a32')]:
            steps = sorted(s for s, ff, dd in values if (ff, dd) == (f, d))
            ax.plot(steps, [values[s,f,d] for s in steps], label=label, color=color, lw=1.2, marker='.', ms=2)
        ax.axvspan(MATCHED[0], MATCHED[-1], color='#16807a', alpha=.06)
        ax.axvspan(LATE[0], LATE[-1], color='#ce8a32', alpha=.07)
        ax.set_title(f'{d} | {METRICS[f]}', fontsize=11)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer checkpoint')
    handles, labels_ = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=3, bbox_to_anchor=(.5,.015))
    fig.suptitle('Persistent deficits, recovery, and late drift', fontsize=17)
    fig.text(.5,.066,'Raw curves; boundary arm ends at ck14639. Different step endpoints must not be read as matched comparisons.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.96))
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT / f'key_dataset_trajectories.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
