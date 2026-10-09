#!/usr/bin/env python3
"""Status + score tables for the weight-space baseline (reads paired/*.json only)."""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import CLS_LARGE, CLS_SMALL, DET, RET_HPA, RET_WITHIN, ROOT, SEG_FIRST, SEG_REST  # noqa: E402

COLS = ['E', 'M', 'WA025', 'WA050', 'WA075', 'L', 'AVG3', 'E+L', 'M+L', 'PCA(E+L)->d']
HEAD = ['E', 'M (α=0)', 'α=.25', 'α=.5', 'α=.75', 'L (α=1)', 'AVG3', 'E+L', 'M+L', 'PCA(E+L)']
WEIGHT = ['WA025', 'WA050', 'WA075', 'AVG3']


def effective_status(d):
    """Relabel a baseline-reproduction mismatch as cross-host solver noise when the E/M/L gap is <= 3e-3."""
    status = d['status']
    if status == 'BASELINE_REPRODUCTION_MISMATCH' and d.get('baseline_fusion'):
        fus = json.loads(Path(d['baseline_fusion']['path']).read_text())['results']
        diff = max(abs(float(v) - float(fus[r][k])) for r in ('E', 'M', 'L') for k, v in d['results'][r].items()
                   if isinstance(v, float) and isinstance(fus[r].get(k), float))
        status = f'REPRO_DIFF_{diff:.1e}_CROSS_HOST' if diff <= 5e-3 else status  # octmnist 2e-3 (1-2 of 1000 test samples); bbbc048 M 3.2e-3 (rare-class balanced acc)
    return status


def fmt(v):
    return '—' if v is None else (f'{v:.4f}' if abs(v) < 10 else f'{v:.2f}')


def primary(task, r):
    if not isinstance(r, dict) or 'status' in r and not any(isinstance(v, float) for v in r.values()):
        return None, None
    if task == 'regression':
        return 'r2', r.get('r2')
    if r.get('macro_auc') is not None and r.get('balanced_accuracy') is None:
        return 'macro_auc', r.get('macro_auc')
    return 'balanced_accuracy', r.get('balanced_accuracy')


def table(title, metric_name, rows):
    out = [f'\n### {title} ({metric_name})\n', '| dataset | ' + ' | '.join(HEAD) + ' | status |', '|---|' + '---:|' * len(HEAD) + '---|']
    for name, vals, status in rows:
        out.append(f'| {name} | ' + ' | '.join(fmt(vals.get(c)) for c in COLS) + f' | {status} |')
    return out


def deltas(rows):
    """Mean (arm - L), (arm - M), (arm - max(E,M,L)) and win counts over complete rows."""
    lines = []
    for arm in WEIGHT + ['E+L', 'M+L']:
        dl, dm, dbest, wins, n = [], [], [], 0, 0
        for _, v, _ in rows:
            if v.get(arm) is None or any(v.get(r) is None for r in ('E', 'M', 'L')):
                continue
            n += 1
            best = max(v['E'], v['M'], v['L'])
            dl.append(v[arm] - v['L']); dm.append(v[arm] - v['M']); dbest.append(v[arm] - best)
            wins += v[arm] >= best
        if n:
            lines.append(f'| {arm} | {n} | {statistics.mean(dl):+.4f} | {statistics.mean(dm):+.4f} | '
                         f'{statistics.mean(dbest):+.4f} | {wins}/{n} |')
    return ['', '| arm | n | mean Δ vs L | mean Δ vs M | mean Δ vs best(E,M,L) | ≥ best(E,M,L) |', '|---|---:|---:|---:|---:|---:|'] + lines


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    md = [f'# Weight-space baseline vs E/M/L coexistence — status {stamp}\n',
          'α interpolates M→L: θ_α = (1−α)θ_M + αθ_L; α=0 is M and α=1 is L (same checkpoint SHA256, '
          'results reused). AVG3 = (θ_E+θ_M+θ_L)/3. E+L / M+L / PCA are the coexistence feature-fusion arms. '
          'Every cell is identity-matched (ordered samples + labels) and uses the pinned v4 evaluators.\n']
    inventory = {}
    # classification / regression
    for family, dsets in (('classification', [d for d in CLS_SMALL + CLS_LARGE]),):
        rows_c, rows_r = [], []
        for ds in dsets:
            p = ROOT / 'paired' / f'classification_{ds}.json'
            q = ROOT / 'paired' / f'regression_{ds}.json'
            path = p if p.exists() else q if q.exists() else None
            if path is None:
                inventory[f'{family}/{ds}'] = 'PENDING'
                continue
            d = json.loads(path.read_text())
            vals, metric = {}, None
            for c in COLS:
                metric_c, v = primary(d['task'], d['results'].get(c, {}))
                vals[c] = v
                metric = metric or metric_c
            status = effective_status(d)
            inventory[f'{d["task"]}/{ds}'] = status
            (rows_r if d['task'] == 'regression' else rows_c).append((f'{ds} [{metric}]', vals, status))
        md += table('Classification (incl. multilabel)', 'balanced accuracy; macro AUC for multilabel', rows_c) + deltas(rows_c)
        md += table('Regression', 'R²', rows_r) + deltas(rows_r)
    # retrieval / clustering
    rows_ret, rows_clu = [], []
    for ds in RET_WITHIN + RET_HPA:
        path = ROOT / 'paired' / f'retrieval_{ds}.json'
        if not path.exists():
            inventory[f'retrieval/{ds}'] = inventory[f'clustering/{ds}'] = 'PENDING'
            continue
        d = json.loads(path.read_text())
        inventory[f'retrieval/{ds}'] = inventory[f'clustering/{ds}'] = d['status']
        if ds == 'hpa-subcellular':
            get = lambda c, key, m: (d['results'].get(c, {}).get(key) or {}).get(m)
            rows_ret.append((f'{ds} [same-gene query→gallery mAP@10]', {c: get(c, 'retrieval:custom-v1-same-gene-query-gallery', 'map_at_10') for c in COLS}, d['status']))
            rows_ret.append((f'{ds} [same-gene R@1]', {c: get(c, 'retrieval:custom-v1-same-gene-query-gallery', 'recall_at_1') for c in COLS}, d['status']))
            rows_clu.append((f'{ds} [all-41 ARI]', {c: get(c, 'clustering:custom-v1-single-location-all41', 'ari') for c in COLS}, d['status']))
            rows_clu.append((f'{ds} [ge10-34 NMI]', {c: get(c, 'clustering:custom-v1-single-location-ge10-34', 'nmi') for c in COLS}, d['status']))
        else:
            rows_ret.append((f'{ds} [R@1]', {c: d['results'].get(c, {}).get('recall_at_1') for c in COLS}, d['status']))
            rows_ret.append((f'{ds} [mAP@10]', {c: d['results'].get(c, {}).get('map_at_10') for c in COLS}, d['status']))
            rows_clu.append((f'{ds} [ARI]', {c: d['results'].get(c, {}).get('ari') for c in COLS}, d['status']))
            rows_clu.append((f'{ds} [NMI]', {c: d['results'].get(c, {}).get('nmi') for c in COLS}, d['status']))
    md += table('Retrieval', 'cosine within-set leave-one-out; HPA locked query/gallery', rows_ret) + deltas(rows_ret)
    md += table('Clustering', 'KMeans seed 0', rows_clu) + deltas(rows_clu)
    # detection proxy
    rows_det = []
    for ds in DET:
        path = ROOT / 'paired' / f'detection_{ds}.json'
        if not path.exists():
            inventory[f'detection_proxy/{ds}'] = 'PENDING'
            continue
        d = json.loads(path.read_text())
        vals = {c: d['arms'].get(c, {}).get('test_patch_f1') for c in COLS}
        status = 'COMPLETE' if all(vals.get(c) is not None for c in ['E', 'M', 'L'] + WEIGHT) else 'PARTIAL'
        status += '' if d['protocol_consistent'] else ' PROTOCOL_MISMATCH'
        inventory[f'detection_proxy/{ds}'] = status
        rows_det.append((f'{ds} [test patch F1]', vals, status))
    md += table('Detection proxy (*not native detection*)', 'test patch F1, B8 224 stretch 5 ep', rows_det) + deltas(rows_det)
    # segmentation
    rows_seg = []
    for ds in SEG_FIRST + SEG_REST:
        path = ROOT / 'paired' / f'segmentation_{ds}.json'
        if not path.exists():
            inventory[f'segmentation/{ds}'] = 'PENDING'
            continue
        d = json.loads(path.read_text())
        by = {}
        for cell in d['cells']:
            key = (cell['tag'], cell['dataset_dir'], cell['budget'])
            for arm, v in cell['arms'].items():
                by.setdefault(key, {}).setdefault(arm, []).append(v['test_mIoU'])
        n_complete = 0
        for (tag, dsdir, budget), arms in sorted(by.items()):
            vals = {c: (statistics.mean(arms[c]) if c in arms and len(arms[c]) == 3 else None) for c in COLS}
            done = all(vals.get(c) is not None for c in ['E', 'M', 'L'] + WEIGHT)
            n_complete += done
            rot = dsdir if ds == 'pannuke' else ''
            rows_seg.append((f'{ds}{" " + rot if rot else ""} E{budget} [mean test mIoU, seeds ' + ','.join(str(len(arms.get(c, []))) for c in ['E', 'M', 'L'] + WEIGHT) + ']',
                             vals, 'COMPLETE' if done else 'PARTIAL'))
        inventory[f'segmentation/{ds}'] = f'{n_complete}/{d["expected_cells"] // 3} budget-cells complete'
    md += table('Segmentation (v3-only, formal splits)', 'test mIoU averaged over seeds 0/1/2 at best-val epoch', rows_seg) + deltas(rows_seg)
    # queue status
    states = {}
    for st in (ROOT / 'claims').glob('*/status.json'):
        try:
            s = json.loads(st.read_text()).get('state')
        except Exception:
            s = 'UNREADABLE'
        states[s] = states.get(s, 0) + 1
    n_tasks = json.loads((ROOT / 'tasks.json').read_text())['n_tasks']
    md += ['\n### Queue\n', f'tasks={n_tasks} claimed={sum(states.values())} ' + ' '.join(f'{k}={v}' for k, v in sorted(states.items())),
           '\nNot scheduled: segmentation/monuseg (BLOCKED_NOT_TESTED), retrieval rxrx1-cross/rxrx3-core, CTC, OOD (no E/M/L baseline). '
           'Missing cells are never counted as zero; no aggregate is a full-v4 claim.']
    out_md = ROOT / f'WEIGHTSPACE_STATUS_{stamp}.md'
    out_md.write_text('\n'.join(md) + '\n')
    (ROOT / 'progress_snapshots' / f'weightspace_inventory_{stamp}.json').write_text(
        json.dumps({'captured_utc': stamp, 'queue': states, 'cells': inventory}, indent=2) + '\n')
    print(out_md)


if __name__ == '__main__':
    main()
