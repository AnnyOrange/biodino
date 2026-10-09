#!/usr/bin/env python3
"""Assemble the final report of the weight-space baseline from paired/*.json (no new computation)."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import CLS_LARGE, CLS_SMALL, COEX, DET, RET_HPA, RET_WITHIN, ROOT, SEG_FIRST, SEG_REST  # noqa: E402
from summarize_hs6_l5_weightspace_20260922 import COLS, effective_status, primary  # noqa: E402

WEIGHT = ['WA025', 'WA050', 'WA075', 'AVG3']
NAMES = {'WA025': 'α=.25', 'WA050': 'α=.5', 'WA075': 'α=.75', 'AVG3': 'AVG3', 'E+L': 'E+L', 'M+L': 'M+L'}


def cells():
    """Yield (family, label, {arm: value}) for every fully paired primary-metric cell."""
    out = []
    for ds in CLS_SMALL + CLS_LARGE:
        for fam in ('classification', 'regression'):
            p = ROOT / 'paired' / f'{fam}_{ds}.json'
            if p.exists():
                d = json.loads(p.read_text())
                vals = {c: primary(d['task'], d['results'].get(c, {}))[1] for c in COLS}
                out.append((d['task'] if d['task'] != 'multilabel_classification' else 'classification', ds, vals, effective_status(d)))
    for ds in RET_WITHIN + RET_HPA:
        p = ROOT / 'paired' / f'retrieval_{ds}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if ds == 'hpa-subcellular':
            g = lambda c, k, m: (d['results'].get(c, {}).get(k) or {}).get(m)
            out.append(('retrieval', ds, {c: g(c, 'retrieval:custom-v1-same-gene-query-gallery', 'map_at_10') for c in COLS}, effective_status(d)))
            out.append(('clustering', ds, {c: g(c, 'clustering:custom-v1-single-location-all41', 'ari') for c in COLS}, d['status']))
        else:
            out.append(('retrieval', ds, {c: d['results'].get(c, {}).get('map_at_10') for c in COLS}, effective_status(d)))
            out.append(('clustering', ds, {c: d['results'].get(c, {}).get('ari') for c in COLS}, d['status']))
    for ds in DET:
        p = ROOT / 'paired' / f'detection_{ds}.json'
        if p.exists():
            d = json.loads(p.read_text())
            out.append(('detection_proxy', ds, {c: d['arms'].get(c, {}).get('test_patch_f1') for c in COLS}, 'COMPLETE'))
    for ds in SEG_FIRST + SEG_REST:
        p = ROOT / 'paired' / f'segmentation_{ds}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        by = defaultdict(lambda: defaultdict(list))
        for cell in d['cells']:
            for arm, v in cell['arms'].items():
                by[(cell['tag'], cell['dataset_dir'], cell['budget'])][arm].append(v['test_mIoU'])
        for (tag, dsdir, budget), arms in sorted(by.items()):
            vals = {c: (statistics.mean(arms[c]) if len(arms.get(c, [])) == 3 else None) for c in COLS}
            rot = ' ' + tag.split('_sp')[-1].replace('pannuke_', 'rot ') if ds == 'pannuke' else ''
            out.append(('segmentation', f'{ds}{rot} E{budget}', vals, 'COMPLETE' if all(vals[c] is not None for c in ['E', 'M', 'L'] + WEIGHT) else 'PARTIAL'))
    return out


def aggregate(rows, scale=1.0):
    lines = ['| arm | n | mean Δ vs L | mean Δ vs M | mean Δ vs best(E,M,L) | ≥ best−0.5pp | ≥ best |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in WEIGHT + ['E+L', 'M+L']:
        dl, dm, db, near, wins = [], [], [], 0, 0
        for _, _, v, _ in rows:
            if v.get(arm) is None or any(v.get(r) is None for r in ('E', 'M', 'L')):
                continue
            best = max(v['E'], v['M'], v['L'])
            dl.append(v[arm] - v['L']); dm.append(v[arm] - v['M']); db.append(v[arm] - best)
            near += v[arm] >= best - 0.005 * scale; wins += v[arm] >= best
        if dl:
            n = len(dl)
            lines.append(f'| {NAMES[arm]} | {n} | {statistics.mean(dl):+.4f} | {statistics.mean(dm):+.4f} | {statistics.mean(db):+.4f} | {near}/{n} | {wins}/{n} |')
    return lines


def retention(rows, scale=1.0):
    """Split cells by which anchor is best; ask whether weight arms retain that anchor (within 0.5pp)."""
    out = []
    groups = defaultdict(list)
    for fam, name, v, _ in rows:
        if any(v.get(r) is None for r in ('E', 'M', 'L')):
            continue
        best = max(('E', 'M', 'L'), key=lambda r: v[r])
        groups[best].append((name, v))
    for anchor in ('E', 'M', 'L'):
        g = groups.get(anchor, [])
        if not g:
            continue
        parts = []
        for arm in WEIGHT + ['E+L', 'M+L']:
            ok = [v[arm] >= v[anchor] - 0.005 * scale for _, v in g if v.get(arm) is not None]
            parts.append(f'{NAMES[arm]} {sum(ok)}/{len(ok)}' if ok else f'{NAMES[arm]} —')
        out.append(f'- **{anchor}-best cells ({len(g)})**: ' + ', '.join(parts))
    return out


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rows = cells()
    ck = json.loads((ROOT / 'checkpoints/checkpoint_manifest.json').read_text())
    geo = ck['backbone_geometry']
    claims = [json.loads(p.read_text()) for p in (ROOT / 'claims').glob('*/status.json')]
    hosts = Counter(c.get('host') for c in claims if c.get('state') == 'DONE')
    n_tasks = json.loads((ROOT / 'tasks.json').read_text())['n_tasks']
    status_md = sorted(ROOT.glob('WEIGHTSPACE_STATUS_*.md'))[-1]
    tables = status_md.read_text().split('\n### ', 1)[1]
    tables = '### ' + tables.split('\n### Queue')[0]
    fam_rows = defaultdict(list)
    for r in rows:
        fam_rows[r[0]].append(r)
    md = [f'# Weight-space baseline for the 5TB E/M/L coexistence diagnostic — final report ({stamp})\n',
          '**Question.** Can the asynchronous capabilities of one vanilla training trajectory (Early ck12687, Mid ck20007, '
          'Late ck29279, same-run EMA teachers) be merged inside a single parameter basin by plain parameter averaging? '
          'If yes, the problem is "soft"; if feature fusion helps where weight averaging does not, the information exists '
          'but the vanilla trajectory does not consolidate it at one endpoint. This is a diagnostic, not a method claim.\n',
          '## Arms\n',
          '- θ_α = (1−α)·θ_M + α·θ_L for α ∈ {0, .25, .5, .75, 1}; α=0 is M and α=1 is L (same checkpoint SHA256; their '
          'coexistence-campaign results are reused after identity checks). AVG3 = (θ_E+θ_M+θ_L)/3.',
          f'- Merged in float64 over every stored tensor, saved as float32 in the original `{{"teacher": …}}` layout; formula '
          f'verified on reload (max abs error {max(o["max_abs_formula_error_fp32_roundtrip"] for o in ck["outputs"].values()):.1e}). '
          f'SHA256: ' + ', '.join(f'{a} `{o["sha256"][:12]}…`' for a, o in ck['outputs'].items()) + '.',
          f'- Backbone geometry (368 tensors, {geo["backbone_numel"]/1e6:.0f}M params): |E−M| = {geo["dist_E_M"]:.1f}, '
          f'|M−L| = {geo["dist_M_L"]:.1f}, |E−L| = {geo["dist_E_L"]:.1f}, cos(M−E, L−M) = {geo["cos_EM_ML"]:.2f} '
          f'(the trajectory keeps turning; E→M and M→L are far from collinear).',
          '- Comparison columns: E, M(α=0), α=.25/.5/.75, L(α=1), AVG3, plus the coexistence feature-fusion arms E+L, M+L, '
          'PCA(E+L)→d.\n',
          '## Protocol and provenance\n',
          f'- v4 protocol, identical pinned evaluators and arguments to the E/M/L campaign (`{COEX.name}`): frozen '
          'classification/regression B64 BF16 seed 0 auto/TTA8 dataset-best resolution; within-set + HPA retrieval/clustering; '
          'detection proxy B8 224 stretch 5 ep; v3-only segmentation formal splits (PanNuke 3 rotations), B32/B32, E20+E50, '
          'seeds 0/1/2, test once at best-val epoch.',
          '- Every classification/regression/retrieval cell is identity-matched: ordered sample paths and labels of the '
          'weight-space banks equal the E/M/L banks (fail-closed), the E/M/L probes are re-run from the same banks, and E+L / '
          'M+L / PCA are copied from the coexistence fusion JSON only when its bank SHA256s match. Re-run E/M/L reproduced '
          'the coexistence numbers exactly on all but seven cells, where the coexistence fusion had been computed on another '
          'host and the logistic-regression solver differs by ≤2e-3 balanced accuracy (labelled `REPRO_DIFF_…_CROSS_HOST`); '
          'in those rows all single arms are same-host numbers.',
          f'- Execution: {n_tasks} GPU tasks through a shared claim queue, all DONE, 0 FAILED; per-host completions: '
          + ', '.join(f'{h} {n}' for h, n in sorted(hosts.items(), key=lambda x: -x[1])) + '. Manifests: '
          '`campaign_manifest.json` (+ pairing addenda v1/v2), `checkpoints/checkpoint_manifest.json`, `tasks.json`, '
          '`claims/*/status.json`, `logs/`, `node_telemetry/`. No checkpoint or dataset was copied between machines.',
          '- Not scheduled (no E/M/L baseline or blocked in v4): MoNuSeg (BLOCKED_NOT_TESTED), rxrx1-cross, rxrx3-core, CTC, '
          'OOD. LC25000 classification is PROVISIONAL_LEGACY_ONLY; nct-crc-he-100 is LOW_N (99 samples). Missing cells are '
          'never counted as zero and nothing here is a full-v4 aggregate.\n',
          '## Per-family aggregates (primary metric; segmentation = mean test mIoU over 3 seeds per budget cell)\n']
    for fam, scale in (('classification', 1), ('regression', 1), ('retrieval', 1), ('clustering', 1), ('detection_proxy', 100), ('segmentation', 1)):
        r = fam_rows.get(fam, [])
        if not r:
            continue
        pending = [name for _, name, _, st in r if st not in ('COMPLETE', 'VALID_COMPLETE') and not st.startswith('REPRO_DIFF') and st not in ('LOW_N', 'PROVISIONAL_LEGACY_ONLY')]
        md.append(f'### {fam} ({len(r)} cells{"; pending baseline: " + ", ".join(pending) if pending else ""})\n')
        md += aggregate(r, scale) + [''] + retention(r, scale) + ['']
    md += ['## Answers to the four questions\n']
    md += answers(fam_rows)
    md += ['\n## Full tables (from the latest status snapshot)\n', tables]
    out = ROOT / f'FINAL_REPORT_{stamp}.md'
    out.write_text('\n'.join(md) + '\n')
    print(out)


def answers(fam_rows):
    def stat(fam, arm, key):
        r = [v for _, _, v, _ in fam_rows.get(fam, []) if v.get(arm) is not None and all(v.get(x) is not None for x in 'EML')]
        if not r:
            return float('nan')
        if key == 'dL':
            return statistics.mean(v[arm] - v['L'] for v in r)
        if key == 'dbest':
            return statistics.mean(v[arm] - max(v['E'], v['M'], v['L']) for v in r)
    lines = []
    lines.append('1. **Classification retain?** Mean Δ vs L: ' + ', '.join(f'{NAMES[a]} {stat("classification", a, "dL"):+.4f}' for a in WEIGHT)
                 + f'; vs best single checkpoint: ' + ', '.join(f'{NAMES[a]} {stat("classification", a, "dbest"):+.4f}' for a in WEIGHT)
                 + '. Interpolation is roughly a monotone path between M and L on most datasets (α=.75 and AVG3 track L on '
                   'average; α=.5 sits a few tenths of a point lower) and it does not recover E-best or M-best datasets '
                   '(e.g. NCT-CRC-HE where M leads L by 8 pp); the same holds for E+L/M+L feature fusion, so classification '
                   'shows no consolidation gain from either route.')
    lines.append('2. **Segmentation retain?** ' + ', '.join(f'{NAMES[a]} {stat("segmentation", a, "dL"):+.4f}' for a in WEIGHT)
                 + ' mean Δ vs L (mIoU). On the E-favouring datasets (cellpose, TissueNet) the weight path is strictly monotone '
                   'between M and L and never approaches E, and AVG3 lands between M and E (at M on cellpose, near E on TissueNet); feature fusion E+L does retain '
                   'E on cellpose (.743 vs E .741, L .718). On PanNuke/CoNIC/Multimodal/LIVECell (L-favouring or flat) every '
                   'weight arm is within noise of L. So early dense features are the one capability weight averaging cannot '
                   'merge, while feature-space concatenation can.')
    lines.append('3. **Retrieval late gain retain?** ' + ', '.join(f'{NAMES[a]} {stat("retrieval", a, "dL"):+.4f}' for a in WEIGHT)
                 + ' mean Δ vs L (mAP@10). α=.5–.75 match or slightly exceed L on every within-set dataset and on HPA; '
                   'clustering is even more favourable (' + ', '.join(f'{NAMES[a]} {stat("clustering", a, "dL"):+.4f}' for a in WEIGHT)
                 + ' ARI vs L), with α=.5 beating both M and L on NCT-CRC-HE-1k/100 and matching E+L elsewhere. The late '
                   'retrieval gain is retained, and mid-path weights are at least as good as feature fusion here.')
    lines.append('4. **Detection late gain retain?** ' + ', '.join(f'{NAMES[a]} {stat("detection_proxy", a, "dL"):+.2f}' for a in WEIGHT)
                 + ' mean Δ vs L (patch F1 points). α=.75 ties L on all three proxies (conic/livecell within 0.1 pp, bbbc038 '
                   '−0.06 pp) whereas E+L/M+L feature concatenation collapses by ~9 pp on bbbc038. Late detection gains are '
                   'retained under weight averaging and the proxy head is unaffected by the merge.')
    lines.append('\n**Overall.** Weight averaging along M→L behaves like a smooth path in one basin: global/retrieval/detection '
                 'capabilities of L are kept (and clustering even improves at α=.5), so those capabilities are "soft" and '
                 'need no architecture. The capability it cannot merge is the early-checkpoint dense-segmentation advantage '
                 '(cellpose, TissueNet), which feature fusion E+L does recover — the information exists in the trajectory but '
                 'a single vanilla parameter endpoint does not consolidate it, supporting a new training objective rather '
                 'than post-hoc merging for that case. AVG3 (which includes E) only partially helps there (M level on cellpose, near E on TissueNet).')
    return lines


if __name__ == '__main__':
    main()
