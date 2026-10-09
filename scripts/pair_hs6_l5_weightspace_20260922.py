#!/usr/bin/env python3
"""CPU pass: identity-matched pairing of finished weight-space arms with E/M/L.

Classification/regression and retrieval banks are re-probed from the pinned
pairing snapshot (source_snapshot_weightspace_v1); outputs paired/<family>_<dataset>.json
are written once and never overwritten. Detection and segmentation results are
gathered into derived paired/<family>_<dataset>.json views (rewritten each run)
after checking that the protocol fields of every arm agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import (  # noqa: E402
    CLS_LARGE, CLS_SMALL, COEX, DET, PYTHON, RET_HPA, RET_WITHIN, ROOT, SEG_CKPT_ID, SEG_FIRST, SEG_REST,
    SEG_RUN_LABEL, SNAP_PAIR, THREADS, WEIGHT_ARMS, task_done, sha256)

PROTOCOL = json.loads((COEX / 'source_snapshot/Evaluation Rules/protocol_v4.json').read_text())
REGRESSION = {d for s in ('tier_a', 'tier_b', 'union_extension') for d in PROTOCOL[s].get('regression', [])}


def log(m):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {m}', flush=True)


def arms_ready(family_dir: str, dataset: str, rows: int = 1) -> bool:
    for arm in WEIGHT_ARMS:
        ok, _ = task_done({'type': 'summary_csv', 'path': str(ROOT / family_dir / dataset / arm / 'summary.csv'),
                           'dataset': dataset, 'rows': rows})
        if not ok:
            return False
    return True


def probe_jobs() -> list[tuple[str, list[str]]]:
    jobs = []
    for ds in CLS_SMALL + CLS_LARGE:
        family = 'regression' if ds in REGRESSION else 'classification'
        out = ROOT / 'paired' / f'{family}_{ds}.json'
        fusion = COEX / 'fusion' / f'{ds}.json'
        if out.exists() or not fusion.exists() or not arms_ready('frozen', ds):
            continue
        jobs.append((f'{family}_{ds}', [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_weightspace_classification',
                                        '--dataset', ds, '--baseline-root', str(COEX / 'frozen'), '--baseline-fusion', str(fusion),
                                        '--arm-root', str(ROOT / 'frozen'), '--arms', *WEIGHT_ARMS, '--output', str(out)]))
    for ds in RET_WITHIN + RET_HPA:
        out = ROOT / 'paired' / f'retrieval_{ds}.json'
        fusion = COEX / 'fusion' / f'retrieval_{ds}.json'
        hpa = ds == 'hpa-subcellular'
        if out.exists() or (not hpa and not fusion.exists()) or not arms_ready('retrieval', ds, 3 if hpa else 1):
            continue
        cmd = [str(PYTHON), '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_weightspace_retrieval', '--dataset', ds,
               '--baseline-root', str(COEX / 'retrieval'), '--arm-root', str(ROOT / 'retrieval'),
               '--arms', *WEIGHT_ARMS, '--output', str(out)]
        if not hpa:
            cmd += ['--baseline-fusion', str(fusion)]
        jobs.append((f'retrieval_{ds}', cmd))
    return jobs


def run_probes(jobs, parallel: int) -> None:
    env = os.environ.copy()
    env.update(THREADS)
    env['PYTHONPATH'] = str(SNAP_PAIR)
    running: list[tuple[str, subprocess.Popen]] = []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < parallel:
            name, cmd = queue.pop(0)
            logp = ROOT / 'logs' / f'pair_{name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.log'
            stream = logp.open('w')
            stream.write(f'# cwd={SNAP_PAIR}\n# cmd={json.dumps(cmd)}\n')
            stream.flush()
            proc = subprocess.Popen(cmd, cwd=SNAP_PAIR, env=env, stdout=stream, stderr=subprocess.STDOUT)
            running.append((name, proc))
            log(f'[pair] {name} pid={proc.pid} log={logp}')
        for name, proc in list(running):
            rc = proc.poll()
            if rc is not None:
                running.remove((name, proc))
                log(f'[pair-{"done" if rc == 0 else "FAILED"}] {name} rc={rc}')
        if running:
            import time
            time.sleep(15)


def gather_detection() -> None:
    for ds in DET:
        arms = {}
        sources = {'E': COEX, 'M': COEX, 'L': COEX, 'E+L': COEX, 'M+L': COEX} if ds == 'bbbc038' else {}
        sources.update({a: ROOT for a in (['E', 'M', 'L'] if ds != 'bbbc038' else []) + WEIGHT_ARMS})
        for arm, root in sources.items():
            p = root / 'detection' / ds / arm / 'results_bio_detection.json'
            if not p.exists():
                arms[arm] = {'status': 'PENDING'}
                continue
            d = json.loads(p.read_text())
            proto = {k: d.get(k) for k in ('dataset', 'image_size', 'epochs', 'batch_size', 'seed', 'conic_split_protocol')}
            arms[arm] = {'status': 'COMPLETE' if math.isfinite(float(d.get('test_patch_f1', float('nan')))) else 'INVALID',
                         'test_patch_f1': d.get('test_patch_f1'), 'val_patch_f1': d.get('val_patch_f1'),
                         'test_patch_precision': d.get('test_patch_precision'), 'test_patch_recall': d.get('test_patch_recall'),
                         'protocol': proto, 'source': str(p), 'sha256': sha256(p)}
        complete = [a for a, v in arms.items() if v['status'] == 'COMPLETE']
        protos = {json.dumps({k: v for k, v in arms[a]['protocol'].items() if k != 'conic_split_protocol' or ds == 'conic'},
                             sort_keys=True) for a in complete}
        out = {'dataset': ds, 'family': 'detection_proxy', 'arms': arms,
               'protocol_consistent': len(protos) <= 1, 'gathered_utc': datetime.now(timezone.utc).isoformat(),
               'note': 'frozen center-to-patch proxy (B8, 224 stretch, 5 epochs); not native detection'}
        (ROOT / 'paired' / f'detection_{ds}.json').write_text(json.dumps(out, indent=2) + '\n')


def seg_results(root: Path, prefix: str, ckpt_id: int | None):
    found = {}
    for run_dir in sorted(root.glob(prefix + '*')):
        tag = run_dir.name[len(prefix):]
        pattern = f'budget*/seed*/*/{ckpt_id}/results.json' if ckpt_id is not None else 'budget*/seed*/*/results.json'
        for res in run_dir.glob(pattern):
            parts = res.relative_to(run_dir).parts
            budget, seed, dsdir = int(parts[0][6:]), int(parts[1][4:]), parts[2]
            d = json.loads(res.read_text())
            meta = d.get('_meta', {})
            if meta.get('test_evaluations') != 1 or 'test' not in d:
                continue
            found[(tag, dsdir, budget, seed)] = {'test_mIoU': d['test']['mIoU'], 'test_mDice': d['test'].get('mDice'),
                                                 'best_val_mIoU': meta.get('best_val_miou'), 'best_epoch': meta.get('best_epoch'),
                                                 'probe_epochs': meta.get('probe_epochs'), 'seed': meta.get('seed'),
                                                 'source': str(res)}
    return found


def gather_segmentation() -> None:
    for ds in SEG_FIRST + SEG_REST:
        base_root = COEX / 'segmentation' / ('results' if ds == 'cellpose' else 'remaining_results')
        cells = {}
        for role in ('E', 'M', 'L'):
            for key, val in seg_results(base_root, f'hs6_l5_{SEG_RUN_LABEL[role]}_', {'E': 12687, 'M': 20007, 'L': 29279}[role]).items():
                if key[1].startswith(ds) or (ds == 'pannuke' and 'pannuke' in key[0]):
                    cells.setdefault(key, {})[role] = val
        for arm in ('E+L', 'M+L', 'PCA_E+L'):
            fused_root = COEX / 'segmentation/fused_results' / arm
            for res in fused_root.glob('budget*/seed*/*/results.json') if fused_root.exists() else []:
                parts = res.relative_to(fused_root).parts
                if not parts[2].startswith(ds):
                    continue
                d = json.loads(res.read_text())
                meta = d.get('_meta', {})
                if meta.get('test_evaluations') != 1 or 'test' not in d:
                    continue
                # fused cells carry no run tag; attach to every baseline tag of this dataset with same budget/seed
                for key in [k for k in cells if k[1] == parts[2] and k[2] == int(parts[0][6:]) and k[3] == int(parts[1][4:])]:
                    cells[key]['PCA(E+L)->d' if arm == 'PCA_E+L' else arm] = {
                        'test_mIoU': d['test']['mIoU'], 'best_val_mIoU': meta.get('best_val_miou'), 'source': str(res)}
        for arm in WEIGHT_ARMS:
            for key, val in seg_results(ROOT / 'segmentation/results', f'hs6_l5_{SEG_RUN_LABEL[arm]}_', SEG_CKPT_ID[arm]).items():
                if key[1].startswith(ds) or (ds == 'pannuke' and 'pannuke' in key[0]):
                    cells.setdefault(key, {})[arm] = val
        rows = [{'tag': k[0], 'dataset_dir': k[1], 'budget': k[2], 'seed': k[3], 'arms': v} for k, v in sorted(cells.items())]
        out = {'dataset': ds, 'family': 'segmentation', 'cells': rows,
               'expected_cells': (3 if ds == 'pannuke' else 1) * 2 * 3,
               'gathered_utc': datetime.now(timezone.utc).isoformat(),
               'note': 'v3-only formal splits; test mIoU at best-validation epoch; pairing key = run tag/budget/seed'}
        (ROOT / 'paired' / f'segmentation_{ds}.json').write_text(json.dumps(out, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parallel', type=int, default=6)
    ap.add_argument('--no-probes', action='store_true')
    args = ap.parse_args()
    (ROOT / 'paired').mkdir(exist_ok=True)
    lock = ROOT / 'locks' / 'pairing_pass'
    if not args.no_probes:
        import time
        try:
            if lock.exists() and time.time() - lock.stat().st_mtime > 8 * 3600:
                lock.rmdir()  # stale
            lock.mkdir()
        except FileExistsError:
            log(f'ANOTHER_PAIRING_PASS_ACTIVE ({lock}); probes skipped, gathers only')
            args.no_probes = True
    try:
        if not args.no_probes:
            jobs = probe_jobs()
            log(f'{len(jobs)} pairing probe jobs ready')
            run_probes(jobs, args.parallel)
        gather_detection()
        gather_segmentation()
        log('gathered detection/segmentation views')
    finally:
        if not args.no_probes and lock.exists():
            lock.rmdir()


if __name__ == '__main__':
    main()
