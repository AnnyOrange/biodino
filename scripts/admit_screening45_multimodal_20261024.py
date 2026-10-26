#!/usr/bin/env python3
"""Append the authorized corrected Multimodal item after existing 44-item work."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/02_eval_runs/screening45_8k_20261024'
HISTORY = ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018/curves_20261019'
EVAL = ROOT / 'outputs/01_training_runs/cooldown_20261013/eval_root'
MM = 'segmentation/multimodal_cellseg'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = list(csv.DictReader((BASE / 'checkpoint_plan.csv').open()))
    manifests = {site: json.loads((BASE/site/'campaign_manifest.json').read_text()) for site in ['main', 'hxw']}
    assert all(not m.get('multimodal_correction_authorized') for m in manifests.values()), 'Already amended'
    main = manifests['main']
    runs = {a['arm']: str(Path(a['path']).parents[2]) for a in main['checkpoint_assets']}
    reuse_keys = set(json.loads((HISTORY/'local_snapshot.json').read_text())['arms']['5tb_peak19519']['scores']['19519'])
    reuse_keys = {k for k in reuse_keys if not k.startswith('detection/') and k != MM}
    assert len(reuse_keys) == 44
    for row in plan:
        if row['needs_fill_or_verification'] != 'False':
            continue
        arm, step = row['arm'], int(row['checkpoint'])
        if arm in runs:
            checkpoint = Path(runs[arm])/'eval'/f'training_{step}'/'teacher_checkpoint.pth'
            config = Path(runs[arm])/'config.yaml'
        elif arm == '20tb_distill_lw5':
            old = ROOT/'outputs/02_eval_runs/l20tb_selected29_20261019'
            checkpoint = old/'assets/20tb_distill_lw5/teacher_checkpoint.pth'
            config = old/'frozen_configs/20tb_distill_lw5.yaml'
        else:
            checkpoint = (EVAL/arm/'adapters'/str(step)/'checkpoint.pth').resolve()
            config = EVAL/arm/'source/config.yaml'
        assert checkpoint.is_file() and config.is_file(), (arm, step)
        source = HISTORY/('lyx_snapshot.json' if arm == '20tb_distill_lw5' else 'local_snapshot.json')
        raw = json.loads(source.read_text())['arms'][arm]['scores'][str(step)]
        scores = {k: raw[k] for k in sorted(reuse_keys)}
        stat = checkpoint.stat()
        main['checkpoint_assets'].append(dict(arm=arm, checkpoint=step, path=str(checkpoint), config=str(config),
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, config_sha256=sha(config), reused_scores=scores,
            reused_source=dict(path=str(source), sha256=sha(source), excluded_metric=MM)))
    readiness_path = ROOT/'outputs/02_eval_runs/unified_paper/multimodal_official_readiness.json'
    readiness = json.loads(readiness_path.read_text())
    source_files = [Path(__file__).resolve(), ROOT/'scripts/run_screening45_multimodal_20261024.py']
    for p in source_files + [readiness_path] + [Path(p) for p in readiness['manifests'].values()]:
        dst = BASE/'source_record'/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for site, m in manifests.items():
        for asset in m['checkpoint_assets']:
            arm, step = asset['arm'], asset['checkpoint']
            config = asset.get('config') or str(Path(asset['path']).parents[2]/'config.yaml')
            key = f'{arm}_ck{step}__segmentation__multimodal_cellseg'
            relative_cell = f'cells/{key}'
            data = ('/data/benchmark' if site == 'hxw' else '/mnt/huawei_deepcad/benchmark') + '/segmentation/Multimodal_CellSeg/neurips22_cellseg'
            m['tasks'].append(dict(key=key, arm=arm, checkpoint=step, family='segmentation', dataset='multimodal_cellseg',
                relative_cell=relative_cell, relative_output=relative_cell+'/results', reserve_mib=4608, ram_mib=32768,
                command=['{PYTHON}', '{REPO}/scripts/run_screening45_multimodal_20261024.py',
                         '--checkpoint', asset['path'], '--checkpoint-id', str(step), '--train-config', config,
                         '--data-root', data, '--cell', '{ROOT}/'+relative_cell]))
        for p in source_files:
            m['source_sha256'][str(p.relative_to(ROOT))] = sha(p)
        m.update(multimodal_correction_authorized=True, enabled_metrics=45, withheld={},
            corrected_protocol_id='SCREENING45_E50_SEED0_MM_AUTHOR_PUBLIC52_20261024',
            multimodal_split=readiness['protocol'], multimodal_manifest_sha256=readiness['manifest_sha256'],
            priority='Existing 44 metrics first; corrected Multimodal tasks appended last',
            aggregate_allowed=False, aggregate_rule='Emit corrected aggregate only when all 45 named metrics are validated')
        target = BASE/site/'campaign_manifest.json'
        shutil.copy2(target, target.with_name('campaign_manifest_before_mm.json'))
        tmp = target.with_suffix('.tmp');tmp.write_text(json.dumps(m, indent=2));tmp.replace(target)
        print(site, len(m['checkpoint_assets']), len(m['tasks']))
    amendment = dict(authorized_at=time.time(), user_instruction='先补其余44项，Multimodal CellSeg按照v4 v5的正确划分来',
        interpretation='Use the currently certified corrected split; stale historical V4/V5 documents describe withdrawn membership.',
        scope_checkpoints=26, protocol=readiness['protocol'], counts=readiness['counts'],
        manifest_sha256=readiness['manifest_sha256'], old_scores_reused=False,
        unchanged_recipe='512 pad last1; feature BF16 batch4; AdamW E50/B16/eval50/seed0; historical screening recipe',
        limitation='Corrected screening45, not complete V4-ID; all other 44 metric definitions remain historical screening.')
    (BASE/'multimodal_correction.json').write_text(json.dumps(amendment, indent=2))


if __name__ == '__main__':
    main()
