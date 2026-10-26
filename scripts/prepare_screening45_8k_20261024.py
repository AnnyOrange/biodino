#!/usr/bin/env python3
"""Prepare the user-approved 8k scope without modifying historical results."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/02_eval_runs/screening45_8k_20261024'
AUDIT = Path('/home/lxy/cpui_20tb_audit_20261024')
REMOTE_REPO = '/home/xzj/screening45_8k_20261024/repo'
REMOTE_BASE = REMOTE_REPO + '/outputs/02_eval_runs/screening45_8k_20261024/hxw'
REF = ROOT / 'outputs/01_training_runs/cooldown_20261013/eval_root/5tb_peak19519'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    plan = list(csv.DictReader((AUDIT / 'screening45_8k_plan.csv').open()))
    shutil.copy2(AUDIT / 'screening45_8k_plan.csv', BASE / 'checkpoint_plan.csv')
    templates = {}
    for p in (REF / 'results/point_19519').glob('*/command_manifest.json'):
        for t in json.loads(p.read_text()):
            templates[(t['task'], t['dataset'])] = t['cmd']
    scores = json.loads((AUDIT / 'screening45_local_current.json').read_text())['5tb_peak19519']['19519']
    cells = sorted({tuple(k.split('/')) for k in scores if not k.startswith(('detection/', 'clustering/'))
                    and k != 'segmentation/multimodal_cellseg'})
    assert len(cells) == 40
    old = ROOT / 'outputs/01_training_runs/cooldown_20261013/eval_root/biodino_eval_git_20260917'
    files = [p for p in (ROOT / 'dinov3').rglob('*.py') if '__pycache__' not in str(p)]
    files += list((ROOT / 'dinov3/configs').rglob('*.yaml'))
    files += list((ROOT / 'Evaluation Rules').glob('*.json'))
    files += list((ROOT / 'Evaluation Rules').glob('*.md'))
    files += [ROOT / 'scripts/run_screening45_8k_20261024.py', ROOT / 'scripts/prepare_screening45_8k_20261024.py']
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in files}
    differences = []
    archive = BASE / 'source_record'
    for p in files:
        rel = p.relative_to(ROOT)
        q = archive / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        if (old / rel).exists() and digest(old / rel) != fingerprints[str(rel)]:
            differences.append(str(rel))
    migration = dict(created=time.time(), canonical_repo=str(ROOT), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_files=subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True),
        changed_from_historical=differences,
        protocol='screening45 historical commands, E50 seed0; Multimodal deferred pending user decision',
        monuseg='Explicit DINOV3_MONUSEG_LEGACY=1 reproduces the historical loader; not current formal V4',
        numerical_recipe='Torch2.8 / torchvision0.23 / NumPy2.2.6 / SciPy1.16.1 / sklearn1.7.1 / Pillow11.3',
        reference_commands_sha256={str(p.relative_to(ROOT)): digest(p) for p in (REF/'results/point_19519').glob('*/command_manifest.json')},
        reuse='Seven historical complete points retained with original protocol; no corrected45 aggregate until MM is resolved')
    (BASE / 'migration.json').write_text(json.dumps(migration, indent=2))
    runs = {
        '5tb_peak19519': str((REF / 'adapters/19519/checkpoint.pth').resolve().parents[2]),
        'nogram_main': str(ROOT / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924'),
        'l20tb_domain': '/home/xzj/storage/merged/l20tb_domain_recoverability_from47823_len19520_hxw_20261019',
        'l20tb_complement_bio5tb': '/home/xzj/storage/merged/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019',
    }
    for site in ['main', 'hxw']:
        dest = BASE / site
        dest.mkdir(exist_ok=True)
        tasks, assets = [], []
        for row in plan:
            if row['needs_fill_or_verification'] != 'True':
                continue
            arm, step = row['arm'], int(row['checkpoint'])
            remote = arm in ['l20tb_domain', 'l20tb_complement_bio5tb'] or (arm == '5tb_peak19519' and step >= 32207)
            if remote != (site == 'hxw'):
                continue
            run = runs[arm]
            if arm == '5tb_peak19519' and remote:
                run = '/data/hs6_l_5tb_nogram_eval_20260921/source'
            checkpoint = f'{run}/eval/training_{step}/teacher_checkpoint.pth'
            config = f'{run}/config.yaml'
            if not remote:
                assert Path(checkpoint).is_file(), checkpoint
                assert Path(config).is_file(), config
                adapter = dest / 'adapters' / arm / str(step)
                adapter.mkdir(parents=True, exist_ok=True)
                link = adapter / 'checkpoint.pth'
                if not link.exists():
                    link.symlink_to(checkpoint)
                stat = Path(checkpoint).stat()
                assets.append(dict(arm=arm, checkpoint=step, path=checkpoint,
                                   bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, config_sha256=digest(Path(config))))
            else:
                assets.append(dict(arm=arm, checkpoint=step, path=checkpoint, config=config))
            for family, dataset in cells:
                key = f'{arm}_ck{step}__{family}__{dataset}'
                relative_cell = f'cells/{key}'
                cell = '{ROOT}/' + relative_cell
                command = list(templates[(family, dataset)])
                command[0] = '{PYTHON}'
                substitutions = {'--checkpoint': checkpoint, '--train-config': config,
                    '--model-name': f'dinov3-{step}', '--checkpoint-iters': str(step),
                    '--checkpoints-dir': '{ROOT}/adapters/' + arm,
                    '--benchmark-root': '/data/benchmark' if remote else '/mnt/huawei_deepcad/benchmark',
                    '--data-root-base': '/data/benchmark/segmentation' if remote else '/mnt/huawei_deepcad/benchmark/segmentation',
                    '--output-dir': cell + '/result', '--output-root': cell + '/results',
                    '--cache-root': cell + '/cache', '--gpu': '{GPU}'}
                for option, value in substitutions.items():
                    if option in command:
                        command[command.index(option) + 1] = value
                reserve = 3584
                if family == 'segmentation':
                    reserve = 4608 if dataset not in ['monuseg', 'livecell'] else 6144
                elif dataset in ['bloodmnist', 'tissuemnist', 'bbbc005', 'bbbc013']:
                    reserve = 6144
                tasks.append(dict(key=key, arm=arm, checkpoint=step, family=family, dataset=dataset,
                    command=command, relative_cell=relative_cell,
                    relative_output=relative_cell + ('/results' if family == 'segmentation' else '/result'),
                    reserve_mib=reserve, ram_mib=32768 if family == 'segmentation' else 8192))
        # Small independent probes establish validated progress before long dense jobs.
        priority = {'breastmnist': 0, 'pneumoniamnist': 1, 'retinamnist': 2, 'nct-crc-he-100': 3}
        tasks.sort(key=lambda t: (priority.get(t['dataset'], 10), t['family']=='segmentation', -t['checkpoint'], t['key']))
        manifest = dict(protocol_id='SCREENING45_E50_SEED0_CANONICAL_8K_20261024', site=site,
            user_authorization='2026-10-24: fill agreed 8k plan; routing/bio weights evaluated on hxw-xzj only',
            created=time.time(), tasks=tasks, checkpoint_assets=assets, expected_screening_metrics=45,
            enabled_metrics=44, withheld={'segmentation/multimodal_cellseg': 'withdrawn mismatched masks; user decision pending'},
            source_sha256=fingerprints, runtime_env={'DINOV3_MONUSEG_LEGACY':'1'},
            aggregate_allowed=False, historical_reference='5TB no-GRAM / 19519',
            repo=str(ROOT) if site=='main' else REMOTE_REPO,
            deployment_root=str(dest) if site=='main' else REMOTE_BASE)
        (dest/'campaign_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(site, len(assets), len(tasks))


if __name__ == '__main__':
    main()
