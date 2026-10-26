#!/usr/bin/env python3
"""Queue the exported local 20TB distillation checkpoints for frozen V4 components."""
import json
import sys
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
PRIOR = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006/campaign_manifest.json'
OUTPUT = REPO / 'outputs/02_eval_runs/v2_20tb_local_distill_20261014'
TRAIN = REPO / 'outputs/01_training_runs/cooldown_20261013/20tb_distill_cooldown_len4880'
SNAPSHOT = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918')


def main():
    manifest_path = OUTPUT / 'campaign_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    sys.path.insert(0, str(SNAPSHOT / 'scripts'))
    import run_retest_fleet_20260918 as fleet
    prior = json.loads(PRIOR.read_text())
    assets = []
    for path in sorted((TRAIN / 'eval').glob('training_*/teacher_checkpoint.pth')):
        if path.stat().st_size < 1_000_000_000:
            continue
        assets.append(dict(arm='distill_lw2_20tb', checkpoint_id=path.parent.name.split('_')[-1],
                           path=str(path), config=str(TRAIN / 'config.yaml'),
                           kind='dinov3', model_id='', reserve_mib=4096))
    if not assets:
        raise RuntimeError('No complete distillation checkpoint found')
    manifest = {**prior, 'campaign_scope': 'LOCAL_20TB_DISTILL_COMPONENTS',
                'authorization': '2026-10-14 user requested all 20TB experiments evaluated',
                'checkpoint_assets': assets, 'training_roots': {'distill_lw2_20tb': str(TRAIN)},
                'tasks': fleet.tasks_for(assets, prior['datasets']),
                'online_checkpoints': False, 'created_unix': time.time()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (OUTPUT / '_state/inputs').mkdir(parents=True, exist_ok=True)
    print('PREPARED', len(assets), 'checkpoints', len(manifest['tasks']), 'tasks', flush=True)


if __name__ == '__main__':
    main()
