#!/usr/bin/env python3
"""Record immutable launch identities for the exploratory v4 coexistence run."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3')
CAMPAIGN = ROOT / 'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
TRAIN = ROOT / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy
    import sklearn
    import torch

    variant = sys.argv[1] if len(sys.argv) > 1 else 'source_snapshot'
    if variant not in {'source_snapshot', 'source_snapshot_fusion_v2',
                       'source_snapshot_retrieval_v3', 'source_snapshot_dense_v4',
                       'source_snapshot_dense_pca_v5', 'source_snapshot_detection_v6',
                       'source_snapshot_lc_v7', 'source_snapshot_dense_controls_v8',
                       'source_snapshot_detection_controls_v9'}:
        raise ValueError(f'Unrecognized source snapshot variant: {variant}')
    snapshot = CAMPAIGN / variant
    launch_names = {'source_snapshot': 'launch_manifest.json',
                    'source_snapshot_fusion_v2': 'fusion_v2_manifest.json',
                    'source_snapshot_retrieval_v3': 'retrieval_v3_manifest.json',
                    'source_snapshot_dense_v4': 'dense_v4_manifest.json',
                    'source_snapshot_dense_pca_v5': 'dense_pca_v5_manifest.json',
                    'source_snapshot_detection_v6': 'detection_v6_manifest.json',
                    'source_snapshot_lc_v7': 'lc_v7_manifest.json',
                    'source_snapshot_dense_controls_v8': 'dense_controls_v8_manifest.json',
                    'source_snapshot_detection_controls_v9': 'detection_controls_v9_manifest.json'}
    launch = CAMPAIGN / launch_names[variant]
    if launch.exists():
        raise FileExistsError(f'Refuse to overwrite launch manifest: {launch}')
    source_hashes = {
        str(p.relative_to(snapshot)): sha256(p)
        for p in sorted(snapshot.rglob('*'))
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh', '.yaml'}
    }
    if variant != 'source_snapshot':
        original = json.loads((CAMPAIGN / 'launch_manifest.json').read_text())
        teachers = original['teachers']
    else:
        teachers = {
            role: {'checkpoint': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
            for role, step in [('E', 12687), ('M', 20007), ('L', 29279)]
            for p in [TRAIN / f'eval/training_{step}/teacher_checkpoint.pth']
        }
    manifest = {
        'protocol': 'bio-eval-union-v4',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_selection': 'E/M retrospective downstream exploratory anchors; L terminal endpoint',
        'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_snapshot': str(snapshot),
        'source_sha256': source_hashes,
        'teachers': teachers,
        'train_config': {'path': str(TRAIN / 'config.yaml'), 'sha256': sha256(TRAIN / 'config.yaml')},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'numpy': numpy.__version__, 'sklearn': sklearn.__version__},
        'full_v4_status': 'INCOMPLETE_UNTIL_EVERY_ADMITTED_CELL_MATCHED; MoNuSeg/CTC blocked',
        'first_gpu_pilot': {'node': 'cpu2', 'dataset': 'breastmnist', 'role': 'E',
                            'pid': 238966, 'launched_before_manifest': True},
        'snapshot_variant': variant,
    }
    launch.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Saved manifest {launch}; source files={len(source_hashes)}")


if __name__ == '__main__':
    main()
