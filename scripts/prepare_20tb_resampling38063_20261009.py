#!/usr/bin/env python3
"""Prepare the sampling-only arm with full model, optimizer, and EMA state."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924'
GROUP = ROOT / 'outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork38063_20261009'


def main():
    run = GROUP / 'resampling_only_38063_8x3090qi'
    run.mkdir(parents=True, exist_ok=True)
    config = run / 'launch_config.yaml'
    if config.exists():
        raise FileExistsError(config)
    runtime = GROUP / 'runtime_resampling'
    runtime.mkdir(exist_ok=False)
    hashes = {}
    for source in (ROOT / 'dinov3').rglob('*'):
        relative = source.relative_to(ROOT)
        if not source.is_file() or {'outputs', '__pycache__'}.intersection(relative.parts):
            continue
        if source.suffix not in {'.py', '.yaml', '.yml', '.json', '.txt'}:
            continue
        dest = runtime / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (runtime / 'source_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    cfg = OmegaConf.load(BASE / 'config.yaml')
    cfg.train.output_dir = str(run)
    cfg.train.dataset_path = 'sourcebalancedmix:/home/bbnc/20tb_resampling_20261009/index/manifest.json'
    cfg.train.max_updates = 43920
    cfg.train.start_iteration_override = None
    cfg.train.batch_size_per_gpu = 16
    cfg.optim.gradient_accumulation_steps = 8
    cfg.recovery.enabled = False
    cfg.gram.use_loss = False
    cfg.gram.rep_update = False
    cfg.evaluation.eval_period_iterations = 488
    cfg.checkpointing.period = 488
    cfg.checkpointing.max_to_keep = None
    cfg.checkpointing.sharded = False
    dest = run / 'ckpt/38063/checkpoint.pth'
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(BASE / 'ckpt/38063/checkpoint.pth', dest)
    OmegaConf.save(cfg, config)
    record = dict(status='PREPARED_REQUIRES_INDEX_PREFLIGHT', run=str(run), runtime=str(runtime),
                  host='3090-qi', initial_checkpoint=38063, last_checkpoint=43919,
                  checkpoint_keep_all=True, optimizer='Original full ck38063 state, no reset',
                  arm='B: sampling only; compare with local C: original sampling plus fixed CLS anchor',
                  source_manifest=str(runtime / 'source_manifest.json'))
    (run / 'launch_manifest_20261009.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
