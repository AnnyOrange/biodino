#!/usr/bin/env python3
"""Prepare an isolated, finite recovery run; this script does not launch it."""
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
    fork = GROUP / 'fork38063'
    manifest = json.loads((fork / 'fork_manifest.json').read_text())
    assert manifest['iteration'] == 38063 and manifest['optimizer_states'] == 357
    assert manifest['adam_step_min_max'] == [38064., 38064.]
    run = GROUP / 'fixed_cls_w1_anchor38063'
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'launch_config.yaml'
    if config_path.exists():
        raise FileExistsError(config_path)
    snapshot = GROUP / 'runtime_original_sampling'
    snapshot.mkdir(exist_ok=True)
    shutil.copytree(ROOT / 'dinov3', snapshot / 'dinov3',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=False)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (snapshot / 'dinov3').rglob('*') if p.is_file()}
    (snapshot / 'source_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    cfg = OmegaConf.load(BASE / 'config.yaml')
    anchor = str(fork / 'anchor/teacher_checkpoint.pth')
    cfg.train.max_updates = 43920
    cfg.train.start_iteration_override = None
    cfg.train.output_dir = str(run)
    cfg.student.resume_from_teacher_chkpt = anchor
    cfg.compute_precision.distributed_mode = 'fsdp'
    cfg.compute_precision.param_dtype = 'bf16'
    cfg.train.batch_size_per_gpu = 16
    cfg.optim.gradient_accumulation_steps = 8
    cfg.gram.use_loss = True
    cfg.gram.require_official_fixed_anchor_contract = False
    cfg.gram.compute_stats = False
    cfg.gram.ckpt = anchor
    cfg.gram.ema_teacher = False
    cfg.gram.it_load_ema_teacher = -1
    cfg.gram.rep_update = False
    cfg.gram.inter_image_loss_weight = 0.
    cfg.gram.global_relation_loss_weight = 0.
    cfg.recovery.enabled = True
    cfg.recovery.mode = 'fixed'
    cfg.recovery.global_tokens = 'cls'
    cfg.recovery.loss_weight = 1.
    cfg.recovery.global_weight = 1.
    cfg.recovery.local_weight = 0.
    cfg.recovery.anchor_momentum = 0.
    cfg.recovery.ridge = .1
    cfg.recovery.warmup = 32
    cfg.evaluation.eval_period_iterations = 488
    cfg.checkpointing.period = 488
    cfg.checkpointing.max_to_keep = None
    cfg.checkpointing.keep_every = 16592
    cfg.checkpointing.sharded = False
    dest = run / 'ckpt/38063/checkpoint.pth'
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(fork / 'ckpt/38063/checkpoint.pth', dest)
    OmegaConf.save(cfg, config_path)
    record = dict(status='PREPARED', run=str(run), runtime=str(snapshot), initial_checkpoint=38063,
                  last_checkpoint=43919, max_updates=43920, anchor=anchor,
                  recovery=OmegaConf.to_container(cfg.recovery), checkpoint_keep_all=True,
                  source_fork_manifest=str(fork / 'fork_manifest.json'),
                  original_config=str(BASE / 'config.yaml'),
                  authorization='User explicitly approved local switch on 2026-10-09')
    (run / 'launch_manifest_20261009.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
