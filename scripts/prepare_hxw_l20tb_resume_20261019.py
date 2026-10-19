#!/usr/bin/env python3
"""Relocate either paused L branch to the verified hxw 20TB data mirror."""

import argparse
import glob
from pathlib import Path

from omegaconf import OmegaConf


HOPTIMUS = Path('/home/xzj/storage/merged/cooldown_20261013/teachers/H-optimus-0')
DINOV3L = Path('/home/xzj/route2_20tb_20261006/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-step', type=int, required=True)
    parser.add_argument('--bio5tb', type=Path)
    parser.add_argument('--microbatch', type=int, default=64)
    args = parser.parse_args()

    route = '/home/xzj/storage/merged/15TB_from_20TB/filtered_projection_20TB_nested-'
    parts = ((0.09, '/data/microscopy-100k-patched/filtered_mixed_train_w*.tar'),
             (0.21, '/home/xzj/storage/merged/4TB/wds_patched_shuffle/filtered_mixed_train*.tar'),
             (0.5581, route + 'r0*.tar'), (0.1419, route + 'r9*.tar'))
    counts = [len(glob.glob(pattern)) for _, pattern in parts]
    if any(count == 0 for count in counts) or counts[2:] != [3671, 283]:
        raise RuntimeError(f'Incomplete hxw 20TB mirror: {counts}')
    required = [HOPTIMUS / 'config.json', HOPTIMUS / 'pytorch_model.bin', DINOV3L,
                args.output_dir / f'ckpt/{args.resume_step}/checkpoint.pth']
    if args.bio5tb:
        required.append(args.bio5tb)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(args.source_config)
    if not cfg.external_distill.enabled or cfg.gram.use_loss:
        raise RuntimeError('Expected the no-GRAM L distillation branch')
    teachers = {teacher.name: teacher for teacher in cfg.external_distill.teachers}
    expected = {'hoptimus0', 'dinov3l'} | ({'bio5tb'} if args.bio5tb else set())
    if set(teachers) != expected:
        raise RuntimeError(f'Unexpected teachers: {list(teachers)}; expected {expected}')
    if args.bio5tb and teachers['bio5tb'].residual_of != 'dinov3l':
        raise RuntimeError('Complementary teacher must retain the DINOv3-L residual')
    if not cfg.train.checkpointing_full or args.microbatch not in (32, 64):
        raise RuntimeError('hxw 4-GPU L resume requires full activation checkpointing and batch 32 or 64')
    cfg.train.batch_size_per_gpu = args.microbatch
    cfg.optim.gradient_accumulation_steps = 1024 // (4 * args.microbatch)

    cfg.train.dataset_path = 'mixwds_robust:' + '||'.join(
        f'{weight}={pattern}' for weight, pattern in parts) + '::pct=1,99'
    cfg.train.output_dir = str(args.output_dir)
    cfg.student.resume_from_teacher_chkpt = str(DINOV3L)
    teachers['hoptimus0'].path = str(HOPTIMUS)
    teachers['dinov3l'].weights = str(DINOV3L)
    if args.bio5tb:
        teachers['bio5tb'].weights = str(args.bio5tb)
    output = args.output_dir / 'candidate_hxw.yaml'
    OmegaConf.save(cfg, output)
    print(f'Prepared {output}: shards={counts}, resume={args.resume_step + 1}, '
          f'end={cfg.train.max_updates}, teachers={sorted(teachers)}, '
          f'batch=4x{args.microbatch}x{cfg.optim.gradient_accumulation_steps}')


if __name__ == '__main__':
    main()
