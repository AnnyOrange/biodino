#!/usr/bin/env python3
"""Relocate the paused L domain-plus-residual run to the verified hxw data mirror."""

import argparse
import glob
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--hoptimus', type=Path, required=True)
    parser.add_argument('--dinov3l', type=Path, required=True)
    parser.add_argument('--bio5tb', type=Path, required=True)
    args = parser.parse_args()

    one_tb = '/data/microscopy-100k-patched/filtered_mixed_train_w*.tar'
    four_tb = '/home/xzj/storage/merged/4TB/wds_patched_shuffle/filtered_mixed_train*.tar'
    route = '/home/xzj/storage/merged/15TB_from_20TB/filtered_projection_20TB_nested-'
    parts = ((0.09, one_tb), (0.21, four_tb), (0.5581, route + 'r0*.tar'),
             (0.1419, route + 'r9*.tar'))
    counts = [len(glob.glob(pattern)) for _, pattern in parts]
    if any(count == 0 for count in counts) or counts[2:] != [3671, 283]:
        raise RuntimeError(f'Incomplete hxw 20TB mirror: {counts}')
    for path in (args.hoptimus / 'config.json', args.hoptimus / 'pytorch_model.bin',
                 args.dinov3l, args.bio5tb, args.output_dir / 'ckpt/49775/checkpoint.pth'):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = OmegaConf.load(args.source_config)
    if not config.external_distill.enabled or config.gram.use_loss:
        raise RuntimeError('Expected the no-GRAM distillation branch')
    config.train.dataset_path = 'mixwds_robust:' + '||'.join(
        f'{weight}={pattern}' for weight, pattern in parts) + '::pct=1,99'
    config.train.output_dir = str(args.output_dir)
    config.student.resume_from_teacher_chkpt = str(args.dinov3l)
    teachers = {teacher.name: teacher for teacher in config.external_distill.teachers}
    if set(teachers) != {'hoptimus0', 'dinov3l', 'bio5tb'}:
        raise RuntimeError(f'Unexpected teachers: {list(teachers)}')
    teachers['hoptimus0'].path = str(args.hoptimus)
    teachers['dinov3l'].weights = str(args.dinov3l)
    teachers['bio5tb'].weights = str(args.bio5tb)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output_config)
    print(f'Prepared {args.output_config}: shard counts={counts}, '
          f'resume=49776, end={config.train.max_updates}')


if __name__ == '__main__':
    main()
