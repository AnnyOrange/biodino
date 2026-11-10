#!/usr/bin/env python3
"""Create the suxin-only validation-search and five-seed training DAG."""
import argparse
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cryo-data-root')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    root = Path(args.campaign).resolve()
    if (root / 'campaign.json').exists():
        raise ValueError('Campaign already exists; do not silently regenerate scientific recipes')
    jobs, branches = [], {}
    for branch, task, sizes, epochs, batch, sex in [
        ('bone_age_image', 'bone_age', [512, 768], 50, 32, False),
        ('bone_age_image_sex', 'bone_age', [512, 768], 50, 32, True),
        ('pneumonia', 'pneumonia', [512, 768], 40, 16, False),
        ('bbbc041', 'bbbc041', [768, 1024], 80, 16, False),
    ]:
        candidates = []
        for lr, size in itertools.product([1e-5, 3e-5, 1e-4], sizes):
            name = f'{branch}_lr{lr:g}_r{size}'
            recipe = dict(task=task, branch=branch, use_sex=sex, resolution=size,
                          train_sizes=([256, 384, 512] if size == 512 else
                                       [512, 640, 768] if size == 768 else [768, 896, 1024]),
                          max_size=size*2, epochs=epochs, effective_batch=batch,
                          encoder_lr=lr, head_lr_multiplier=10., weight_decay=.05, layer_decay=.75,
                          gradient_clip=1., warmup_fraction=.05, activation_checkpointing=True,
                          num_classes=7 if task == 'bbbc041' else 1,
                          score_threshold_grid=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9],
                          checkpoint_sha256='f63544ca11ab72d7c3e05b5765c613ebc9d6374ba45ee5ca0c619eaa2f4cc423',
                          train_config_sha256='e33cb26b1eecbec0efa7a466b17fb59f16f7a77ae25aa6b4b138028aa95eabe3')
            path = root / 'recipes' / (name + '.json')
            save_json(path, recipe)
            candidates.append(dict(name=name, recipe=str(path), resolution=size, lr=lr))
            for seed in (101, 102):
                jobs.append(dict(id=f'{name}_seed{seed}', branch=branch, task=task, seed=seed,
                                 phase='development', recipe=str(path), microbatch=1,
                                 output=str(root / 'runs' / f'{name}_seed{seed}')))
        branches[branch] = dict(candidates=candidates, task=task)
        for seed in range(5):
            jobs.append(dict(id=f'{branch}_formal_seed{seed}', branch=branch, task=task, seed=seed,
                             phase='formal', recipe=str(root / 'selected' / branch / 'recipe.json'),
                             recipe_lock=str(root / 'selected' / branch / 'lock.json'), microbatch=1,
                             output=str(root / 'runs' / f'{branch}_formal_seed{seed}')))
    save_json(root / 'campaign.json', dict(schema='ood-transfer-campaign-v1', host='suxin-8H100-1',
              checkpoint=args.checkpoint, train_config=args.train_config, python=args.python,
              inputs=args.inputs, data_root=args.data_root,
              task_data_roots={'cryo':args.cryo_data_root} if args.cryo_data_root else {}, jobs=jobs, branches=branches,
              development_seeds=[101, 102], formal_seeds=list(range(5)), enabled=False,
              frozen_jobs=[dict(id=task+'_frozen',task=task,phase='frozen',output=str(root/'runs'/(task+'_frozen')))
                           for task in ('pxct','cryo','film')],
              blocked_tasks=[dict(task='cryo_pose',reason='Missing canonical pose manifest and label audit')]))
    save_json(root / 'caps.json', dict(gpus=list(range(8)), max_jobs=8, gpu_peak_mib=50000,
              gpu_reserve_fraction=.20, ram_peak_gib=24, ram_reserve_fraction=.20, max_cpu_util=.88,
              min_disk_free_gib=120, threads=2, nice=15, allow_test=False))
    print('Created 48 development +20 formal training jobs; enabled=false, allow_test=false')


if __name__ == '__main__':
    main()
