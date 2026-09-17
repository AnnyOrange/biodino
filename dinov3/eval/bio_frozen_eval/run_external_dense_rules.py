"""FM14 dense entrypoint, using the same cached probe as HS0/HS6."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from .external_fm_protocol import MODELS, SEGMENTATION, PANNUKE_PROTOCOLS, dense_recipe, validate_best_head

BENCH = Path('/mnt/huawei_deepcad/benchmark_model')
HOLD = BENCH/'fair_plot_20260915/FM_ALIGNMENT_HOLD_20260917.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--datasets', nargs='+', choices=SEGMENTATION, default=[d for d in SEGMENTATION if d!='bbbc038'])
    parser.add_argument('--comparison-view', choices=('primary-last','dataset-best'), default='primary-last')
    parser.add_argument('--observation', action='store_true')
    parser.add_argument('--benchmark-root', type=Path, default=Path('/mnt/huawei_deepcad/benchmark'))
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    plans=[]
    for dataset in args.datasets:
        folds=PANNUKE_PROTOCOLS if dataset=='pannuke' else (None,)
        for model in args.models:
            for fold in folds:
                recipe=dense_recipe(model,dataset,args.comparison_view,split_protocol=fold,observation=args.observation)
                # Depth is established from the loaded architecture, not guessed here.
                recipe['layer_resolution_status']='PENDING_ARCHITECTURE_INSPECTION'
                if model not in ('cytoself','cytoimagenet'):
                    recipe['resolved_layers']=None
                    recipe['layer_indices']=[]
                    recipe['fallback_reason']=''
                plans.append(recipe)
    if args.plan_only:
        print(json.dumps(plans,indent=2));return 0
    if HOLD.exists():raise RuntimeError('FM alignment hold: experiments remain stopped pending complete preflight')
    if args.out_root is None:parser.error('--out-root is required for experiments')

    import numpy as np
    import torch
    from .external_fm_features import RuleFMFeatures
    from ..bio_segmentation.feature_extractor import _build_dataset
    from ..bio_segmentation.linear_probe import DATASET_CONFIGS, run_cached_linear_probe
    from ..bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root

    spec=importlib.util.spec_from_file_location('rule_dense_cache',BENCH/'run_dense_probe_benchmark.py')
    cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
    for plan in plans:
        np.random.seed(0);torch.manual_seed(0)
        requested='last' if args.comparison_view=='primary-last' else SEGMENTATION[plan['dataset']][2]
        extractor=RuleFMFeatures(plan['model'],args.device,requested)
        depth=len(extractor.blocks) if extractor.blocks is not None else None
        recipe=dense_recipe(plan['model'],plan['dataset'],args.comparison_view,depth,
                            plan['split_protocol'],args.observation)
        root=args.out_root/recipe['scope']/args.comparison_view/plan['dataset']/plan['model']/plan['split_protocol']
        root.mkdir(parents=True,exist_ok=True)
        manifest=root/'recipe.json'
        if manifest.exists() and json.loads(manifest.read_text())!=recipe:
            raise RuntimeError('Output contains a different recipe; use a fresh output directory')
        manifest.write_text(json.dumps(recipe,indent=2)+'\n')
        extractor.n_layers=len(extractor.layer_record['layer_indices']) or 1
        cache.OUT_ROOT=root
        cache.DenseFeatureExtractor=lambda *a,**k: extractor
        data_root=_resolve_data_root(args.benchmark_root/'segmentation',plan['dataset'])
        cache.build_bioseg_dataset=lambda dataset,split,size: _build_dataset(
            dataset,str(data_root),split,size,resize_mode=recipe['resize_mode'],
            augment=False,do_normalize=False,multichannel=True,
            dataset_split_protocol=recipe['split_protocol'])
        settings=SimpleNamespace(img_size=recipe['image_size'],device=args.device,feature_canonical=True,
            overwrite_cache=True,extract_batch_size=32,num_workers=2,max_feature_side=0)
        try:
            caches={split:cache.extract_cache(settings,plan['model'],plan['dataset'],split) for split in ('train','val','test')}
        finally:
            extractor.close()
            cache.DenseFeatureExtractor=None
        del extractor;torch.cuda.empty_cache()
        cfg=DATASET_CONFIGS[plan['dataset']]
        for budget in recipe['probe_budgets']:
            for seed in recipe['seeds']:
                output=root/f'E{budget}'/f'seed{seed}'
                if (output/'results.json').exists():
                    raise RuntimeError('Existing result requires evidence audit; it is not silently overwritten')
                run_cached_linear_probe(train_cache=str(caches['train']),val_cache=str(caches['val']),
                    test_cache=str(caches['test']),output_dir=str(output),**cfg,
                    epochs=budget,lr=recipe['lr'],batch_size=recipe['probe_batch_size'],
                    weight_decay=recipe['weight_decay'],dropout=recipe['dropout'],num_workers=2,
                    eval_every=1,seed=seed,class_weight_mode=recipe['class_weight_mode'])
                result=json.loads((output/'results.json').read_text())
                validate_best_head(result['_meta'])
    return 0


if __name__=='__main__':raise SystemExit(main())
