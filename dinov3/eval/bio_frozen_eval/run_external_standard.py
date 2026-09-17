"""Strict shared frozen components for external FMs with a real CLS token."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path('/mnt/huawei_deepcad/benchmark_model')
SUPPORTED = ('dinov2','mae','phikon2','uni','virchow2','gigapath','hoptimus0')


def cls_patch_feature(tokens, prefix):
    if tokens.ndim != 3 or prefix < 1 or tokens.shape[1] <= prefix:
        raise ValueError('A real CLS token and nonempty final patch tokens are required')
    return F.normalize(torch.cat((tokens[:,0].float(),tokens[:,prefix:].float().mean(1)),dim=1),dim=1)


class StandardExternalEncoder:
    def __init__(self, name, size, resize):
        if name not in SUPPORTED: raise ValueError('External native readout is not certified for this strict CLS+patch comparison: '+name)
        sys.path.insert(0,str(BENCH))
        spec = importlib.util.spec_from_file_location('shared_dense_loaders',BENCH/'run_dense_probe_benchmark.py')
        dense = importlib.util.module_from_spec(spec); spec.loader.exec_module(dense)
        self.native = dense.DenseFeatureExtractor(name,'cuda:0',canonical=True)
        self.name = name; self.size = size; self.resize = resize
        self.transform = transforms.Compose([transforms.Resize(resize,interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.CenterCrop(size),transforms.ToTensor()])
        if name == 'mae': self.native.model.config.mask_ratio = 0.0
        for parameter in self.native.model.parameters(): parameter.requires_grad_(False)
        self.encoder_input_size = None

    @torch.inference_mode()
    def encode_images(self, images):
        from run_external_fm_linear_probe import ExternalEncoderAdapter
        rgb = [ExternalEncoderAdapter._tensor_first3_to_pil(x) if torch.is_tensor(x) else x.convert('RGB') for x in images]
        x = self.native._resize(torch.stack([self.transform(image) for image in rgb])).cuda()
        self.encoder_input_size = list(x.shape[-2:])
        x = self.native._norm(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if self.native.spec.kind == 'transformers':
                model = self.native.model
                kwargs = dict(pixel_values=x,interpolate_pos_encoding=True)
                if self.name == 'mae':
                    patches = (x.shape[-2]//self.native.patch_size)*(x.shape[-1]//self.native.patch_size)
                    kwargs['noise'] = torch.arange(patches,device=x.device,dtype=torch.float32).expand(x.shape[0],-1)
                try: output = model(**kwargs)
                except TypeError:
                    del kwargs['interpolate_pos_encoding']
                    output = model(**kwargs)
                tokens = output.last_hidden_state
                prefix = 1 + int(getattr(model.config,'num_register_tokens',0))
            elif self.native.spec.kind == 'timm':
                tokens = self.native.model.forward_features(x)
                prefix = int(self.native.model.num_prefix_tokens)
            else: raise ValueError('Unsupported strict external token architecture')
        return cls_patch_feature(tokens,prefix).cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=SUPPORTED,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--task',choices=('classification','regression','retrieval'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--num-workers',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args = parser.parse_args()
    if (args.batch_size,args.num_workers,args.seed)!=(64,2,0): raise ValueError('Rules require batch64/workers2/seed0')
    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_shared_frozen_stage1_20260917 as queue
    import run_external_fm_linear_probe as probe
    from .run_classification import resolve_image_size,resolve_dataset_resize_size
    invocation = json.loads((args.output/'invocation_manifest.json').read_text())
    for path,digest in invocation.get('external_source_hashes',{}).items():
        if queue.sha256(path)!=digest: raise ValueError('External source changed after campaign registration')
    size = resolve_image_size(args.dataset,'best',224) if args.task!='retrieval' else 224
    resize = resolve_dataset_resize_size(args.dataset,size,0) if args.task!='retrieval' else 256
    encoder = StandardExternalEncoder(args.model,size,resize)
    settings = SimpleNamespace(model=args.model,output_dir=str(args.output),feature_root=None,
        benchmark_root=args.benchmark,max_samples=None,max_per_class=None,batch_size=64,num_workers=2,
        overwrite_features=True,overwrite_results=True,no_save_features=True,save_paths=False,
        train_fraction=.8,seed=0)
    if args.task == 'retrieval':
        from . import run_retrieval_clustering as retrieval
        settings = retrieval.parse_args(['--checkpoint',str(probe.MODEL_REGISTRY[args.model].path),
            '--train-config','/external-not-used','--output-dir',str(args.output),'--datasets',args.dataset,
            '--batch-size','64','--num-workers','2','--seed','0','--no-save-features','--overwrite-results','--overwrite-features'])
        settings.benchmark_root=args.benchmark; settings.model_name=args.model
        retrieval.Dinov3CkptEncoder=lambda **kwargs: encoder
        if retrieval.main(['--checkpoint',str(probe.MODEL_REGISTRY[args.model].path),
            '--train-config','/external-not-used','--output-dir',str(args.output),'--datasets',args.dataset,
            '--benchmark-root',str(args.benchmark),'--model-name',args.model,
            '--batch-size','64','--num-workers','2','--seed','0','--no-save-features','--overwrite-results','--overwrite-features']):
            raise RuntimeError('Native retrieval component failed')
        row = json.loads((args.output/'last_result.json').read_text())
    else:
        row = probe.evaluate_dataset(settings,encoder,args.dataset)
    targets = row.get('rows',[row])
    for target in targets:
        target.update(checkpoint=str(probe.MODEL_REGISTRY[args.model].path),batch_size=64,seed=0,
            image_size=size,resize_size=resize,encoder_input_size=encoder.encoder_input_size,
            encoder_preprocess='dataset-best/model-published-normalization/nearest-legal-patch-grid',
            feature_layers='final-cls-plus-final-patch-mean',dtype='bf16',channel_policy='auto',channel_mapping='deterministic-first3-repeat-if-needed',
            channel_tta_samples=8,channel_policy_seed=0)
    queue.save(args.output/'last_result.json',row)


if __name__ == '__main__': main()
