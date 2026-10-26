"""Generate missing frozen spatial caches with the existing DINOv3 extractor."""
import argparse, gc, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def main():
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    parser=argparse.ArgumentParser();parser.add_argument('--cell-index',type=int,required=True)
    args=parser.parse_args()
    cell=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells'][args.cell_index]
    spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==cell['model'])
    recipe=json.loads(Path(cell['invocation']).read_text())
    if cell.get('source_file_fingerprints'):
        for filename, expected in cell['source_file_fingerprints'].items():
            with Path(filename).open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError(f'Resident segmentation source differs from canonical file: {filename}')
    from dinov3.eval.bio_segmentation import feature_extractor as extractor
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root=_resolve_data_root(Path(cell['input_spec']['data_root_base']),cell['dataset'])
    if cell['dataset']=='monuseg':
        # Historical common v4 cells explicitly override the loader's 24/6 default.
        # Bind to that same approved 30/7/14 manifest, including its source hash.
        original_recipe=json.loads(Path(recipe['template_invocation']).read_text())
        source_hashes=original_recipe['template_source_hashes'] if 'template_source_hashes' in original_recipe else original_recipe['dataset']['source_hashes']
        manifest=root/'monuseg2018_train30_extra7val_test14_manifest.json'
        approved=source_hashes[str(manifest)]
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==approved
        from dinov3.eval.bio_segmentation.datasets.monuseg_official import official_paths
        for part,count in recipe['expected_counts'].items():
            images,labels=official_paths(root,part,approved,manifest_file=manifest.name,counts=recipe['expected_counts'])
            assert len(images)==len(labels)==count
        os.environ.update(DINOV3_MONUSEG_MANIFEST_FILE=manifest.name,
            DINOV3_MONUSEG_MANIFEST_SHA256=approved,
            DINOV3_MONUSEG_COUNTS=json.dumps(recipe['expected_counts']))
        os.environ.pop('DINOV3_MONUSEG_LEGACY',None)
    cache=Path(cell['cell'])/'cache';cache.mkdir(parents=True,exist_ok=True)
    commands=[];observed={}
    for split in ['train','val','test']:
        target=Path(cell['caches'][split])
        if target.exists():
            with np.load(target) as z:previous_count=int(z['num_samples'])
            if previous_count!=recipe['expected_counts'][split]:
                rejected=target.parent/'rejected_membership';rejected.mkdir(exist_ok=True)
                archive=rejected/f'{split}_n{previous_count}.npz'
                if archive.exists():raise ValueError('Repeated invalid cache; inspect retained evidence')
                target.replace(archive)
                (rejected/f'{split}_reason.json').write_text(json.dumps({'actual':previous_count,'expected':recipe['expected_counts'][split],
                    'reason':'Default loader membership rejected before head training; retry binds original approved common manifest'},indent=2)+'\n')
        if not target.exists():
            argv=['feature_extractor','--dataset',cell['dataset'],'--data-root',str(root),
                '--checkpoint',spec['checkpoint'],'--train-config',spec['train_config'],
                '--output-dir',str(cache),'--split',split,'--dataset-split-protocol',cell['split'],
                '--img-size',str(cell['input_spec']['image_size']),'--resize-mode',cell['input_spec']['resize_mode'],
                '--batch-size','2','--autocast-dtype','bf16','--num-workers','2','--channel-policy','auto',
                '--chunked-cache','--no-compress-cache']
            original=sys.argv;sys.argv=argv
            try:extractor.main()
            finally:sys.argv=original
            commands.append(argv)
            files=list(cache.glob(f"{cell['dataset']}_{split}_*last1*.npz"))
            if len(files)!=1:raise ValueError(f'Ambiguous extraction outputs: {files}')
            files[0].replace(target)
            gc.collect();torch.cuda.empty_cache()
        with np.load(target) as z:
            count=int(z['num_samples']);assert int(z['n_layers'])==1
            assert count==recipe['expected_counts'][split],(split,count,recipe['expected_counts'])
            observed[split]={'path':str(target),'samples':count,'feature_batch_size':int(z['feature_batch_size'])}
    result=Path(cell['cache_completion'])
    data={'model':cell['model'],'dataset':cell['dataset'],'split':cell['split'],
        'checkpoint':spec['checkpoint'],'recipe':cell['invocation'],
        'recipe_sha256':hashlib.sha256(Path(cell['invocation']).read_bytes()).hexdigest(),
        'caches':observed,'commands':commands,'frozen_encoder':True,
        'adaptation':'Existing DINOv3 last-layer spatial extractor and exact shared split. Inference microbatch2 only; downstream B32 budgets unchanged.'}
    temp=result.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(result)
    print('COMPLETE FROZEN CACHES',cell['model'],cell['dataset'],flush=True)

if __name__=='__main__':main()
