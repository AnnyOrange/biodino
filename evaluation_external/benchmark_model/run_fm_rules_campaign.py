"""Resource-managed FM14 campaign, pinned to the common DINOv3 probe code."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('DINOV3_CODE_ROOT','/mnt/huawei_deepcad/dinov3'))
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(Path(__file__).parent)]
import run_shared_frozen_stage1_20260917 as queue
from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head

original_command=queue.command
original_validator=queue.validate_cell


def command(task,output,benchmark):
    spec=task['dataset']
    if spec['task']!='segmentation':return original_command(task,output,benchmark)
    result=[sys.executable,'-u','-m','dinov3.eval.bio_frozen_eval.run_external_dense_rules',
        '--models',task['asset']['model_id'],'--datasets',spec['dataset'],
        '--comparison-view',spec['comparison_view'],'--benchmark-root',str(benchmark),
        '--out-root',str(output),'--device','cuda:0','--extract-batch-size','32','--probe-batch-size','32']
    if spec['dataset']=='pannuke':result+=['--split-protocol',spec['split']]
    if spec['dataset']=='bbbc038':result+=['--observation']
    return result


def validate(task,directory,invocation):
    spec=task['dataset']
    if spec['task']!='segmentation':return original_validator(task,directory,invocation)
    import math
    recipes=list(directory.rglob('recipe.json'))
    if len(recipes)!=1:raise ValueError('Exactly one resolved dense recipe required')
    recipe=json.loads(recipes[0].read_text())
    for key,value in (('model',task['asset']['model_id']),('dataset',spec['dataset']),
                      ('image_size',spec['image_size']),('resize_mode',spec['resize_mode']),
                      ('split_protocol',spec['split']),('comparison_view',spec['comparison_view'])):
        if recipe[key]!=value:raise ValueError(f'Dense recipe mismatch: {key}')
    results={}
    for budget in (20,50):
        for seed in (0,1,2):
            path=recipes[0].parent/f'E{budget}'/f'seed{seed}'/'results.json'
            result=json.loads(path.read_text());validate_best_head(result['_meta'])
            if result.get('error') or not math.isfinite(result['test']['mDice']):raise ValueError('Invalid dense metric')
            meta=result['_meta']
            if meta['probe_epochs']!=budget or meta['seed']!=seed:raise ValueError('Wrong probe budget/seed')
            if meta['full_train_samples']!=spec['counts']['train']:raise ValueError('Wrong dense training count')
            results[str(path)]=queue.sha256(path)
    report=dict(status='VALID_COMPLETE',scope='INDEPENDENT_COMPONENT_ONLY',full_v3_aggregate_allowed=False,
        result_sha256=results,recipe_sha256=queue.sha256(recipes[0]),input_fingerprint=queue.fingerprint(invocation),
        expected_counts=spec['counts'],validator_commit=invocation['git_commit'])
    queue.save(directory/'validation_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--gpus',nargs='+',type=int,required=True)
    parser.add_argument('--target-per-gpu',type=int,default=5)
    parser.add_argument('--max-host-jobs',type=int,default=40)
    parser.add_argument('--max-global-jobs',type=int,default=80)
    parser.add_argument('--task-family',choices=('mixed','frozen','segmentation'),default='mixed')
    args=parser.parse_args()
    manifest=json.loads((args.output/'campaign_manifest.json').read_text())
    for path,digest in manifest['external_source_hashes'].items():
        if queue.sha256(path)!=digest:raise RuntimeError(f'Unregistered external code change: {path}')
    queue.command=command;queue.validate_cell=validate
    queue.worker(args)


if __name__=='__main__':main()
