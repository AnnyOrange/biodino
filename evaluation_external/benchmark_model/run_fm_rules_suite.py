"""Reviewed replacement for legacy classification/regression/retrieval CLIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(os.environ.get('DINOV3_CODE_ROOT','/mnt/huawei_deepcad/dinov3'))
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_shared_frozen_stage1_20260917 as queue
from benchmark_eval.encoders import MODEL_REGISTRY
from dinov3.eval.bio_frozen_eval.external_fm_protocol import MODELS


def main(family='classification',argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--datasets','--dataset',nargs='+')
    parser.add_argument('--benchmark-root',type=Path,default=Path('/mnt/huawei_deepcad/benchmark'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    protocol=json.loads((ROOT/'Evaluation Rules/protocol_v3.json').read_text())
    allowed={name:task for tier in ('tier_a','tier_b') for task in ('classification','regression','retrieval') for name in protocol[tier][task]}
    names=args.datasets or [name for name,task in allowed.items() if task==family and name!='rxrx3-core']
    if any(name not in allowed or name=='rxrx3-core' for name in names):parser.error('Use registered frozen datasets; RxRx3 requires its dedicated compact3 entrypoint')
    if (args.batch_size,args.num_workers,args.seed)!=(64,2,0):parser.error('New frozen evaluations require batch64/workers2/seed0')
    plans=[]
    for name in names:
        task=allowed[name]
        spec=queue.retrieval_preflight(name,args.benchmark_root) if task=='retrieval' else queue.dataset_preflight(name,task,args.benchmark_root)
        for model in args.models:plans.append(dict(asset=dict(kind='external',arm='fm_'+model,checkpoint_id='pretrained',
            model_id=model,path=str(MODEL_REGISTRY[model].path),config=''),dataset=spec))
    if args.plan_only:print(json.dumps(plans,indent=2));return 0
    if (Path(__file__).parent/'fair_plot_20260915/FM_ALIGNMENT_HOLD_20260917.json').exists():raise RuntimeError('FM alignment hold')
    if queue.git('status','--porcelain'):raise RuntimeError('Use a clean fixed DINOV3_CODE_ROOT for formal evaluation')
    hashes=json.loads((ROOT/'dinov3/eval/bio_frozen_eval/external_fm_source_hashes.json').read_text())['files']
    queue.save(args.output_dir/'campaign_manifest.json',dict(git_commit=queue.git('rev-parse','HEAD'),
        tasks=plans,full_v3_aggregate_allowed=False,external_source_hashes=hashes))
    for task in plans:
        dataset=task['dataset'];model=task['asset']['model_id']
        output=args.output_dir/dataset['task']/dataset['dataset']/model
        if (output/'last_result.json').exists():raise RuntimeError('Existing result needs audit; not overwritten')
        invocation=dict(git_commit=queue.git('rev-parse','HEAD'),host=os.uname().nodename,
            checkpoint=queue.checkpoint_record(args.output_dir,task['asset']),dataset=dataset,
            external_source_hashes=hashes,batch_size=64,seed=0,num_workers=2)
        queue.save(output/'invocation_manifest.json',invocation)
        subprocess.run(queue.command(task,output,str(args.benchmark_root)),check=True,cwd=ROOT)
        queue.validate_cell(task,output,invocation)
    return 0
