#!/usr/bin/env python3
"""Five-seed mean/SD and independent-unit bootstrap intervals (not ensembles)."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dinov3.eval.ood_transfer.common import process_policy, read_json, save_json, sha256
from dinov3.eval.ood_transfer.evaluate import score


class Records:
    def __len__(self):return len(self.records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=10000)
    p.add_argument('--bootstrap-seed',type=int,default=20261110)
    args=p.parse_args();process_policy()
    root=Path(args.campaign);cfg=read_json(root/'campaign.json');report={}
    for branch in cfg['branches']:
        runs=[root/'runs'/f'{branch}_formal_seed{seed}' for seed in range(5)]
        if not all((r/'test_result.json').exists() for r in runs):
            report[branch]=dict(status='INCOMPLETE_FIVE_SEEDS',complete_seeds=[i for i,r in enumerate(runs) if (r/'test_result.json').exists()])
            continue
        results=[read_json(r/'test_result.json') for r in runs]
        fingerprints={(r['identity']['checkpoint_sha256'],r['identity']['recipe_sha256'],r['test_manifest_sha256'],r['identity']['code']['commit']) for r in results}
        if len(fingerprints)!=1 or [r['identity']['seed'] for r in results]!=list(range(5)):
            raise ValueError('Cannot pool different models/recipes/test sets/code or seeds')
        task=cfg['branches'][branch]['task']
        key={'bone_age':'mae_months','pneumonia':'rsna_mean_precision','bbbc041':'bbox_ap_50_95'}[task]
        values=[r['metrics'][key] for r in results]
        row=dict(status='FIVE_SEEDS_COMPLETE',metric=key,values=values,mean=float(np.mean(values)),
                 sd=float(np.std(values,ddof=1)),n_downstream_seeds=5,n_pretrained_checkpoints=1,
                 result_sha256=[sha256(r/'test_result.json') for r in runs])
        if args.bootstrap_replicates:
            payloads=[]
            for r,result in zip(runs,results):
                if sha256(r/'test_predictions.pth')!=result['predictions_sha256']:raise ValueError('Predictions changed')
                payloads.append(torch.load(r/'test_predictions.pth',map_location='cpu',weights_only=False))
            ids=[[r['sample_id'] for r in payload['records']] for payload in payloads]
            if any(value!=ids[0] for value in ids[1:]):raise ValueError('Seed sample order differs')
            groups=defaultdict(list)
            for i,r in enumerate(payloads[0]['records']):groups[r.get('group_id',r['sample_id'])].append(i)
            group_indices=list(groups.values());rng=np.random.default_rng(args.bootstrap_seed);samples=[]
            recipe=read_json(runs[0]/'runtime_recipe.json')
            for replicate in range(args.bootstrap_replicates):
                chosen=rng.integers(0,len(group_indices),size=len(group_indices))
                indices=[i for g in chosen for i in group_indices[g]]
                seed_scores=[]
                for payload,result in zip(payloads,results):
                    data=Records();data.records=[payload['records'][i] for i in indices]
                    data.classes=payload['classes'];data.recipe=recipe
                    predictions=[payload['predictions'][i] for i in indices]
                    seed_scores.append(score(data,predictions,result['metrics'].get('score_threshold'))[key])
                samples.append(float(np.mean(seed_scores)))
                if replicate%100==0:print('BOOTSTRAP',branch,replicate,args.bootstrap_replicates,flush=True)
            row.update(bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),bootstrap_replicates=len(samples),
                       bootstrap_seed=args.bootstrap_seed,independent_groups=len(groups),
                       interval_scope='Test-unit uncertainty conditional on the five fixed trained runs; SD reports optimizer variability separately',
                       group_limitation='Uses only available group IDs; case/image units do not certify patient/slide independence')
        report[branch]=row
        save_json(root/'summary.json',report)
    save_json(root/'summary.json',report)


if __name__=='__main__':main()
