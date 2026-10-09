"""Adapt the existing full native CTC implementation to one method checkpoint."""
import argparse,json
from pathlib import Path
import torch
import run_ctc_native_full_hs6 as native

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=native.ROOT
    native.PLAN=root/'Evaluation Rules/plans/deepcad_method_full_v4_fleet_20260928.md'
    candidate=dict(model=a.arm,family='hs6_l5_method',checkpoint_step=a.step,
                   checkpoint_path=str(a.checkpoint),config_path=str(a.config),layers=[23])
    native.candidates=lambda:[candidate]
    data_path=native.CACHE/'data_manifest.json';data=native.load_cache_manifest(data_path)
    assert data['source_split_manifest_sha256']=='7a0f5bc2579f6ae22a2b8f2b16103530cd678466a4b2bbd1f617692b8e3cac27'
    assert len(data['domains'])==20 and len(data['folds'])==5
    assert native.base.EPOCHS==50 and native.base.TRAIN_BATCH_SIZE==8
    manifest_path=native.prepare_campaign(a.output,data_path)
    manifest=json.loads(manifest_path.read_text())
    # Correct the historical fixed-12 description before this manifest is used
    # to fit any head; the native prediction/scoring code is unchanged.
    description='One method checkpoint, all 5 fixed folds and all 20 CTC domains'
    if manifest['candidate_selection']!=description:
        manifest['candidate_selection']=description;native.atomic_json(manifest_path,manifest)
    assert len(manifest['models'])==1 and manifest['models'][0]['model']==a.arm
    torch.cuda.set_device(0)
    result=native.run_candidate(a.output,manifest_path,data_path,manifest['models'][0],torch.device('cuda:0'))
    assert len(result['folds'])==5 and {r['domain'] for r in result['domain_rows']}==set(native.CTC_DOMAINS)
    assert all(int(r['ctc_metrics']['Valid'])==1 for r in result['domain_rows'])
    native.atomic_json(a.output/'validation_report.json',dict(status='VALID_COMPLETE',expected_models=1,valid_models=1,
                       expected_folds=5,expected_domains=20,campaign_manifest_sha256=native.sha256(manifest_path)))

if __name__=='__main__':main()
