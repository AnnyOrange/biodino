#!/usr/bin/env python3
"""Shared native opacity detector on fixed official RSNA test with patient purge."""
import argparse,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from dinov3.eval.bio_detection.rsna import RSNADataset,image_precision
from dinov3.eval.bio_detection.native import sha256,coco_bbox_metrics
from dinov3.eval.bio_detection.campaign import collate
from run_unified_paper_native_detection import fit,dump
OUT=ROOT/'outputs/02_eval_runs/unified_paper';MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json'

def evaluate(model,dataset,save=None):
 model.eval();preds=[]
 with torch.inference_mode():
  for images,_ in torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0,collate_fn=collate):preds.extend({k:v.cpu() for k,v in p.items()} for p in model([x.cuda() for x in images]))
 vals=[image_precision(r['boxes'],p['boxes'].numpy(),p['scores'].numpy()) for r,p in zip(dataset.records,preds)];valid=[x for x in vals if x is not None];metrics=coco_bbox_metrics(dataset,preds,dataset.classes);metrics.update(rsna_mean_precision=float(np.mean(valid)) if valid else 0.,scored_images=len(valid),negative_no_prediction_ignored=len(vals)-len(valid),evaluated_images=len(vals))
 if save is not None:
  torch.save(dict(predictions=preds,records=dataset.records,classes=dataset.classes,per_image_precision=vals),save)
 return metrics

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command();p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--seed',type=int,choices=range(5),required=True);a=p.parse_args();torch.set_num_threads(2)
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model);manifest=json.loads(MANIFEST.read_text());out=OUT/'results'/a.model/'rsna_pneumonia';seedout=out/f'seed{a.seed}';seedout.mkdir(parents=True,exist_ok=True);target=seedout/'component_result.json'
 if target.exists():return
 from unified_paper_resident_native_inputs import verify_resident_inputs
 verify_resident_inputs('rsna_pneumonia',MANIFEST,manifest)
 selected=out/'selected_recipe.json'
 if not selected.exists():
  assert a.seed==0;candidates=[]
  for lr in [.001,.005]:
   model,history,state=fit(spec,manifest,0,lr,out/'selection'/f'lr{lr:g}',dataset_factory=RSNADataset,evaluator=evaluate);candidates.append(dict(learning_rate=lr,validation=history[-1]['validation'],state=str(state)));del model;torch.cuda.empty_cache()
  winner=max(candidates,key=lambda x:(x['validation']['rsna_mean_precision'],-x['learning_rate']));dump(selected,dict(selected=winner,candidates=candidates,test_used=False,manifest_sha256=sha256(MANIFEST)))
 selection=json.loads(selected.read_text());assert selection['manifest_sha256']==sha256(MANIFEST);lr=selection['selected']['learning_rate'];directory=Path(selection['selected']['state']).parent if a.seed==0 else seedout/'training'
 model,history,state=fit(spec,manifest,a.seed,lr,directory,dataset_factory=RSNADataset,evaluator=evaluate);test=RSNADataset(manifest,'test');prediction=seedout/'test_predictions.pth';metrics=evaluate(model,test,prediction)
 dump(target,dict(model=a.model,checkpoint=spec['checkpoint'],dataset='xray_chest',task='box_detection',seed=a.seed,protocol='rsna-native-fixed-patient-purged-v1',metrics=metrics,manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),prediction_path=str(prediction),prediction_sha256=sha256(prediction),epochs=10,batch_size=2,selection=str(selected),head_state=str(state),training_history=history,grouping=manifest['grouping'],limitation=manifest['limitation']));print('COMPLETE RSNA',a.model,a.seed,metrics,flush=True)
if __name__=='__main__':main()
