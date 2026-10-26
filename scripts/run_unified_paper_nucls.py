#!/usr/bin/env python3
"""Shared NuCLS hybrid head; publisher fold1 selects LR, folds2-5 evaluate."""
import argparse,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
import run_unified_paper_native_detection as common
from dinov3.eval.bio_detection.nucls import NuCLSDataset,build_hybrid_detector,compact_prediction,native_metrics
from dinov3.eval.bio_detection.native import sha256

def model_for(spec,classes):
 if spec['kind']=='biodino':
  encoder=common.Dinov3CkptEncoder(Path(spec['checkpoint']),Path(spec['train_config']),'cuda:0',1,False,torch.bfloat16)
  body=encoder.model.backbone;layers=[body.n_blocks-1]
 else:
  from unified_paper_native_backbone import NativeFMBackbone
  body=NativeFMBackbone(spec['model']);layers=[0]
 return build_hybrid_detector(body,layers).cuda()

def evaluate(model,dataset,save=None):
 model.eval();predictions=[]
 with torch.inference_mode():
  for images,_ in torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0,collate_fn=common.collate):
   output=model([x.cuda() for x in images])
   predictions.extend(compact_prediction(x) for x in output)
 metrics,rows=native_metrics(dataset,predictions)
 if save is not None:
  payload={'predictions':predictions,'records':dataset.records,'classes':dataset.classes,'sample_metrics':rows,
   'schema':'Source FOV/patient identifiers and full true objects; scored boxes/classes; lossless RLE of predicted masks thresholded at .5, cropped native coordinates'}
  temp=save.with_suffix('.tmp');torch.save(payload,temp);temp.replace(save)
 return metrics

def fit(spec,manifest,fold,seed,lr,path):
 return common.fit(spec,manifest,seed,lr,path,dataset_factory=lambda m,s:NuCLSDataset(m,fold,s),model_factory=model_for,evaluator=evaluate)

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--fold',type=int,choices=[2,3,4,5],required=True);parser.add_argument('--seed',type=int,choices=range(5),required=True);a=parser.parse_args()
 torch.set_num_threads(2);config_path=OUT/'nucls_native_config.json';cfg=json.loads(config_path.read_text())
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 if spec['kind']!='biodino':
  import unified_paper_native_backbone  # Import before fit resets the training RNG.
 source=Path(cfg['manifest'].replace('/mnt/huawei_deepcad/dinov3',str(ROOT)));assert sha256(source)==cfg['manifest_sha256'];manifest=json.loads(source.read_text())
 from unified_paper_resident_native_inputs import verify_resident_inputs
 verify_resident_inputs('nucls_qc',source,manifest)
 base=OUT/'results'/a.model/'nucls_qc';out=base/f'fold{a.fold}'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 target=out/'component_result.json'
 if target.exists():return
 selected=base/'selected_recipe.json'
 if not selected.exists():
  if (a.fold,a.seed)!=(2,0):raise ValueError('NuCLS fold1 selection must complete before dependent fits')
  candidates=[]
  for lr in cfg['learning_rates']:
   directory=base/'selection_fold1'/f'lr{lr:g}'
   try:
    model,history,state=fit(spec,manifest,1,0,lr,directory)
    candidates.append({'status':'complete','learning_rate':lr,'validation':history[-1]['validation'],'state':str(state)});del model
   except ValueError as error:
    if str(error)!='Nonfinite native detection loss':raise
    candidate={'status':'invalid_training_nonfinite','learning_rate':lr,'reason':str(error),'test_used':False}
    common.dump(directory/'candidate_failure.json',candidate);candidates.append(candidate)
   torch.cuda.empty_cache()
  valid=[c for c in candidates if c['status']=='complete' and np.isfinite(c['validation']['bbox_ap_50_95'])]
  if not valid:raise ValueError('Every prescribed NuCLS LR failed')
  winner=max(valid,key=lambda c:(c['validation']['bbox_ap_50_95'],-c['learning_rate']))
  common.dump(selected,{'selected':winner,'candidates':candidates,'selection_fold':1,'selection_metric':'final epoch10 validation bbox AP',
    'fold2_5_scores_used':False,'config_sha256':sha256(config_path),'invalid_candidate_rule':cfg['invalid_candidate_rule']})
 selection=json.loads(selected.read_text());lr=selection['selected']['learning_rate']
 model,history,state=fit(spec,manifest,a.fold,a.seed,lr,out/'training')
 test=NuCLSDataset(manifest,a.fold,'test');prediction=out/'test_predictions.pth';metrics=evaluate(model,test,prediction)
 row={'model':a.model,'checkpoint':spec['checkpoint'],'dataset':'nucls_qc','task':'nuclei_detection_segmentation','fold':a.fold,'seed':a.seed,'required_seeds':5,
  'protocol':cfg['protocol_id'],'metrics':metrics,'epochs':10,'batch_size':2,'frozen_encoder':True,'selection':str(selected),'selection_sha256':sha256(selected),
  'selected_learning_rate':lr,'config':str(config_path),'config_sha256':sha256(config_path),'manifest':str(source),'manifest_sha256':sha256(source),
  'prediction_path':str(prediction),'prediction_sha256':sha256(prediction),'head_state':str(state),'head_state_sha256':sha256(state),
  'training_history':history,'seed_semantics':cfg['seed_semantics'],'grouping':'TCGA patient; official institution/slide folds',
  'limitation':cfg['limitations'],'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
 common.dump(target,row);print('COMPLETE NUCLS',a.model,a.fold,a.seed,metrics,flush=True)
if __name__=='__main__':main()
