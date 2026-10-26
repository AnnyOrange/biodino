#!/usr/bin/env python3
"""Shared frozen native Faster R-CNN: fixed members, validation LR selection, five fits."""
import argparse,json,os,random,sys,time
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
os.environ['DINOV3_CODE_ROOT']=str(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from dinov3.eval.bio_detection.native import NativeDetectionDataset,build_detector,coco_bbox_metrics,sha256
from dinov3.eval.bio_detection.campaign import collate,seed_all
from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def evaluate(model,dataset,save=None):
 model.eval();predictions=[]
 with torch.inference_mode():
  for images,_ in torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0,collate_fn=collate):
   predictions.extend({k:v.cpu() for k,v in p.items()} for p in model([x.cuda() for x in images]))
 metrics=coco_bbox_metrics(dataset,predictions,dataset.classes)
 if save is not None:
  payload={'predictions':predictions,'records':dataset.records,'classes':dataset.classes,
    'schema':'One record per source image; boxes/labels/scores in native coordinates; true boxes/labels and source-image group in records'}
  t=save.with_suffix('.tmp');torch.save(payload,t);t.replace(save)
 return metrics

def model_for(spec,classes):
 if spec['kind']=='biodino':
  encoder=Dinov3CkptEncoder(Path(spec['checkpoint']),Path(spec['train_config']),'cuda:0',1,False,torch.bfloat16)
  body=encoder.model.backbone;layers=[body.n_blocks-1]
 else:
  from unified_paper_native_backbone import NativeFMBackbone
  body=NativeFMBackbone(spec['model']);layers=[0]
 return build_detector(body,layers,len(classes)+1,min_size=512,max_size=1024).cuda()

def fit(spec,manifest,seed,lr,directory,*,dataset_factory=None,model_factory=None,evaluator=None):
 directory.mkdir(parents=True,exist_ok=True);seed_all(seed)
 model=(model_factory or model_for)(spec,manifest['classes'])
 dataset_factory=dataset_factory or NativeDetectionDataset;evaluator=evaluator or evaluate
 train=dataset_factory(manifest,'train');val=dataset_factory(manifest,'val');val.classes=manifest['classes']
 generator=torch.Generator().manual_seed(seed)
 loader=torch.utils.data.DataLoader(train,batch_size=2,shuffle=True,num_workers=0,collate_fn=collate,generator=generator)
 optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=lr,momentum=.9,weight_decay=.0005)
 scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.1)
 state_path=directory/'training_state.pth';history=[]
 if state_path.exists():
  state=torch.load(state_path,map_location='cpu',weights_only=False)
  incompatible=model.load_state_dict(state['head_state'],strict=False)
  assert not incompatible.unexpected_keys and all(k.startswith('backbone.body.') for k in incompatible.missing_keys)
  optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);history=state['history']
  random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['loader_rng'])
 for epoch in range(len(history),10):
  model.train();losses=[];started=time.time()
  for images,targets in loader:
   targets=[{k:v.cuda() for k,v in target.items()} for target in targets]
   lossdict=model([x.cuda() for x in images],targets);loss=sum(lossdict.values())
   if not torch.isfinite(loss):
    dump(directory/'nonfinite_training.json',{'epoch':epoch+1,'step':len(losses),'seed':seed,'learning_rate':lr,'losses':{k:str(float(v.detach())) for k,v in lossdict.items()},'test_evaluated':False})
    raise ValueError('Nonfinite native detection loss')
   optimizer.zero_grad(set_to_none=True);loss.backward()
   if epoch==0 and not losses:
    if any(p.grad is not None for p in model.backbone.body.parameters()):raise ValueError('Encoder unexpectedly received a training gradient')
    grad=model.backbone.projection.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum():raise ValueError('Trainable native neck has no finite nonzero gradient')
    dump(directory/'first_train_step.json',{'encoder_frozen':True,'neck_gradient_finite_nonzero':True,'batch_size':len(images),'losses':{k:float(v.detach()) for k,v in lossdict.items()}})
    print('REAL TRAIN STEP',spec['model'],manifest['dataset'],seed,lr,float(loss.detach()),flush=True)
   optimizer.step();losses.append(float(loss.detach()))
  scheduler.step();metrics=evaluator(model,val) if len(val) else None
  history.append({'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'validation':metrics,'seconds':time.time()-started})
  state={'head_state':{k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.body.')},
   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'history':history,'seed':seed,'learning_rate':lr,
   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()}
  t=state_path.with_suffix('.tmp');torch.save(state,t);t.replace(state_path);dump(directory/'training_history.json',history)
  print('EPOCH',spec['model'],manifest['dataset'],seed,lr,epoch+1,metrics,flush=True)
 return model,history,state_path

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['bccd','bbbc041'],required=True);p.add_argument('--seed',type=int,choices=range(5),required=True);a=p.parse_args()
 torch.set_num_threads(2);cfg=json.loads((OUT/'native_detection_config.json').read_text());spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 if spec['kind']!='biodino':
  import unified_paper_native_backbone  # Import before fit resets the training RNG.
 source=Path(cfg['manifests'][a.dataset]['path'].replace('/mnt/huawei_deepcad/dinov3',str(ROOT)));assert sha256(source)==cfg['manifests'][a.dataset]['sha256'];manifest=json.loads(source.read_text())
 from unified_paper_resident_native_inputs import verify_resident_inputs
 verify_resident_inputs(a.dataset,source,manifest)
 out=OUT/'results'/a.model/'native_box'/a.dataset;out.mkdir(parents=True,exist_ok=True);seedout=out/f'seed{a.seed}';seedout.mkdir(exist_ok=True);target=seedout/'component_result.json'
 if target.exists():return
 selected=out/'selected_recipe.json'
 if not selected.exists():
  if a.seed!=0:raise ValueError('Seed0 validation search must finish before seed1-4')
  candidates=[]
  for lr in cfg['search']['learning_rate']:
   directory=out/'selection'/f'lr{lr:g}'
   try:
    model,history,state=fit(spec,manifest,0,lr,directory)
    candidates.append({'status':'complete','learning_rate':lr,'validation':history[-1]['validation'],'state':str(state)})
    del model
   except ValueError as error:
    if str(error)!='Nonfinite native detection loss':raise
    candidate={'status':'invalid_training_nonfinite','learning_rate':lr,'reason':str(error),'test_used':False}
    dump(directory/'candidate_failure.json',candidate);candidates.append(candidate)
   torch.cuda.empty_cache()
  valid=[c for c in candidates if c['status']=='complete' and np.isfinite(c['validation']['bbox_ap_50_95'])]
  if not valid:raise ValueError('Every prescribed LR failed; no valid native selection')
  winner=max(valid,key=lambda c:(c['validation']['bbox_ap_50_95'],-c['learning_rate']))
  dump(selected,{'selected':winner,'candidates':candidates,'selection_metric':'validation bbox_ap_50_95 at epoch10','invalid_candidate_rule':'Nonfinite training loss makes a candidate ineligible; retain failure evidence; never use test to resolve it; all-invalid task fails','test_used':False,'config_sha256':sha256(OUT/'native_detection_config.json')})
 selection=json.loads(selected.read_text());lr=selection['selected']['learning_rate']
 directory=Path(selection['selected']['state']).parent if a.seed==0 else seedout/'training'
 model,history,state=fit(spec,manifest,a.seed,lr,directory)
 test=NativeDetectionDataset(manifest,'test');test.classes=manifest['classes'];prediction=seedout/'test_predictions.pth';metrics=evaluate(model,test,prediction)
 row={'model':a.model,'checkpoint':spec['checkpoint'],'dataset':a.dataset,'task':'box_detection','fold':'fixed','seed':a.seed,'required_seeds':5,
  'protocol':cfg['protocol_id'],'metrics':metrics,'selection':str(selected),'selection_sha256':sha256(selected),'selected_learning_rate':lr,
  'epochs':10,'batch_size':2,'optimizer':'SGD','momentum':.9,'weight_decay':.0005,'scheduler':'StepLR step3 gamma0.1','head':'shared frozen-encoder FasterRCNN, projection256/RPN/RoI',
  'manifest':str(source),'manifest_sha256':sha256(source),'prediction_path':str(prediction),'prediction_sha256':sha256(prediction),'head_state':str(state),
  'head_state_sha256':sha256(state),'seed_semantics':cfg['seed_semantics'],'training_history':history,
  'input_adaptation':'Published FM normalization and channels; common stride16 feature grid; new4 published224 encoder input; own DINO original native512/1024 input; no label-coordinate changes',
  'grouping':manifest['grouping'],'limitation':manifest['limitation'],'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
 dump(target,row);print('COMPLETE NATIVE BOX',a.model,a.dataset,a.seed,metrics,flush=True)

if __name__=='__main__':main()
