#!/usr/bin/env python3
"""Frozen slice encoder + shared 3D centroid head on the PAC-MAP official split.

This is a project-defined frozen-backbone probe, NOT a reproduction of PAC-MAP
training. No LN18 weak pretraining. All recipe choices are locked before test.
"""
import argparse, hashlib, json, os, sys, time, socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
U=ROOT/'outputs/02_eval_runs/unified_paper'
MANIFEST=U/'spheroid_official_manifest.json'
PROTOCOL='spheroid-official-patches-frozen-slice-3d-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def prepare():
 import numpy as np
 audit=json.loads((U/'spheroid_spatial_audit.json').read_text())
 records=audit['records'];train=[i for i,r in enumerate(records) if r['split']=='train'];test=[i for i,r in enumerate(records) if r['split']=='test']
 order=np.random.RandomState(0).permutation(train).tolist()
 for r in records:
  assert sha(r['image'])==r['sha256'];r['points_sha256']=sha(r['points'])
 cfg=dict(protocol=PROTOCOL,records=records,splits=dict(train=order[:24],val=order[24:],test=test),seeds=list(range(5)),
  epochs=150,patience=10,lr=1e-4,batch_size=4,microbatch=1,selection='minimum validation MSE; earliest tie; threshold selected on validation F1 only',
  thresholds=[.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9],target_sigma_um=2.,
  head='Conv3d C->32 at native grid; trilinear64; Conv3d32->16; trilinear128; Conv3d16->8->1; sigmoid; trilinear native256',
  loss='unweighted MSE to max-composed isotropic Gaussian centroids sigma2um',
  augmentation='none',optimizer='Adam lr1e-4 betas0.9,0.999 eps1e-8 wd0',
  preprocessing='Released normalized intensity clipped0..1, each Z plane replicated RGB, no crop/rescale, ImageNet norm for BioDINO; final spatial layer; published FM adapter for FM',
  matching_um=5.,border_um=5.,nms_um=5.,official_split=True,spheroid_disjoint=False,
  split_note='Author 30 training /18 test members unchanged; fixed seed0 24/6 train/val for every model and head seed. Shared spheroids and overlapping pixels retained by user authorization.',
  method_note='New shared frozen feature probe, not author PAC-MAP UNet training; no LN18 weak pretraining; five downstream seeds, not backbone repeats',
  sources=['https://zenodo.org/records/11636385','https://github.com/DeVosLab/PAC-MAP/blob/main/configs/SH-SY5Y.py'])
 assert len(test)==18 and len(order)==30
 if MANIFEST.exists():assert json.loads(MANIFEST.read_text())==cfg
 else:dump(MANIFEST,cfg)
 return cfg

def main():
 p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--prepare',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.prepare:print(prepare()['splits']);return
 if not a.self_test:
  from unified_paper_gpu_admission import reserve_for_current_command
  reserve_for_current_command()
 import numpy as np,pandas as pd,tifffile,torch
 from torch import nn
 from torch.nn import functional as F
 from scipy.ndimage import maximum_filter
 from dinov3.eval.bio_detection.spheroid import VOXEL_UM,SHAPE,centroid_metrics
 torch.set_num_threads(2)
 class Head(nn.Module):
  def __init__(self,c):
   super().__init__();self.a=nn.Conv3d(c,32,1);self.b=nn.Conv3d(32,16,3,padding=1);self.c=nn.Conv3d(16,8,3,padding=1);self.d=nn.Conv3d(8,1,1)
  def forward(self,x):
   x=F.gelu(self.a(x));x=F.interpolate(x,size=(46,64,64),mode='trilinear',align_corners=False);x=F.gelu(self.b(x))
   x=F.interpolate(x,size=(46,128,128),mode='trilinear',align_corners=False);x=torch.sigmoid(self.d(F.gelu(self.c(x))))
   return F.interpolate(x,size=(46,256,256),mode='trilinear',align_corners=False)
 def target(points):
  out=np.zeros(tuple(SHAPE),np.float32);radius=np.ceil(6/VOXEL_UM).astype(int)
  for pt in points:
   lo=np.maximum(0,np.floor(pt).astype(int)-radius);hi=np.minimum(SHAPE,np.ceil(pt).astype(int)+radius+1)
   axes=[(np.arange(l,h)-v)*s for l,h,v,s in zip(lo,hi,pt,VOXEL_UM)]
   g=np.exp(-sum(x*x for x in np.meshgrid(*axes,indexing='ij'))/8).astype(np.float32)
   sl=tuple(slice(l,h) for l,h in zip(lo,hi));out[sl]=np.maximum(out[sl],g)
  return out
 def peaks(volume,threshold):
  # 3x3x3 local maxima followed by physical Euclidean greedy NMS.
  mask=(volume>=threshold)&(volume==maximum_filter(volume,size=3,mode='constant',cval=-1))
  coords=np.argwhere(mask);values=volume[tuple(coords.T)]
  order=np.argsort(-values,kind='stable');kept=[];physical=[]
  for ix in order:
   v=coords[ix]*VOXEL_UM
   if physical and np.any(np.sum((np.asarray(physical)-v)**2,axis=1)<25.):continue
   kept.append(coords[ix]);physical.append(v)
  return np.asarray(kept,dtype=float).reshape(-1,3)
 def combine(rows):
  tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
  return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)
 if a.self_test:
  pts=np.array([[12.,80.,80.],[25.,170.,170.]])
  pred=peaks(target(pts),.5);assert centroid_metrics(pts,pred)['f1']==1.
  assert centroid_metrics(pts,[])['f1']==0.
  assert centroid_metrics([],pred)['fp']==2
  assert centroid_metrics([[0,10,10]],[[0,10,10]])['tp']==0
  print('PASS Gaussian -> physical NMS -> border/matching, missing/false positives');return
 cfg=json.loads(MANIFEST.read_text());assert cfg['protocol']==PROTOCOL
 spec=next(m for m in json.loads((U/'config.json').read_text())['models'] if m['model']==a.model)
 out=U/'results'/a.model/'spheroid_lsfm';out.mkdir(parents=True,exist_ok=True)
 if (out/'component_result.json').exists():return
 cache=out/'cache';cache.mkdir(exist_ok=True)
 identity=dict(manifest_sha256=sha(MANIFEST),checkpoint=spec['checkpoint'],checkpoint_sha256=sha(spec['checkpoint']),train_config_sha256=sha(spec['train_config']) if spec.get('train_config') else None)
 if (cache/'identity.json').exists():assert json.loads((cache/'identity.json').read_text())==identity
 else:dump(cache/'identity.json',identity)
 records=cfg['records'];truth=[]
 for r in records:
  assert sha(r['image'])==r['sha256'] and sha(r['points'])==r['points_sha256']
  truth.append(pd.read_csv(r['points'])[['axis-0','axis-1','axis-2']].to_numpy(float))
 missing=[i for i in range(len(records)) if not (cache/f'{i}.npy').exists()]
 if missing:
  if spec['kind']=='biodino':
   from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
   enc=Dinov3CkptEncoder(Path(spec['checkpoint']),Path(spec['train_config']),'cuda:0',1,False,torch.bfloat16)
   body=enc.model.backbone;layer=[body.n_blocks-1]
  else:
   from unified_paper_native_backbone import NativeFMBackbone
   body=NativeFMBackbone(a.model);layer=[0]
  body.eval()
  for param in body.parameters():param.requires_grad_(False)
  for i in missing:
   vol=tifffile.imread(records[i]['image']).astype(np.float32);assert vol.shape==tuple(SHAPE) and np.isfinite(vol).all()
   features=[]
   for z in range(0,46,4):
    x=torch.from_numpy(vol[z:z+4]).cuda().clamp(0,1)[:,None].expand(-1,3,-1,-1)
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):f=body.get_intermediate_layers(x,n=layer,reshape=True)[0]
    features.append(f.float().cpu().numpy().astype(np.float16))
   values=np.concatenate(features).transpose(1,0,2,3)
   tmp=cache/f'{i}.tmp.npy';np.save(tmp,values);tmp.replace(cache/f'{i}.npy');print('FEATURE',a.model,i+1,len(records),values.shape,flush=True)
  del body
  if spec['kind']=='biodino':del enc
  torch.cuda.empty_cache()
 channels=np.load(cache/'0.npy',mmap_mode='r').shape[0]
 def feature(i):return torch.from_numpy(np.load(cache/f'{i}.npy').astype(np.float32))[None].cuda()
 def y(i):return torch.from_numpy(target(truth[i]))[None,None].cuda()
 def save_state(path,obj):
  tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
 results=[]
 for seed in cfg['seeds']:
  directory=out/f'seed{seed}';directory.mkdir(exist_ok=True);result=directory/'component_result.json'
  if result.exists():results.append(json.loads(result.read_text()));continue
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.RandomState(seed)
  head=Head(channels).cuda();opt=torch.optim.Adam(head.parameters(),lr=cfg['lr'])
  statepath=directory/'training_state.pth';bestpath=directory/'best_head.pth';history=[];best=float('inf');stale=0
  if statepath.exists():
   state=torch.load(statepath,map_location='cpu',weights_only=False);assert state['identity']==identity
   head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];stale=state['stale'];rng.set_state(state['rng'])
  for epoch in range(len(history),cfg['epochs']):
   if stale>=cfg['patience']:break
   head.train();order=rng.permutation(cfg['splits']['train']);losses=[];opt.zero_grad(set_to_none=True);start=time.time()
   for step,i in enumerate(order):
    output=head(feature(int(i)));loss=F.mse_loss(output,y(int(i)));assert torch.isfinite(loss)
    (loss/4).backward();losses.append(float(loss.detach()))
    if (step+1)%4==0:opt.step();opt.zero_grad(set_to_none=True)
   head.eval()
   with torch.inference_mode():vl=float(np.mean([F.mse_loss(head(feature(i)),y(i)).item() for i in cfg['splits']['val']]))
   history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=vl,seconds=time.time()-start))
   if vl<best:
    best=vl;stale=0;save_state(bestpath,dict(head=head.state_dict(),epoch=epoch+1,identity=identity))
   else:stale+=1
   save_state(statepath,dict(head=head.state_dict(),optimizer=opt.state_dict(),history=history,best=best,stale=stale,rng=rng.get_state(),identity=identity))
   dump(directory/'history.json',history);print('EPOCH',a.model,seed,history[-1],flush=True)
  selected=torch.load(bestpath,map_location='cpu',weights_only=False);head.load_state_dict(selected['head']);head.eval()
  validation=[]
  with torch.inference_mode():
   for threshold in cfg['thresholds']:
    rows=[]
    for i in cfg['splits']['val']:
     volume=head(feature(i))[0,0].cpu().numpy();rows.append(centroid_metrics(truth[i],peaks(volume,threshold)))
    validation.append(dict(threshold=threshold,**combine(rows)))
  winner=max(validation,key=lambda r:(r['f1'],-r['threshold']));dump(directory/'selection.json',dict(epoch=selected['epoch'],validation=validation,selected=winner,test_used=False))
  predictions=[]
  with torch.inference_mode():
   for i in cfg['splits']['test']:
    points=peaks(head(feature(i))[0,0].cpu().numpy(),winner['threshold'])
    predictions.append(dict(index=i,image=records[i]['image'],group=records[i]['sample'],truth_zyx=truth[i].tolist(),prediction_zyx=points.tolist(),metrics=centroid_metrics(truth[i],points)))
  prediction=directory/'test_predictions.json';dump(prediction,predictions);metrics=combine([r['metrics'] for r in predictions])
  payload=dict(model=a.model,checkpoint=spec['checkpoint'],dataset='spheroid_lsfm',task='centroid_detection_3d',seed=seed,protocol=PROTOCOL,metrics=metrics,manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),prediction_path=str(prediction),prediction_sha256=sha(prediction),head_state=str(bestpath),head_state_sha256=sha(bestpath),identity=identity,selected_epoch=selected['epoch'],threshold=winner['threshold'],n_train=24,n_val=6,n_test=18,required_seeds=5,host=socket.gethostname(),peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
  dump(result,payload);results.append(payload);del head,opt;torch.cuda.empty_cache();print('SEED COMPLETE',a.model,seed,metrics,flush=True)
 dump(out/'component_result.json',dict(model=a.model,protocol=PROTOCOL,completed_seeds=5,required_seeds=5,seeds=results,metrics={k:float(np.mean([r['metrics'][k] for r in results])) for k in ['f1','precision','recall']},manifest_sha256=sha(MANIFEST)))
 # Cache belongs only to this model's bundled five-seed job. Formal predictions/head remain.
 import shutil
 shutil.rmtree(cache)
if __name__=='__main__':main()
