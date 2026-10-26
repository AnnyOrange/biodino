"""Published FM processing wrapped for the existing frozen native RCNN neck."""
import torch
from benchmark_eval.rules_features import RuleFMFeatures
from run_new_fm_v4_detection_20261016 import NewSpatialBackbone,MODELS

class NativeFMBackbone(torch.nn.Module):
 def __init__(self,name):
  super().__init__();self.name=name;self.patch_size=16;self.n_blocks=1
  self.new=NewSpatialBackbone(name) if name in MODELS else None
  self.features=None if self.new is not None else RuleFMFeatures(name,'cuda:0',layers='last')
  self.model=self.new.model if self.new is not None else self.features.model
  for p in self.parameters():p.requires_grad_(False)
  self.eval()
  with torch.no_grad():self.embed_dim=self.get_intermediate_layers(torch.zeros(1,3,224,224,device='cuda'),n=[0],reshape=True)[0].shape[1]
 def train(self,mode=True):
  super().train(False);return self
 @torch.no_grad()
 def get_intermediate_layers(self,images,n=1,reshape=True,**kwargs):
  if not reshape or n not in (1,[0]):raise ValueError('Native primary-last requests exactly one final feature map')
  maps=[];grid=(images.shape[-2]//16,images.shape[-1]//16)
  for x in images.split(1):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if self.new is not None:
     resized=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
     value=self.new.get_intermediate_layers(resized,n=1,reshape=True)[0]
    else:
     mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
     value=self.features((x*std+mean).clamp(0,1))
   maps.append(torch.nn.functional.interpolate(value.float(),size=grid,mode='bilinear',align_corners=False).clone())
  return [torch.cat(maps)]
