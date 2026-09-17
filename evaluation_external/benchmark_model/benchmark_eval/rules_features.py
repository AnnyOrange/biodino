"""Common FM14 spatial adapter; unsupported geometry fails instead of resizing."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType

import torch
import torch.nn.functional as F

from dinov3.eval.bio_frozen_eval.external_fm_protocol import MODELS,resolve_dense_layers
from dinov3.eval.bio_frozen_eval import external_fm_protocol as protocol_module

BENCH=Path('/mnt/huawei_deepcad/benchmark_model')


def tokens_to_spatial(tokens,prefix,grid):
    if tokens.ndim!=3:raise ValueError('Expected B,N,D tokens')
    if prefix < 0 or min(grid) < 1 or tokens.shape[-1] < 1:
        raise ValueError('Invalid prefix, spatial grid or embedding width')
    patches=tokens[:,prefix:];spatial=grid[0]*grid[1]
    if patches.shape[1] == 0:raise ValueError('Nonempty spatial tokens are required')
    if patches.shape[1] % spatial:raise ValueError('Token count does not match actual spatial/channel grid')
    channels=patches.shape[1]//spatial
    patches=patches.reshape(tokens.shape[0],channels,spatial,tokens.shape[-1]).mean(1)
    return patches.transpose(1,2).reshape(tokens.shape[0],tokens.shape[-1],*grid)


def interpolate_position(pos,grid,prefix=1):
    squeeze=pos.ndim==2
    p=pos.unsqueeze(0) if squeeze else pos
    n=p.shape[1]-prefix;old=math.isqrt(n)
    if old*old!=n:raise ValueError('Non-square published spatial position grid')
    if grid==(old,old):return pos
    spatial=p[:,prefix:].reshape(1,old,old,p.shape[-1]).permute(0,3,1,2)
    spatial=F.interpolate(spatial.float(),grid,mode='bicubic',align_corners=False).to(pos.dtype)
    spatial=spatial.permute(0,2,3,1).reshape(1,grid[0]*grid[1],p.shape[-1])
    output=torch.cat((p[:,:prefix],spatial),1)
    return output.squeeze(0) if squeeze else output


def sdpa_channel_attention(self,x):
    batch,length,width=x.shape
    q,k,v=self.qkv(x).reshape(batch,length,3,self.num_heads,width//self.num_heads).permute(2,0,3,1,4).unbind(0)
    output=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
    output=output.transpose(1,2).reshape(batch,length,width)
    return self.proj_drop(self.proj(output)),None


class RuleFMFeatures:
    def __init__(self,name,device='cuda:0',layers='last'):
        if name not in MODELS:raise ValueError('FM14 model required')
        hashes=json.loads(Path(protocol_module.__file__).with_name('external_fm_source_hashes.json').read_text())
        for path,expected in hashes['files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('External asset loader changed; re-register and preflight its code fingerprint')
        sys.path[:0]=[str(BENCH),str(BENCH/'_vendor')]
        spec=importlib.util.spec_from_file_location('rule_native_fm_loaders',BENCH/'run_dense_probe_benchmark.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.native=module.DenseFeatureExtractor(name,device,canonical=True)
        self.name=name;self.device=torch.device(device);self.kind=self.native.spec.kind
        self.model=self.native.model;self.patch_size=self.native.patch_size
        self.blocks=None;self.norm=None;self.prefix=0;self.tower=self.model
        if self.kind=='transformers':
            self.tower=getattr(self.model,'vision_model',self.model)
            self.blocks=getattr(self.tower.encoder,'layer',None)
            if self.blocks is None:self.blocks=getattr(self.tower.encoder,'layers',None)
            self.norm=getattr(self.tower,'layernorm',getattr(self.tower,'post_layernorm',None))
            self.prefix=0 if name=='siglip2' else 1+int(getattr(self.tower.config,'num_register_tokens',0))
            if name=='mae':self.model.config.mask_ratio=0.
        elif self.kind in ('timm','conch','channelvit','pe_openphenom'):
            if self.kind=='conch':self.tower=self.model.visual.trunk
            if self.kind=='pe_openphenom':self.tower=self.model.encoder.vit_backbone
            self.blocks=getattr(self.tower,'blocks',None);self.norm=getattr(self.tower,'norm',None)
            self.prefix=int(getattr(self.tower,'num_prefix_tokens',1 if getattr(self.tower,'cls_token',None) is not None else 0))
            if self.kind=='channelvit':
                # Published ChannelViT CellAugmentation statistics (raw 0..255).
                self.native.mean=torch.tensor([4.031743599139058,1.565935237087539,3.77367898215863,
                    3.4605251427133257,4.1723172504050225]).view(1,5,1,1)/255.
                self.native.std=torch.tensor([17.318438884455695,12.015918256263747,16.966058078452495,
                    15.064776266287147,17.964118200870608]).view(1,5,1,1)/255.
                for block in self.blocks:block.attn.forward=MethodType(sdpa_channel_attention,block.attn)
            if self.kind in ('conch','pe_openphenom'):
                self.tower.patch_embed.strict_img_size=False
                # Position interpolation is performed explicitly; keep flat patch tokens.
                self.tower.dynamic_img_size=False
                if self.kind=='conch':self.tower.patch_embed.flatten=True
        elif self.kind=='open_clip':
            self.tower=self.model.visual;self.blocks=self.tower.transformer.resblocks
            self.norm=self.tower.ln_post;self.prefix=1
        elif self.kind=='keras_torch':
            import os
            os.environ.setdefault('KERAS_BACKEND','torch')
            from keras.applications.efficientnet import EfficientNetB0
            self.model=EfficientNetB0(include_top=False,weights=None,pooling=None,input_shape=(None,None,3))
            self.model.load_weights(self.native.spec.path/'efficientnetb0_weights-notop.h5')
            self.model.to(self.device).eval();self.native.model=self.model
        self.layer_record=resolve_dense_layers(layers,len(self.blocks) if self.blocks is not None else None)
        self.captured={};self.handles=[]
        for index in self.layer_record['layer_indices']:
            def capture(_module,_args,out,index=index):
                value=out[0] if isinstance(out,(tuple,list)) else out
                self.captured[index]=value
            self.handles.append(self.blocks[index].register_forward_hook(capture))
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.metadata=dict(model=name,channel_policy='auto',autocast_dtype='bf16',**self.layer_record)
        if self.kind=='channelvit':
            self.metadata['normalization_source']='channelvit/config/transformations/cell.yaml:first-five-fluorescence-channels'
            self.metadata['native_max_channels']=self.native.in_chans
        self.final_global=None

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.captured={}

    def _rgb(self,x):
        if x.shape[1]<3:x=torch.cat((x,x[:,-1:].expand(-1,3-x.shape[1],-1,-1)),1)
        return x[:,:3]

    @torch.inference_mode()
    def __call__(self,imgs):
        self.captured={};self.final_global=None
        original=list(imgs.shape[-2:]);x=self.native._resize(imgs).to(self.device)
        grid=(x.shape[-2]//self.patch_size,x.shape[-1]//self.patch_size)
        self.metadata.update(requested_input_size=original,encoder_input_size=list(x.shape[-2:]))
        if self.kind not in ('channelvit','pe_openphenom'):x=self._rgb(x)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            if self.kind=='transformers':
                x=self.native._norm(x);kwargs=dict(pixel_values=x,interpolate_pos_encoding=True)
                if self.name=='mae':kwargs['noise']=torch.arange(grid[0]*grid[1],device=x.device,dtype=torch.float32).expand(x.shape[0],-1)
                out=self.tower(**kwargs)
                if self.prefix==0:self.final_global=getattr(out,'pooler_output',None)
            elif self.kind=='timm':self.model.forward_features(self.native._norm(x))
            elif self.kind=='conch':
                saved=self.tower.pos_embed
                self.tower.pos_embed=torch.nn.Parameter(interpolate_position(saved,grid,self.prefix),requires_grad=False)
                try:self.tower.forward_features(self.native._norm(x))
                finally:self.tower.pos_embed=saved
            elif self.kind=='pe_openphenom':
                saved=self.tower.pos_embed
                base_grid=self.tower.patch_embed.grid_size
                base=saved[:,:self.prefix+base_grid[0]*base_grid[1]]
                position=interpolate_position(base,grid,self.prefix)
                position=torch.cat((position[:,:self.prefix],position[:,self.prefix:].repeat(1,x.shape[1],1)),1)
                self.tower.pos_embed=torch.nn.Parameter(position,requires_grad=False)
                spatial=grid[0]*grid[1]*x.shape[1]
                noise=torch.arange(spatial,device=x.device,dtype=torch.float32).expand(x.shape[0],-1)
                try:self.model.encoder.forward_masked(self.model.input_norm(x*255.),0.,noise)
                finally:self.tower.pos_embed=saved
            elif self.kind=='channelvit':
                count=min(x.shape[1],self.native.in_chans);x=x[:,:count]
                if count>len(self.native.mean.flatten()):
                    raise ValueError('Published native-channel normalization must be certified before >3-channel JUMP evaluation')
                x=(x-self.native.mean[:,:count].to(x.device))/self.native.std[:,:count].to(x.device)
                channels=torch.arange(count,device=x.device).expand(x.shape[0],-1)
                self.model(x,extra_tokens={'channels':channels})
            elif self.kind=='open_clip':
                v=self.tower;x=self.native._norm(x);y=v.conv1(x).flatten(2).transpose(1,2)
                y=torch.cat((v.class_embedding.to(y.dtype).expand(y.shape[0],1,-1),y),1)
                y=v.ln_pre(y+interpolate_position(v.positional_embedding,grid).to(y.dtype))
                batch_first=bool(getattr(v.transformer,'batch_first',False))
                v.transformer(y if batch_first else y.transpose(0,1))
            elif self.kind=='cytoself':
                gray=x.mean(1,keepdim=True)
                dense=self.model(gray.repeat(1,self.native.in_channels,1,1)).float()
                self.metadata['channel_mapping']='published-gray-repeat'
                self.final_global=dense.mean((2,3));return dense
            elif self.kind=='keras_torch':
                dense=self.model((x*255.).permute(0,2,3,1).contiguous(),training=False).permute(0,3,1,2).float()
                self.final_global=dense.mean((2,3));return dense
            else:raise ValueError('Unsupported FM architecture')
        maps=[]
        for index in self.layer_record['layer_indices']:
            if index not in self.captured:raise ValueError('Requested block did not execute')
            tokens=self.captured[index]
            if self.kind=='open_clip' and not batch_first:tokens=tokens.transpose(0,1)
            if self.norm is not None:tokens=self.norm(tokens.float())
            if index==self.layer_record['layer_indices'][-1] and self.prefix:self.final_global=tokens[:,0].float()
            maps.append(tokens_to_spatial(tokens,self.prefix,grid).float())
        if not maps:raise ValueError('No spatial layer captured')
        return torch.cat(maps,1)

    @torch.inference_mode()
    def frozen_feature(self,imgs):
        if self.layer_record['resolved_layers']!='last':raise ValueError('Non-dense tasks must use final layer only')
        spatial=self(imgs);mean=spatial.mean((2,3))
        if self.prefix:
            feature=torch.cat((self.final_global,mean),1);readout='final-cls-plus-final-patch-mean'
        elif self.name=='siglip2' and self.final_global is not None:
            feature=torch.cat((self.final_global.float(),mean),1);readout='no-cls:native-global-plus-final-patch-mean'
        else:feature=mean;readout='no-cls:final-spatial-mean'
        self.metadata['feature_layers']=readout
        return F.normalize(feature.float(),dim=1)
