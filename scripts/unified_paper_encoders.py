"""Formal global readout: final CLS and patch mean, published normalization."""
import numpy as np
import torch
import torch.nn.functional as F
from run_new_fm_v4_linear_20261016 import NewEncoder

class NewFormalGlobalEncoder(NewEncoder):
    @torch.inference_mode()
    def encode_images(self, images):
        out=[]
        mb=min(self.batch_size,2 if self.name in ('genbio_pathfm','openpath') else 32)
        for i in range(0,len(images),mb):
            x=torch.stack([self._prepare(v) for v in images[i:i+mb]]).to(self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                if self.name=='morphem':
                    cls=[];patch=[]
                    for c in range(3):
                        g=x[:,c:c+1];g=(g-g.mean((2,3),keepdim=True))/g.std((2,3),keepdim=True).clamp_min(1e-7)
                        f=self.model.forward_features(g);cls.append(f['x_norm_clstoken']);patch.append(f['x_norm_patchtokens'])
                    global_feature=torch.cat(cls,dim=-1);patches=torch.cat(patch,dim=-1)
                else:
                    mean=x.new_tensor(self.mean)[None,:,None,None];std=x.new_tensor(self.std)[None,:,None,None];x=(x-mean)/std
                    if self.name=='distillpath':
                        f=self.model.forward_features(x);global_feature=f[:,0];patches=f[:,1:]
                    elif self.name=='genbio_pathfm':
                        global_feature,patches=self.model.forward_with_patches(x)
                    else:
                        f=self.model.forward_features(x);global_feature=f['x_norm_clstoken'];patches=f['x_norm_patchtokens']
                feature=torch.cat([global_feature.float(),patches.float().mean(1)],dim=1)
            out.append(F.normalize(feature,dim=1).cpu().numpy().astype(np.float16))
        return np.concatenate(out)
