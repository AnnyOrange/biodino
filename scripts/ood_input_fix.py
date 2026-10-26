"""OOD-only correction of raw floating CyclOps intensities; identities unchanged."""
import numpy as np
import torch
from torch.utils.data import Dataset
from dinov3.utils.bio_io import read_bio_image_as_numpy
REVISION='cyclops_raw_float_percentile_0p5_99p5_v2'
class CorrectedIDDataset(Dataset):
 def __init__(self,original):
  self.original=original;self.datasets=original.datasets
  self.starts=np.cumsum([0]+[len(d) for d in original.datasets])
 def __len__(self):return len(self.original)
 def __getitem__(self,index):
  part_index=int(np.searchsorted(self.starts,index,side='right')-1);part=self.datasets[part_index]
  if part.source!='cyclops':return self.original[index]
  local=int(index-self.starts[part_index]);base=part.dataset.dataset;source_index=part.dataset.indices[local];path,label=base.samples[source_index]
  # The selection already declares these percentiles. Apply to RAW intensities,
  # before conversion; float TIFF data are not assumed to be in [0,1].
  raw=read_bio_image_as_numpy(path,target_channels=3,normalize=False).astype(np.float32)
  if not np.isfinite(raw).all():raise ValueError(f'Nonfinite raw image {path}')
  lo,hi=np.percentile(raw,[.5,99.5]);image=np.clip((raw-lo)/max(float(hi-lo),1e-12),0,1)
  return torch.from_numpy(image).permute(2,0,1).contiguous(),int(label),{'path':f'cyclops:{local}','source':'cyclops'}
