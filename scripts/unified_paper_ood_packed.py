"""Lossless transport of the fixed, already-normalized ID reference inputs."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
class PreparedRGB:
    def __init__(self,array):
        if array.ndim==2:array=np.repeat(array[None],3,axis=0)
        self.prepared_rgb=torch.from_numpy(np.ascontiguousarray(array))
class PackedIDDataset(torch.utils.data.Dataset):
    def __init__(self,root):
        self.root=Path(root);self.meta=json.loads((self.root/'manifest.json').read_text());self.input_revision=self.meta['input_revision'];self.archive=None;self.archive_pid=None
        for name,digest in self.meta['archive_sha256'].items():
            with (self.root/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    def __len__(self):return len(self.meta['identities'])
    def __getitem__(self,index):
        if self.archive is None or self.archive_pid!=os.getpid():
            if self.archive is not None:self.archive.close()
            self.archive=np.load(self.root/self.meta.get('archive_file','id_inputs.npz'),allow_pickle=False);self.archive_pid=os.getpid()
        array=self.archive[f'image_{index:06d}'];kind=self.meta['kinds'][index]
        image=PreparedRGB(array) if kind=='dinov3_spatial_uint8_v1' else torch.from_numpy(array) if kind=='tensor_CHW' else Image.fromarray(array) if kind=='PIL_HWC' else array
        return image,self.meta['labels'][index],{'source':self.meta['identities'][index]['source'],'path':self.meta['identities'][index]['sample_id']}
