"""Canonical identities for the already adopted CoNIC/LIVECell center probes."""
import csv,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BENCH=Path('/mnt/huawei_deepcad/benchmark')
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def signature(path):
 s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
def manifest_path(dataset):return ROOT/'outputs/02_eval_inputs/formal_v4'/f'{dataset}_center_probe_manifest.json'
def membership(dataset,benchmark=BENCH):
 from dinov3.eval.bio_detection import center_probe
 benchmark=Path(benchmark);sources=set();records={}
 for split in ['train','val','test']:
  if dataset=='conic':
   folder=benchmark/'segmentation/conic/extracted';images,labels,indices=center_probe.get_conic_paths(str(folder),split,split_protocol='official-baseline-fold0-nested-v1');info=sorted(folder.rglob('patch_info.csv'))[0];names=list(csv.DictReader(info.open()));column='patch_info'if 'patch_info'in names[0]else next(iter(names[0]));names=[r[column].strip()for r in names];assert len(names)==len(center_probe.np.load(images,mmap_mode='r'))==len(center_probe.np.load(labels,mmap_mode='r'))
   sources.update([str(info),images,labels]);rows=[dict(sample_id=names[i],group_id=names[i].split('-')[0],array_index=int(i),image_path=images,mask_path=labels)for i in indices]
  else:
   assert dataset=='livecell';coco,folder=center_probe.get_livecell_paths(str(benchmark/'segmentation/LIVECell'),split);sources.add(coco);meta=json.loads(Path(coco).read_text());ds=center_probe.LiveCellCenterDataset(str(benchmark/'segmentation/LIVECell'),split,image_size=224,max_samples=0,seed=0);assert len(ds)==len(meta['images']);rows=[]
   for record,sample in zip(meta['images'],ds.samples):
    image,width,height,centers=sample;assert Path(image).name==Path(record['file_name']).name and width==record['width']and height==record['height'];sources.add(image)
    rows.append(dict(sample_id=record['file_name']+'#'+str(record['id']),group_id=record['file_name'],image_path=image,coco_image_id=record['id'],coco_path=coco))
  assert len({r['sample_id']for r in rows})==len(rows)
  records[split]=rows
 # LIVECell's released train/val annotations share 30 file_names. This probe
 # uses a predeclared final epoch and no validation-based selection. Preserve
 # official membership and disclose the diagnostic validation overlap.
 pairs=[('train','test'),('val','test')]+([('train','val')]if dataset=='conic'else[])
 for a,b in pairs:assert not({r['group_id']for r in records[a]}&{r['group_id']for r in records[b]}),(dataset,a,b)
 return records,sorted(sources)
def load_verified(dataset,benchmark):
 p=manifest_path(dataset);raw=json.loads(p.read_text());benchmark=Path(benchmark)
 def actual(value):return benchmark/Path(value).relative_to(BENCH)
 for value,expected in raw['source_hashes'].items():
  path=actual(value)
  if signature(path)!=raw['source_stats'][value]:assert sha(path)==expected,value
 records,_=membership(dataset,benchmark)
 canonical=json.loads(json.dumps(records).replace(str(benchmark),str(BENCH)))
 assert canonical==raw['records'],'Shared center-probe membership changed'
 return raw

class CachedCenterDataset:
 """Exact deterministic input tensors; preserves the original training sampler."""
 def __init__(self,dataset,split):
  import numpy as np
  self.base=ROOT/'outputs/02_eval_runs/unified_paper/prepared_inputs/center_probes'/dataset
  proof=json.loads((self.base/'cache.json').read_text());assert proof['manifest_sha256']==sha(manifest_path(dataset))
  files=proof['splits'][split];self.paths=[self.base/files[k]for k in ['images','labels']]
  for path in self.paths:
   if signature(path)!=proof['file_stats'][path.name]:assert sha(path)==proof['file_sha256'][path.name]
  self.images=np.load(self.paths[0],mmap_mode='r');self.labels=np.load(self.paths[1],mmap_mode='r');assert len(self.images)==len(self.labels)==files['count']
 def __len__(self):return len(self.labels)
 def __getitem__(self,index):
  import torch
  return torch.from_numpy(self.images[index].copy()),torch.from_numpy(self.labels[index].copy())

def prepare_tensor_cache(dataset):
 import fcntl,time,numpy as np,torch
 from torch.utils.data import DataLoader
 from dinov3.eval.bio_detection import center_probe
 out=ROOT/'outputs/02_eval_runs/unified_paper';base=out/'prepared_inputs/center_probes'/dataset;base.mkdir(parents=True,exist_ok=True);proof=base/'cache.json'
 with(out/'locks'/f'{dataset}_center_tensor_cache.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  if proof.exists():return
  torch.set_num_threads(2);m=json.loads(manifest_path(dataset).read_text());spec=dict(manifest_sha256=sha(manifest_path(dataset)),splits={},file_sha256={},file_stats={},validation={},format='Exact shared center_probe float32 tensors; original image normalization and centroid labels; no random augmentation')
  benchmark=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT',str(BENCH))
  load_verified(dataset,benchmark)
  source_stats={str(Path(benchmark)/Path(path).relative_to(BENCH)):signature(Path(benchmark)/Path(path).relative_to(BENCH))for path in m['source_hashes']}
  for split in ['train','val','test']:
   source=center_probe.build_center_dataset(dataset,benchmark,split,224,0,0);assert len(source)==len(m['records'][split]);paths=[base/f'{split}_{kind}.npy'for kind in ['images','labels']];pending=[p.with_suffix('.pending.npy')for p in paths];image,label=source[0]
   arrays=[np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(source),*shape))for path,shape in zip(pending,[image.shape,label.shape])];offset=0
   for images,labels in DataLoader(source,batch_size=16,num_workers=4,shuffle=False):
    count=len(images);arrays[0][offset:offset+count]=images.numpy();arrays[1][offset:offset+count]=labels.numpy();offset+=count
    if offset%256==0:
     state=out/'states'/f'{dataset}_center_tensor_cache.json';tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),status='preparing_exact_input_tensors',split=split,completed=offset,total=len(source))));tmp.replace(state)
   assert offset==len(source)
   for array in arrays:array.flush()
   del arrays
   for a,b in zip(pending,paths):a.replace(b)
   saved=[np.load(path,mmap_mode='r')for path in paths]
   checks=sorted({0,len(source)//2,len(source)-1})
   for i in checks:
    original=source[i]
    for j in [0,1]:assert np.array_equal(saved[j][i],original[j].numpy()),(dataset,split,i,j)
   spec['splits'][split]=dict(images=paths[0].name,labels=paths[1].name,count=len(source));spec['validation'][split]=dict(bitwise_recomputed_rows=checks,all_rows_from_shared_loader=True)
   for path in paths:spec['file_sha256'][path.name]=sha(path);spec['file_stats'][path.name]=signature(path)
  for path,before in source_stats.items():assert signature(path)==before,'Source changed while caching: '+path
  spec['source_stats']=source_stats
  tmp=proof.with_suffix('.tmp');tmp.write_text(json.dumps(spec,indent=2)+'\n');tmp.replace(proof)
  state=out/'states'/f'{dataset}_center_tensor_cache.json';tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),status='complete_verified',cache=str(proof),manifest_sha256=spec['manifest_sha256'])));tmp.replace(state)
