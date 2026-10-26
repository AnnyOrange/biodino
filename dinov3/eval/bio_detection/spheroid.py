"""PAC-MAP source-coordinate adapter and physical centroid scorer.

The released SH-SY5Y split is patch-level, not spheroid-independent. Strict
independence admission fails without altering the author's official cohort.
"""
from pathlib import Path
import ast,json,hashlib
import numpy as np,pandas as pd,tifffile
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
VOXEL_UM=np.array([1.9999,.3594,.3594])
SHAPE=np.array([46,256,256]);BORDER_UM=5.;MATCH_UM=5.
def centroid_metrics(truth,prediction):
 truth=np.asarray(truth,dtype=float).reshape(-1,3)*VOXEL_UM;prediction=np.asarray(prediction,dtype=float).reshape(-1,3)*VOXEL_UM
 # Source author excludes points in a5um boundary zone before assignment.
 upper=SHAPE*VOXEL_UM-BORDER_UM
 truth=truth[np.all((truth>=BORDER_UM)&(truth<upper),axis=1)];prediction=prediction[np.all((prediction>=BORDER_UM)&(prediction<upper),axis=1)]
 if len(truth) and len(prediction):
  dist=cdist(truth,prediction,'sqeuclidean');dist[dist>MATCH_UM**2]=max(float(dist.max()),MATCH_UM**2+1);a,b=linear_sum_assignment(dist);tp=int((dist[a,b]<=MATCH_UM**2).sum())
 else:tp=0
 fp=len(prediction)-tp;fn=len(truth)-tp
 return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.)
def prepare(root):
 root=Path(root);rows=[];arrays={}
 for split in ['train','test']:
  for image in sorted((root/split/split/'patches').glob('*/*.tif')):
   if image.name.startswith('.'):continue
   with tifffile.TiffFile(image) as f:meta=f.imagej_metadata;a=f.asarray()
   assert tuple(a.shape)==tuple(SHAPE) and np.isfinite(a).all()
   points=root/split/split/'points_csv'/image.parent.name/(image.stem+'.csv');df=pd.read_csv(points);xyz=df[['axis-0','axis-1','axis-2']].to_numpy(float);assert np.isfinite(xyz).all() and np.all(xyz>=0) and np.all(xyz<SHAPE)
   grid=ast.literal_eval(meta['patches_structure']);stride=ast.literal_eval(meta['patch_stride']);ix=int(image.stem.split('_')[-1]);start=(np.array(np.unravel_index(ix,grid))*stride)[[0,2,3]]
   rows.append(dict(sample=image.parent.name,split=split,patch=ix,start=start.tolist(),end=(start+SHAPE).tolist(),image=str(image),points=str(points),n_points=len(xyz),sha256=hashlib.sha256(image.read_bytes()).hexdigest()));arrays[str(image)]=a
 assert sum(r['split']=='train' for r in rows)==30 and sum(r['split']=='test' for r in rows)==18
 overlap=[]
 for a in rows:
  for b in rows:
   if a['split']!='train' or b['split']!='test' or a['sample']!=b['sample']:continue
   lo=np.maximum(a['start'],b['start']);hi=np.minimum(a['end'],b['end']);size=hi-lo
   if np.all(size>0):
    aa=arrays[a['image']][tuple(slice(l,h) for l,h in zip(lo-a['start'],hi-a['start']))];bb=arrays[b['image']][tuple(slice(l,h) for l,h in zip(lo-b['start'],hi-b['start']))]
    overlap.append(dict(train=a['image'],test=b['image'],overlap_voxels=int(np.prod(size)),identical_overlap_pixels=bool(np.array_equal(aa,bb)),max_abs_difference=float(np.max(abs(aa-bb)))))
 train_groups={r['sample'] for r in rows if r['split']=='train'};test_groups={r['sample'] for r in rows if r['split']=='test'}
 return dict(records=rows,overlap_pairs=overlap,train_samples=sorted(train_groups),test_samples=sorted(test_groups),shared_spheroids=sorted(train_groups&test_groups),status='BLOCKED_OFFICIAL_PATCH_SPLIT_NOT_SPHEROID_DISJOINT',reason='Both complete official cohorts use same2spheroids;74pixel-overlap pairs involve29/30trainingpatches. Removing overlaps leaves1trainingpatch; whole-spheroid exclusion leaves0. Cannot claim grouped independent generalization or silently invent a new split.',voxel_um=VOXEL_UM.tolist(),matching_um=MATCH_UM,border_um=BORDER_UM,weak_pretraining_included=False)
