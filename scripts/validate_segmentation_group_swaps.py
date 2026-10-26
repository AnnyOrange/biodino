"""Verify grouped pooled-Dice swap algebra against explicit label predictions."""
import sys,json,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from unified_paper_segmentation_statistics import metric,dump,OUT
rng=np.random.default_rng(17);truth=rng.integers(3,size=(4,2,40));a=rng.integers(3,size=(5,4,2,40));b=rng.integers(3,size=a.shape)
def counts(pred):
 out=np.zeros((5,4,3,3))
 for seed in range(5):
  for group in range(4):out[seed,group]=np.bincount((truth[group]*3+pred[seed,group]).ravel(),minlength=9).reshape(3,3)
 return out
ac,bc=counts(a),counts(b);errors=[]
for draw in itertools.product([False,True],repeat=4):
 mask=np.asarray(draw);aa=np.where(mask[None,:,None,None],b,a);bb=np.where(mask[None,:,None,None],a,b)
 expected=metric(counts(aa).sum(1))-metric(counts(bb).sum(1))
 moved=np.einsum('n,snij->sij',mask,ac-bc);actual=metric(ac.sum(1)-moved)-metric(bc.sum(1)+moved)
 errors.append(abs(expected-actual))
assert max(errors)<1e-12
# Explicit patient bootstrap repeats every image in selected groups together.
indices=np.array([3,1,3,0]);weights=np.bincount(indices,minlength=4)
expected=metric(ac[:,indices].sum(1));actual=metric(np.einsum('n,snij->sij',weights,ac));assert abs(expected-actual)<1e-12
p={'status':'passed','all_group_swaps':16,'training_seeds':5,'images_per_group':2,'max_swap_error':float(max(errors)),'bootstrap_repeated_group_error':float(abs(expected-actual)),'benchmark_result':False}
dump(OUT/'validation/segmentation_group_swaps.json',p);print(json.dumps(p))
