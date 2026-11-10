"""FILM/SPEND author self-permutation data, shared frozen reconstruction task.

Author preprocessing is imported from the released GPL-3.0 SPEND package.
No age labels or denoised algorithm outputs are used as supervision.
"""
from contextlib import contextmanager
import ast
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import tifffile

DATA=Path('/mnt/huawei_deepcad/benchmark/ood/ood_classification/datasets/FILM/extracted')
AUTHOR=DATA/'SPEND/SPEND-main'
RAW=DATA/'Fig2 SPEND example/hyper raw.tif'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@contextmanager
def fixed_randomness(seed):
    python_state=random.getstate();numpy_state=np.random.get_state()
    random.seed(seed);np.random.seed(seed)
    try:yield
    finally:random.setstate(python_state);np.random.set_state(numpy_state)


def permutation_axis(raw):
    # Load only the author's two pure functions, never its top-level file I/O.
    path=AUTHOR/'datasplit_with_aug_choose.py'
    tree=ast.parse(path.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('noise_level','perm_direc')]
    if len(functions)!=2:raise ValueError('Author axis functions changed')
    scope={'np':np,'print':lambda *a,**k:None}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),scope)
    return int(scope['perm_direc'](raw,7,7,50))


def permutation_pairs(raw,axis):
    if raw.shape[axis]%2:raise ValueError('Author permutation requires an even axis')
    for flip in [(),(1,),(2,),(1,2)]:
        image=np.flip(raw,flip) if flip else raw
        even=np.take(image,np.arange(0,image.shape[axis],2),axis=axis)
        odd=np.take(image,np.arange(1,image.shape[axis],2),axis=axis)
        yield np.concatenate((even,odd),axis=axis),np.concatenate((odd,even),axis=axis),'ZYX',None


def training_pairs(seed=0):
    sys.path.insert(0,str(AUTHOR))
    import csbdeep
    if not Path(csbdeep.__file__).resolve().is_relative_to(AUTHOR):raise ValueError('Wrong CSBDeep version imported')
    from csbdeep.data import RawData,create_patches
    raw=tifffile.imread(RAW)
    if raw.shape!=(126,200,200) or not np.isfinite(raw).all():raise ValueError('FILM released raw cube changed')
    axis=permutation_axis(raw)
    source=RawData(lambda:permutation_pairs(raw,axis),4,'FILM Fig2; author four flips and axis permutation')
    with fixed_randomness(seed):
        x,y,axes=create_patches(source,patch_size=(16,64,64),n_patches_per_image=128,verbose=False)
    if axes!='SCZYX' or x.shape!=(512,1,16,64,64):raise ValueError('Author patch layout changed')
    # Exactly the split fraction and tail selection in author load_training_data.
    count=int(round(len(x)*.1))
    return x[:,0],y[:,0],dict(axis=axis,n_train=len(x)-count,n_val=count,seed=seed)


def describe():
    x,y,meta=training_pairs()
    sources=[RAW,AUTHOR/'datasplit_with_aug_choose.py',AUTHOR/'1_datapre_SPEND.py',
             AUTHOR/'2_training_SPEND.py',AUTHOR/'3_prediction_SPEND.py']
    sources+=sorted((AUTHOR/'csbdeep').rglob('*.py'))
    cfg=AUTHOR/'models/my_model_20240908_test_4unet/config.json';sources.append(cfg)
    return dict(protocol='film-spend-author-self-supervised-frozen-v1',dataset='FILM',
        source='https://doi.org/10.6084/m9.figshare.31302607',
        author_code='https://zenodo.org/records/19236368',source_hashes={str(p):sha(p) for p in sources},
        raw=str(RAW),shape=[126,200,200],pairing=meta,patch_size=[16,64,64],
        pairs_sha256={'input':hashlib.sha256(x.tobytes()).hexdigest(),'target':hashlib.sha256(y.tobytes()).hexdigest()},
        official_test=None,split_origin='Author patch validation fraction0.1; fixed shared preprocessing seed0 added for FM reproducibility',
        evaluation_scope='Within-image self-supervised reconstruction; patches can overlap. NOT independent held-out-test generalization.',
        supervision='Author odd/even self-permutation of RAW data only. No clean ground truth released for this example.',
        author_training={'epochs':100,'steps_per_epoch':20,'batch_size':16,'learning_rate':4e-5,'loss':'mae','optimizer':'Adam','reduce_lr_factor':.5,'reduce_lr_patience':10,'reduce_lr_min_delta':0},
        fm_adaptation='Frozen per-spectral-plane last spatial features; common224 input and14x14 grid; linear 3D convolutional residual readout. Equal task, data, head family and2000 optimizer updates.',
        head={'kernel':[3,3,3],'out_channels':1,'residual':True,'activation':None},seeds=list(range(5)),
        primary_metric='validation_self_permutation_mae',
        forbidden_metrics=['clean_reference_psnr','clean_reference_ssim','official_test_accuracy'],
        outputs='Full126-band denoised cube, validation noisy-pair residuals, spectra, head and history; no age classifier',
        published_comparators={name:str(DATA/f'Fig2 SPEND example/hyper after {name}.tif') for name in ['SPEND','BM4D','N2V']})
