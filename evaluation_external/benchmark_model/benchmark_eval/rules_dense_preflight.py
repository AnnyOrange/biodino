"""Audit canonical dense split identities without extracting features."""
import csv
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def audit_dense(dataset,benchmark,protocol=None,observation=False):
    from dinov3.eval.bio_frozen_eval.external_fm_protocol import dense_recipe
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root
    from dinov3.eval.bio_segmentation.datasets import DATASET_REGISTRY
    recipe=dense_recipe('dinov2',dataset,split_protocol=protocol,observation=observation)
    root=_resolve_data_root(Path(benchmark)/'segmentation',dataset)
    expected={'conic':(3469,494,1018),'livecell':(3253,570,1564),
              'monuseg':(23,7,14),'multimodal_cellseg':(855,172,100)}
    counts={};identities={};sources={};inventory=[]
    for split in ('train','val','test'):
        if dataset=='tissuenet':
            path=Path(DATASET_REGISTRY[dataset][1](str(root),split=split))
            with zipfile.ZipFile(path) as archive:
                with archive.open('X.npy') as handle:
                    version=np.lib.format.read_magic(handle)
                    shape,_,_=np.lib.format._read_array_header(handle,version)
                inventory.append((str(path),path.stat().st_size,path.stat().st_mtime_ns,
                    [(i.filename,i.CRC,i.file_size) for i in archive.infolist()]))
            identities[split]=[f'{path}:{i}' for i in range(shape[0])];counts[split]=shape[0]
            continue
        ds=_build_dataset(dataset,str(root),split,recipe['image_size'],recipe['resize_mode'],
                          augment=False,do_normalize=False,dataset_split_protocol=recipe['split_protocol'])
        counts[split]=len(ds)
        if hasattr(ds,'img_paths'):
            identities[split]=[str(Path(p).resolve()) for p in ds.img_paths]
            for path in ds.img_paths+ds.mask_paths:
                p=Path(path);s=p.stat();inventory.append((str(p),s.st_size,s.st_mtime_ns))
        elif dataset=='conic':
            identities[split]=[str(i) for i in ds.indices]
            for array in (ds.images,ds.labels):
                p=Path(array.filename);stat=p.stat();inventory.append((str(p),stat.st_size,stat.st_mtime_ns))
        elif dataset=='livecell':
            identities[split]=[str(i) for i in ds._img_ids]
            existing={p.name for p in Path(ds.img_root).rglob('*') if p.is_file()}
            for record in ds._images.values():
                p=Path(ds.img_root)/record['file_name']
                if not p.exists():
                    if Path(record['file_name']).name not in existing:raise FileNotFoundError(record['file_name'])
        elif dataset=='pannuke':
            fold={'train':recipe['split_protocol'].split('-')[1],
                  'val':recipe['split_protocol'].split('-')[3],
                  'test':recipe['split_protocol'].split('-')[5]}[split]
            identities[split]=[f'{fold}:{i}' for i in range(len(ds))]
            for pair in ds._folds:
                for array in pair:
                    p=Path(array.filename);stat=p.stat();inventory.append((str(p),stat.st_size,stat.st_mtime_ns))
        else:raise ValueError(f'No dense identity auditor for {dataset}')
        del ds;gc.collect()
    if dataset in expected and tuple(counts.values())!=expected[dataset]:raise ValueError(f'Canonical count mismatch: {dataset} {counts}')
    if dataset!='livecell':
        for left,right in (('train','val'),('train','test'),('val','test')):
            if set(identities[left])&set(identities[right]):raise ValueError(f'{dataset} {left}/{right} identity overlap')
    if dataset=='conic':
        path=next(root.rglob('patch_info.csv'));sources[str(path)]=sha256(path)
        rows=list(csv.DictReader(path.open()));column='patch_info' if 'patch_info' in rows[0] else next(iter(rows[0]))
        groups={s:{rows[int(i)][column].split('-')[0] for i in ids} for s,ids in identities.items()}
        for left,right in (('train','val'),('train','test'),('val','test')):
            if groups[left]&groups[right]:raise ValueError('CoNIC source leakage')
    elif dataset=='livecell':
        lock=json.loads((Path(__file__).resolve().parents[2]/'dinov3/Evaluation Rules/protocol_v3.json').read_text())
        for split in counts:
            path=Path(DATASET_REGISTRY[dataset][1](str(root),split=split)[0])
            sources[str(path)]=sha256(path)
            if sources[str(path)]!=lock['dense_splits']['livecell']['annotation_sha256'][split]:raise ValueError('LIVECell official annotation hash mismatch')
    elif dataset=='monuseg':
        path=root/'monuseg_val_indices.npy';sources[str(path)]=sha256(path)
        if sources[str(path)]!='932a09d0e936bd2ee83438145f6e6955dac224b747f2536ae06d31c764ff5c91':raise ValueError('MoNuSeg val index changed')
    elif dataset=='multimodal_cellseg':
        for name in ('train.csv','val.csv','test_source_heldout.csv'):
            path=root/'splits'/name;sources[str(path)]=sha256(path)
        groups={s:{r['source_dataset'] for r in csv.DictReader((root/'splits'/name).open())}
                for s,name in (('train','train.csv'),('val','val.csv'),('test','test_source_heldout.csv'))}
        if groups['test'] & (groups['train']|groups['val']):raise ValueError('Multimodal source leakage')
    return dict(status='PASS',task='segmentation',dataset=dataset,split=recipe['split_protocol'],counts=counts,
        source_hashes=sources,split_identity_sha256=digest(identities),dataset_inventory_sha256=digest(inventory),
        image_size=recipe['image_size'],resize_mode=recipe['resize_mode'],resize_size=recipe['image_size'],
        feature_batch_size=32,requires_pyarrow=False,requires_modules=['cv2','skimage'])
