"""Recover retained-head predictions and verify their ordered source identities."""
import hashlib
import json
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
from dinov3.eval.bio_segmentation.linear_probe import CachedFeatureDataset, LinearSegHead, evaluate_cached, DATASET_CONFIGS
from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root


def recover_predictions(cell, budget, seed, output):
    previous_audit = output/'prediction_recovery_audit.json'
    if previous_audit.exists() and not json.loads(previous_audit.read_text()).get('pass'):
        archive = output/'rejected_prediction_recovery'/hashlib.sha256(previous_audit.read_bytes()).hexdigest()
        archive.mkdir(parents=True, exist_ok=True)
        for name in ['prediction_recovery_audit.json','test_semantic_predictions.npz']:
            path = output/name
            if path.exists():path.replace(archive/name)
    old = next(x for x in cell['existing'] if x['budget'] == budget and x['seed'] == seed)
    source = Path(old['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == old['sha256']
    result = json.loads(source.read_text())
    head_path = source.with_name('best_head.pth')
    with np.load(cell['caches']['test']) as cache:
        if 'features' in cache:
            features, masks = cache['features'], cache['sem_masks']
        else:
            features = [cache[f'features_{i:04d}'] for i in range(int(cache['num_chunks']))]
            masks = [cache[f'sem_masks_{i:04d}'] for i in range(int(cache['num_chunks']))]
        shape = int(cache['orig_H']), int(cache['orig_W'])
    dim = features[0].shape[1] if isinstance(features, list) else features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = LinearSegHead(dim, DATASET_CONFIGS[cell['dataset']]['num_classes'], result['_meta']['dropout']).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    loader = DataLoader(CachedFeatureDataset(features, masks, None), batch_size=32, shuffle=False, num_workers=0)
    score = evaluate_cached(head, loader, orig_size=shape, device=device,
        semantic_only=True, prediction_path=str(output/'test_semantic_predictions.npz'), **DATASET_CONFIGS[cell['dataset']])
    differences = {k: float(score[k] - result['test'][k]) for k in ['mDice', 'mIoU']}
    # Failed reproduction remains explicit and never replaces a historical score.
    audit = {'source_result': str(source), 'source_sha256': old['sha256'],
        'cache_path':cell['caches']['test'], 'cache_revision':cell.get('cache_revision','original_retained_cache'),
        'head_path': str(head_path), 'head_sha256': hashlib.sha256(head_path.read_bytes()).hexdigest(),
        'score_differences': differences, 'tolerance': 1e-6,
        'tolerance_basis': 'Absolute metric tolerance on 0–1 scale for CPU/GPU floating-point logits near argmax ties; original scores retained verbatim',
        'inference_device': str(device),
        'pass': all(abs(x) <= 1e-6 for x in differences.values()),
        'mode': 'Inference from original validation-selected head; no new training repetition'}
    (output/'prediction_recovery_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    if not audit['pass']:
        raise ValueError(f'Retained head failed exact score reproduction: {differences}')
    result['prediction_recovery'] = audit
    return result


def attach_source_manifest(cell, output):
    """Verify every cached test mask against the same loader before assigning IDs.

    Initial certified datasets: Cellpose original images and MoNuSeg TCGA patients.
    Other datasets retain prediction row indices but remain pending group metadata.
    """
    if cell['dataset'] not in {'cellpose', 'monuseg', 'conic', 'livecell', 'pannuke', 'tissuenet', 'multimodal_cellseg'}:
        return {'sample_manifest_status': 'pending_dataset_group_adapter'}
    if 'input_spec' in cell:
        spec = cell['input_spec']
    else:
        invocation = json.loads(Path(cell['invocation']).read_text())
        cmd = invocation['command']
        arg = lambda key: cmd[cmd.index(key)+1]
        spec = {'data_root_base':arg('--data-root-base'), 'image_size':int(arg('--feature-img-size')), 'resize_mode':arg('--resize-mode')}
    root = _resolve_data_root(Path(spec['data_root_base']), cell['dataset'])
    dataset = _build_dataset(cell['dataset'], str(root), 'test', int(spec['image_size']),
        resize_mode=spec['resize_mode'], dataset_split_protocol=cell['split'], do_normalize=False)
    if cell['dataset'] == 'conic':
        info = next(root.rglob('patch_info.csv'))
        with info.open() as stream:
            reader = csv.DictReader(stream);column = 'patch_info' if 'patch_info' in reader.fieldnames else reader.fieldnames[0]
            names = [r[column].strip() for r in reader if r[column].strip()]
        sources = [(names[i], f'{root}/images.npy#{i}', f'{root}/labels.npy#{i}') for i in dataset.indices]
    elif cell['dataset'] == 'livecell':
        from dinov3.eval.bio_segmentation.datasets.livecell import get_livecell_paths
        annotation,_ = get_livecell_paths(str(root),split='test')
        sources = [(dataset._images[i]['file_name']+f'#image_id={i}',dataset._find_image_path(dataset._images[i]['file_name']),f'{annotation}#image_id={i}') for i in dataset._img_ids]
    elif cell['dataset'] == 'pannuke':
        sources=[]
        for images,labels in dataset._folds:
            sources.extend((f'{Path(images.filename).relative_to(root)}#{i}',f'{images.filename}#{i}',f'{labels.filename}#{i}') for i in range(len(images)))
    elif cell['dataset'] == 'tissuenet':
        from dinov3.eval.bio_segmentation.datasets.tissuenet import get_tissuenet_paths
        archive=get_tissuenet_paths(str(root),split='test')
        sources=[(f'{Path(archive).name}#{i}',f'{archive}#X[{i}]',f'{archive}#y[{i}]') for i in range(len(dataset))]
    else:
        sources = [(str(Path(p).relative_to(root)), p, dataset.mask_paths[i]) for i,p in enumerate(dataset.img_paths)]
    pred_path = output/'test_semantic_predictions.npz'
    with np.load(pred_path) as data:
        masks = data['true_label']
        indices = data['sample_index']
    assert len(dataset) == len(masks) and np.array_equal(indices, np.arange(len(dataset)))
    # Ground-truth/source verification is model-independent. Reuse only a proof
    # with identical ordered source files, transforms and every saved label.
    # File changes invalidate the proof; predictions and trained heads are never reused.
    source_signature=[]
    for _, image_path, label_path in sources:
        for path in (image_path,label_path):
            file=Path(str(path).split('#')[0]);stat=file.stat()
            source_signature.append((str(file),stat.st_size,stat.st_mtime_ns))
    label_digests=[hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest() for mask in masks]
    proof_key=hashlib.sha256(json.dumps(dict(dataset=cell['dataset'],split=cell['split'],spec=spec,sources=source_signature,labels=label_digests),sort_keys=True).encode()).hexdigest()
    shared=Path(__file__).resolve().parents[1]/'outputs/02_eval_runs/unified_paper/validation/source_manifest_cache'
    shared.mkdir(parents=True,exist_ok=True);proof_path=shared/f'{proof_key}.json'
    reused=None
    if proof_path.exists():reused=json.loads(proof_path.read_text())
    elif cell['dataset']=='multimodal_cellseg':
        # Adopt a completed pre-cache verification only while all source files
        # predate that proof, and its IDs and per-sample ground truth match exactly.
        base=Path(__file__).resolve().parents[1]/'outputs/02_eval_runs/unified_paper/results'
        for candidate in base.glob(f'*/segmentation_refit/{cell["dataset"]}/{cell["split"]}/*/*/test_sample_manifest.json'):
            if any(row[2]>candidate.stat().st_mtime_ns for row in source_signature):continue
            old=json.loads(candidate.read_text());rows=old.get('samples',[])
            if old.get('verification')!='All ordered source masks exactly equal saved test labels':continue
            if len(rows)==len(sources) and all(r['sample_id']==s[0] and r['image_path']==s[1] and r['label_path']==s[2] and r['label_sha256']==digest for r,s,digest in zip(rows,sources,label_digests)):
                reused=old;break
    if reused is not None:
        payload=dict(reused,prediction_path=str(pred_path),source_invocation=cell['invocation'],source_verification_cache_key=proof_key)
        manifest=output/'test_sample_manifest.json';manifest.write_text(json.dumps(payload,indent=2)+'\n')
        if not proof_path.exists():
            tmp=proof_path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(proof_path)
        return {'sample_manifest':str(manifest),'sample_manifest_status':'verified','patient_group_verified':payload['primary_independent_groups_verified'],'sample_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    rows = []
    for i, (sample_id, path, label_path) in enumerate(sources):
        actual = dataset[i][1].numpy().astype(np.uint8)
        if not np.array_equal(actual, masks[i]):
            raise ValueError(f'Cached test mask/order no longer matches source sample {i}: {path}')
        stem = Path(path).stem
        if cell['dataset'] == 'monuseg':
            if not stem.startswith('TCGA-') or len(stem.split('-')) < 3:
                raise ValueError(f'Missing TCGA patient ID: {stem}')
            group = '-'.join(stem.split('-')[:3])
        elif cell['dataset'] == 'conic':
            group = sample_id.split('-')[0]
        elif cell['dataset'] in {'livecell','pannuke','tissuenet','multimodal_cellseg'}:
            group=sample_id.split('#image_id=')[0]
        else:
            group = stem.removesuffix('_img')
        rows.append({'sample_index': i, 'sample_id': sample_id,
            'group_id': group, 'stratum': group.split('_')[0] if cell['dataset']=='conic' else 'all',
            'image_path': path, 'label_path': label_path,
            'label_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    manifest = output/'test_sample_manifest.json'
    conditional=cell['dataset'] in {'livecell','pannuke','tissuenet','multimodal_cellseg'}
    payload = {'dataset': cell['dataset'], 'split': cell['split'], 'samples': rows,
        'group_unit': {'monuseg':'TCGA patient','conic':'published source image, preserving cohort strata','cellpose':'original benchmark image'}.get(cell['dataset'],'published image or array record; not verified biological independence'),
        'group_limitation': '' if cell['dataset'] == 'monuseg' else 'No donor/acquisition mapping beyond published source images; CI conditional on source image groups',
        'primary_independent_groups_verified':not conditional,
        'verification': 'All ordered source masks exactly equal saved test labels',
        'prediction_path': str(pred_path), 'source_invocation': cell['invocation']}
    temp = manifest.with_suffix('.tmp'); temp.write_text(json.dumps(payload, indent=2)+'\n'); temp.replace(manifest)
    temp=proof_path.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(proof_path)
    return {'sample_manifest': str(manifest), 'sample_manifest_status': 'verified',
        'patient_group_verified':not conditional,
        'sample_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
