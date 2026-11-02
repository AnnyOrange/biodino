"""Fail-closed inputs for RxRx3, fixed OOD, and native CTC companions."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

SHARED = Path('/mnt/huawei_deepcad/dinov3')
BENCHMARK = Path('/mnt/huawei_deepcad/benchmark')
RXRX3 = SHARED / 'outputs/02_eval_inputs/formal_v3/rxrx3-core'
CTC = SHARED / 'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def audit_rxrx3():
    manifest = RXRX3 / 'split_manifest.jsonl'
    lock = json.loads((SHARED / 'Evaluation Rules/protocol_v3.json').read_text())['retrieval_splits']['rxrx3-core']
    if sha256(manifest) != lock['manifest_sha256']:
        raise ValueError('RxRx3 locked manifest changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    metadata = json.loads((RXRX3 / 'metadata.json').read_text())
    if metadata['rows'] != rows or metadata['manifest_sha256'] != lock['manifest_sha256']:
        raise ValueError('RxRx3 cache/manifest order mismatch')
    if metadata['input_mapping'] != lock['input_mapping']:
        raise ValueError('RxRx3 requires common six-channel compact3 mapping')
    gallery = {row['gene']: row for row in rows if row['split'] == 'gallery'}
    query = {row['gene']: row for row in rows if row['split'] == 'query'}
    if gallery.keys() != query.keys() or len(rows) != 2 * len(gallery) or len(gallery) != 734:
        raise ValueError('RxRx3 must contain all 734 eligible genes once per role')
    for gene in gallery:
        if gallery[gene]['well_id'] == query[gene]['well_id'] or gallery[gene]['plate'] == query[gene]['plate']:
            raise ValueError('RxRx3 same-well/plate leakage')
    labels = np.load(RXRX3 / 'labels.npy')
    images = np.load(RXRX3 / 'images.npy', mmap_mode='r')
    if labels.tolist() != [row['gene'] for row in rows] or images.shape != (1468, 224, 224, 3):
        raise ValueError('RxRx3 cache shape/labels mismatch')
    return dict(status='PASS', task='retrieval', component='rxrx3', dataset='rxrx3-core',
                split=lock['protocol'], counts=dict(n_query=734, n_gallery=734), image_size=224,
                feature_batch_size=64, resize_size=256,
                source_hashes={str(path): sha256(path) for path in RXRX3.iterdir() if path.is_file()},
                split_identity_sha256=lock['manifest_sha256'], requires_modules=[], requires_pyarrow=False)


def ood_datasets(domain):
    from dinov3.eval.eval_ood.datasets import (
        CryoParticleDataset, XrayTomogramSliceDataset, build_id_reference_dataset)
    identity = build_id_reference_dataset(BENCHMARK, transform=None,
        dataset_names=('bloodmnist', 'bbbc048', 'cyclops'), max_samples=3000, seed=0)
    if domain == 'xray':
        target = XrayTomogramSliceDataset(BENCHMARK / 'ood', transform=None,
            slices_per_volume=8, input_mode='three_slices', percentiles=(0.5, 99.5))
    elif domain == 'cryo':
        target = CryoParticleDataset(BENCHMARK / 'ood', transform=None,
            percentiles=(0.5, 99.5), invert=False, max_particles_per_project=20000, seed=0)
    else:
        raise ValueError('Only formally registered xray/cryo OOD domains')
    return identity, target


def ood_identity(identity, target):
    selection = []
    for part in identity.datasets:
        subset = part.dataset
        indices = list(map(int, subset.indices)) if hasattr(subset, 'indices') else list(range(len(subset)))
        original = subset.dataset if hasattr(subset, 'indices') else subset
        if hasattr(original, 'samples'):
            selection.append(dict(source=part.source, indices=indices,
                                  samples=[original.samples[index] for index in indices]))
        else:
            path = BENCHMARK / 'Classification/bloodmnist_64.npz?download=1'
            if not path.exists(): path = BENCHMARK / 'Classification/bloodmnist_64.npz'
            selection.append(dict(source=part.source, indices=indices, array_path=str(path)))
    records = json.loads(json.dumps([dataclasses.asdict(record) for record in target.records], default=str))
    return dict(id_selection=selection, ood_records=records,
                id_bank_fraction=0.7, k=10, seed=0, resize=256, crop=224,
                percentiles=[0.5, 99.5], cryo_max_particles_per_project=20000,
                xray_slices_per_volume=8, xray_input_mode='three_slices', cryo_invert=False)


def locked_ood_datasets(selection, domain):
    from torch.utils.data import ConcatDataset, Subset
    from dinov3.eval.bio_classification.datasets.benchmark import ArrayClassificationDataset, ImagePathClassificationDataset
    from dinov3.eval.eval_ood.datasets import (
        CryoParticleDataset, CryoRecord, XrayRecord, XrayTomogramSliceDataset, _WrappedDataset)
    parts = []
    for part in selection['id_selection']:
        if 'samples' in part:
            dataset = ImagePathClassificationDataset(part['samples'], [], transform=None)
        else:
            with np.load(part['array_path']) as arrays:
                dataset = ArrayClassificationDataset(arrays['train_images'], arrays['train_labels'], transform=None)
            dataset = Subset(dataset, part['indices'])
        parts.append(_WrappedDataset(dataset, part['source']))
    cls, record_cls = (XrayTomogramSliceDataset, XrayRecord) if domain == 'xray' else (CryoParticleDataset, CryoRecord)
    target = cls.__new__(cls)
    target.transform = None
    target.percentiles = tuple(selection['percentiles'])
    target.input_mode = selection['xray_input_mode']
    target.invert = selection['cryo_invert']
    target._memmaps, target._stacks = {}, {}
    records = []
    for serialized in selection['ood_records']:
        values = dict(serialized)
        for key in ('raw_path', 'mrc_path', 'json_path', 'cs_path'):
            if key in values: values[key] = Path(values[key])
        if 'raw_shape_xyz' in values: values['raw_shape_xyz'] = tuple(values['raw_shape_xyz'])
        records.append(record_cls(**values))
    target.records = records
    return ConcatDataset(parts), target


def audit_ood(domain):
    identity, target = ood_datasets(domain)
    selection = ood_identity(identity, target)
    if len(identity) != 3000 or not len(target):
        raise ValueError('Fixed OOD benchmark requires the complete 3000-image ID reference')
    inventory = []
    source_hashes = {}
    for part in selection['id_selection']:
        if 'array_path' in part:
            source_hashes[part['array_path']] = sha256(part['array_path'])
        for path, _ in part.get('samples', []):
            stat = Path(path).stat()
            inventory.append((path, stat.st_size, stat.st_mtime_ns))
    for record in target.records:
        for field in ('raw_path', 'mrc_path', 'json_path', 'cs_path'):
            path = getattr(record, field, None)
            if path is not None:
                stat = path.stat()
                inventory.append((str(path), stat.st_size, stat.st_mtime_ns))
                if field in ('json_path', 'cs_path') and str(path) not in source_hashes:
                    source_hashes[str(path)] = sha256(path)
    for path in (BENCHMARK / 'ood/cryo_em_foundation_model/extracted').glob('*/scores.json') if domain == 'cryo' else ():
        source_hashes[str(path)] = sha256(path)
    return dict(status='PASS', task='ood', component='ood', dataset=domain,
                split='fixed-common-ID3000-OOD-native-records-v1', counts=dict(n_id=3000, n_ood=len(target)),
                image_size=224, resize_size=256, feature_batch_size=64,
                split_identity_sha256=fingerprint(selection), dataset_inventory_sha256=fingerprint(sorted(set(inventory))),
                source_hashes=source_hashes, requires_modules=[], requires_pyarrow=False,
                selection=selection)


def audit_ctc(fold):
    from dinov3.eval.bio_tracking.ctc_native import CTC_DOMAINS, load_cache_manifest
    data = load_cache_manifest(CTC)
    lock = json.loads((SHARED / 'Evaluation Rules/protocol_v3.json').read_text())['cell_tracking_splits']['ctc']
    split = SHARED / 'outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl'
    if sha256(split) != lock['manifest_sha256'] or data['source_split_manifest_sha256'] != lock['manifest_sha256']:
        raise ValueError('CTC locked split mismatch')
    if set(data['domains']) != set(CTC_DOMAINS):
        raise ValueError('CTC requires all 20 native 2-D/3-D domains')
    current = next(row for row in data['folds'] if row['fold'] == fold)
    if set(current['train_domains']) & set(current['test_domains']) or len(current['test_domains']) != 4:
        raise ValueError('CTC held-out domain leakage')
    return dict(status='PASS', task='cell_tracking', component='ctc', dataset='ctc', fold=fold,
                split=lock['protocol'], counts=dict(n_train=sum(len(data['domains'][name]['train_samples']) for name in current['train_domains']),
                    n_test_domains=4, n_test_frames=sum(len(data['domains'][name]['test_frames']) for name in current['test_domains'])),
                image_size=256, feature_batch_size=8, reserve_mib=20000,
                train_domains=current['train_domains'], test_domains=current['test_domains'],
                source_hashes={str(CTC): sha256(CTC), str(split): lock['manifest_sha256']},
                split_identity_sha256=lock['manifest_sha256'], requires_modules=['cv2', 'skimage'], requires_pyarrow=False)
