"""Identity, atomic persistence and reproducibility helpers."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def code_identity():
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=normal'], text=True)
    return dict(commit=commit, dirty=bool(dirty), source_root=str(root))


def resolve_input(root, name):
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Input must stay within the declared dataset root')
    return target


def verify_inputs(manifest_path, data_root, receipt_path):
    manifest = read_json(manifest_path)
    seen = set()
    for row in manifest['files'] if 'files' in manifest else manifest['records']:
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(row['sample_id'])
        path = resolve_input(data_root, row['path'])
        if sha256(path) != row['input_sha256']:
            raise ValueError('Source bytes differ: ' + str(path))
    receipt = dict(manifest_sha256=sha256(manifest_path), data_root=str(Path(data_root).resolve()),
                   verified_files=len(seen), status='ALL_SOURCE_BYTES_VERIFIED')
    save_json(receipt_path, receipt)
    return receipt


def require_inputs(manifest_path, data_root, receipt_path):
    receipt = read_json(receipt_path)
    if (receipt.get('status') != 'ALL_SOURCE_BYTES_VERIFIED'
            or receipt['manifest_sha256'] != sha256(manifest_path)
            or receipt['data_root'] != str(Path(data_root).resolve())):
        raise ValueError('Missing or mismatched resident-input receipt')


def process_policy():
    torch.set_num_threads(int(os.environ.get('OOD_THREADS', '2')))
    if os.name == 'posix' and os.nice(0) < 15:
        os.nice(15 - os.nice(0))
