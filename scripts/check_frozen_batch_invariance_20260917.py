"""GPU preflight only: verify teacher BF16 features at batch 1 versus batch 64."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--batch-size', type=int, required=True, choices=[64])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, default=Path('/mnt/huawei_deepcad/benchmark'))
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    if commit != args.expected_commit or dirty:
        raise RuntimeError('GPU preflight requires the expected clean Git checkout')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    import sklearn
    from torch.utils.data import DataLoader, Subset
    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder, pil_collate
    from dinov3.eval.bio_frozen_eval.registry import build_dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, path in (('checkpoint', args.checkpoint), ('config', args.train_config),
                       ('dataset', args.benchmark_root / 'Classification/MedMNIST/pathmnist.npz')):
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Input still changing: {path}')
        records[name] = dict(path=str(path.resolve()), sha256=digest,
                             bytes=after.st_size, mtime_ns=after.st_mtime_ns)
    report = dict(
        status='RUNNING', scope='GPU_PREFLIGHT_ONLY_NOT_TASK_METRICS',
        reportability='EXPERIMENTAL_NOT_REPORTABLE', git_commit=commit, git_status_porcelain=dirty,
        hostname=os.uname().nodename, pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        python=sys.version, python_executable=sys.executable, torch=torch.__version__, sklearn=sklearn.__version__,
        command=sys.argv, inputs=records, teacher_branch='teacher', dataset='pathmnist', split='official-train',
        sample_indices=list(range(64)), feature='l2(final_cls_concat_final_patch_mean)',
        resolution=224, resize=256, n_last_blocks=1, use_avgpool=True, autocast_dtype='bf16',
        batch_sizes=[1, 64], num_workers=2, seed=0,
        acceptance=dict(min_cosine=0.9999, max_relative_l2=0.02),
        started_at=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output)

    save()
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        dataset, task = build_dataset('pathmnist', 'train', None, None, benchmark_root=args.benchmark_root)
        loader = DataLoader(Subset(dataset, range(64)), batch_size=64, shuffle=False,
                            num_workers=2, collate_fn=pil_collate)
        images, labels, paths = next(iter(loader))
        report['sample_paths'] = list(paths)
        report['sample_order_sha256'] = hashlib.sha256(json.dumps(list(paths)).encode()).hexdigest()
        encoder = Dinov3CkptEncoder(args.checkpoint, args.train_config, 'cuda:0', 1, True,
                                    torch.bfloat16, image_size=224, resize_size=256,
                                    channel_policy='auto', channel_policy_seed=0)
        torch.cuda.reset_peak_memory_stats()
        reference = np.concatenate([encoder.encode_pil([image]) for image in images]).astype(np.float32)
        batched = encoder.encode_pil(images).astype(np.float32)
        torch.cuda.synchronize()
        cosine = (reference * batched).sum(1) / (np.linalg.norm(reference, axis=1) * np.linalg.norm(batched, axis=1))
        relative = np.linalg.norm(reference-batched, axis=1) / np.linalg.norm(reference, axis=1)
        passed = bool(np.isfinite(reference).all() and np.isfinite(batched).all()
                      and cosine.min() >= 0.9999 and relative.max() <= 0.02)
        report.update(status='PASS' if passed else 'FAIL', feature_dimension=int(batched.shape[1]),
                      min_cosine=float(cosine.min()), max_relative_l2=float(relative.max()),
                      max_absolute_difference=float(np.abs(reference-batched).max()),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
