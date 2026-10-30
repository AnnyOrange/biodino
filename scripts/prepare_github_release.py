#!/usr/bin/env python3
"""Export a source-only snapshot without changing the working tree or Git index.

Uses only the standard library and Git. Never follows symlinks, copies Git history,
starts workloads, or publishes to a remote. Secret checks are heuristic, not proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT_FILES = {
    'LICENSE.md', 'MODEL_CARD.md', 'setup.py', 'pyproject.toml',
    'requirements.txt', 'requirement.txt', 'requirements-dev.txt', 'hubconf.py',
    '.gitignore', '.gitattributes', '.docstr.yaml', 'MANIFEST.in',
}
ROOT_DIRS = {'dinov3', 'scripts', 'tests'}
EXCLUDED_PARTS = {
    '__pycache__', '.git', '.venv', 'venv', 'cache', 'archived_cache',
    'outputs', 'logs', 'build', 'dist', '.ipynb_checkpoints', 'node_modules',
    'ChannelViT', 'dataset_webdataset', 'smart_sharding',
}
SOURCE_SUFFIXES = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.txt', '.cfg', '.ini', '.cu', '.cuh', '.cpp', '.h', '.hpp', '.c'}
SECRET_PATTERNS = [
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(rb'''(?i)(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
]
# A fresh repository must not inherit workstation rules that hide source shell
# scripts or nested READMEs. This generated file is hashed like every other file.
PUBLIC_GITIGNORE = b"""# Generated files and local environments
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.vscode/
.cursor/
.env
.env.*
!.env.example
*.pem
*.key
*.bak
*.bak_*
# Runtime artifacts, datasets and weights
/outputs/
/logs/
/benchmark_runs/
/benchmark_data/
/cache/
/ckpt/
/plot/
/.github/workflows/
/docs/
/Evaluation Rules/
README.md
README_20TB_7B.md
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.npy
*.npz
*.tar
*.tar.gz
*.zip
"""
MAX_BYTES = 10 * 1024 * 1024


def source_bytes(root: Path, name: str) -> bytes:
    return PUBLIC_GITIGNORE if name == '.gitignore' else (root / name).read_bytes()


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args])


def included(name: str) -> bool:
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        return False
    if any(part in EXCLUDED_PARTS for part in p.parts):
        return False
    if p.name == 'AGENTS.md' or '.bak' in p.name:
        return False
    if p.name.lower().startswith('readme'):
        return False
    if len(p.parts) == 1:
        return name in ROOT_FILES
    if p.parts[0] not in ROOT_DIRS or p.suffix not in SOURCE_SUFFIXES:
        return False
    if p.parts[0] == 'scripts' and re.search(r'(^|_)(plot|fig[0-9]|figure|visual)', p.stem, re.I):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--check', action='store_true', help='Audit only; write nothing')
    args = parser.parse_args()
    root = args.source.resolve()
    if not args.check and args.destination is None:
        parser.error('--destination is required unless --check is used')
    dest = args.destination.resolve() if args.destination else None
    if not args.check and dest.exists():
        parser.error('destination must not exist; existing directories are never overwritten')
    candidates = set(git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'))
    selected = sorted(n for n in candidates if n and included(n))
    records = []
    problems = []
    total = 0
    # Read at most one bounded file at a time, avoiding model/data allocations.
    for name in selected:
        path = root / name
        if path.is_symlink() or path.resolve() != root / name:
            problems.append(f'{name}: symlink or redirected parent')
            continue
        if not path.is_file():
            problems.append(f'{name}: tracked file missing')
            continue
        if path.stat().st_size > MAX_BYTES:
            problems.append(f'{name}: exceeds source file limit of 10 MiB')
            continue
        data = source_bytes(root, name)
        if b'\0' in data:
            problems.append(f'{name}: unexpected binary content')
            continue
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            problems.append(f'{name}: possible credential (value suppressed)')
            continue
        records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        total += len(data)
    required = ROOT_FILES - {'.gitattributes', '.docstr.yaml'}
    for name in sorted(required - {r['path'] for r in records}):
        problems.append(f'{name}: required publication file missing or rejected')
    if problems:
        print(json.dumps({'status': 'blocked', 'problems': problems}, indent=2))
        return 1
    summary = {'status': 'checked', 'files': len(records), 'bytes': total,
               'excluded_candidates': len(candidates - {''}) - len(selected)}
    if not args.check:
        dest.mkdir(parents=True, exist_ok=False)
        for record in records:
            source = root / record['path']
            data = source_bytes(root, record['path'])
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise RuntimeError(f'Source changed during export: {record["path"]}; discard this partial snapshot')
            target = dest / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        manifest = {
            'format': 1, 'source_commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'includes_working_tree_changes': True,
            'generated_files': ['.gitignore'],
            'scope': 'Source only; excludes Git history, operational records, data, weights and caches',
            'secret_scan': 'Heuristic patterns only; not a guarantee of no secrets',
            'files': records,
        }
        (dest / 'release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        summary.update(status='exported', destination=str(dest))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
