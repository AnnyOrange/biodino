#!/usr/bin/env python3
"""Verify GitHub publication, sync clean worktrees, then test all four machines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {
    "5090-hxw-xzj": {"old": "/home/xzj/biodino_eval_git_20260917_1d2330a",
                     "git": "/home/xzj/git_sync_tools_20260917/bin/git",
                     "python": "/home/xzj/miniconda3/envs/dinov3/bin/python"},
    "5090-lyx-xr": {"old": "/data/xuzijing/biodino_eval_git_20260917_1d2330a",
                    "git": "git", "python": "/home/server/miniconda3/envs/dinov3/bin/python"},
    "suxin-8H100-1": {"old": "/data_2/suxin/biodino_eval_git_20260917_1d2330a",
                      "git": "git", "python": "/data_2/suxin/envs/dinov3/bin/python"},
}
REMOTE = r'''
import hashlib, importlib.metadata, importlib.util, json, os, pathlib, subprocess, sys
request = json.loads(sys.argv[1])
old, new = pathlib.Path(request['old']), pathlib.Path(request['new'])
def git(root, *args, timeout=60):
    return subprocess.check_output([request['git'], '-c', 'safe.directory='+str(root), '-C', str(root), *args], text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
old_head = git(old, 'rev-parse', 'HEAD')
if git(old, 'status', '--porcelain'):
    raise RuntimeError('Refuse to sync from dirty existing evaluation worktree')
origin = git(old, 'remote', 'get-url', 'origin')
if origin != 'https://github.com/AnnyOrange/biodino.git':
    raise RuntimeError('Unexpected GitHub origin')
transport = 'GitHub HTTPS'
try:
    git(old, '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=15', 'fetch', 'origin', 'main', timeout=30)
except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    git(old, 'fetch', request['bundle'], 'refs/heads/main')
    transport = 'Published GitHub commit via locally verified code-only bundle'
git(old, 'cat-file', '-t', request['commit'])
if not new.exists():
    git(old, 'worktree', 'add', '--detach', str(new), request['commit'])
if git(new, 'rev-parse', 'HEAD') != request['commit'] or git(new, 'status', '--porcelain'):
    raise RuntimeError('New worktree commit/cleanliness mismatch')
if git(old, 'rev-parse', 'HEAD') != old_head:
    raise RuntimeError('Existing worktree HEAD changed')
registry_sha = hashlib.sha256((new/'Evaluation Rules/unprotocolized_protocols.json').read_bytes()).hexdigest()
if registry_sha != request['registry_sha256']:
    raise RuntimeError('Registry hash mismatch')
env = dict(os.environ, PYTHONPATH=str(new))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    env[key] = '1'
env['PATH'] = str(pathlib.Path(request['git']).parent)+':'+env.get('PATH', '') if '/' in request['git'] else env.get('PATH', '')
tests = subprocess.run([sys.executable, '-m', 'unittest', 'dinov3.tests.test_unprotocolized_campaign', 'dinov3.tests.test_cellfmcount'], cwd=new, env=env, text=True, capture_output=True, timeout=120)
versions = {name: importlib.metadata.version(name) for name in ('torch','numpy','scipy','scikit-learn','Pillow','omegaconf','torchvision')}
print(json.dumps(dict(host=request['host'], hostname=os.uname().nodename, code_root=str(new),
 git_commit=request['commit'], code_clean=True, registry_sha256=registry_sha, environment_pass=tests.returncode==0,
 python=sys.executable, git_executable=request['git'], versions=versions, test_log=tests.stdout+tests.stderr, transport=transport,
 hest_dependencies=dict(h5py=importlib.util.find_spec('h5py') is not None), previous_checkout_unchanged=True)))
'''


def checked(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = checked(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    published = checked(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/main"]).split()[0]
    if commit != published:
        raise RuntimeError("Local benchmark commit is not published GitHub main")
    bundle = output / f"published_{commit[:12]}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "main", "^1d2330accebbb91e539b50924db3440fac598e7e"], check=True)
    registry_sha = hashlib.sha256((ROOT / "Evaluation Rules/unprotocolized_protocols.json").read_bytes()).hexdigest()
    local = ROOT.parent / f"dinov3_unprotocolized_eval_{commit[:12]}"
    if not local.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--quiet", "--detach", str(local), commit], check=True)
    local_git = ["git", "-c", "safe.directory=" + str(local), "-C", str(local)]
    env = dict(__import__("os").environ, PYTHONPATH=str(local))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    tests = subprocess.run([sys.executable, "-m", "unittest", "dinov3.tests.test_unprotocolized_campaign",
                            "dinov3.tests.test_cellfmcount"], cwd=local, env=env, text=True, capture_output=True, timeout=120)
    import importlib.metadata
    import importlib.util
    rows = [{"host": "local", "hostname": __import__("socket").gethostname(),
             "git_commit": checked([*local_git, "rev-parse", "HEAD"]),
             "code_root": str(local), "code_clean": not bool(checked([*local_git, "status", "--porcelain"])),
             "registry_sha256": hashlib.sha256((local / "Evaluation Rules/unprotocolized_protocols.json").read_bytes()).hexdigest(),
             "environment_pass": tests.returncode == 0, "python": sys.executable,
             "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "Pillow", "omegaconf", "torchvision")},
             "hest_dependencies": {"h5py": importlib.util.find_spec("h5py") is not None},
             "test_log": tests.stdout + tests.stderr, "transport": "clean local detached worktree"}]
    for host, config in HOSTS.items():
        remote_bundle = str(Path(config["old"]).parent / bundle.name)
        subprocess.run(["scp", "-q", str(bundle), f"{host}:{remote_bundle}"], check=True, timeout=90)
        request = {**config, "host": host, "commit": commit, "bundle": remote_bundle,
                   "registry_sha256": registry_sha,
                   "new": str(Path(config["old"]).parent / f"biodino_unprotocolized_eval_{commit[:12]}")}
        response = checked(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                            shlex.join([config["python"], "-c", REMOTE, json.dumps(request)])], timeout=240)
        row = json.loads(response)
        rows.append(row)
        print(host, row["git_commit"], "environment_pass=" + str(row["environment_pass"]), flush=True)
    from dinov3.eval.bio_frozen_eval.protocol_campaign import validate_sync, write_new
    path = output / "sync_manifest.json"
    write_new(path, {"benchmark_commit": commit, "machines": rows,
                     "scope": "candidate frozen regression only; no claim of full HEST/registration/detection readiness"})
    validate_sync(path, commit, registry_sha)
    print(path, flush=True)


if __name__ == "__main__":
    main()
