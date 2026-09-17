#!/usr/bin/env python3
"""Publish-verified code sync, offline environment overlays and runtime tests.

An unreachable host is recorded as failed, never silently waived. Existing
training checkouts/environments are preserved. All source changes originate in
the authoritative local commit, including this installer and its test suite.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from sync_unprotocolized_campaign import HOSTS

ROOT = Path(__file__).resolve().parents[1]
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
TESTS = ["test_unprotocolized_campaign", "test_cellfmcount", "test_grouped_benchmarks",
         "test_opencell_transloc", "test_bio_registration", "test_hest_benchmark",
         "test_native_detection", "test_native_detection_campaign", "test_expansion_campaign"]
REMOTE_CODE = r'''
import json, pathlib, subprocess, sys
r=json.loads(sys.argv[1]); old=pathlib.Path(r['old']); new=pathlib.Path(r['new'])
def git(root,*args,timeout=60):
 return subprocess.check_output([r['git'],'-c','safe.directory='+str(root),'-C',str(root),*args],text=True,stderr=subprocess.STDOUT,timeout=timeout).strip()
head=git(old,'rev-parse','HEAD')
if git(old,'status','--porcelain'): raise RuntimeError('Existing sync worktree dirty')
if git(old,'remote','get-url','origin')!='https://github.com/AnnyOrange/biodino.git': raise RuntimeError('Unexpected origin')
transport='GitHub HTTPS'
try: git(old,'fetch','origin','main',timeout=30)
except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
 git(old,'fetch',r['bundle'],'refs/heads/main'); transport='Published GitHub commit through verified code-only bundle'
if not new.exists(): git(old,'worktree','add','--detach',str(new),r['commit'])
if git(new,'rev-parse','HEAD')!=r['commit'] or git(new,'status','--porcelain'): raise RuntimeError('Sync worktree mismatch')
if git(old,'rev-parse','HEAD')!=head: raise RuntimeError('Existing checkout changed')
print(json.dumps({'code_root':str(new),'transport':transport,'code_clean':True}))
'''
INVENTORY = r'''
import importlib.metadata as m,json
print(json.dumps({d.metadata['Name'].lower().replace('_','-'):m.version(d.metadata['Name']) for d in m.distributions()}))
'''
REMOTE_VERIFY = r'''
import hashlib,json,os,pathlib,subprocess,sys
r=json.loads(sys.argv[1]); root=pathlib.Path(r['new']); env=dict(os.environ,**r['threads'],PYTHONPATH=str(root))
if '/' in r['git']: env['PATH']=str(pathlib.Path(r['git']).parent)+':'+env.get('PATH','')
tests=subprocess.run([sys.executable,'-m','pytest',*[str(root/'dinov3/tests'/(name+'.py')) for name in r['tests']],'-q'],cwd=root,env=env,capture_output=True,text=True,timeout=300)
state=json.loads(subprocess.check_output([sys.executable,str(root/'scripts/evaluation_environment.py'),'--require-cuda'],env=env,text=True))
registries={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'Evaluation Rules').glob('*.json')}
print(json.dumps(dict(state,host=r['host'],code_root=str(root),git_commit=r['commit'],code_clean=True,
 environment_pass=tests.returncode==0,registries=registries,registry_sha256=registries['Evaluation Rules/unprotocolized_protocols.json'],
 test_log=tests.stdout+tests.stderr,git_executable=r['git'],transport=r['transport'])))
'''


def command(args, timeout=120, **kwargs):
    return subprocess.check_output(args, text=True, timeout=timeout, **kwargs).strip()


def remote(host, python, source, request=None, timeout=120):
    args = [python, "-c", source]
    if request is not None:
        args.append(json.dumps(request))
    return command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(args)], timeout=timeout)


def copy(host, local, target):
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    str(local), f"{host}:{target}"], check=True, timeout=1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hosts", nargs="+", choices=list(HOSTS), default=list(HOSTS))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if command(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/main"]).split()[0] != commit:
        raise RuntimeError("The authoritative commit is not published on GitHub")
    bundle = output / f"published_{commit[:12]}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "main",
                        "^1d2330accebbb91e539b50924db3440fac598e7e"], check=True)
    local = ROOT.parent / f"dinov3_unprotocolized_eval_{commit[:12]}"
    if not local.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(local), commit], check=True)
    if command(["git", "-c", f"safe.directory={local}", "-C", str(local), "status", "--porcelain"]):
        raise RuntimeError("Local synchronized worktree is dirty")
    env = dict(os.environ, **THREADS, PYTHONPATH=str(local))
    state = json.loads(command([sys.executable, str(local / "scripts/evaluation_environment.py")], env=env))
    tests = subprocess.run([sys.executable, "-m", "pytest", *[str(local / "dinov3/tests" / (name + ".py"))
                           for name in TESTS], "-q"], cwd=local, env=env, text=True, capture_output=True, timeout=300)
    registries = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (local / "Evaluation Rules").glob("*.json")}
    rows = [{**state, "host": "local", "code_root": str(local), "git_commit": commit,
             "code_clean": True, "environment_pass": tests.returncode == 0, "registries": registries,
             "registry_sha256": registries["Evaluation Rules/unprotocolized_protocols.json"],
             "test_log": tests.stdout + tests.stderr, "transport": "local clean detached worktree"}]
    for host in args.hosts:
        config = HOSTS[host]
        try:
            inventory = json.loads(remote(host, config["python"], INVENTORY, timeout=30))
            base = output / f"{host}_base_inventory.json"
            base.write_text(json.dumps(inventory, indent=2) + "\n")
            parent = Path(config["old"]).parent
            remote_bundle = str(parent / bundle.name)
            copy(host, bundle, remote_bundle)
            request = {**config, "host": host, "commit": commit, "bundle": remote_bundle,
                       "new": str(parent / f"biodino_unprotocolized_eval_{commit[:12]}")}
            synced = json.loads(remote(host, config["python"], REMOTE_CODE, request))
            archive = output / f"{host}_environment_delta.tar"
            if not archive.exists():
                with (output / f"{host}_export.log").open("w") as log:
                    subprocess.run([sys.executable, str(local / "scripts/export_evaluation_environment.py"),
                                    "--output", str(archive), "--base-inventory", str(base)], check=True, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=600)
            archive_manifest = json.loads(archive.with_suffix(".tar.json").read_text())
            remote_archive = str(parent / archive.name)
            copy(host, archive, remote_archive)
            venv = str(parent / "eval_envs/hs6_protocol_v2")
            install = [config["python"], str(Path(request["new"]) / "scripts/evaluation_environment.py"),
                       "--install", venv, "--environment-archive", remote_archive,
                       "--archive-sha256", archive_manifest["sha256"], "--require-cuda"]
            with (output / f"{host}_install.log").open("w") as log:
                subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                                shlex.join(install)], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            request.update(threads=THREADS, tests=TESTS, transport=synced["transport"])
            row = json.loads(remote(host, venv + "/bin/python", REMOTE_VERIFY, request, timeout=360))
            if row["environment_sha256"] != state["environment_sha256"]:
                raise RuntimeError("Resulting pinned numerical/environment fingerprint differs from local")
            row["environment_archive_sha256"] = archive_manifest["sha256"]
            rows.append(row)
            print(host, "PASS", commit, row["environment_sha256"], flush=True)
        except Exception as error:
            rows.append({"host": host, "git_commit": commit, "code_clean": False, "environment_pass": False,
                         "status": "FAILURE", "error": str(error)})
            print(host, "FAILURE", str(error), flush=True)
        (output / "sync_manifest.json").write_text(json.dumps({"benchmark_commit": commit, "machines": rows}, indent=2) + "\n")
    if {row["host"] for row in rows} != {"local", *HOSTS} or not all(row["environment_pass"] for row in rows):
        raise SystemExit("Not all four machines synchronized; full benchmark admission remains closed")
    print(output / "sync_manifest.json")


if __name__ == "__main__":
    main()
