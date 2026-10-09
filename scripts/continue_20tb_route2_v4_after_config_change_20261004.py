#!/usr/bin/env python3
"""Continue unfinished route2 v4 cells after the training config was rewritten.

The original campaign remains paused and immutable. This campaign freezes each
run's config at admission, so later training restarts cannot change its inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929")
PYTHON = Path("/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python")
PRIOR = ROOT / "outputs/02_eval_runs/20tb_route2_remaining_v4_monu30_20261002"
OUTPUT = ROOT / "outputs/02_eval_runs/20tb_route2_v4_monu30_config_frozen_20261004"
DENSE_HOSTS = ("cpu1", "cpu2", "cpu10", "cpu12", "cpu15")
FROZEN_HOSTS = ("cpu5", "cpu8", "cpu9", "cpu11", "cpu19")
ONLINE_AFTER = {"5090route2": 36111, "3090qiroute2": 7319}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fleet_module():
    os.environ["DINOV3_CODE_ROOT"] = str(SNAPSHOT)
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    import run_retest_fleet_20260918 as fleet
    return fleet


def prepare(_args):
    if (OUTPUT / "campaign_manifest.json").exists():
        raise FileExistsError("Continuation campaign already exists")
    active = {path.stem for path in (PRIOR / "_state/running").glob("*.json")}
    old = json.loads((PRIOR / "campaign_manifest.json").read_text())
    pending = [task for task in old["tasks"]
               if task["key"] not in active and
               not (PRIOR / "_state/done" / f"{task['key']}.json").exists()]
    if not pending:
        raise RuntimeError("No unfinished cells")
    assets = {}
    for task in pending:
        original = task["asset"]
        key = original["path"]
        if key not in assets:
            frozen = OUTPUT / "frozen_configs" / f"{original['arm']}.yaml"
            frozen.parent.mkdir(parents=True, exist_ok=True)
            if not frozen.exists():
                shutil.copyfile(original["config"], frozen)
            assets[key] = {**original, "config": str(frozen)}
        task["asset"] = assets[key]
    frozen_hashes = {asset["config"]: sha256(Path(asset["config"])) for asset in assets.values()}
    manifest = {**old,
                "created_unix": time.time(),
                "prior_campaign": str(PRIOR),
                "continuation_reason": "Training resume rewrote config.yaml; prior input fingerprint paused the queue",
                "prior_inflight_excluded": sorted(active),
                "checkpoint_assets": list(assets.values()),
                "tasks": pending,
                "online_checkpoints": False,
                "frozen_config_sha256": frozen_hashes,
                "external_source_hashes": {**old["external_source_hashes"],
                    str(Path(__file__).resolve()): sha256(Path(__file__).resolve()), **frozen_hashes},
                "scheduling": {"dense_hosts": DENSE_HOSTS, "frozen_hosts": FROZEN_HOSTS,
                               "jobs_per_gpu": 1, "reason": "Avoid the prior concurrent-cell CUDA OOM failures"}}
    fleet_module().queue.save(OUTPUT / "campaign_manifest.json", manifest)
    (OUTPUT / "_state/inputs").mkdir(parents=True)
    print(json.dumps({"assets": len(assets), "tasks": len(pending),
                      "frozen_config_sha256": frozen_hashes}), flush=True)


def worker(args):
    fleet = fleet_module()
    dense = args.role == "dense"

    def guard(_gpu, _actual, task):
        if (task["dataset"]["task"] == "segmentation") != dense:
            return False
        if dense:
            for path in (PRIOR / "_state/running").glob("*.json"):
                try:
                    if json.loads(path.read_text()).get("host") == args.host:
                        return False
                except (OSError, ValueError):
                    continue
        return True

    ns = argparse.Namespace(output=OUTPUT, host=args.host, gpus=[0], target_per_gpu=1,
                            max_host_jobs=1, max_global_jobs=100,
                            task_family="mixed", admission_guard=guard)
    fleet.worker(ns)


def watch(_args):
    fleet = fleet_module()
    lock = (OUTPUT / "_state/online_watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        with (OUTPUT / "_state/manifest.lock").open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            manifest = json.loads((OUTPUT / "campaign_manifest.json").read_text())
            known = {asset["path"] for asset in manifest["checkpoint_assets"]}
            additions = []
            for arm, floor in ONLINE_AFTER.items():
                root = Path(manifest["training_roots"][arm])
                config = OUTPUT / "frozen_configs" / f"{arm}.yaml"
                for path in sorted((root / "eval").glob("training_*/teacher_checkpoint.pth")):
                    checkpoint_id = int(path.parent.name.split("_")[-1])
                    if checkpoint_id <= floor or str(path) in known:
                        continue
                    stat = path.stat()
                    if stat.st_size < 1_000_000_000 or time.time() - stat.st_mtime < 180:
                        continue
                    asset = dict(arm=arm, checkpoint_id=str(checkpoint_id), path=str(path),
                                 config=str(config), kind="dinov3", model_id="", reserve_mib=4096)
                    fingerprint = fleet.queue.checkpoint_record(OUTPUT, asset)
                    with (OUTPUT / "_state/checkpoint_admission.jsonl").open("a") as stream:
                        stream.write(json.dumps({"time": time.time(), "asset": asset,
                                                 "fingerprint": fingerprint}) + "\n")
                    additions.append(asset)
            if additions:
                manifest["checkpoint_assets"] += additions
                manifest["tasks"] += fleet.tasks_for(additions, manifest["datasets"])
                manifest["tasks"].sort(key=lambda task: "__segmentation__monuseg__" not in task["key"])
                fleet.queue.save(OUTPUT / "campaign_manifest.json", manifest)
                print("ADMITTED", [f"{a['arm']}:{a['checkpoint_id']}" for a in additions],
                      flush=True)
        time.sleep(120)


def launch(_args):
    if not (OUTPUT / "campaign_manifest.json").is_file():
        raise FileNotFoundError("Prepare campaign first")
    for host, role in ([(host, "dense") for host in DENSE_HOSTS] +
                       [(host, "frozen") for host in FROZEN_HOSTS]):
        log = OUTPUT / "logs" / f"{host}.{role}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", str(Path(__file__).resolve()),
                   "worker", "--host", host, "--role", role]
        remote = (f"cd {shlex.quote(str(SNAPSHOT))} && setsid -f {shlex.join(command)}"
                  f" >> {shlex.quote(str(log))} 2>&1 < /dev/null")
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                 host, remote], capture_output=True, text=True)
        print(json.dumps({"host": host, "role": role, "returncode": result.returncode,
                          "stderr": result.stderr.strip()[-300:]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "watch", "launch"))
    parser.add_argument("--host", default=os.uname().nodename)
    parser.add_argument("--role", choices=("dense", "frozen"), default="frozen")
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "watch": watch, "launch": launch}[args.mode](args)
