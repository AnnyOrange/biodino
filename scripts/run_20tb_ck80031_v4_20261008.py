#!/usr/bin/env python3
"""Run an isolated component and v4 evaluation for the 20TB ck80031 teacher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
PAIR_ROOT = REPO / "outputs/02_eval_runs/v2_20tb_ck80031_20261008"
V4_ROOT = REPO / "outputs/02_eval_runs/v2_full_v4_ck80031_20261008"
PRIOR_PAIR = REPO / "outputs/02_eval_runs/v2_20tb_paired_20261006"
BASE_RUN = REPO / "outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork36111_20261004/cls_slow2"
CHECKPOINT = BASE_RUN / "eval/training_80031/teacher_checkpoint.pth"
BASE_WORKER = REPO / "scripts/run_v2_progress_fleet_fill_20261006.py"
V4_WORKER = REPO / "scripts/run_v2_full_v4_20261007.py"
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_fleet():
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    import run_retest_fleet_20260918 as fleet
    return fleet


def prepare_base(_args):
    fleet = import_fleet()
    if (PAIR_ROOT / "campaign_manifest.json").exists():
        raise FileExistsError(PAIR_ROOT)
    prior = json.loads((PRIOR_PAIR / "campaign_manifest.json").read_text())
    config = PAIR_ROOT / "frozen_inputs/ck80031_config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE_RUN / "config.yaml", config)
    asset = dict(arm="cls_slow2_20tb", checkpoint_id="80031", path=str(CHECKPOINT),
                 config=str(config), kind="dinov3", model_id="", reserve_mib=4096)
    stat = CHECKPOINT.stat()
    if stat.st_size < 1_000_000_000 or time.time() - stat.st_mtime < 180:
        raise RuntimeError("Checkpoint is incomplete or still being written")
    manifest = {**prior,
                "campaign_scope": "SINGLE_CHECKPOINT_20TB_CK80031_COMPONENTS",
                "authorization": "2026-10-08 user requested 20TB ck80031 evaluation on 3090-qi followed by v4",
                "prior_campaign": str(PRIOR_PAIR),
                "checkpoint_assets": [asset],
                "training_roots": {"cls_slow2_20tb": str(BASE_RUN)},
                "tasks": fleet.tasks_for([asset], prior["datasets"]),
                "online_checkpoints": False,
                "external_source_hashes": {**prior["external_source_hashes"],
                    str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                    str(config): sha256(config)},
                "created_unix": time.time()}
    fleet.queue.save(PAIR_ROOT / "campaign_manifest.json", manifest)
    (PAIR_ROOT / "_state/inputs").mkdir(parents=True)
    print(json.dumps({"checkpoint": str(CHECKPOINT), "sha256": sha256(CHECKPOINT),
                      "config": str(config), "tasks": len(manifest["tasks"])}), flush=True)


def prepare_v4(_args):
    if not (PAIR_ROOT / "campaign_manifest.json").is_file():
        raise FileNotFoundError(PAIR_ROOT / "campaign_manifest.json")
    if V4_ROOT.exists() and (V4_ROOT / "campaign_manifest.json").exists():
        raise FileExistsError(V4_ROOT)
    module = runpy.run_path(str(V4_WORKER))
    globals_ = module["prepare"].__globals__
    globals_.update(PAIRED=PAIR_ROOT, ROOT=V4_ROOT)
    module["prepare"]()
    manifest_path = V4_ROOT / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["authorization"] = "2026-10-08 user requested v4 evaluation for 20TB ck80031 on 3090-qi"
    manifest["runtime_hashes"][str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    manifest["source_component_campaign"] = str(PAIR_ROOT)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"assets": manifest["assets"],
                      "extension_executions": manifest["extension_executions"]}), flush=True)


def worker_base(args):
    spec = importlib.util.spec_from_file_location("v2_progress_fill", BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.worker(Namespace(output=PAIR_ROOT, host=args.host, gpus=args.gpus,
        target_per_gpu=args.target_per_gpu, memory_target=args.memory_target,
        max_host_jobs=args.max_jobs, max_global_jobs=args.max_jobs,
        task_family="mixed"))


def worker_v4(args):
    module = runpy.run_path(str(V4_WORKER))
    globals_ = module["worker"].__globals__
    globals_.update(PAIRED=PAIR_ROOT, ROOT=V4_ROOT)
    module["worker"](Namespace(host=args.host, gpus=args.gpus, max_jobs=args.max_jobs,
        ram_reserve=args.ram_reserve, family="all"))


def inventory(_args):
    module = runpy.run_path(str(V4_WORKER))
    globals_ = module["inventory"].__globals__
    globals_.update(PAIRED=PAIR_ROOT, ROOT=V4_ROOT)
    module["inventory"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-base", "worker-base", "prepare-v4", "worker-v4", "inventory"))
    parser.add_argument("--host", default=os.uname().nodename)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--target-per-gpu", type=int, default=8)
    parser.add_argument("--memory-target", type=float, default=0.75)
    parser.add_argument("--max-jobs", type=int, default=40)
    parser.add_argument("--ram-reserve", type=float, default=96)
    args = parser.parse_args()
    {"prepare-base": prepare_base, "worker-base": worker_base,
     "prepare-v4": prepare_v4, "worker-v4": worker_v4,
     "inventory": inventory}[args.mode](args)


if __name__ == "__main__":
    main()
