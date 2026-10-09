#!/usr/bin/env python3
"""Evaluate untested 20TB route2 teachers on single-3090 cpu nodes.

The 2026-09-28 campaign already completed its 31 selected checkpoints. This
campaign contains the other stable teachers and uses the derived 30/7/14
MoNuSeg snapshot for every cell. Frozen jobs share a GPU with one dense job
until measured usage reaches the target; dense jobs retain the queue's
single-dense-per-GPU admission rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929")
PYTHON = Path("/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python")
PROTOCOL = ROOT / "Evaluation Rules/protocol_v4.json"
PRIOR = ROOT / "outputs/02_eval_runs/20tb_route2_union_v4_online_20260928"
MONUSEG = ROOT / "outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929"
OUTPUT = ROOT / "outputs/02_eval_runs/20tb_route2_remaining_v4_monu30_20261002"
RUNS = {
    "5090route2": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924",
    "3090qiroute2": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_r0r9_8x3090qi_20260927",
    "3090qiablation": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_15tb07_5tbboundary03_ablation_8x3090qi_20260928",
}
DENSE_HOSTS = ("cpu1", "cpu2", "cpu10", "cpu12", "cpu15")
FROZEN_ONLY_HOSTS = ("cpu5", "cpu8", "cpu9", "cpu11", "cpu19")


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
        raise FileExistsError("Campaign already prepared")
    fleet = fleet_module()
    old = json.loads((PRIOR / "campaign_manifest.json").read_text())
    monu = json.loads((MONUSEG / "campaign_manifest.json").read_text())
    if old["protocol_id"] != "bio-eval-union-v4":
        raise ValueError("Prior campaign is not v4")
    selected = {(a["arm"], str(a["checkpoint_id"])) for a in old["checkpoint_assets"]}
    for asset in old["checkpoint_assets"]:
        keys = [t["key"] for t in old["tasks"] if t["asset"]["path"] == asset["path"]]
        if any(not (PRIOR / "_state/done" / f"{key}.json").exists() for key in keys):
            raise RuntimeError(f"Prior checkpoint is incomplete: {asset['arm']}:{asset['checkpoint_id']}")
        monu_key = f"{asset['arm']}_ck{asset['checkpoint_id']}__segmentation__monuseg__primary-last__formal-static-v1"
        if not (MONUSEG / "_state/done" / f"{monu_key}.json").exists():
            raise RuntimeError(f"Prior MoNuSeg checkpoint is incomplete: {monu_key}")
    assets = []
    for arm, root in RUNS.items():
        for path in sorted((root / "eval").glob("training_*/teacher_checkpoint.pth"),
                           key=lambda p: int(p.parent.name.split("_")[-1])):
            checkpoint_id = path.parent.name.split("_")[-1]
            if (arm, checkpoint_id) in selected:
                continue
            stat = path.stat()
            if stat.st_size < 1_000_000_000 or time.time() - stat.st_mtime < 180:
                raise RuntimeError(f"Checkpoint is not stable: {path}")
            assets.append(dict(arm=arm, checkpoint_id=checkpoint_id, path=str(path),
                               config=str(root / "config.yaml"), kind="dinov3", model_id="",
                               reserve_mib=4096))
    if not assets:
        raise RuntimeError("No untested checkpoints")
    monu_spec = next(d for d in monu["datasets"] if d["dataset"] == "monuseg")
    if monu_spec["counts"] != {"train": 30, "val": 7, "test": 14}:
        raise RuntimeError("Unexpected MoNuSeg split")
    datasets = [monu_spec if d["dataset"] == "monuseg" else d for d in old["datasets"]]
    source = json.loads((SNAPSHOT / "source_snapshot.json").read_text())
    if source != monu["source_snapshot"]:
        raise RuntimeError("MoNuSeg evaluator snapshot changed")
    script = Path(__file__).resolve()
    external = {str(p): sha256(p) for p in (script, PROTOCOL,
                 SNAPSHOT / "scripts/run_retest_fleet_20260918.py",
                 ROOT / "outputs/02_eval_inputs/shared_fleet_reference_20260930/raw_campaign_manifest.json")}
    manifest = dict(
        protocol_id="bio-eval-union-v4", campaign_scope="V4_SHARED_COMPONENTS_PLUS_MONUSEG_30_7_14",
        full_v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
        explicit_user_authorization="2026-10-02: test 20TB checkpoints with v4 and MoNuSeg 30/7/14 on cpu* 3090 GPUs",
        source_snapshot=source, source_snapshot_path=str(SNAPSHOT), git_commit=source["git_commit"],
        numerical_environment=monu["numerical_environment"], external_source_hashes=external,
        protocol_rules_path=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
        prior_campaign=str(PRIOR), prior_monuseg_campaign=str(MONUSEG),
        checkpoint_assets=assets, training_roots={k: str(v) for k, v in RUNS.items()},
        checkpoint_selection_rule="All stable teacher checkpoints absent from completed prior v4 and 30/7/14 campaigns",
        datasets=datasets, expected_unique_dataset_counts=old["expected_unique_dataset_counts"],
        benchmark_root=old["benchmark_root"], batch_size=64, autocast_dtype="bf16", seed=0,
        num_workers=2, online_checkpoints=False, legacy_reuse=False,
        isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True,
        scheduling=dict(dense_hosts=DENSE_HOSTS, frozen_only_hosts=FROZEN_ONLY_HOSTS,
                        frozen_target_gpu_memory_fraction=0.55,
                        frozen_max_gpu_memory_fraction=0.75), created_unix=time.time())
    manifest["tasks"] = fleet.tasks_for(assets, datasets)
    OUTPUT.mkdir(parents=True)
    fleet.queue.save(OUTPUT / "campaign_manifest.json", manifest)
    (OUTPUT / "_state/inputs").mkdir(parents=True)
    print(json.dumps(dict(assets=len(assets), tasks=len(manifest["tasks"]),
                          per_checkpoint=len(manifest["tasks"]) // len(assets),
                          arms={arm: sum(a["arm"] == arm for a in assets) for arm in RUNS},
                          monuseg_split=monu_spec["split_identity_sha256"])), flush=True)


def dense_running(host: str) -> bool:
    for path in (OUTPUT / "_state/running").glob("*__segmentation__*.json"):
        try:
            if json.loads(path.read_text()).get("host") == host:
                return True
        except (OSError, ValueError):
            continue
    return False


def predicted_gpu_fraction(gpu: int, host: str) -> float:
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                                   "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    used, total = (float(v) for v in out.strip().split(","))
    now = time.time()
    recent = []
    for path in (OUTPUT / "_state/running").glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("host") in (host, host + "-co") and now - float(record.get("time", 0)) < 180:
            recent.append(record)
    if recent and now - max(float(r["time"]) for r in recent) < 45:
        return 1.0
    return (used + 4096 * len(recent)) / total


def worker(args):
    fleet = fleet_module()
    if args.role == "dense":
        guard = lambda gpu, actual, task: task["dataset"]["task"] == "segmentation"
        target = 1
    else:
        def guard(gpu, actual, task):
            if task["dataset"]["task"] == "segmentation":
                return False
            if args.role == "cofrozen" and not dense_running(args.host.removesuffix("-co")):
                return False
            host = args.host.removesuffix("-co")
            return predicted_gpu_fraction(gpu, host) < 0.55
        target = 8
    ns = argparse.Namespace(output=OUTPUT, host=args.host, gpus=[0], target_per_gpu=target,
                            max_host_jobs=target, max_global_jobs=400,
                            task_family="mixed", admission_guard=guard)
    fleet.worker(ns)


def launch(_args):
    if not (OUTPUT / "campaign_manifest.json").is_file():
        raise FileNotFoundError("Prepare campaign first")
    plan = ([(h, "dense") for h in DENSE_HOSTS] +
            [(h, "cofrozen") for h in DENSE_HOSTS] +
            [(h, "frozen") for h in FROZEN_ONLY_HOSTS])
    for host, role in plan:
        name = f"{host}-co" if role == "cofrozen" else host
        log = OUTPUT / "logs" / f"{name}.{role}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", str(Path(__file__).resolve()),
                   "worker", "--host", name, "--role", role]
        remote = f"cd {shlex.quote(str(SNAPSHOT))} && setsid -f {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1 < /dev/null"
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
                                capture_output=True, text=True)
        print(json.dumps(dict(host=host, role=role, returncode=result.returncode,
                              stderr=result.stderr.strip()[-300:])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "launch"))
    parser.add_argument("--host", default=os.uname().nodename)
    parser.add_argument("--role", choices=("dense", "cofrozen", "frozen"), default="frozen")
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "launch": launch}[args.mode](args)
