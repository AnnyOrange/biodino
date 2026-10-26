#!/usr/bin/env python3
"""Export rolling 1PB teachers and admit them to the pinned V4 component queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
RUN = REPO / "outputs/01_training_runs/HS6_L_random1pb_rolling_wds_8x5090zxr_ddp_b32_20261012"
BASE = REPO / "outputs/01_training_runs/HS6_L_random1pb1m_clean_robust_biosafe256_gb2048_lr1e4_e1_seed0_8x5090zxr_ddp_b32_20261012"
OUT = Path("/mnt/huawei_blm/rolling_1pb_eval_v4_20261019")
REFERENCE = REPO / "outputs/02_eval_runs/v2_20tb_paired_20261006/campaign_manifest.json"
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def full_checkpoint_records(run_dir: Path = RUN) -> list[dict]:
    journal = run_dir / "rolling_controller.jsonl"
    if not journal.exists():
        return []
    records = []
    for line in journal.read_text().splitlines():
        item = json.loads(line)
        step = int(item["checkpoint_iteration"] if "checkpoint_iteration" in item else
                   Path(item["checkpoint"]).parent.name)
        batch = item["batch_id"]
        config = run_dir / "segment_configs" / f"{batch}.yaml"
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_file():
            retained = run_dir / "ckpt" / "retained" / str(step) / "checkpoint.pth"
            if retained.is_file():
                checkpoint = retained
        records.append(dict(step=step, checkpoint=checkpoint, config=config))
    return records


def export_teacher(record: dict, teacher_root: Path | None = None) -> tuple[Path, Path] | None:
    import torch

    step = record["step"]
    source = record["checkpoint"]
    config = record["config"]
    directory = (teacher_root or OUT / "teachers") / f"training_{step}"
    target = directory / "teacher_checkpoint.pth"
    frozen_config = directory / "config.yaml"
    provenance = directory / "export.json"
    if target.is_file() and frozen_config.is_file() and provenance.is_file():
        return target, frozen_config
    if not source.is_file() or not config.is_file():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("iteration") != step:
        raise RuntimeError(f"Checkpoint iteration mismatch for {source}")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError(f"No model weights in {source}")
    teacher = {key.removeprefix("teacher."): value for key, value in model.items()
               if key.startswith("teacher.")}
    if not any(key.startswith("backbone.") for key in teacher):
        raise RuntimeError(f"No EMA teacher backbone in {source}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    torch.save({"teacher": teacher}, temporary)
    os.replace(temporary, target)
    temporary_config = frozen_config.with_name(f".{frozen_config.name}.{os.getpid()}.partial")
    shutil.copyfile(config, temporary_config)
    os.replace(temporary_config, frozen_config)
    atomic_json(provenance, dict(step=step, source=str(source), source_sha256=digest(source),
                                 teacher=str(target), teacher_sha256=digest(target),
                                 teacher_tensors=len(teacher), config_source=str(config),
                                 config_sha256=digest(frozen_config), exported_unix=time.time()))
    print(f"EXPORTED step={step} tensors={len(teacher)} bytes={target.stat().st_size}", flush=True)
    return target, frozen_config


def evaluator():
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    import run_retest_fleet_20260918 as fleet
    return fleet


def prepare() -> None:
    destination = OUT / "campaign_manifest.json"
    if destination.exists():
        with (OUT / "_state/manifest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = json.loads(destination.read_text())
            if manifest.get("campaign_scope") != "ROLLING_1PB_ALL_COMPLETED_CHECKPOINTS_COMPONENTS":
                raise RuntimeError(f"Unexpected campaign at {destination}")
            if manifest.get("no_checkpoint_or_data_transfers") is not False:
                manifest["no_checkpoint_or_data_transfers"] = False
                manifest["checkpoint_export"] = "EMA teacher exported from each completed training checkpoint"
                atomic_json(destination, manifest)
        return
    reference = json.loads(REFERENCE.read_text())
    manifest = {key: reference[key] for key in (
        "protocol_id", "datasets", "benchmark_root", "numerical_environment",
        "source_snapshot", "git_commit", "source_snapshot_path", "batch_size",
        "autocast_dtype", "n_last_blocks", "use_avgpool", "seed", "num_workers")}
    manifest.update(
        authorization="2026-10-19 user requested concurrent rolling 1PB testing",
        campaign_scope="ROLLING_1PB_ALL_COMPLETED_CHECKPOINTS_COMPONENTS",
        checkpoint_assets=[], tasks=[], online_checkpoints=True,
        isolate_runtime_failures=True, full_v3_aggregate_allowed=False,
        v4_aggregate_allowed=False, no_checkpoint_or_data_transfers=False,
        checkpoint_export="EMA teacher exported from each completed training checkpoint",
        training_roots={"rolling1pb": str(RUN)}, created_unix=time.time(),
        scheduling={"host": "3090-qi", "gpus": [0, 1, 3, 4],
                    "target_per_gpu": 5, "memory_target": 0.60})
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, manifest)
    (OUT / "_state/inputs").mkdir(parents=True, exist_ok=True)
    print(f"PREPARED {destination}", flush=True)


def admit(arm: str, step: int, checkpoint: Path, config: Path) -> bool:
    fleet = evaluator()
    manifest_path = OUT / "campaign_manifest.json"
    lock_path = OUT / "_state/manifest.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        if any(item["path"] == str(checkpoint) for item in manifest["checkpoint_assets"]):
            return False
        asset = dict(arm=arm, checkpoint_id=str(step), path=str(checkpoint),
                     config=str(config), kind="dinov3", model_id="", reserve_mib=4096)
        manifest["checkpoint_assets"].append(asset)
        manifest["tasks"].extend(fleet.tasks_for([asset], manifest["datasets"]))
        atomic_json(manifest_path, manifest)
    print(f"ADMITTED step={step} tasks={len(manifest['tasks'])}", flush=True)
    return True


def prioritize() -> None:
    manifest_path = OUT / "campaign_manifest.json"
    with (OUT / "_state/manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        if not manifest["tasks"]:
            return
        latest = max(int(asset["checkpoint_id"]) for asset in manifest["checkpoint_assets"])
        first_arm = manifest["checkpoint_assets"][0]["arm"]
        order = {task["key"].split("__", 1)[1]: index for index, task in
                 enumerate(task for task in manifest["tasks"] if task["asset"]["arm"] == first_arm)}

        def priority(task: dict) -> tuple[int, int, int]:
            step = int(task["asset"]["checkpoint_id"])
            tier = 0 if step == latest else 1 if step == 487 else 2
            return order.get(task["key"].split("__", 1)[1], 1000), tier, -step

        sorted_tasks = sorted(manifest["tasks"], key=priority)
        if sorted_tasks != manifest["tasks"]:
            manifest["tasks"] = sorted_tasks
            atomic_json(manifest_path, manifest)


def scan() -> None:
    prepare()
    baseline = BASE / "eval/training_487/teacher_checkpoint.pth"
    if baseline.is_file():
        admit("rolling1pb_baseline", 487, baseline, BASE / "config.yaml")
    for record in full_checkpoint_records():
        result = export_teacher(record)
        if result is not None:
            admit("rolling1pb", record["step"], *result)
    prioritize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("once", "watch"))
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            scan()
        except Exception as error:
            print(f"SCAN_ERROR {type(error).__name__}: {error}", flush=True)
            if args.mode == "once":
                raise
        if args.mode == "once":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
