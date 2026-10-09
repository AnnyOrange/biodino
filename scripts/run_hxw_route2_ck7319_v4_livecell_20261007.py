#!/usr/bin/env python3
"""Run the identity-matched v4 LIVECell cell for route2 checkpoint 7319 on hxw GPU 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/xzj/monuseg_t30v7_remote_20260930")
SNAPSHOT = BASE / "snapshot"
OUTPUT = BASE / "route2_r0r9_ck7319_v4_livecell"
BENCHMARK = Path("/data/benchmark")
CHECKPOINT = Path("/home/xzj/route2_20tb_20261006/output/ckpt/7319/checkpoint.pth")
CONFIG = Path("/home/xzj/route2_20tb_20261006/output/config.yaml")
PROTOCOL = BASE / "protocol_v4_route2_20261007.json"
EXPECTED_SPLIT = "9f0329bab2f895d9c9224024ec8c0eddb58c56422500bcd8334e1557b7cd9c54"
EXPECTED_INVENTORY = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fleet_module():
    os.environ["DINOV3_CODE_ROOT"] = str(SNAPSHOT)
    os.environ["BENCHMARK_MODEL_ROOT"] = str(SNAPSHOT / "evaluation_external/benchmark_model")
    sys.path[:0] = [str(SNAPSHOT / "scripts"), str(SNAPSHOT)]
    import run_retest_fleet_20260918 as fleet
    return fleet


def prepare():
    if (OUTPUT / "campaign_manifest.json").exists():
        raise FileExistsError("Existing LIVECell campaign")
    fleet = fleet_module()
    from benchmark_eval.rules_dense_preflight import audit_dense

    spec = audit_dense("livecell", BENCHMARK)
    if (spec["split_identity_sha256"], spec["dataset_inventory_sha256"]) != (
        EXPECTED_SPLIT, EXPECTED_INVENTORY
    ):
        raise RuntimeError("LIVECell identity differs from the existing v4 campaign")
    if not CHECKPOINT.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("Resident checkpoint or training config missing")
    if json.loads(PROTOCOL.read_text())["protocol_id"] != "bio-eval-union-v4":
        raise ValueError("Expected v4 protocol")

    OUTPUT.mkdir(parents=True)
    frozen_config = OUTPUT / "config_7319.yaml"
    shutil.copy2(CONFIG, frozen_config)
    spec.update(comparison_view="primary-last", reserve_mib=16000)
    asset = dict(arm="route2_r0r9", checkpoint_id="7319", path=str(CHECKPOINT),
                 config=str(frozen_config), kind="dinov3", model_id="", reserve_mib=4096)
    source = json.loads((SNAPSHOT / "source_snapshot.json").read_text())
    manifest = dict(
        protocol_id="bio-eval-union-v4", campaign_scope="IDENTITY_MATCHED_LIVECELL_COMPONENT_ONLY",
        full_v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
        source_snapshot=source, source_snapshot_path=str(SNAPSHOT), git_commit=source["git_commit"],
        numerical_environment=fleet.NUMERICAL,
        external_source_hashes={str(PROTOCOL): sha256(PROTOCOL),
                                str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                                str(frozen_config): sha256(frozen_config)},
        protocol_rules_path=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
        checkpoint_assets=[asset], datasets=[spec], tasks=fleet.tasks_for([asset], [spec]),
        benchmark_root=str(BENCHMARK), batch_size=64, autocast_dtype="bf16", seed=0,
        num_workers=2, online_checkpoints=False, legacy_reuse=False,
        isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True,
        created_unix=time.time(),
    )
    fleet.queue.save(OUTPUT / "campaign_manifest.json", manifest)
    (OUTPUT / "_state/inputs").mkdir(parents=True)
    print(json.dumps({"tasks": len(manifest["tasks"]), "split": EXPECTED_SPLIT}), flush=True)


def gpu_idle(gpu: int) -> bool:
    pids = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return not pids and int(used) < 1024


def worker():
    fleet = fleet_module()
    args = argparse.Namespace(output=OUTPUT, host="hxw-route2-livecell-gpu5", gpus=[5],
                              target_per_gpu=1, max_host_jobs=1, max_global_jobs=2,
                              task_family="segmentation",
                              admission_guard=lambda gpu, _actual, _task: gpu_idle(gpu))
    fleet.worker(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "worker"))
    mode = parser.parse_args().mode
    {"prepare": prepare, "worker": worker}[mode]()
