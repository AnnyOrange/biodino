#!/usr/bin/env python3
"""Register stable 5TB recovery teachers with resident v4 and MoNuSeg queues."""

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


SITES = {
    "lyx": {
        "arm": "global_cls",
        "run": Path("/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls"),
        "eval": Path("/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/global_cls"),
        "config": Path("/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/bin/config.yaml"),
        "monu": Path("/data/xuzijing/monuseg_t30v7_remote_20260930/campaign_v3"),
        "snapshot": Path("/data/xuzijing/monuseg_t30v7_remote_20260930/snapshot"),
    },
    "hxw": {
        "arm": "global_cls_w3",
        "run": Path("/home/xzj/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls_w3"),
        "eval": Path("/data/hs6_l5_v2_recovery_eval_20260930/global_cls_w3"),
        "config": Path("/data/hs6_l_5tb_nogram_eval_20260921/source/config.yaml"),
        "monu": Path("/data/v2_monu30_resume_20261009"),
        "snapshot": Path("/data/v2_monu30_resume_20261009/snapshot"),
    },
}


def stable_teachers(run):
    for checkpoint in sorted((run / "eval").glob("training_*/teacher_checkpoint.pth")):
        try:
            ck = int(checkpoint.parent.name.split("_")[-1])
            stat = checkpoint.stat()
        except (OSError, ValueError):
            continue
        if ck <= 35135 or ck > 50263 or stat.st_size < 1_000_000_000:
            continue
        if time.time() - stat.st_mtime < 180:
            continue
        if not (run / "ckpt" / str(ck) / "checkpoint.pth").is_file():
            continue
        yield ck, checkpoint


def ensure_link(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected symlink target: {link}")
    elif link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected existing file: {link}")
    else:
        link.symlink_to(target)


def fleet_module(snapshot):
    os.environ["DINOV3_CODE_ROOT"] = str(snapshot)
    sys.path.insert(0, str(snapshot / "scripts"))
    spec = importlib.util.spec_from_file_location("resident_v4_fleet", snapshot / "scripts/run_retest_fleet_20260918.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_monu(site, teachers):
    campaign = site["monu"]
    manifest_path = campaign / "campaign_manifest.json"
    if not manifest_path.is_file():
        print("MONUSEG_CAMPAIGN_WAIT", campaign, flush=True)
        return 0
    fleet = fleet_module(site["snapshot"])
    lock_path = campaign / "_state/manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        datasets = manifest["datasets"]
        if len(datasets) != 1 or datasets[0]["dataset"] != "monuseg":
            raise RuntimeError("Unexpected MoNuSeg campaign datasets")
        if datasets[0]["counts"] != {"train": 30, "val": 7, "test": 14}:
            raise RuntimeError("MoNuSeg split is not 30/7/14")
        if not manifest["site_membership_reference"]["matched"]:
            raise RuntimeError("MoNuSeg site membership is not verified")
        known = {item["path"] for item in manifest["checkpoint_assets"]}
        new = []
        for ck, checkpoint in teachers:
            if str(checkpoint) not in known:
                new.append({"arm": "v2_" + site["arm"], "checkpoint_id": str(ck),
                            "path": str(checkpoint), "config": str(site["config"]),
                            "kind": "dinov3", "model_id": "", "reserve_mib": 4096})
        if new:
            manifest["checkpoint_assets"].extend(new)
            manifest["tasks"].extend(fleet.tasks_for(new, datasets))
            manifest["online_checkpoints"] = True
            manifest.setdefault("admission_history", []).append({
                "time": time.time(), "arms": [item["arm"] for item in new],
                "authorization": "2026-10-09 test ongoing 5TB w=1 and w=3 checkpoints with v4 and MonuSeg 30/7/14",
            })
            fleet.queue.save(manifest_path, manifest)
        return len(new)


def scan(site):
    teachers = list(stable_teachers(site["run"]))
    ensure_link(site["eval"] / "source/config.yaml", site["config"])
    for ck, checkpoint in teachers:
        ensure_link(site["eval"] / "adapters" / str(ck) / "checkpoint.pth", checkpoint)
    (site["eval"] / "results/_state/done").mkdir(parents=True, exist_ok=True)
    added = register_monu(site, teachers)
    print("SCAN", site["arm"], "stable", [ck for ck, _ in teachers],
          "monuseg_added", added, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=SITES, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    site = SITES[args.site]
    if not site["config"].is_file():
        raise FileNotFoundError(site["config"])
    lock_path = site["eval"] / "resume_v4_watcher.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            scan(site)
            if args.once:
                return
            time.sleep(120)


if __name__ == "__main__":
    main()
