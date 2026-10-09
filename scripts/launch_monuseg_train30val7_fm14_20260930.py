#!/usr/bin/env python3
"""FM14 part of the MoNuSeg train30 / extra7 val / test14 re-test (2026-09-30).

The main re-test (monuseg_train30val7_test14_retest_20260929) failed all 14 FM cells: the live
benchmark_model run_fm_dense_rules.py / rules_features.py changed on 2026-09-23 and no longer match
the registered external-loader fingerprint.  This campaign uses dinov3_monuseg_train30val7_fm_snapshot_20260930,
which pins those two files to the registered 09-18 copies inside the snapshot.

Original description of the main re-test:

User instruction 2026-09-29: all MoNuSeg numbers must use the 30/7/14 split; the 24/6 seed split
and the legacy 37-pool random 7-val index are superseded.  Evaluator: the derived snapshot
dinov3_monuseg_train30val7_snapshot_20260929 (only MoNuSeg split admission differs from the v4
evaluator snapshot).  Models: every asset of the two legacy MoNuSeg campaigns (HS6 1TB / 5TB /
5TB+GRAM / S+, FM14, HS0) plus the 20TB route2 arms, whose teacher snapshots are thinned with
the v4 campaign rule and followed online.  MoNuSeg is dense: one cell per GPU at a time, started
only when no other project evaluation runs on that GPU (evaluator rule).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_fm_snapshot_20260930")
OUTPUT = ROOT / "outputs/02_eval_runs/monuseg_train30val7_test14_fm14_20260930"
BENCHMARK = Path("/mnt/huawei_deepcad/benchmark")
PYTHON = "/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python"
LEGACY = [Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs") / name for name in
          ("scaling_union_monuseg_legacy_seg_20260921", "fm-hs0_union_monuseg_legacy_seg_20260921")]
RUNS = {}  # FM14 only; the 20TB arms stay in the main re-test
HOSTS = ("cpu1", "cpu2", "cpu5", "cpu7", "cpu8", "cpu9", "cpu10", "cpu11", "cpu12",
         "cpu15", "cpu18", "cpu19", "cpu20")
WATCH_HOST = "cpu9"
EVAL_PERIOD, THIN, THIN_PHASE = 488, 4, 2


def selected(checkpoint_id: int) -> bool:
    return (checkpoint_id + 1) % EVAL_PERIOD == 0 and ((checkpoint_id + 1) // EVAL_PERIOD) % THIN == THIN_PHASE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def stable_teachers(arm: str, root: Path, known: set[str]) -> list[dict]:
    assets = []
    for path in sorted((root / "eval").glob("training_*/teacher_checkpoint.pth"),
                       key=lambda p: int(p.parent.name.split("_")[-1])):
        checkpoint_id = int(path.parent.name.split("_")[-1])
        if not selected(checkpoint_id) or str(path) in known:
            continue
        stat = path.stat()
        if stat.st_size < 1_000_000_000 or time.time() - stat.st_mtime < 180:
            continue
        assets.append(dict(arm=arm, checkpoint_id=str(checkpoint_id), path=str(path),
                           config=str(root / "config.yaml"), kind="dinov3", model_id="", reserve_mib=4096))
    return assets


def fleet_module():
    os.environ["DINOV3_CODE_ROOT"] = str(SNAPSHOT)
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    import run_retest_fleet_20260918 as fleet  # noqa: E402  (derived snapshot, hash-registered)
    return fleet


def prepare(_args) -> None:
    if (OUTPUT / "campaign_manifest.json").exists():
        raise FileExistsError("Existing campaign; refusing overwrite")
    fleet = fleet_module()
    from benchmark_eval.rules_dense_preflight import audit_dense
    if Path(audit_dense.__code__.co_filename).resolve().parents[2] != (SNAPSHOT / "evaluation_external").resolve():
        raise RuntimeError("dense preflight not imported from the derived snapshot")
    spec = audit_dense("monuseg", BENCHMARK)
    if spec["counts"] != {"train": 30, "val": 7, "test": 14}:
        raise RuntimeError(f"unexpected MoNuSeg counts {spec['counts']}")
    spec.update(comparison_view="primary-last", reserve_mib=16000 if spec["image_size"] >= 512 else 8192,
                component="segmentation/monuseg", split_protocol_id="monuseg2018-train30-extra7val-test14-v1")
    source = json.loads((SNAPSHOT / "source_snapshot.json").read_text())
    legacy = [json.loads((c / "campaign_manifest.json").read_text()) for c in LEGACY]
    seen, assets = set(), []
    for manifest in legacy:
        for asset in manifest["checkpoint_assets"]:
            if asset.get("kind") != "external":
                continue
            key = (asset["arm"], str(asset["checkpoint_id"]))
            if key not in seen:
                seen.add(key)
                assets.append(asset)
    for arm, root in RUNS.items():
        assets += stable_teachers(arm, root, {a["path"] for a in assets})
    missing = [a["path"] for a in assets if not Path(a["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"missing assets: {missing[:3]}")
    # Register the external sources as they are now (this is what will run); record which ones
    # differ from the legacy campaigns.  The FM dense runner takes dinov3 (dataset splits) from
    # DINOV3_CODE_ROOT = this derived snapshot, so MoNuSeg membership comes from the new lock.
    external, changed_since_legacy = {}, []
    for manifest in legacy:
        for path, digest in manifest.get("external_source_hashes", {}).items():
            if "dinov3_" in path and "snapshot" in path:
                continue  # previous evaluator snapshot files are replaced by the derived snapshot
            if not Path(path).exists():
                raise FileNotFoundError(f"external source missing: {path}")
            current = sha256(Path(path))
            if current != digest and path not in changed_since_legacy:
                changed_since_legacy.append(path)
            external[path] = current
    this = Path(__file__).resolve()
    for path in (this, SNAPSHOT / "scripts/run_retest_fleet_20260918.py",
                 SNAPSHOT / "dinov3/eval/bio_frozen_eval/run_external_dense_rules.py",
                 SNAPSHOT / "dinov3/eval/bio_frozen_eval/external_fm_source_hashes.json",
                 BENCHMARK / "segmentation/monuseg/extracted/monuseg2018_train30_extra7val_test14_manifest.json"):
        external[str(path)] = sha256(path)
    manifest = dict(
        protocol_id="monuseg2018-train30-extra7val-test14-retest-v1-fm14",
        explicit_user_authorization="2026-09-29: re-test all MoNuSeg with the official 30/7/14 split",
        supersedes=[str(c) for c in LEGACY], superseded_split="monuseg2018-official30-seed42-val6-test14-v1 and legacy 37-pool random val7",
        git_commit=source["git_commit"], source_snapshot=source, source_snapshot_path=str(SNAPSHOT),
        numerical_environment=legacy[0]["numerical_environment"], external_source_hashes=external,
        external_sources_changed_since_legacy=changed_since_legacy,
        checkpoint_assets=assets, training_roots={k: str(v) for k, v in RUNS.items()},
        checkpoint_selection_rule_20tb=f"(id+1)%{EVAL_PERIOD}==0 and ((id+1)//{EVAL_PERIOD})%{THIN}=={THIN_PHASE}",
        datasets=[spec], benchmark_root=str(BENCHMARK), batch_size=64, autocast_dtype="bf16", seed=0,
        num_workers=2, full_v3_aggregate_allowed=False, legacy_reuse=False, online_checkpoints=True,
        isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True, created_unix=time.time())
    manifest["tasks"] = fleet.tasks_for(assets, [spec])
    save(OUTPUT / "campaign_manifest.json", manifest)
    (OUTPUT / "_state/inputs").mkdir(parents=True, exist_ok=True)
    arms = {}
    for a in assets:
        arms[a["arm"]] = arms.get(a["arm"], 0) + 1
    print(json.dumps(dict(output=str(OUTPUT), tasks=len(manifest["tasks"]), arms=arms,
                          split_identity=spec["split_identity_sha256"][:12])), flush=True)


def watch(_args) -> None:
    fleet = fleet_module()
    queue = fleet.queue
    lock = (OUTPUT / "_state/online_watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        with (OUTPUT / "_state/manifest.lock").open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            manifest = json.loads((OUTPUT / "campaign_manifest.json").read_text())
            known = {asset["path"] for asset in manifest["checkpoint_assets"]}
            additions = [a for arm, root in RUNS.items() for a in stable_teachers(arm, root, known)]
            for asset in additions:
                queue.checkpoint_record(OUTPUT, asset)
            if additions:
                manifest["checkpoint_assets"] += additions
                manifest["tasks"] += fleet.tasks_for(additions, manifest["datasets"])
                queue.save(OUTPUT / "campaign_manifest.json", manifest)
                print(time.strftime("%FT%TZ", time.gmtime()), "ADMITTED",
                      [f"{a['arm']}:{a['checkpoint_id']}" for a in additions], flush=True)
        time.sleep(300)


def worker(args) -> None:
    fleet = fleet_module()
    ns = argparse.Namespace(output=OUTPUT, host=args.host, gpus=[0], target_per_gpu=1, max_host_jobs=1,
                            max_global_jobs=400, task_family="mixed",
                            admission_guard=lambda gpu, actual, task: task["dataset"]["task"] == "segmentation")
    fleet.worker(ns)


def launch(_args) -> None:
    if not (OUTPUT / "campaign_manifest.json").is_file():
        raise FileNotFoundError("Prepare campaign first")
    this = str(Path(__file__).resolve())
    for host, mode in [(h, ["worker", "--host", h]) for h in HOSTS]:
        log = OUTPUT / "logs" / f"{host}.{mode[0]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, "-u", this] + mode
        remote = f"cd {shlex.quote(str(SNAPSHOT))} && setsid -f {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1 < /dev/null"
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
                                text=True, capture_output=True)
        print(json.dumps(dict(host=host, mode=mode[0], returncode=result.returncode,
                              stderr=result.stderr.strip()[-200:])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "watch", "worker", "launch"))
    p.add_argument("--host", default=os.uname().nodename)
    a = p.parse_args()
    {"prepare": prepare, "watch": watch, "worker": worker, "launch": launch}[a.mode](a)
