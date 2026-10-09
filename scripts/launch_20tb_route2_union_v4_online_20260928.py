#!/usr/bin/env python3
"""Online Union-v4 campaign for the two 20TB route2 HS6 L runs on the single-3090 cpu fleet.

Reuses the frozen v4 shared-component evaluator (dinov3_20tb_online_snapshot_20260918,
same recipe as 20tb_corrected_union_v4_20260922).  Changes are scheduling-only:
  * checkpoint thinning: evaluate teacher snapshots with (id+1)/488 == 2 (mod 4),
    i.e. every ~2M images, which includes training_12687 (the 5TB comparison point);
  * host roles: segmentation hosts run one dense cell at a time (evaluator rule);
    frozen hosts keep stacking frozen cells while GPU memory use is below 50%;
  * an admission guard enforces both rules for every task, including tasks
    admitted later by the online watcher.
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
SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")
REFERENCE = ROOT / "outputs/02_eval_inputs/shared_fleet_reference_20260930/raw_campaign_manifest.json"
PROTOCOL = ROOT / "Evaluation Rules/protocol_v4.json"
OUTPUT = ROOT / "outputs/02_eval_runs/20tb_route2_union_v4_online_20260928"
PYTHON = "/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python"
RUNS = {
    "5090route2": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924",
    "3090qiroute2": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_r0r9_8x3090qi_20260927",
    # 2026-09-28 ablation: route2 15TB (0.70) + route2 boundary 5TB (0.30), replaces 3090qiroute2
    "3090qiablation": ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_15tb07_5tbboundary03_ablation_8x3090qi_20260928",
}
SEGMENTATION_HOSTS = ("cpu1", "cpu2", "cpu8", "cpu10", "cpu11", "cpu12")
FROZEN_HOSTS = ("cpu5", "cpu9", "cpu19", "cpu20", "cpu7", "cpu18")
WATCH_HOST = "cpu12"
MEMORY_FRACTION = 0.5
EVAL_PERIOD = 488
THIN = 4
THIN_PHASE = 2
SELECTION_RULE = f"(checkpoint_id+1) % {EVAL_PERIOD} == 0 and ((checkpoint_id+1)//{EVAL_PERIOD}) % {THIN} == {THIN_PHASE}"


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
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    import run_retest_fleet_20260918 as fleet  # noqa: E402  (frozen evaluator, hash-registered)
    return fleet


def prepare(_args) -> None:
    if (OUTPUT / "campaign_manifest.json").exists():
        raise FileExistsError("Existing campaign; refusing overwrite")
    fleet = fleet_module()
    reference = json.loads(REFERENCE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["protocol_id"] != "bio-eval-union-v4":
        raise ValueError("Expected v4 protocol")
    source = json.loads((SNAPSHOT / "source_snapshot.json").read_text())
    if reference["source_snapshot"] != source:
        raise RuntimeError("Frozen evaluator source has changed")
    datasets = reference["datasets"]
    assets = [a for arm, root in RUNS.items() for a in stable_teachers(arm, root, set())]
    frozen_script = SNAPSHOT / "scripts/run_retest_fleet_20260918.py"
    this = Path(__file__).resolve()
    manifest = dict(
        protocol_id="bio-eval-union-v4", campaign_scope="V4_SHARED_V3_COMPONENTS_ONLY",
        v4_aggregate_allowed=False, old_results_relabelled=False,
        explicit_user_authorization="2026-09-28 20TB route2 v4 evaluation on the single-3090 cpu fleet",
        protocol_sha256=sha256(PROTOCOL), protocol_rules_path=str(PROTOCOL),
        git_commit=source["git_commit"], source_snapshot=source, source_snapshot_path=str(SNAPSHOT),
        external_source_hashes={str(PROTOCOL): sha256(PROTOCOL), str(this): sha256(this),
                                str(frozen_script): sha256(frozen_script)},
        checkpoint_assets=assets, training_roots={k: str(v) for k, v in RUNS.items()},
        checkpoint_selection_rule=SELECTION_RULE,
        scheduling=dict(segmentation_hosts=list(SEGMENTATION_HOSTS), frozen_hosts=list(FROZEN_HOSTS),
                        frozen_stack_until_gpu_memory_fraction=MEMORY_FRACTION,
                        segmentation_single_job_per_gpu=True),
        checkpoint_config_sha256={a["arm"]: sha256(Path(a["config"])) for a in assets},
        expected_unique_dataset_counts=protocol["expected_unique_dataset_counts"],
        datasets=datasets, benchmark_root=str(reference["benchmark_root"]),
        batch_size=64, autocast_dtype="bf16", n_last_blocks=1, use_avgpool=True, seed=0, num_workers=2,
        numerical_environment=reference["numerical_environment"],
        full_v3_aggregate_allowed=False, legacy_reuse=False, online_checkpoints=True,
        isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True, created_unix=time.time())
    manifest["tasks"] = fleet.tasks_for(assets, datasets)
    save(OUTPUT / "campaign_manifest.json", manifest)
    (OUTPUT / "_state/inputs").mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(output=str(OUTPUT), checkpoints=len(assets), tasks=len(manifest["tasks"]),
                          per_checkpoint=len(manifest["tasks"]) // max(len(assets), 1),
                          assets=[f"{a['arm']}:{a['checkpoint_id']}" for a in assets])), flush=True)


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
                inputs = queue.checkpoint_record(OUTPUT, asset)
                with (OUTPUT / "_state/checkpoint_admission.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(time=time.time(), asset=asset, fingerprint=inputs,
                                                 source_snapshot_sha256=manifest["source_snapshot"]["sha256"])) + "\n")
            if additions:
                manifest["checkpoint_assets"] += additions
                manifest["tasks"] += fleet.tasks_for(additions, manifest["datasets"])
                queue.save(OUTPUT / "campaign_manifest.json", manifest)
                print(time.strftime("%FT%TZ", time.gmtime()), "ADMITTED",
                      [f"{a['arm']}:{a['checkpoint_id']}" for a in additions], flush=True)
        time.sleep(120)


_MEMORY_CACHE: dict[int, tuple[float, float]] = {}
SETTLE_SECONDS = 45


def gpu_memory_fraction(gpu: int) -> float:
    cached = _MEMORY_CACHE.get(gpu)
    if cached and time.time() - cached[0] < 2:
        return cached[1]
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    used, total = (float(x) for x in out.strip().split(","))
    _MEMORY_CACHE[gpu] = (time.time(), used / total)
    return used / total


PENDING_SECONDS = 180
PENDING_RESERVE_MIB = 4096


def running_records(hosts: tuple[str, ...]) -> list[tuple[str, dict]]:
    records = []
    for path in (OUTPUT / "_state/running").glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("host") in hosts:
            records.append((path.stem, record))
    return records


def predicted_fraction(gpu: int, hosts: tuple[str, ...]) -> float:
    """Current use plus a reserve for cells started in the last few minutes that have
    not allocated yet: stacking on a lagging reading overshot to 72% and OOM-killed a
    cell next to another user's job on cpu18 (2026-09-28)."""
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    used, total = (float(x) for x in out.strip().split(","))
    now = time.time()
    fresh = [r for _, r in running_records(hosts) if now - float(r.get("time", 0)) < PENDING_SECONDS]
    last = max((float(r.get("time", 0)) for r in fresh), default=0.0)
    if now - last < SETTLE_SECONDS:
        return 1.0
    return (used + PENDING_RESERVE_MIB * len(fresh)) / total


def dense_running(host: str) -> bool:
    return any("__segmentation__" in key for key, _ in running_records((host,)))


def worker(args) -> None:
    """segmentation: one dense cell at a time (the evaluator starts dense only on an
    empty GPU).  frozen: stack frozen cells while predicted memory < 50%.
    cofrozen: runs on a segmentation host under the name <host>-co; stacks frozen cells
    only while that host has a dense cell running, so the GPU drains afterwards and
    the next dense cell can start."""
    fleet = fleet_module()
    role, name = args.role, args.host
    partner = name[:-3] if name.endswith("-co") else name
    gpu_hosts = (partner, partner + "-co")
    def guard(gpu, actual, task):
        dense = task["dataset"]["task"] == "segmentation"
        if role == "segmentation":
            return dense
        if dense:
            return False
        if role == "cofrozen" and not dense_running(partner):
            return False
        return predicted_fraction(gpu, gpu_hosts) < MEMORY_FRACTION
    ns = argparse.Namespace(
        output=OUTPUT, host=name, gpus=[0],
        target_per_gpu=1 if role == "segmentation" else 32,
        max_host_jobs=1 if role == "segmentation" else 32,
        max_global_jobs=400, task_family="mixed", admission_guard=guard)
    fleet.worker(ns)


def launch(_args) -> None:
    if not (OUTPUT / "campaign_manifest.json").is_file():
        raise FileNotFoundError("Prepare campaign first")
    manifest = json.loads((OUTPUT / "campaign_manifest.json").read_text())
    if manifest["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("V4 protocol modified after manifest freeze")
    this = str(Path(__file__).resolve())
    plan = ([(h, "segmentation") for h in SEGMENTATION_HOSTS] + [(h, "frozen") for h in FROZEN_HOSTS]
            + [(h, "cofrozen") for h in SEGMENTATION_HOSTS] + [(WATCH_HOST, "watch")])
    plan = [(h, r) for h, r in plan if r in _args.roles]
    for host, role in plan:
        name = f"{host}-co" if role == "cofrozen" else host
        log = OUTPUT / "logs" / f"{name}.{role}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        mode = ["watch"] if role == "watch" else ["worker", "--host", name, "--role", role]
        command = [PYTHON, "-u", this] + mode
        remote = f"cd {shlex.quote(str(ROOT))} && setsid -f {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1 < /dev/null"
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
                                text=True, capture_output=True)
        print(json.dumps(dict(host=host, role=role, returncode=result.returncode,
                              stderr=result.stderr.strip()[-300:], log=str(log))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "watch", "worker", "launch"))
    p.add_argument("--host", default=os.uname().nodename)
    p.add_argument("--role", choices=("segmentation", "frozen", "cofrozen"), default="frozen")
    p.add_argument("--roles", nargs="+", default=["segmentation", "frozen", "cofrozen", "watch"])
    a = p.parse_args()
    {"prepare": prepare, "watch": watch, "worker": worker, "launch": launch}[a.mode](a)
