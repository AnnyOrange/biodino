#!/usr/bin/env python3
"""Register sparse endpoint screening for finished 20TB training branches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "outputs/01_training_runs"
CAMPAIGN = REPO / "outputs/02_eval_runs/20tb_completed_sparse_20261016"
WORKER = REPO / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
BENCHMARK = Path("/mnt/huawei_deepcad/benchmark")
ARM_SPECS = (
    ("quality16k", "HS6_L_quality_long_noreplace_20tb_ddp_gb1024_16k_20261009", (15999, 7999), 4000, "cpu8"),
    ("resampling", "hs6_l_20tb_v2_recovery_fork38063_20261009/resampling_only_38063_8x3090qi", (43919,), 488, "cpu11"),
    ("fixed_cls", "hs6_l_20tb_v2_recovery_fork38063_20261009/fixed_cls_w1_anchor38063", (40015,), 488, "cpu2"),
    ("cls_slow2", "hs6_l_20tb_v2_recovery_fork36111_20261004/cls_slow2", (80031,), 488, "local"),
)
PYTHONS = {
    "cpu8": "/home/inspur/anaconda3/envs/dinov3/bin/python",
    "cpu11": "/home/inspur/anaconda3/envs/dinov3/bin/python",
    "cpu2": "/home/inspur/anaconda3/envs/dinov3/bin/python",
    "local": "/home/lxy/miniconda3/envs/dinov3/bin/python",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    os.umask(0)
    if (CAMPAIGN / "campaign_manifest.json").exists():
        raise FileExistsError("Campaign already registered")
    targets = {host: [] for host in PYTHONS}
    dirty = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=REPO, text=True).splitlines()
    manifest = dict(protocol="2026-10-16 completed-run sparse screening, 11 lanes / 48 metrics",
                    formal_v4=False, numerical_cross_host_aggregate_allowed=False,
                    reason="User requested evaluation of completed training and spaced checkpoints on single-3090 fleet",
                    created_unix=time.time(), worker=str(WORKER), worker_sha256=sha(WORKER),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    dirty_files=dirty,
                    rules_sha256=sha(REPO / "Evaluation Rules/protocol_v4.json"), arms=[])
    for arm, trainrel, checkpoints, period, host in ARM_SPECS:
        train = TRAIN / trainrel
        config = train / "config.yaml"
        assert config.is_file()
        root = CAMPAIGN / "arms" / arm
        frozen = root / "source/config.yaml"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, frozen)
        assert sha(frozen) == sha(config)
        checks = []
        for ck in checkpoints:
            teacher = train / "eval" / f"training_{ck}" / "teacher_checkpoint.pth"
            if teacher.stat().st_size < 500_000_000:
                raise RuntimeError(f"Incomplete teacher: {teacher}")
            checks.append(dict(id=ck, path=str(teacher), bytes=teacher.stat().st_size,
                               mtime_ns=teacher.stat().st_mtime_ns))
            targets[host].append(dict(arm=arm, checkpoint=ck, snapshot_root=str(train / "eval"),
                                      full_eval_period=period, official_epoch_length=4098,
                                      config_sha256=sha(frozen)))
        manifest["arms"].append(dict(arm=arm, training_root=str(train),
                                     training_config_sha256=sha(config), frozen_config=str(frozen),
                                     host=host, checkpoints=checks, eval_period=period))
    for host, entries in targets.items():
        cfg = dict(control_root=str(CAMPAIGN / "dispatch"), eval_root=str(CAMPAIGN / "arms"),
                   worker=str(WORKER), worker_sha256=sha(WORKER), python=PYTHONS[host],
                   evaluator_repo=str(REPO), benchmark=str(BENCHMARK), targets=entries,
                   ram_reserve_gib=32)
        save(CAMPAIGN / "configs" / f"{host}.json", cfg)
    manifest["configs"] = {h: str(CAMPAIGN / "configs" / f"{h}.json") for h in targets}
    source = CAMPAIGN / "source_record"
    source.mkdir(parents=True, exist_ok=True)
    files = {WORKER, REPO / "scripts/dispatch_20tb_priority_20261016.py",
             REPO / "scripts/register_completed_sparse_20261016.py",
             REPO / "scripts/run_bio_benchmark_all.sh"}
    for part in ("eval", "models", "layers", "utils", "checkpointer", "data"):
        files.update((REPO / "dinov3" / part).rglob("*.py"))
    files.update((REPO / "Evaluation Rules").rglob("*.json"))
    for entry in dirty:
        rel = entry[3:]
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        path = REPO / rel
        if path.is_file() and (rel.startswith("scripts/") or rel.startswith("dinov3/")):
            files.add(path)
    source_hashes = {}
    for path in sorted(files):
        if not path.is_file():
            continue
        dest = source / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(path.relative_to(REPO))] = sha(dest)
    manifest["source_archive"] = str(source)
    manifest["source_sha256"] = source_hashes
    manifest["launch_command"] = "scripts/dispatch_20tb_priority_20261016.py --config <host config> --host <host> --gpu <gpu> --group <frozen|dense> --slot <slot>"
    save(CAMPAIGN / "campaign_manifest.json", manifest)
    print(json.dumps(dict(campaign=str(CAMPAIGN), arms=len(ARM_SPECS), checkpoints=sum(map(len, targets.values())))))


if __name__ == "__main__":
    main()
