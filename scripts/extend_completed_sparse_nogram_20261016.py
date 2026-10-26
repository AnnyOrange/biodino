#!/usr/bin/env python3
"""Append the two completed no-GRAM anchors to the sparse campaign."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "outputs/02_eval_runs/20tb_completed_sparse_20261016"
TRAIN = REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924"
SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    os.umask(0)
    manifest_path = CAMPAIGN / "campaign_manifest.json"
    cfg_path = CAMPAIGN / "configs/cpu2.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    arm = "nogram_main"
    if any(a["arm"] == arm for a in manifest["arms"]):
        raise RuntimeError("No-GRAM anchors are already registered")
    root = CAMPAIGN / "arms" / arm
    frozen = root / "source/config.yaml"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAIN / "config.yaml", frozen)
    if sha(frozen) != sha(TRAIN / "config.yaml"):
        raise RuntimeError("Training config copy failed")
    checks = []
    for ck in (47823, 74663):
        teacher = TRAIN / "eval" / f"training_{ck}" / "teacher_checkpoint.pth"
        if teacher.stat().st_size < 500_000_000:
            raise RuntimeError(f"Incomplete teacher: {teacher}")
        checks.append(dict(id=ck, path=str(teacher), bytes=teacher.stat().st_size,
                           mtime_ns=teacher.stat().st_mtime_ns))
        cfg["targets"].append(dict(arm=arm, checkpoint=ck,
                                   snapshot_root=str(TRAIN / "eval"), full_eval_period=488,
                                   official_epoch_length=4098, config_sha256=sha(frozen)))
    source = CAMPAIGN / "source_record" / "scripts" / SCRIPT.name
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, source)
    manifest["source_sha256"][f"scripts/{SCRIPT.name}"] = sha(source)
    manifest["arms"].append(dict(arm=arm, training_root=str(TRAIN),
                                 training_config_sha256=sha(frozen), frozen_config=str(frozen),
                                 host="cpu2", checkpoints=checks, eval_period=488))
    manifest["extensions"] = manifest.get("extensions", []) + [dict(
        added_unix=time.time(), script=str(SCRIPT), script_sha256=sha(SCRIPT),
        reason="User requested evaluation of completed no-GRAM training anchors")]
    save(manifest_path, manifest)
    save(cfg_path, cfg)
    print(json.dumps({"arm": arm, "checkpoints": [c["id"] for c in checks],
                      "host": "cpu2", "after": "fixed_cls"}))


if __name__ == "__main__":
    main()
