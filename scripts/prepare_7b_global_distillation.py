#!/usr/bin/env python3
"""Prepare four CLS-only 7B continuation arms from 21503; never launch training.

Use the actual run's archived config AFTER launch overrides, BEFORE LR scaling
(the local setup_config writes config.yaml at exactly that point). All input
paths and the complete origin DCP must already be available on the target host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-origin-iteration", type=int, default=21503, choices=[21503],
                        help="common student/optimizer origin: 21503; 13311 is only an auxiliary teacher")
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, help="same additional update budget for all four arms")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_config.resolve(strict=True)
    checkpoint = args.source_checkpoint.resolve(strict=True)
    if not checkpoint.name.isdigit() or not (checkpoint / ".metadata").is_file():
        raise ValueError("source-checkpoint must be a numeric full DCP directory")
    if (checkpoint / "checkpoint.pth").exists():
        raise ValueError("Expected the 7B DCP format")
    origin = int(checkpoint.name)
    if origin != args.expected_origin_iteration:
        raise ValueError("source-checkpoint differs from the explicitly selected origin iteration")
    base = OmegaConf.load(source)
    if str(base.student.arch) != "vit_7b" or int(base.student.patch_size) != 16:
        raise ValueError("Expected the original ViT-7B/16 run's config")
    if str(base.compute_precision.get("distributed_mode", "fsdp")) != "fsdp":
        raise ValueError("7B continuation requires FSDP")
    if base.train.get("start_iteration_override") is not None:
        raise ValueError("Logical-step/fresh-optimizer overrides are not full continuation")
    if (base.get("external_distill", {}).get("enabled", False)
            or base.get("joint_teacher", {}).get("enabled", False)
            or base.optim.get("cooldown", {}).get("enabled", False)):
        raise ValueError("Expected the original 7B branch without auxiliary distillation or a cooldown override")
    end = origin + 1 + args.updates
    schedule_end = int(base.optim.epochs) * int(base.train.OFFICIAL_EPOCH_LENGTH)
    source_end = min(schedule_end, int(base.train.get("max_updates") or schedule_end))
    if args.updates <= 0 or end > source_end:
        raise ValueError("Choose a positive update budget within the original schedule; extensions need a new schedule")
    specs_raw = json.loads(args.teacher_manifest.read_text())
    specs = specs_raw["teachers"] if isinstance(specs_raw, dict) else specs_raw
    selected = {}
    for name, kind in (("dinov3_7b", "dinov3"), ("history_7b_13311", "dinov3"),
                       ("hoptimus0", "timm_dir"), ("genbio_pathfm", "genbio_pathfm")):
        matches = [s for s in specs if s["name"] == name and s["kind"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Manifest must contain exactly one {name}/{kind}")
        selected[name] = matches[0]
    official = selected["dinov3_7b"]
    teacher_config = OmegaConf.load(official["config"])
    if (str(teacher_config.student.arch) != "vit_7b" or int(official["output_dim"]) != 8192
            or official.get("native_preprocess", "published_model") != "published_model"
            or official.get("checkpoint_branch", "teacher") != "teacher"
            or official.get("provenance", {}).get("checkpoint_origin") != "official_dinov3_release"):
        raise ValueError("dinov3_7b must identify the official frozen 7B release, with its own config and hashes")
    historical = selected["history_7b_13311"]
    historical_config = OmegaConf.load(historical["config"])
    history_provenance = historical.get("provenance", {})
    if (str(historical_config.student.arch) != "vit_7b" or int(historical["output_dim"]) != 8192
            or historical.get("native_preprocess") != "checkpoint_config"
            or historical.get("checkpoint_branch") != "teacher"
            or history_provenance.get("checkpoint_origin") != "historical_7b_ema"
            or history_provenance.get("checkpoint_iteration") != 13311
            or int(historical["img_size"]) != int(historical_config.crops.global_crops_size)):
        raise ValueError("history_7b_13311 must identify frozen EMA 13311 with its original preprocessing/config")
    if not Path(historical["weights"]).is_file():
        raise ValueError("Historical teacher requires a consolidated EMA/backbone file; a DCP directory cannot be used here")
    output = args.output_root.resolve()
    if output == source.parent or output.is_relative_to(checkpoint.parent.parent):
        raise ValueError("Use a separate output root, outside the source training run")
    # Exclusive creation prevents replacing an existing branch/config/checkpoint.
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = output / "source_training_config.yaml"
    source_snapshot.write_bytes(source.read_bytes())
    records = []
    arms = {
        "cls_dinov3": ["dinov3_7b"],
        "cls_history13311": ["history_7b_13311"],
        "cls_dinov3_hoptimus0": ["dinov3_7b", "hoptimus0"],
        "cls_dinov3_genbio_pathfm": ["dinov3_7b", "genbio_pathfm"],
    }
    for arm, names in arms.items():
        branch = output / arm
        (branch / "ckpt").mkdir(parents=True)
        manifest = branch / "teacher_manifest.json"
        manifest.write_text(json.dumps({"teachers": [selected[name] for name in names]}, indent=2) + "\n")
        overlay = OmegaConf.load(root / f"dinov3/configs/train/7b_{arm}.yaml")
        if (int(overlay.external_distill.resume_origin_iteration) != origin
                or int(overlay.external_distill.start_update) != origin + 1):
            raise ValueError(f"{arm}: overlay origin differs from the selected checkpoint")
        if (list(overlay.external_distill.teacher_names) != names
                or overlay.external_distill.global_readout != "cls"
                or float(overlay.external_distill.patchmean_weight) != 0
                or float(overlay.external_distill.patch_weight) != 0):
            raise ValueError(f"{arm}: overlay differs from the requested CLS-only teacher combination")
        weights = [float(w) for w in overlay.external_distill.teacher_weights]
        if (len(weights) != len(names) or any(not math.isfinite(w) or w <= 0 for w in weights)
                or not math.isfinite(sum(weights))):
            raise ValueError(f"{arm}: invalid teacher weights")
        overlay.external_distill.teacher_manifest = str(manifest)
        overlay.external_distill.resume_origin_iteration = origin
        overlay.external_distill.start_update = origin + 1
        cfg = OmegaConf.merge(base, overlay)
        cfg.train.output_dir = str(branch)
        cfg.train.max_updates = end
        cfg.train.sharded_eval_checkpoint = True
        # Full DCP supplies both student and EMA; avoid reloading release weights
        # first. Runtime metadata guards reject incomplete backbone/optimizer state.
        cfg.student.pretrained_weights = ""
        cfg.student.resume_from_teacher_chkpt = ""
        cfg.setdefault("checkpointing", {})
        cfg.checkpointing.sharded = True
        cfg.checkpointing.save_final = True
        cfg.checkpointing.period_origin_iteration = origin
        cfg.checkpointing.period = 1024
        cfg.evaluation.eval_period_iterations = 1024
        cfg.checkpointing.max_to_keep = None
        path = branch / "config.yaml"
        OmegaConf.save(cfg, path)
        (branch / "ckpt" / str(origin)).symlink_to(checkpoint, target_is_directory=True)
        records.append({"arm": arm, "teachers": names, "readout": "cls",
                        "normalized_teacher_weights": [w / sum(weights) for w in weights],
                        "config": str(path), "config_sha256": digest(path),
                        "teacher_manifest": str(manifest), "teacher_manifest_sha256": digest(manifest),
                        "expected_start_iteration": origin + 1, "last_checkpoint_iteration": end - 1})
    (output / "PREPARED.json").write_text(json.dumps({
        "status": "PREPARED_NOT_LAUNCHED_NOT_RUNTIME_VALIDATED",
        "source_config": str(source), "source_config_sha256": digest(source_snapshot),
        "source_checkpoint": str(checkpoint), "source_metadata_sha256": digest(checkpoint / ".metadata"),
        "student_origin_iteration": 21503,
        "historical_auxiliary_teacher_iteration": 13311,
        "checkpoint_content_validation": "pending runtime DCP guard; metadata hash is not a shard content hash",
        "additional_updates": args.updates, "branches": records,
        "checkpoint_and_ema_interval_updates": 1024,
        "checkpoint_period_origin_iteration": origin,
        "scheduled_checkpoint_iterations": sorted(set(
            list(range(origin + 1024, end, 1024)) + [end - 1]
        )),
        "protocol": "CLS-only auxiliary distillation; original DINO/iBOT retained; equal teacher mean",
        "schedule": "inherited; only stop budget changed; no cooldown",
        "preserved": ["batch", "accumulation", "LR rule/schedule", "temperature/EMA schedule", "seed", "data/augmentation", "Gram policy"],
    }, indent=2) + "\n")
    print(json.dumps({"prepared": str(output), "branches": records}, indent=2))


if __name__ == "__main__":
    main()
