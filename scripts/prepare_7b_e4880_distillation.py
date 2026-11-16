#!/usr/bin/env python3
"""Prepare the 7B E4880 continuation from the full 21503 state; never launch training.

E recipe = the objective of the H+ arm E4880_mt3_lw5weights (2026-11-12): 4880-update linear cooldown
(LR -> 1e-6, EMA momentum -> 1, teacher temperature frozen, weight decay continues) + three-teacher CLS and patch
recoverability distillation (official DINOv3-7B, H-optimus-0, GenBio-PathFM; lw5 inverse-variance balance,
loss weight 5, 64 patches, ridge 0.1, decoder momentum 0.995).  DINO / iBOT / KoLeo are kept.

Use the actual run's archived config AFTER launch overrides, BEFORE LR scaling (same rule as
prepare_7b_global_distillation.py).  The teacher manifest is the same FrozenJointTeacher manifest used for the
CLS arms (names dinov3_7b, hoptimus0, genbio_pathfm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

TEACHERS = (("dinov3_7b", "dinov3"), ("hoptimus0", "timm_dir"), ("genbio_pathfm", "genbio_pathfm"))
PERIOD = 976          # = the H+ E4880 export cadence (16591 + 976k); 21503 + 976k here


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-origin-iteration", type=int, default=21503, choices=[21503])
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4880,
                        help="cooldown length in optimizer updates; 4880 = the tested recipe (do not sample-match)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_config.resolve(strict=True)
    checkpoint = args.source_checkpoint.resolve(strict=True)
    if not checkpoint.name.isdigit() or not (checkpoint / ".metadata").is_file():
        raise ValueError("source-checkpoint must be a numeric full DCP directory")
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
        raise ValueError("Expected the original 7B SSL run without auxiliary distillation or a cooldown override")
    end = origin + 1 + args.updates                     # train.max_updates (exclusive); last update = end - 1
    schedule_end = int(base.optim.epochs) * int(base.train.OFFICIAL_EPOCH_LENGTH)
    if args.updates <= 0 or end > schedule_end:
        raise ValueError("The cooldown must end inside the original schedule (the override replaces its tail)")
    specs_raw = json.loads(args.teacher_manifest.read_text())
    specs = specs_raw["teachers"] if isinstance(specs_raw, dict) else specs_raw
    selected = []
    for name, kind in TEACHERS:
        matches = [s for s in specs if s["name"] == name and s["kind"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Manifest must contain exactly one {name}/{kind}")
        selected.append(matches[0])
    official = selected[0]
    # E recipe: the official-7B teacher sees the student's own global crop (256 px -> 16x16 patch grid, no resize, no grid
    # interpolation; same as the official-H+ teacher in the H+ runs).  The CLS-only arms may have used the manifest default
    # (224 -> 14x14 grid, which the patch loss would have to bilinearly upsample); here it is set explicitly.
    student_crop = int(base.crops.global_crops_size if isinstance(base.crops.global_crops_size, int)
                       else max(base.crops.global_crops_size))
    manifest_img_size = int(official.get("img_size", 224))
    official = dict(official, img_size=student_crop)
    selected[0] = official
    if (str(OmegaConf.load(official["config"]).student.arch) != "vit_7b"
            or official.get("provenance", {}).get("checkpoint_origin") != "official_dinov3_release"):
        raise ValueError("dinov3_7b must identify the official frozen 7B release")
    output = args.output_root.resolve()
    if output == source.parent or output.is_relative_to(checkpoint.parent.parent):
        raise ValueError("Use a separate output root, outside the source training run")
    output.mkdir(parents=True, exist_ok=False)          # never replace an existing branch
    (output / "source_training_config.yaml").write_bytes(source.read_bytes())
    branch = output / "e4880_mt3_lw5"
    (branch / "ckpt").mkdir(parents=True)
    manifest = branch / "teacher_manifest.json"
    manifest.write_text(json.dumps({"teachers": selected}, indent=2) + "\n")
    overlay = OmegaConf.load(root / "dinov3/configs/train/7b_e4880_mt3_lw5.yaml")
    if (list(overlay.external_distill.teacher_names) != [n for n, _ in TEACHERS]
            or str(overlay.external_distill.mode) != "phis_spatial_recovery"
            or str(overlay.external_distill.weighting) != "inverse_variance"):
        raise ValueError("overlay differs from the tested E recipe")
    overlay.external_distill.teacher_manifest = str(manifest)
    overlay.optim.cooldown.start_iteration = origin + 1
    overlay.optim.cooldown.length = args.updates
    cfg = OmegaConf.merge(base, overlay)
    cfg.train.output_dir = str(branch)
    cfg.train.max_updates = end
    cfg.train.sharded_eval_checkpoint = True
    cfg.student.pretrained_weights = ""                  # the full DCP supplies student, EMA, heads and AdamW
    cfg.student.resume_from_teacher_chkpt = ""
    cfg.setdefault("checkpointing", {})
    cfg.checkpointing.sharded = True
    cfg.checkpointing.save_final = True
    cfg.checkpointing.period_origin_iteration = origin
    cfg.checkpointing.period = PERIOD
    cfg.checkpointing.max_to_keep = None                 # keep every full state
    if "keep_every" in cfg.checkpointing:
        del cfg.checkpointing["keep_every"]
    cfg.evaluation.eval_period_iterations = PERIOD
    path = branch / "config.yaml"
    OmegaConf.save(cfg, path)
    (branch / "ckpt" / str(origin)).symlink_to(checkpoint, target_is_directory=True)
    saves = sorted(set(list(range(origin + PERIOD, end, PERIOD)) + [end - 1]))
    (output / "PREPARED.json").write_text(json.dumps({
        "status": "PREPARED_NOT_LAUNCHED_NOT_RUNTIME_VALIDATED",
        "source_config": str(source), "source_config_sha256": digest(output / "source_training_config.yaml"),
        "source_checkpoint": str(checkpoint), "source_metadata_sha256": digest(checkpoint / ".metadata"),
        "student_origin_iteration": origin, "expected_start_iteration": origin + 1,
        "cooldown_updates": args.updates, "last_update": end - 1,
        "config": str(path), "config_sha256": digest(path),
        "teacher_manifest": str(manifest), "teacher_manifest_sha256": digest(manifest),
        "scheduled_checkpoint_iterations": saves,
        "official_7b_teacher_img_size": student_crop, "official_7b_teacher_img_size_in_input_manifest": manifest_img_size,
        "recipe": "E4880: linear cooldown (LR->1e-6, momentum->1, teacher temp frozen, wd continues) + "
                  "3-teacher CLS+patch recoverability, lw5 inverse-variance balance, loss_weight 5, 64 patches",
        "preserved": ["batch", "accumulation", "data/augmentation", "seed", "DINO/iBOT/KoLeo", "Gram policy (off)"],
    }, indent=2) + "\n")
    print(json.dumps({"prepared": str(branch), "config": str(path), "saves": saves}, indent=2))


if __name__ == "__main__":
    main()
