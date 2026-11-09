"""Frozen external teachers, with image-level recoverability objectives.

CLS and mean-pooled patch tokens have separate ridge decoders. There is no
spatial token matching and no new optimizer parameter. This is not CorrCA.
The frozen teacher loader is shared with the manifest-pinned joint-teacher code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import torch
from torch import nn

from dinov3.train.external_distill import ExternalRecoveryStream
from dinov3.train.joint_teacher_models import FrozenJointTeacher


class GlobalTeacherDistillation(nn.Module):
    def __init__(self, cfg, student_dim: int):
        super().__init__()
        ed = cfg.external_distill
        if float(ed.patch_weight) != 0 or len(ed.teachers):
            raise ValueError("global_recovery requires patch_weight=0 and teachers=[]; use teacher_manifest/names")
        if bool(cfg.multidistillation.enabled):
            raise ValueError("global_recovery currently requires a single student/world process group")
        if str(cfg.compute_precision.get("distributed_mode", "fsdp")) != "fsdp":
            raise ValueError("This full-DCP continuation path requires FSDP")
        if int(cfg.train.batch_size_per_gpu) < 2:
            raise ValueError("Global recovery needs at least two images per rank for fit/monitor halves")
        self.readout = str(ed.global_readout)
        if self.readout not in ("cls", "cls_patchmean"):
            raise ValueError("global_readout must be cls or cls_patchmean")
        self.part_weights = {"cls": float(ed.cls_weight)}
        if self.readout == "cls_patchmean":
            self.part_weights["patchmean"] = float(ed.patchmean_weight)
        if any(not math.isfinite(w) or w <= 0 for w in self.part_weights.values()):
            raise ValueError("Enabled global readout weights must be finite and positive")
        if not math.isfinite(float(ed.loss_weight)) or float(ed.loss_weight) <= 0:
            raise ValueError("Global loss_weight must be finite and positive")
        if not (float(ed.ridge) > 0 and 0 <= float(ed.momentum) < 1
                and int(ed.warmup) > 0 and 0 < int(ed.solve_every) <= int(ed.warmup)):
            raise ValueError("Invalid ridge, momentum, warmup or solve_every")
        if ed.resume_origin_iteration is None:
            raise ValueError("global_recovery requires resume_origin_iteration from a full DCP checkpoint")
        self.origin_iteration = int(ed.resume_origin_iteration)
        self.start_update = int(ed.start_update)
        self.ramp_updates = int(ed.ramp_updates)
        if self.origin_iteration < 0 or self.start_update != self.origin_iteration + 1 or self.ramp_updates < 0:
            raise ValueError("start_update must be origin_iteration + 1; ramp_updates must be nonnegative")

        manifest = json.loads(Path(ed.teacher_manifest).read_text())
        specs = manifest["teachers"] if isinstance(manifest, dict) else manifest
        names = list(getattr(ed, "teacher_names", []) or [])
        single_name = str(getattr(ed, "teacher_name", ""))
        if names and single_name:
            raise ValueError("Use either teacher_names or legacy teacher_name, not both")
        self.names = [str(name) for name in (names or [single_name])]
        if not all(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Select nonempty, unique teacher names")
        weights = list(getattr(ed, "teacher_weights", []) or [1.0] * len(self.names))
        if len(weights) != len(self.names) or any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights):
            raise ValueError("teacher_weights must contain one finite positive value per teacher")
        # Keep the auxiliary coefficient fixed when adding a teacher. The
        # per-teacher gradient norms may still differ; this is not norm matching.
        weight_sum = sum(float(w) for w in weights)
        if not math.isfinite(weight_sum):
            raise ValueError("teacher_weights sum must be finite")
        self.teacher_weights = dict(zip(self.names, [float(w) / weight_sum for w in weights]))
        teachers, streams, teacher_identities = {}, {}, []
        for name in self.names:
            selected = [dict(spec) for spec in specs if spec["name"] == name]
            if len(selected) != 1:
                raise ValueError(f"Manifest must contain exactly one selected teacher: {name}")
            spec = selected[0]
            if spec.get("domains") or spec.get("residual_of"):
                raise ValueError("This global branch uses all images, without routing or residual targets")
            spec["chunk_size"] = int(ed.teacher_chunk_size)
            teacher = FrozenJointTeacher(
                spec, student_mean=list(cfg.crops.rgb_mean), student_std=list(cfg.crops.rgb_std),
                device="cpu", dtype=torch.bfloat16,
            )
            teachers[name] = teacher
            for part in self.part_weights:
                streams[self._stream_key(name, part)] = ExternalRecoveryStream(
                    student_dim, teacher.embed_dim, ridge=float(ed.ridge), momentum=float(ed.momentum),
                    warmup=int(ed.warmup), solve_every=int(ed.solve_every),
                )
            teacher_identity = {key: spec[key] for key in ("name", "kind", "weights_sha256", "config_sha256")}
            teacher_identity.update(
                code_sha256=sorted(dict(spec.get("code_sha256", {})).values()),
                teacher_dim=teacher.embed_dim, img_size=teacher.img_size, chunk_size=teacher.chunk_size,
                mean=teacher.signature["mean"], std=teacher.signature["std"],
                checkpoint_branch=teacher.signature["checkpoint_branch"],
                weight=self.teacher_weights[name],
            )
            teacher_identities.append(teacher_identity)
        self.teachers = nn.ModuleDict(teachers)
        self.streams = nn.ModuleDict(streams)

        # Bind saved decoder statistics to the target and objective, while allowing
        # the same checked teacher files to move to another machine/path.
        identity = dict(
            schema="multi_global_recovery_v2", student_dim=int(student_dim), teachers=teacher_identities,
            readout=self.readout, part_weights=self.part_weights, loss_weight=float(ed.loss_weight),
            student_mean=list(cfg.crops.rgb_mean), student_std=list(cfg.crops.rgb_std),
            ridge=float(ed.ridge), momentum=float(ed.momentum), warmup=int(ed.warmup),
            solve_every=int(ed.solve_every), start_update=self.start_update, ramp_updates=self.ramp_updates,
            student_observation="existing_ibot_masked_first_global_crop",
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
        self._expected_identity = digest
        self.register_buffer("objective_identity", torch.tensor(list(digest), dtype=torch.uint8))
        self._validated_checkpoint_iteration = None
        logging.getLogger("dinov3").info(
            "[GLOBAL EXTD] teachers=%s weights=%s readout=%s student_dim=%d chunk=%d origin=%d ramp=%d; "
            "BF16 frozen teacher, FP32 ridge buffers, first global crop, no spatial matching",
            self.names, self.teacher_weights, self.readout, student_dim, int(ed.teacher_chunk_size),
            self.origin_iteration, self.ramp_updates,
        )

    def _stream_key(self, name, part):
        return part if len(self.names) == 1 else f"{name}__{part}"

    def teacher_state_keys(self):
        return [key for key in self.state_dict() if key.startswith("teachers.")]

    def reset_buffers(self):
        for stream in self.streams.values():
            stream.reset_buffers()

    def validate_checkpoint(self, checkpoint, model):
        """Check DCP metadata before the generic loader's partial-load pathway.

        Only the first fork may omit the new objective buffers. Backbone, EMA,
        heads and every trainable parameter's Adam state must already exist.
        No model/optimizer shard is materialized by this metadata inspection.
        """
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        if checkpoint is None:
            raise ValueError("global_recovery requires full-state resume; scratch/EMA-only initialization is forbidden")
        checkpoint = Path(checkpoint)
        if (checkpoint / "checkpoint.pth").exists() or not (checkpoint / ".metadata").is_file():
            raise ValueError("This 7B continuation branch requires a full DCP directory")
        iteration = int(checkpoint.name)
        if iteration < self.origin_iteration:
            raise ValueError("Checkpoint predates the configured fork")
        metadata = FileSystemReader(checkpoint).read_metadata()
        saved = metadata.state_dict_metadata
        prefix = "model.external_distill."
        saved_aux = {key for key in saved if key.startswith(prefix)}
        initial_fork = iteration == self.origin_iteration
        if initial_fork and saved_aux:
            raise ValueError("The fork origin must be the base 7B run, without an external-distillation state")
        current = get_model_state_dict(model)
        expected_keys = {"model." + key for key in current}
        unexpected = set()
        for key in saved:
            if not key.startswith("model.") or key in expected_keys:
                continue
            alias = "model.teacher." + key.removeprefix("model.model_ema.")
            if (key.startswith("model.model_ema.") and model.model_ema is model.teacher
                    and alias in expected_keys):
                continue
            unexpected.add(key)
        if unexpected:
            raise ValueError(f"Unexpected model state in continuation checkpoint: {sorted(unexpected)[:10]}")
        for name, value in current.items():
            key = "model." + name
            if initial_fork and key.startswith(prefix):
                continue
            # model_ema may be an alias omitted by DCP's canonical state dict.
            if key not in saved and name.startswith("model_ema.") and model.model_ema is model.teacher:
                key = "model.teacher." + name.removeprefix("model_ema.")
            if key not in saved:
                raise ValueError(f"Incomplete continuation checkpoint: missing {key}")
            if hasattr(value, "shape") and tuple(saved[key].size) != tuple(value.shape):
                raise ValueError(f"Checkpoint shape mismatch: {key}")
        if not initial_fork and saved_aux != {"model." + key for key in current if key.startswith("external_distill.")}:
            raise ValueError("Saved external objective keys differ from this branch")
        if "iteration" not in saved or not any(k.startswith("optimizer.param_groups.") for k in saved):
            raise ValueError("Full optimizer groups and iteration are required")
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            name = name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
            for state in ("exp_avg", "exp_avg_sq", "step"):
                key = f"optimizer.state.{name}.{state}"
                if key not in saved:
                    raise ValueError(f"Missing AdamW state: {key}")
                if state != "step" and tuple(saved[key].size) != tuple(parameter.shape):
                    raise ValueError(f"AdamW shape mismatch: {key}")
        # Check all referenced shard extents, not just a directory/.metadata marker.
        sizes = {}
        for storage in metadata.storage_data.values():
            path = checkpoint / storage.relative_path
            if path not in sizes:
                sizes[path] = path.stat().st_size
            if sizes[path] < storage.offset + storage.length:
                raise ValueError(f"Truncated DCP shard: {path}")
        self._validated_checkpoint_iteration = iteration

    def validate_restored(self, start_iteration: int):
        if self._validated_checkpoint_iteration is None or start_iteration != self._validated_checkpoint_iteration + 1:
            raise ValueError("Restored iteration disagrees with the inspected DCP directory")
        actual = bytes(self.objective_identity.detach().cpu().tolist())
        if actual != self._expected_identity:
            raise ValueError("Saved decoder belongs to a different teacher or global objective")
        if start_iteration > self.start_update and any(int(s.steps) <= 0 for s in self.streams.values()):
            raise ValueError("Resumed global decoder statistics were not restored")

    def forward(self, *, student_cls, student_patches, images, iteration,
                teacher_cls_bottleneck=None, teacher_domains=None):
        if iteration < self.start_update:
            raise ValueError("Global distillation called before its configured fork")
        # Preserve the existing student's iBOT masking; the frozen expert sees
        # the corresponding clean augmented image. No extra 7B forward is added.
        total = student_cls.new_zeros(())
        stats = {}
        for name, teacher in self.teachers.items():
            targets = teacher(images[0], readout=self.readout)
            if self.readout == "cls":
                teacher_cls = targets
            else:
                teacher_cls, teacher_mean = targets.chunk(2, dim=-1)
            pairs = {"cls": (student_cls[0], teacher_cls)}
            if self.readout == "cls_patchmean":
                pairs["patchmean"] = (student_patches[0].float().mean(dim=1), teacher_mean)
            for part, (student, target) in pairs.items():
                loss, info = self.streams[self._stream_key(name, part)](student.unsqueeze(1), target.unsqueeze(1))
                total = total + self.teacher_weights[name] * self.part_weights[part] * loss
                stats[f"extd_{name}_{part}_loss"] = loss.detach()
                stats[f"extd_{name}_{part}_nerr"] = info["nerr"]
                stats[f"extd_{name}_{part}_calibrated"] = info["calibrated"]
            stats[f"extd_{name}_weight"] = self.teacher_weights[name]
        scale = min(1.0, (iteration - self.start_update + 1) / max(1, self.ramp_updates))
        total = total * scale
        stats.update(extd_loss=total.detach(), extd_global_ramp=scale)
        return total, stats
