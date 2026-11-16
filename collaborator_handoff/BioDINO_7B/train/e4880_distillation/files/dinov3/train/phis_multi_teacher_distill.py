"""Multi-teacher CLS + patch recoverability distillation with PHI-S-style teacher balancing (NEW mode).

``external_distill.mode = 'phis_spatial_recovery'``.  The lw5 objective (``external_distill.py``,
mode ``spatial_recovery``) is NOT modified; this file only re-uses its ridge-decoder stream.

Same objective family as lw5: each frozen teacher's CLS and (64 sampled) patch features must be linearly
recoverable from the student's pre-head features through an exact ridge decoder refreshed from running
sufficient statistics (fit / monitor = even / odd images).  What differs is how teachers are balanced:

* ``weighting='inverse_variance'`` - exactly the lw5 stream: per-dimension inverse-variance weights
  (clamped, mean 1).  Not invariant to a teacher's overall feature scale.
* ``weighting='phis'`` - PHI-S (Ranzinger et al., "PHI-S: distribution balancing for label-free multi-teacher
  distillation").  PHI-S maps every teacher to y' = H U^T (y - mu) / phi: PCA rotation U, Hadamard rotation H
  and ONE isotropic scale phi chosen so every channel has the same variance.  Our loss is a plain squared
  error and the ridge decoder is solved independently per output channel, so both are invariant to the
  orthogonal part H U^T (the decoder simply rotates with the targets; predictions and errors are rotated
  identically).  Under MSE + ridge the transform is therefore EXACTLY equivalent to dividing each teacher's
  squared error by its mean channel variance (phi^2 = trace(Sigma)/D).  We implement that equivalent form:
  uniform channel weights 1/mean(var); every teacher's loss is its unexplained-variance fraction, so teachers
  with large feature norms (e.g. GenBio, 4608-d concatenated RGB encodings) cannot dominate.

Teachers are loaded through ``FrozenJointTeacher`` (SHA256-pinned manifest, image-only SSL provenance,
bf16 weights so three ~1B teachers fit next to an H+ student on a 32 GB card).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.train.external_distill import ExternalRecoveryStream
from dinov3.train.joint_teacher_models import FrozenJointTeacher

logger = logging.getLogger("dinov3")

WEIGHTINGS = ("inverse_variance", "phis")


class BalancedRecoveryStream(ExternalRecoveryStream):
    """``ExternalRecoveryStream`` with a selectable per-teacher balancing (see module docstring)."""

    def __init__(self, in_dim, out_dim, *, weighting: str, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.weighting = weighting

    def forward(self, student: Tensor, target: Tensor, sample_weight: Tensor | None = None):
        if self.weighting == "inverse_variance":
            return super().forward(student, target, sample_weight)  # the lw5 stream, bit for bit
        if sample_weight is not None:
            raise ValueError("phis weighting does not support domain-routed teachers")
        if student.ndim != 3 or target.ndim != 3 or student.shape[:2] != target.shape[:2] or len(student) < 2:
            raise ValueError(f"expects [B,N,Ds]/[B,N,Dt] with B>=2, got {tuple(student.shape)} {tuple(target.shape)}")
        if student.shape[-1] != self.in_dim or target.shape[-1] != self.out_dim:
            raise ValueError(f"stream {self.in_dim}->{self.out_dim}, got {student.shape[-1]}->{target.shape[-1]}")
        with torch.autocast(device_type=student.device.type, enabled=False):
            x, y = student.float(), target.detach().float()
            fit_x, fit_y = x[::2].flatten(0, 1), y[::2].flatten(0, 1)
            val_x, val_y = x[1::2].flatten(0, 1), y[1::2].flatten(0, 1)
            decoder = self.delta.detach().clone()
            raw, flat_y = x.flatten(0, 1), y.flatten(0, 1)
            prediction = torch.cat((raw, torch.ones_like(raw[:, :1])), dim=1) @ decoder
            variance = (self.second - self.mean.square()).clamp_min(1e-8)
            scale = variance.mean().clamp_min(1e-8).reciprocal().detach()   # 1/phi^2, one scalar per teacher
            calibrated = int(self.steps) >= self.warmup
            loss = (prediction - flat_y).square().mean() * scale if calibrated else prediction.sum() * 0
            with torch.no_grad():
                va = torch.cat((val_x, torch.ones_like(val_x[:, :1])), dim=1)
                held_out = self._average((va @ decoder - val_y).square().mean(0))
                self.monitor.lerp_(held_out, 0.1)
                step = int(self.steps)
                fa = torch.cat((fit_x.detach(), torch.ones_like(fit_x[:, :1])), dim=1)
                cov = self._average(fa.T @ fa / len(fa))
                cross = self._average(fa.T @ fit_y / len(fa))
                mu = self._average(fit_y.mean(0))
                sq = self._average(fit_y.square().mean(0))
                rate = 1.0 / (step + 1) if step < self.warmup else 1.0 - self.momentum
                self.xx.lerp_(cov, rate)
                self.xr.lerp_(cross, rate)
                self.mean.lerp_(mu, rate)
                self.second.lerp_(sq, rate)
                if (step + 1) % self.solve_every == 0:
                    regularized = self.xx + self.ridge * torch.eye(self.in_dim + 1, device=x.device)
                    self.delta.copy_(torch.linalg.solve(regularized, self.xr))
                self.steps.add_(1)
            # held-out unexplained-variance fraction (1.0 == predicting the teacher mean)
            nerr = (self.monitor.mean() / variance.mean().clamp_min(1e-8)).detach()
            return loss, {"error": held_out.mean().detach(), "nerr": nerr, "calibrated": float(calibrated)}


class _PatchTeacher(nn.Module):
    """(cls, patches) readout of a manifest-pinned ``FrozenJointTeacher`` (its own forward only gives pooled CLS)."""

    def __init__(self, frozen: FrozenJointTeacher):
        super().__init__()
        self.frozen = frozen
        native = frozen.embed_dim
        self.embed_dim = native

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        t = self.frozen
        cls_out, patch_out = [], []
        for chunk in images.split(t.chunk_size):
            x = t.preprocess(chunk)
            with torch.autocast(device_type=x.device.type, dtype=t.storage_dtype,
                                enabled=t.storage_dtype in (torch.bfloat16, torch.float16)):
                if t.kind == "genbio_pathfm":
                    cls, patches = t.model.forward_with_patches(x)
                elif t.kind == "timm_dir":
                    tokens = t.model.forward_features(x)
                    cls, patches = tokens[:, 0], tokens[:, t.num_prefix_tokens:]
                else:
                    features = t.model.forward_features(x)
                    cls, patches = features["x_norm_clstoken"], features["x_norm_patchtokens"]
            cls, patches = cls.float(), patches.float()
            if cls.shape[-1] != self.embed_dim or patches.shape[-1] != self.embed_dim:
                raise RuntimeError(f"{t.name}: unexpected feature dims {tuple(cls.shape)} {tuple(patches.shape)}")
            if not (torch.isfinite(cls).all() and torch.isfinite(patches).all()):
                raise FloatingPointError(f"{t.name}: nonfinite teacher features")
            cls_out.append(cls)
            patch_out.append(patches)
        return torch.cat(cls_out), torch.cat(patch_out)


class PhisMultiTeacherDistillation(nn.Module):
    """Multi-teacher CLS + patch recoverability on the first global crop (lw5 family, new balancing)."""

    def __init__(self, cfg, student_dim: int):
        super().__init__()
        ed = cfg.external_distill
        self.cls_weight, self.patch_weight = float(ed.cls_weight), float(ed.patch_weight)
        self.n_patches = int(ed.patches)
        self.weighting = str(ed.weighting)
        names = list(ed.teacher_names)
        weights = [float(w) for w in ed.teacher_weights]
        if not names or len(names) != len(weights) or len(set(names)) != len(names):
            raise ValueError(f"teacher_names / teacher_weights mismatch: {names} {weights}")
        manifest = json.loads(Path(ed.teacher_manifest).read_text())
        specs = {t["name"]: t for t in manifest["teachers"]}
        student_mean, student_std = list(cfg.crops.rgb_mean), list(cfg.crops.rgb_std)
        teachers, streams = {}, {}
        for name in names:
            frozen = FrozenJointTeacher(specs[name], student_mean=student_mean, student_std=student_std,
                                        device="cpu", dtype=torch.bfloat16)
            teachers[name] = _PatchTeacher(frozen)
            for part in ("cls", "patch"):
                streams[f"{name}__{part}"] = BalancedRecoveryStream(
                    student_dim, teachers[name].embed_dim, weighting=self.weighting, ridge=float(ed.ridge),
                    momentum=float(ed.momentum), warmup=int(ed.warmup), solve_every=int(ed.solve_every))
            logger.info("[PHIS] teacher %s kind=%s dim=%d img=%d weighting=%s", name, frozen.kind,
                        teachers[name].embed_dim, frozen.img_size, self.weighting)
        self.teachers = nn.ModuleDict(teachers)
        self.streams = nn.ModuleDict(streams)
        self.teacher_weights = dict(zip(names, weights))
        self.signatures = {n: t.frozen.signature for n, t in teachers.items()}

    def teacher_state_keys(self) -> list[str]:
        return [k for k in self.state_dict().keys() if k.startswith("teachers.")]

    def reset_buffers(self) -> None:
        for s in self.streams.values():
            s.reset_buffers()

    def forward(self, *, student_cls: Tensor, student_patches: Tensor, images: Tensor,
                teacher_cls_bottleneck: Tensor | None = None, teacher_domains=None):
        s_cls = student_cls[0].unsqueeze(1)    # [B,1,Ds]   first global crop only (as lw5)
        s_patch = student_patches[0]           # [B,P,Ds]
        x = images[0]
        B, P, _ = s_patch.shape
        grid = int(round(P ** 0.5))
        idx = torch.linspace(0, P - 1, min(self.n_patches, P), device=s_patch.device).long()
        total = s_cls.new_zeros(())
        stats: dict[str, Tensor | float] = {}
        for name, teacher in self.teachers.items():
            t_cls, t_patch = teacher(x)
            if t_patch.shape[1] != P:          # e.g. GenBio 224/16 = 14x14 -> student 16x16 (lw5 does the same)
                tg = int(round(t_patch.shape[1] ** 0.5))
                t_patch = F.interpolate(t_patch.transpose(1, 2).reshape(B, -1, tg, tg), size=(grid, grid),
                                        mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            cls_loss, cls_stats = self.streams[f"{name}__cls"](s_cls, t_cls.unsqueeze(1))
            patch_loss, patch_stats = self.streams[f"{name}__patch"](s_patch[:, idx], t_patch[:, idx])
            w = self.teacher_weights[name]
            total = total + w * (self.cls_weight * cls_loss + self.patch_weight * patch_loss)
            stats[f"extd_{name}_cls_loss"] = cls_loss.detach()
            stats[f"extd_{name}_patch_loss"] = patch_loss.detach()
            stats[f"extd_{name}_cls_nerr"] = cls_stats["nerr"]
            stats[f"extd_{name}_patch_nerr"] = patch_stats["nerr"]
            stats[f"extd_{name}_calibrated"] = cls_stats["calibrated"]
        stats["extd_loss"] = total.detach()
        return total, stats
