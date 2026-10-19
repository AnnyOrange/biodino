"""External-expert recoverability distillation (multi-teacher, no new trainable parameters).

For every frozen external teacher ``t`` (a pathology FM, the general-domain DINOv3 release,
an earlier microscopy teacher, ...) the student is trained so that the teacher's CLS and
patch features are *linearly recoverable* from the student's own pre-head features::

    L_t = mean_d  w_d * ( [s, 1] @ W_t - f_t(x) )_d^2

``W_t`` is the ridge-regression decoder fit on running, cross-rank averaged sufficient
statistics of (student, teacher) pairs and refreshed every ``solve_every`` updates; ``w_d`` is
the inverse running variance of teacher dimension ``d`` (so teachers with different feature
scales contribute comparably).  Gradients flow only into the student.  Because the decoder
is the exact least-squares optimum rather than an SGD-trained adaptor, resuming a trunk
checkpoint needs no optimizer surgery, and the objective is exactly what frozen linear
probes measure downstream.  Fit and monitor images are disjoint (even/odd positions in the
batch), so the reported ``error`` is a held-out recoverability error.

This generalises :class:`dinov3.train.selective_recoverability.RecoveryStream` (same-dimension,
residual decoder, in-run EMA anchor) to arbitrary-dimension external teachers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger("dinov3")


def _as_tuple3(values, name: str) -> tuple[float, float, float]:
    vals = tuple(float(v) for v in values)
    if len(vals) != 3:
        raise ValueError(f"{name} must have 3 entries, got {vals}")
    return vals


class ExternalRecoveryStream(nn.Module):
    """Ridge-decoder recoverability of ``out_dim`` teacher features from ``in_dim`` student features."""

    def __init__(self, in_dim: int, out_dim: int, ridge: float = 0.1, momentum: float = 0.98,
                 warmup: int = 32, solve_every: int = 8):
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.ridge, self.momentum = float(ridge), float(momentum)
        self.warmup, self.solve_every = int(warmup), int(solve_every)
        d_in, d_out = self.in_dim + 1, self.out_dim
        for name, shape in {"xx": (d_in, d_in), "xr": (d_in, d_out), "delta": (d_in, d_out),
                            "mean": (d_out,), "second": (d_out,), "monitor": (d_out,)}.items():
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def reset_buffers(self) -> None:
        for buf in self.buffers():
            buf.zero_()

    @staticmethod
    def _average(x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x)
            x /= dist.get_world_size()
        return x

    def forward(self, student: Tensor, target: Tensor, sample_weight: Tensor | None = None
                ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if student.ndim != 3 or target.ndim != 3 or student.shape[:2] != target.shape[:2] or len(student) < 2:
            raise ValueError(
                "recoverability expects >=2 images [B,N,D_student] / [B,N,D_teacher] with matching B,N; "
                f"got {tuple(student.shape)} vs {tuple(target.shape)}"
            )
        if student.shape[-1] != self.in_dim or target.shape[-1] != self.out_dim:
            raise ValueError(
                f"stream built for {self.in_dim}->{self.out_dim}, got {student.shape[-1]}->{target.shape[-1]}"
            )
        if sample_weight is not None:
            return self._forward_routed(student, target, sample_weight)
        with torch.autocast(device_type=student.device.type, enabled=False):
            x, y = student.float(), target.detach().float()
            fit_x, fit_y = x[::2].flatten(0, 1), y[::2].flatten(0, 1)
            val_x, val_y = x[1::2].flatten(0, 1), y[1::2].flatten(0, 1)
            decoder = self.delta.detach().clone()
            raw, flat_y = x.flatten(0, 1), y.flatten(0, 1)
            aug = torch.cat((raw, torch.ones_like(raw[:, :1])), dim=1)
            prediction = aug @ decoder
            variance = (self.second - self.mean.square()).clamp_min(1e-8)
            # Inverse-variance weights, clamped relative to the teacher's typical variance so
            # (near-)constant teacher dimensions cannot dominate, then normalised to mean 1.
            floor = 0.05 * variance.mean().clamp_min(1e-8)
            weights = variance.clamp_min(floor).reciprocal()
            weights = (weights / weights.mean()).detach()
            calibrated = int(self.steps) >= self.warmup
            errors = (prediction - flat_y).square()
            loss = (errors * weights).mean() if calibrated else prediction.sum() * 0
            with torch.no_grad():
                va = torch.cat((val_x, torch.ones_like(val_x[:, :1])), dim=1)
                held_out = self._average(((va @ decoder - val_y).square() * weights).mean(0))
                self.monitor.lerp_(held_out, 0.1)
                step = int(self.steps)
                fx = fit_x.detach()
                fa = torch.cat((fx, torch.ones_like(fx[:, :1])), dim=1)
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
            # Normalised held-out error: 1.0 == predicting the teacher mean; <1 == information recovered.
            explained = (self.monitor / variance.clamp_min(floor)).mean()
            return loss, {"error": held_out.mean().detach(), "nerr": explained.detach(),
                          "calibrated": float(calibrated)}

    def _forward_routed(self, student, target, sample_weight):
        """Fit only eligible images; keep the original per-image loss coefficient.

        Weighted sufficient-statistic sums and counts are reduced separately.
        This handles ranks with zero pathology images without rank-wise bias or
        collective mismatches. Even/odd image positions remain fit/monitor sets.
        """
        if sample_weight.shape != student.shape[:1] or not torch.isfinite(sample_weight).all():
            raise ValueError('Expected one finite domain weight per image')
        if (sample_weight < 0).any() or (sample_weight > 1).any():
            raise ValueError('Domain weights must be in [0,1]')
        with torch.autocast(device_type=student.device.type, enabled=False):
            x, y = student.float(), target.detach().float()
            sw = sample_weight.detach().float()[:, None].expand(x.shape[:2])
            raw = x.flatten(0, 1)
            aug = torch.cat((raw, torch.ones_like(raw[:, :1])), dim=1)
            decoder = self.delta.detach().clone()
            prediction = aug @ decoder
            variance = (self.second - self.mean.square()).clamp_min(1e-8)
            floor = .05 * variance.mean().clamp_min(1e-8)
            weights = variance.clamp_min(floor).reciprocal()
            weights = (weights / weights.mean()).detach()
            calibrated = int(self.steps) >= self.warmup
            # Divide by ALL images, not active images: rare H&E batches do not
            # silently amplify the original lw5 coefficient by ~10x.
            loss = (((prediction - y.flatten(0, 1)).square() * weights)
                    * sw.flatten()[:, None]).mean() if calibrated else prediction.sum() * 0
            with torch.no_grad():
                fx, fy = x[::2].detach().flatten(0, 1), y[::2].flatten(0, 1)
                vx, vy = x[1::2].detach().flatten(0, 1), y[1::2].flatten(0, 1)
                fw, vw = sw[::2].flatten(), sw[1::2].flatten()
                counts = self._average(torch.stack((fw.sum(), vw.sum())))
                fa = torch.cat((fx, torch.ones_like(fx[:, :1])), dim=1)
                va = torch.cat((vx, torch.ones_like(vx[:, :1])), dim=1)
                held_out = self._average((((va @ decoder - vy).square() * weights) * vw[:, None]).sum(0))
                if counts[1] > 0:
                    held_out /= counts[1]
                    self.monitor.lerp_(held_out, .1)
                if counts[0] > 0:
                    cov = self._average(fa.T @ (fa * fw[:, None])) / counts[0]
                    cross = self._average(fa.T @ (fy * fw[:, None])) / counts[0]
                    mu = self._average((fy * fw[:, None]).sum(0)) / counts[0]
                    sq = self._average((fy.square() * fw[:, None]).sum(0)) / counts[0]
                    step = int(self.steps)
                    rate = 1 / (step + 1) if step < self.warmup else 1 - self.momentum
                    self.xx.lerp_(cov, rate)
                    self.xr.lerp_(cross, rate)
                    self.mean.lerp_(mu, rate)
                    self.second.lerp_(sq, rate)
                    if (step + 1) % self.solve_every == 0:
                        regularized = self.xx + self.ridge * torch.eye(self.in_dim + 1, device=x.device)
                        self.delta.copy_(torch.linalg.solve(regularized, self.xr))
                    self.steps.add_(1)
            return loss, {'error': held_out.mean().detach(),
                          'nerr': (self.monitor / variance.clamp_min(floor)).mean().detach(),
                          'calibrated': float(calibrated)}


class _TeacherBase(nn.Module):
    """Frozen teacher operating on the *student-normalised* global crops."""

    embed_dim: int

    def __init__(self, name: str, img_size: int, mean, std, student_mean, student_std):
        super().__init__()
        self.name, self.img_size = str(name), int(img_size)
        self.register_buffer("t_mean", torch.tensor(_as_tuple3(mean, "mean")).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("t_std", torch.tensor(_as_tuple3(std, "std")).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("s_mean", torch.tensor(_as_tuple3(student_mean, "student mean")).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("s_std", torch.tensor(_as_tuple3(student_std, "student std")).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, x: Tensor) -> Tensor:
        # student-normalised -> [0,1] -> teacher-normalised, resized to the teacher's grid
        x01 = (x.float() * self.s_std + self.s_mean).clamp_(0.0, 1.0)
        if x01.shape[-1] != self.img_size or x01.shape[-2] != self.img_size:
            x01 = F.interpolate(x01, size=(self.img_size, self.img_size), mode="bicubic",
                                align_corners=False, antialias=True).clamp_(0.0, 1.0)
        return (x01 - self.t_mean) / self.t_std

    def features(self, x: Tensor) -> tuple[Tensor, Tensor]:  # -> cls [N,D], patches [N,P,D]
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            cls, patches = self.features(self.preprocess(x))
        return cls.float(), patches.float()


class TimmDirTeacher(_TeacherBase):
    """A timm ViT stored as a HF-style directory (config.json + pytorch_model.bin / model.safetensors)."""

    def __init__(self, name, path, img_size, mean, std, student_mean, student_std, model_args=None):
        super().__init__(name, img_size, mean, std, student_mean, student_std)
        import timm

        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        args: dict[str, Any] = dict(cfg.get("model_args", {}))
        args.setdefault("num_classes", cfg.get("num_classes", 0))
        args.update({k: v for k, v in (model_args or {}).items()})
        args.setdefault("img_size", self.img_size)
        args.setdefault("dynamic_img_size", False)
        model = timm.create_model(cfg["architecture"], pretrained=False, **args)
        weight_path = path / "model.safetensors"
        if weight_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(weight_path))
        else:
            state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("[EXTD] timm teacher %s (%s): %d params, missing=%d unexpected=%d",
                    self.name, cfg["architecture"], sum(p.numel() for p in model.parameters()),
                    len(missing), len(unexpected))
        if any("blocks" in k for k in missing):
            raise RuntimeError(f"teacher {self.name}: transformer weights missing from {path}: {missing[:5]}")
        self.model = model.eval().requires_grad_(False)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
        self.embed_dim = int(model.num_features)

    def features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.model.forward_features(x)  # [N, prefix + P, D]
        return tokens[:, 0], tokens[:, self.num_prefix_tokens:]


class DinoV3Teacher(_TeacherBase):
    """A DINOv3 backbone built from the run's own student config and loaded from a flat release file
    or a training export (``{"teacher": {"backbone.*": ...}}``)."""

    def __init__(self, name, weights, img_size, mean, std, student_mean, student_std, cfg):
        super().__init__(name, img_size, mean, std, student_mean, student_std)
        from dinov3.models import build_model

        # build_model_from_cfg() places the network on the meta device (the trunk is materialised by
        # FSDP later); a frozen replica teacher must be materialised and initialised here instead.
        backbone, embed_dim = build_model(cfg.student, only_teacher=True, img_size=self.img_size, device="cpu")
        backbone.init_weights()  # also fills the non-persistent RoPE / mask buffers
        raw = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "teacher" in raw:
            state = {k.removeprefix("backbone."): v for k, v in raw["teacher"].items() if k.startswith("backbone.")}
        elif isinstance(raw, dict) and "model" in raw:
            state = {k.removeprefix("teacher.backbone."): v for k, v in raw["model"].items()
                     if k.startswith("teacher.backbone.")}
        else:
            state = dict(raw)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        core_missing = [k for k in missing if "rope" not in k and "bias_mask" not in k]
        logger.info("[EXTD] dinov3 teacher %s from %s: missing=%d (core %d) unexpected=%d",
                    self.name, weights, len(missing), len(core_missing), len(unexpected))
        if core_missing:
            raise RuntimeError(f"teacher {self.name}: backbone keys missing: {core_missing[:5]}")
        self.model = backbone.eval().requires_grad_(False)
        self.embed_dim = int(embed_dim)

    def features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.model(x, is_training=True)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]


def build_teacher(spec, cfg, student_mean, student_std) -> _TeacherBase:
    kind = str(spec.kind)
    common = dict(name=spec.name, img_size=int(spec.img_size), mean=list(spec.mean), std=list(spec.std),
                  student_mean=student_mean, student_std=student_std)
    if kind == "timm_dir":
        model_args = dict(getattr(spec, "model_args", None) or {})
        return TimmDirTeacher(path=spec.path, model_args=model_args, **common)
    if kind == "dinov3":
        return DinoV3Teacher(weights=spec.weights, cfg=cfg, **common)
    raise ValueError(f"unknown external teacher kind {kind!r} (timm_dir|dinov3)")


class TeacherResidualizer(nn.Module):
    """Remove knowledge linearly explained by a frozen reference teacher.

    Use the decoder from preceding minibatches for the current residual, then
    update its fit/monitor statistics. Both teachers and this decomposition are
    detached; the student cannot reduce its target by changing the reference.
    """

    def __init__(self, reference_dim, target_dim, **kwargs):
        super().__init__()
        self.fit = ExternalRecoveryStream(reference_dim, target_dim, **kwargs)

    @torch.no_grad()
    def forward(self, reference, target, sample_weight=None):
        with torch.autocast(device_type=reference.device.type, enabled=False):
            x, y = reference.detach().float(), target.detach().float()
            augmented = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
            residual = y - augmented @ self.fit.delta.detach()
            _, stats = self.fit(x, y, sample_weight)
        return residual, stats


class ExternalDistillation(nn.Module):
    """Multi-teacher recoverability loss on the first global crop."""

    def __init__(self, cfg, student_dim: int):
        super().__init__()
        ed = cfg.external_distill
        self.cls_weight = float(ed.cls_weight)
        self.patch_weight = float(ed.patch_weight)
        self.n_patches = int(ed.patches)
        student_mean, student_std = list(cfg.crops.rgb_mean), list(cfg.crops.rgb_std)
        if not len(ed.teachers):
            raise ValueError("external_distill.enabled requires at least one entry in external_distill.teachers")
        teachers, streams, weights = {}, {}, {}
        self.teacher_domains = {}
        self.residual_references = {}
        self.self_targets: list[str] = []
        for spec in ed.teachers:
            domains = tuple(getattr(spec, "domains", []) or [])
            if any(d not in ("pathology", "other") for d in domains):
                raise ValueError(f'Unsupported teacher domains: {domains}')
            self.teacher_domains[spec.name] = domains
            weights[spec.name] = float(getattr(spec, "weight", 1.0))
            if str(spec.kind) == "self_bottleneck":
                # Self-readout alignment: the EMA teacher's own DINO-head bottleneck (the coordinates that define its
                # prototype assignment, i.e. the geometry kNN/NMI readouts see) must be linearly recoverable from the
                # student's raw pre-head CLS. No network, no new trainable parameters; CLS stream only.
                out_dim = int(cfg.dino.head_bottleneck_dim)
                streams[f"{spec.name}__cls"] = ExternalRecoveryStream(
                    student_dim, out_dim, ridge=float(ed.ridge), momentum=float(ed.momentum),
                    warmup=int(ed.warmup), solve_every=int(ed.solve_every))
                self.self_targets.append(spec.name)
                logger.info("[EXTD] self-readout target %s: teacher DINO-head bottleneck dim=%d weight=%.3g",
                            spec.name, out_dim, weights[spec.name])
                continue
            teacher = build_teacher(spec, cfg, student_mean, student_std)
            teachers[spec.name] = teacher
            reference = getattr(spec, 'residual_of', None)
            if reference:
                self.residual_references[spec.name] = str(reference)
            for part in ("cls", "patch"):
                streams[f"{spec.name}__{part}"] = ExternalRecoveryStream(
                    student_dim, teacher.embed_dim, ridge=float(ed.ridge), momentum=float(ed.momentum),
                    warmup=int(ed.warmup), solve_every=int(ed.solve_every))
            logger.info("[EXTD] teacher %s: kind=%s dim=%d img=%d weight=%.3g", spec.name, spec.kind,
                        teacher.embed_dim, int(spec.img_size), weights[spec.name])
        self.teachers = nn.ModuleDict(teachers)
        self.streams = nn.ModuleDict(streams)
        self.teacher_weights = weights
        residualizers = {}
        for name, reference in self.residual_references.items():
            if reference == name or reference not in teachers:
                raise ValueError(f'Residual teacher {name} requires a separate frozen reference teacher')
            if list(teachers).index(reference) >= list(teachers).index(name):
                raise ValueError('The reference teacher must precede its residual teacher')
            if self.teacher_domains[reference]:
                raise ValueError('The residual reference must cover all training images')
            for part in ('cls', 'patch'):
                residualizers[f'{name}__{part}'] = TeacherResidualizer(
                    teachers[reference].embed_dim, teachers[name].embed_dim,
                    ridge=float(ed.ridge), momentum=float(ed.momentum),
                    warmup=int(ed.warmup), solve_every=int(ed.solve_every))
        self.residualizers = nn.ModuleDict(residualizers)

    def teacher_state_keys(self) -> list[str]:
        return [k for k in self.state_dict().keys() if k.startswith("teachers.")]

    def reset_buffers(self) -> None:
        for s in self.streams.values():
            s.reset_buffers()
        for projection in self.residualizers.values():
            projection.fit.reset_buffers()

    def forward(self, *, student_cls: Tensor, student_patches: Tensor, images: Tensor,
                teacher_cls_bottleneck: Tensor | None = None, teacher_domains=None):
        # Only the first global crop: fit/monitor halves must not share an image.
        s_cls = student_cls[0].unsqueeze(1)  # [B,1,Ds]
        s_patch = student_patches[0]  # [B,P,Ds]
        x = images[0]  # [B,3,H,W]
        B, P, _ = s_patch.shape
        grid = int(round(P ** 0.5))
        idx = torch.linspace(0, P - 1, min(self.n_patches, P), device=s_patch.device).long()
        total = s_cls.new_zeros(())
        stats: dict[str, Tensor | float] = {}
        reference_features = {}
        for name in self.self_targets:
            if teacher_cls_bottleneck is None:
                raise ValueError("self_bottleneck target needs teacher_global['cls_bottleneck']")
            t = F.normalize(teacher_cls_bottleneck[0].detach().float(), dim=-1)  # [B,256] unit-sphere coordinates
            cls_loss, cls_stats = self.streams[f"{name}__cls"](s_cls, t.unsqueeze(1))
            total = total + self.teacher_weights[name] * self.cls_weight * cls_loss
            stats[f"extd_{name}_cls_loss"] = cls_loss.detach()
            stats[f"extd_{name}_cls_nerr"] = cls_stats["nerr"]
            stats[f"extd_{name}_calibrated"] = cls_stats["calibrated"]
        for name, teacher in self.teachers.items():
            domains = self.teacher_domains[name]
            sample_weight = None
            selected = torch.ones(B, device=x.device, dtype=torch.bool)
            if domains:
                if teacher_domains is None or len(teacher_domains) != B:
                    raise ValueError('Domain-routed teacher requires training metadata for every image')
                selected = torch.tensor([d in domains for d in teacher_domains], device=x.device)
                sample_weight = selected.float()
                stats[f'extd_{name}_eligible_fraction'] = sample_weight.mean()
            # Frozen teachers have no distributed collectives. Skip their
            # expensive forward on ineligible images, including empty ranks.
            if selected.any():
                selected_cls, selected_patch = teacher(x[selected])
                if selected_patch.shape[1] != P:
                    tg = int(round(selected_patch.shape[1] ** .5))
                    selected_patch = F.interpolate(
                        selected_patch.transpose(1, 2).reshape(-1, teacher.embed_dim, tg, tg),
                        size=(grid, grid), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
                t_cls = selected_cls.new_zeros((B, teacher.embed_dim))
                t_patch = selected_patch.new_zeros((B, P, teacher.embed_dim))
                t_cls[selected], t_patch[selected] = selected_cls, selected_patch
            else:
                t_cls = x.new_zeros((B, teacher.embed_dim), dtype=torch.float32)
                t_patch = x.new_zeros((B, P, teacher.embed_dim), dtype=torch.float32)
            if t_patch.shape[1] != P:
                tg = int(round(t_patch.shape[1] ** 0.5))
                t_patch = F.interpolate(
                    t_patch.transpose(1, 2).reshape(B, -1, tg, tg), size=(grid, grid),
                    mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            w = self.teacher_weights[name]
            cls_target, patch_target = t_cls.unsqueeze(1), t_patch[:, idx]
            if name in self.residual_references.values():
                reference_features[name] = (cls_target.detach(), patch_target.detach())
            if name in self.residual_references:
                ref_cls, ref_patch = reference_features[self.residual_references[name]]
                cls_target, ref_cls_stats = self.residualizers[f'{name}__cls'](
                    ref_cls, cls_target, sample_weight)
                patch_target, ref_patch_stats = self.residualizers[f'{name}__patch'](
                    ref_patch, patch_target, sample_weight)
                stats[f'extd_{name}_reference_cls_nerr'] = ref_cls_stats['nerr']
                stats[f'extd_{name}_reference_patch_nerr'] = ref_patch_stats['nerr']
            cls_loss, cls_stats = self.streams[f"{name}__cls"](s_cls, cls_target, sample_weight)
            patch_loss, patch_stats = self.streams[f"{name}__patch"](s_patch[:, idx], patch_target, sample_weight)
            total = total + w * (self.cls_weight * cls_loss + self.patch_weight * patch_loss)
            stats[f"extd_{name}_cls_loss"] = cls_loss.detach()
            stats[f"extd_{name}_patch_loss"] = patch_loss.detach()
            stats[f"extd_{name}_cls_nerr"] = cls_stats["nerr"]
            stats[f"extd_{name}_patch_nerr"] = patch_stats["nerr"]
            stats[f"extd_{name}_calibrated"] = cls_stats["calibrated"]
        stats["extd_loss"] = total.detach()
        return total, stats
