"""Prequential, mask-matched historical-feature recovery for DDP SSL.

The decoder is fit on previous/even-image observations, while its controller
observes held-out odd images BEFORE adding the current fit observations.
All sufficient statistics and controllers are synchronized across DDP ranks.
This is a retention surrogate, not a guarantee on arbitrary downstream scores.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class RecoveryStream(nn.Module):
    def __init__(self, dim, mode="adaptive", ridge=0.1, tolerance=0.02,
                 momentum=0.98, warmup=32, solve_every=8, dual_lr=0.05):
        super().__init__()
        self.dim, self.mode = dim, mode
        self.ridge, self.tolerance = ridge, tolerance
        self.momentum, self.warmup = momentum, warmup
        self.solve_every, self.dual_lr = solve_every, dual_lr
        for name, shape in {"xx": (dim+1, dim+1), "xr": (dim+1, dim),
                            "delta": (dim+1, dim), "mean": (dim,),
                            "second": (dim,), "floor": (dim,),
                            "monitor": (dim,), "dual": (dim,)}.items():
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @staticmethod
    def average(x):
        if dist.is_initialized():
            dist.all_reduce(x)
            x /= dist.get_world_size()
        return x

    def forward(self, student, anchor):
        if student.shape != anchor.shape or student.ndim != 3 or len(student) < 2:
            raise ValueError("recovery expects >=2 independent images [B,N,D] in both tensors")
        with torch.autocast(device_type=student.device.type, enabled=False):
            x, y = student.float(), anchor.detach().float()
            fit_x, fit_y = x[::2].flatten(0, 1), y[::2].flatten(0, 1)
            val_x, val_y = x[1::2].flatten(0, 1), y[1::2].flatten(0, 1)
            # Clone because sufficient statistics are updated before backward.
            decoder = self.delta.detach().clone()
            raw = x.flatten(0, 1)
            aug = torch.cat((raw, torch.ones_like(raw[:, :1])), dim=1)
            prediction = raw + aug @ decoder
            variance = (self.second - self.mean.square()).clamp_min(0.05)
            weights = variance.reciprocal().clamp_max(20).detach().clone()
            errors = (prediction - y.flatten(0, 1)).square()
            gate = self.dual.detach().clone() if self.mode == "adaptive" else torch.ones_like(weights)
            calibrated = int(self.steps) >= self.warmup
            loss = (errors * weights * gate).mean() if calibrated else prediction.sum() * 0
            with torch.no_grad():
                va = torch.cat((val_x, torch.ones_like(val_x[:, :1])), dim=1)
                ve = self.average(((val_x + va @ decoder - val_y).square() * weights).mean(0))
                self.monitor.lerp_(ve, 0.1)
                step = int(self.steps)
                if step < self.warmup:
                    # Use the terminal calibration EMA, not the average of
                    # poorly conditioned initial decoder transients.
                    self.floor.copy_(self.monitor)
                else:
                    # Consolidate newly achieved recoverability too. The
                    # tolerance absorbs small monitor fluctuations.
                    self.floor.copy_(torch.minimum(self.floor, self.monitor))
                    self.dual.add_(self.dual_lr * (self.monitor-self.floor-self.tolerance)).clamp_(0, 1)
                fx = fit_x.detach()
                fa = torch.cat((fx, torch.ones_like(fx[:, :1])), dim=1)
                cov = self.average(fa.T @ fa / len(fa))
                cross = self.average(fa.T @ (fit_y-fx) / len(fa))
                mu = self.average(fit_y.mean(0))
                sq = self.average(fit_y.square().mean(0))
                rate = 1.0 / (step+1) if step < self.warmup else 1-self.momentum
                self.xx.lerp_(cov, rate); self.xr.lerp_(cross, rate)
                self.mean.lerp_(mu, rate); self.second.lerp_(sq, rate)
                if (step+1) % self.solve_every == 0:
                    regularized = self.xx + self.ridge * torch.eye(self.dim+1, device=x.device)
                    self.delta.copy_(torch.linalg.solve(regularized, self.xr))
                self.steps.add_(1)
            return loss, {"error": ve.mean().detach(), "gate": gate.mean().detach(),
                          "floor": self.floor.mean().detach(), "calibrated": float(calibrated)}


class SelectiveRecoverability(nn.Module):
    def __init__(self, dim, mode="adaptive", ridge=0.1, tolerance=0.02,
                 warmup=32, patches=16, global_weight=1.0, local_weight=1.0):
        super().__init__()
        # Both streams are always constructed so checkpoints keep the same keys
        # across arms; ``local_weight == 0`` skips the patch stream entirely.
        self.local = RecoveryStream(dim, mode, ridge, tolerance, warmup=warmup)
        self.global_stream = RecoveryStream(dim, mode, ridge, tolerance, warmup=warmup)
        self.patches, self.global_weight, self.local_weight = patches, global_weight, local_weight

    def forward(self, gram):
        sp, ap = gram["student_patches"], gram["teacher_patches"]
        # First global crop only; a given image cannot appear in both fit/monitor.
        b = gram["student_cls"].shape[1]
        s = torch.stack((gram["student_cls"][0], sp[:b].mean(1)), dim=1)
        a = torch.stack((gram["teacher_cls"][0], ap[:b].mean(1)), dim=1)
        global_loss, gm = self.global_stream(s, a)
        stats = {f"recovery_global_{k}": v for k,v in gm.items()}
        stats.update(recovery_global_loss=global_loss.detach())
        total = self.global_weight * global_loss
        if self.local_weight > 0:
            idx = torch.linspace(0, sp.shape[1]-1, self.patches, device=sp.device).long()
            local, lm = self.local(sp[:b, idx], ap[:b, idx])
            stats.update({f"recovery_local_{k}": v for k,v in lm.items()})
            stats.update(recovery_local_loss=local.detach())
            total = total + self.local_weight * local
        stats.update(recovery_local_weight=float(self.local_weight), recovery_global_weight=float(self.global_weight))
        return total, stats
