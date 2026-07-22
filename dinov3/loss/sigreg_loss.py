# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Spectral Independence Gradient Regularization (SIGReg) loss."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.nn import functional as dist_nn


class DistributedSIGReg(nn.Module):
    """Match random projections of representations to a standard normal ECF."""

    def __init__(self, num_slices: int = 1024, range_max: float = 5.0, n_knots: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.range_max = range_max
        self.n_knots = n_knots

        t = torch.linspace(-range_max, range_max, n_knots)
        weights = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("weights", weights)
        self.register_buffer("target_phi", weights.clone())

    def reset_buffers(self) -> None:
        t = torch.linspace(self.range_max * -1, self.range_max, self.n_knots, device=self.t.device)
        weights = torch.exp(-0.5 * t.square())
        self.t.copy_(t)
        self.weights.copy_(weights)
        self.target_phi.copy_(weights)

    def forward(self, z: torch.Tensor, seed_step: int | None = None) -> torch.Tensor:
        _, feature_dim = z.shape
        generator = torch.Generator(device=z.device)
        if seed_step is not None:
            generator.manual_seed(int(seed_step))

        directions = torch.randn(
            feature_dim,
            self.num_slices,
            device=z.device,
            generator=generator,
            dtype=z.dtype,
        )
        directions = directions / directions.norm(p=2, dim=0, keepdim=True)
        projections = z @ directions

        arguments = projections.unsqueeze(-1) * self.t.view(1, 1, -1)
        local_cos = torch.cos(arguments).mean(dim=0)
        local_sin = torch.sin(arguments).mean(dim=0)
        stats = torch.stack([local_cos, local_sin])

        if dist.is_initialized():
            # The autograd-aware collective propagates every rank's ECF gradient.
            stats = dist_nn.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats = stats / dist.get_world_size()

        ecf_real, ecf_imag = stats[0], stats[1]
        diff_real = ecf_real - self.target_phi.view(1, -1)
        loss_unreduced = (diff_real.square() + ecf_imag.square()) * self.weights.view(1, -1)
        return torch.trapezoid(loss_unreduced, self.t, dim=-1).mean()
