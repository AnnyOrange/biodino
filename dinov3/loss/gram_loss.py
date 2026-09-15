# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn import functional as dist_nn


class GramLoss(nn.Module):
    """Implementation of the gram loss"""

    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        """Compute the MSE loss between the gram matrix of the input and target features.

        Args:
            output_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            target_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            img_level: bool, if true gram computed at the image level only else over the entire batch
        Returns:
            loss: scalar
        """

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)


class CrossRankGramLoss(nn.Module):
    """Match per-view sample relations against a frozen global anchor."""

    def __init__(
        self,
        *,
        apply_norm: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
        relation_scope: str = "global",
    ) -> None:
        super().__init__()
        if relation_scope not in {"local", "global"}:
            raise ValueError(
                "relation_scope must be 'local' or 'global', got "
                f"{relation_scope!r}"
            )
        self.relation_scope = relation_scope
        self.gram_loss = GramLoss(
            apply_norm=apply_norm,
            remove_neg=remove_neg,
            remove_only_teacher_neg=remove_only_teacher_neg,
        )

    def _gather_views(self, features: torch.Tensor, *, with_grad: bool) -> torch.Tensor:
        if (
            self.relation_scope != "global"
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return features
        features = features.contiguous()
        if with_grad:
            return torch.cat(dist_nn.all_gather(features), dim=1)
        with torch.no_grad():
            gathered = [torch.empty_like(features) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, features)
            return torch.cat(gathered, dim=1)

    def forward(self, output_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        if output_feats.ndim != 3 or target_feats.ndim != 3:
            raise ValueError(
                "Cross-rank Gram features must have shape [views, batch, dim], got "
                f"student={tuple(output_feats.shape)} teacher={tuple(target_feats.shape)}"
            )
        if output_feats.shape != target_feats.shape:
            raise ValueError(
                "Student and global-anchor features must have identical local shapes, got "
                f"student={tuple(output_feats.shape)} teacher={tuple(target_feats.shape)}"
            )
        global_output = self._gather_views(output_feats, with_grad=True)
        global_target = self._gather_views(target_feats.detach(), with_grad=False)
        return self.gram_loss(global_output, global_target, img_level=True)
