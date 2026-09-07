# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Small, auditable primitives for DoReMi-style domain reweighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DomainWeightStep:
    step: int
    excess_losses: tuple[float, ...]
    weights: tuple[float, ...]
    averaged_weights: tuple[float, ...]


class DoReMiDomainWeights:
    """Exponentiated-gradient domain weights from DoReMi Algorithm 1.

    Inputs are already domain-averaged proxy and reference losses.  The
    optimizer clips their difference at zero, applies exponentiated gradient
    ascent, smooths toward uniform, and tracks the trajectory average used to
    train the final large model.
    """

    def __init__(
        self,
        domain_names: Sequence[str],
        *,
        step_size: float = 1.0,
        smoothing: float = 1e-3,
    ) -> None:
        self.domain_names = tuple(str(name) for name in domain_names)
        if not self.domain_names:
            raise ValueError("at least one domain is required")
        if len(set(self.domain_names)) != len(self.domain_names):
            raise ValueError(f"domain names must be unique, got {self.domain_names}")
        if step_size < 0:
            raise ValueError(f"step_size must be non-negative, got {step_size}")
        if not 0 <= smoothing <= 1:
            raise ValueError(f"smoothing must lie in [0, 1], got {smoothing}")

        self.step_size = float(step_size)
        self.smoothing = float(smoothing)
        self.weights = np.full(len(self.domain_names), 1.0 / len(self.domain_names))
        self._weight_sum = np.zeros_like(self.weights)
        self.num_steps = 0

    @staticmethod
    def _as_vector(values: Iterable[float], *, size: int, name: str) -> np.ndarray:
        vector = np.asarray(tuple(values), dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(f"{name} must contain {size} values, got shape {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains non-finite values: {vector}")
        return vector

    @property
    def averaged_weights(self) -> np.ndarray:
        if self.num_steps == 0:
            return self.weights.copy()
        return self._weight_sum / self.num_steps

    def update(
        self,
        proxy_losses: Iterable[float],
        reference_losses: Iterable[float],
        *,
        present: Iterable[bool] | None = None,
    ) -> DomainWeightStep:
        size = len(self.domain_names)
        proxy = self._as_vector(proxy_losses, size=size, name="proxy_losses")
        reference = self._as_vector(reference_losses, size=size, name="reference_losses")
        excess = np.maximum(proxy - reference, 0.0)
        if present is not None:
            present_vector = np.asarray(tuple(present), dtype=bool)
            if present_vector.shape != (size,):
                raise ValueError(f"present must contain {size} values, got {present_vector.shape}")
            excess = np.where(present_vector, excess, 0.0)

        # Work in log space so long runs or high-loss domains cannot overflow.
        logits = np.log(np.maximum(self.weights, np.finfo(np.float64).tiny))
        logits += self.step_size * excess
        logits -= np.max(logits)
        updated = np.exp(logits)
        updated /= updated.sum()
        uniform = np.full(size, 1.0 / size)
        self.weights = (1.0 - self.smoothing) * updated + self.smoothing * uniform

        self.num_steps += 1
        self._weight_sum += self.weights
        return DomainWeightStep(
            step=self.num_steps,
            excess_losses=tuple(float(value) for value in excess),
            weights=tuple(float(value) for value in self.weights),
            averaged_weights=tuple(float(value) for value in self.averaged_weights),
        )
