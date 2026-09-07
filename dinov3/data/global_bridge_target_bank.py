"""CPU lookup for full-bank cross-domain bridge targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class GlobalBridgeTargetBatch:
    sample_indices: Tensor
    anchor_features: Tensor
    target_features: Tensor

    @property
    def size(self) -> int:
        return int(self.sample_indices.numel())


class GlobalBridgeTargetBank:
    def __init__(self, path: str | Path, *, control: bool = False) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Global bridge target bank does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as payload:
            target_name = "control_target_features" if control else "target_features"
            required = {"keys", "anchor_features", target_name}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"Bridge target bank is missing arrays: {sorted(missing)}")
            self.keys = np.asarray(payload["keys"]).astype(str)
            self.anchor_features = np.asarray(payload["anchor_features"], dtype=np.float16)
            self.target_features = np.asarray(payload[target_name], dtype=np.float16)
            self.feature_protocol = (
                str(np.asarray(payload["feature_protocol"]).item()).lower()
                if "feature_protocol" in payload.files
                else None
            )
            self.input_normalization = (
                str(np.asarray(payload["input_normalization"]).item()).lower()
                if "input_normalization" in payload.files
                else None
            )
            self.observation_crop_size = (
                int(np.asarray(payload["observation_crop_size"]).item())
                if "observation_crop_size" in payload.files
                else None
            )
        expected = (len(self.keys),)
        if self.anchor_features.ndim != 2 or self.target_features.shape != self.anchor_features.shape:
            raise ValueError("Bridge anchor/target features must share shape [samples, dimensions]")
        if self.anchor_features.shape[0] != expected[0]:
            raise ValueError("Bridge keys and feature rows differ")
        if len(set(self.keys.tolist())) != len(self.keys):
            raise ValueError("Bridge target bank contains duplicate keys")
        if not np.isfinite(self.anchor_features).all() or not np.isfinite(self.target_features).all():
            raise ValueError("Bridge target bank contains non-finite features")
        self.key_to_row = {key: row for row, key in enumerate(self.keys)}
        self.control = bool(control)

    @property
    def feature_dim(self) -> int:
        return int(self.anchor_features.shape[1])

    def lookup(
        self,
        sample_keys: Sequence[str],
        *,
        device: torch.device | str,
    ) -> GlobalBridgeTargetBatch:
        sample_indices = []
        rows = []
        for index, key_value in enumerate(sample_keys):
            row = self.key_to_row.get(str(key_value))
            if row is not None:
                sample_indices.append(index)
                rows.append(row)
        indices = torch.tensor(sample_indices, device=device, dtype=torch.long)
        if not rows:
            empty = torch.empty((0, self.anchor_features.shape[1]), device=device)
            return GlobalBridgeTargetBatch(indices, empty, empty.clone())
        anchor = torch.from_numpy(np.asarray(self.anchor_features[rows])).to(device=device)
        target = torch.from_numpy(np.asarray(self.target_features[rows])).to(device=device)
        return GlobalBridgeTargetBatch(indices, anchor, target)
