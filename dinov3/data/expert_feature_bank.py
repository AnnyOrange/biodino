"""CPU lookup for offline expert features keyed by packed WebDataset sample."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ExpertFeatureBatch:
    sample_indices: Tensor
    features: tuple[Tensor, ...]
    weights: Tensor
    domains: tuple[str, ...]
    organisms: tuple[str, ...]
    acquisition_families: tuple[str, ...]
    sample_types: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.sample_indices.numel())


class _SingleExpertBank:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Expert feature bank does not exist: {self.path}")
        payload = np.load(self.path, allow_pickle=False)
        required = {"keys", "features"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Expert bank {self.path} is missing arrays: {sorted(missing)}")
        keys = np.asarray(payload["keys"]).astype(str)
        features = np.asarray(payload["features"])
        if keys.ndim != 1 or features.ndim != 2 or features.shape[0] != keys.shape[0]:
            raise ValueError(
                f"Expert bank {self.path} needs keys [N] and features [N,D], got "
                f"{keys.shape} and {features.shape}"
            )
        if len(set(keys.tolist())) != len(keys):
            raise ValueError(f"Expert bank {self.path} contains duplicate sample keys")
        reliability = (
            np.asarray(payload["reliability"], dtype=np.float32)
            if "reliability" in payload.files
            else np.ones(len(keys), dtype=np.float32)
        )
        if reliability.shape != (len(keys),):
            raise ValueError(
                f"Expert bank {self.path} reliability must have shape {(len(keys),)}, "
                f"got {reliability.shape}"
            )
        if not np.isfinite(features).all() or not np.isfinite(reliability).all():
            raise ValueError(f"Expert bank {self.path} contains non-finite values")
        if bool((reliability < 0).any()):
            raise ValueError(f"Expert bank {self.path} reliability must be non-negative")
        self.keys = keys
        self.features = features.astype(np.float16, copy=False)
        self.reliability = reliability
        self.key_to_row = {key: row for row, key in enumerate(keys)}
        self.metadata = {}
        for name in ("domain", "organism", "acquisition_family", "sample_type"):
            values = (
                np.asarray(payload[name]).astype(str)
                if name in payload.files
                else np.full(len(keys), "", dtype=str)
            )
            if values.shape != (len(keys),):
                raise ValueError(
                    f"Expert bank {self.path} {name} must have shape {(len(keys),)}, "
                    f"got {values.shape}"
                )
            self.metadata[name] = values


class ExpertFeatureBank:
    """Look up samples present in every expert bank without touching CUDA at init."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("At least one expert bank path is required")
        self.banks = tuple(_SingleExpertBank(path) for path in paths)

    def lookup(
        self,
        sample_keys: Sequence[str],
        *,
        device: torch.device | str,
    ) -> ExpertFeatureBatch:
        selected_indices = []
        rows_by_expert: list[list[int]] = [[] for _ in self.banks]
        for sample_index, key_value in enumerate(sample_keys):
            key = str(key_value)
            rows = [bank.key_to_row.get(key) for bank in self.banks]
            if any(row is None for row in rows):
                continue
            selected_indices.append(sample_index)
            for expert_index, row in enumerate(rows):
                rows_by_expert[expert_index].append(int(row))

        sample_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
        if not selected_indices:
            return ExpertFeatureBatch(
                sample_indices=sample_indices,
                features=tuple(),
                weights=torch.empty((len(self.banks), 0), device=device),
                domains=tuple(),
                organisms=tuple(),
                acquisition_families=tuple(),
                sample_types=tuple(),
            )

        features = tuple(
            torch.from_numpy(np.asarray(bank.features[rows])).to(device=device)
            for bank, rows in zip(self.banks, rows_by_expert)
        )
        weights = torch.stack(
            [
                torch.from_numpy(np.asarray(bank.reliability[rows])).to(
                    device=device, dtype=torch.float32
                )
                for bank, rows in zip(self.banks, rows_by_expert)
            ],
            dim=0,
        )
        metadata_by_name: dict[str, tuple[str, ...]] = {}
        for name in ("domain", "organism", "acquisition_family", "sample_type"):
            aligned_values = []
            for rows in zip(*rows_by_expert):
                values = [str(bank.metadata[name][row]).strip() for bank, row in zip(self.banks, rows)]
                known_values = {value for value in values if value}
                if len(known_values) > 1:
                    raise ValueError(
                        f"Expert banks disagree on {name} for a shared sample: {sorted(known_values)}"
                    )
                aligned_values.append(next(iter(known_values), ""))
            metadata_by_name[name] = tuple(aligned_values)

        return ExpertFeatureBatch(
            sample_indices=sample_indices,
            features=features,
            weights=weights,
            domains=metadata_by_name["domain"],
            organisms=metadata_by_name["organism"],
            acquisition_families=metadata_by_name["acquisition_family"],
            sample_types=metadata_by_name["sample_type"],
        )


_UNKNOWN_METADATA = frozenset({"", "unknown", "unresolved", "none", "nan", "n/a", "na"})


def stable_metadata_codes(values: Sequence[str], device: torch.device | str) -> Tensor:
    """Encode labels for tensor-only distributed gathering; zero means unknown."""
    codes = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized in _UNKNOWN_METADATA:
            codes.append(0)
            continue
        digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
        code = int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)
        codes.append(code or 1)
    return torch.tensor(codes, device=device, dtype=torch.long)


def _different_known_pairs(values: Sequence[str], device: torch.device | str) -> Tensor:
    normalized = [str(value).strip().lower() for value in values]
    known = torch.tensor(
        [value not in _UNKNOWN_METADATA for value in normalized],
        device=device,
        dtype=torch.bool,
    )
    label_to_index: dict[str, int] = {}
    encoded = []
    for value in normalized:
        if value not in label_to_index:
            label_to_index[value] = len(label_to_index)
        encoded.append(label_to_index[value])
    labels = torch.tensor(encoded, device=device, dtype=torch.long)
    return known[:, None] & known[None, :] & (labels[:, None] != labels[None, :])


def build_cross_domain_edge_mask(
    batch: ExpertFeatureBatch,
    *,
    scope: str,
    device: torch.device | str,
) -> Tensor:
    """Build auditable cross-species/acquisition candidate edges."""

    valid_scopes = {
        "all",
        "cross_organism",
        "cross_acquisition",
        "cross_organism_or_acquisition",
        "cross_organism_and_acquisition",
    }
    if scope not in valid_scopes:
        raise ValueError(f"Unknown expert-consensus edge scope {scope!r}; expected {sorted(valid_scopes)}")
    size = batch.size
    if scope == "all":
        return torch.ones((size, size), device=device, dtype=torch.bool)
    cross_organism = _different_known_pairs(batch.organisms, device)
    cross_acquisition = _different_known_pairs(batch.acquisition_families, device)
    if scope == "cross_organism":
        return cross_organism
    if scope == "cross_acquisition":
        return cross_acquisition
    if scope == "cross_organism_or_acquisition":
        return cross_organism | cross_acquisition
    return cross_organism & cross_acquisition
