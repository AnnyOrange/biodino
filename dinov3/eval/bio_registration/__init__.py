"""Coordinate-aware registration, not image-vector retrieval."""

from .core import (RegistrationResult, apply_affine, fit_correspondences,
                   match_descriptors, patch_coordinates, register_descriptors,
                   registration_metrics, voxel_coordinates)
from .datasets import (RegistrationPair, anhir_pairs, cima_pairs,
                       grouped_partition, load_landmarks, preflight_pairs)

__all__ = ["RegistrationResult", "RegistrationPair", "apply_affine",
           "fit_correspondences", "match_descriptors", "patch_coordinates",
           "register_descriptors", "registration_metrics", "voxel_coordinates",
           "anhir_pairs", "cima_pairs", "grouped_partition", "load_landmarks",
           "preflight_pairs", "evaluate_registration"]


def __getattr__(name):
    if name == "evaluate_registration":
        from .api import evaluate_registration
        return evaluate_registration
    raise AttributeError(name)
