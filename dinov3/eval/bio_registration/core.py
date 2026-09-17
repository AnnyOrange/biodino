"""Landmark registration with existing OpenCV robust estimators.

Source coordinates always map to target coordinates. Coordinates are XY for
images, XYZ in micrometers for calibrated volumes; never array-order ZYX.
Ground-truth landmarks are scoring inputs only, never matching/fitting inputs.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class RegistrationResult:
    matrix: np.ndarray
    success: bool
    matches: int
    inliers: int
    reason: str = ""


def _points(points, dim=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Coordinates must have shape (N,2) or (N,3)")
    if dim is not None and points.shape[1] != dim:
        raise ValueError("Coordinate dimensions disagree")
    if not np.isfinite(points).all():
        raise ValueError("Coordinates contain NaN/Inf")
    return points


def patch_coordinates(grid_hw, original_hw, processed_hw=None, offset_xy=(0, 0)):
    """Original-image XY patch centers for a resize WITHOUT padding/cropping.

    For crops pass original_hw as the crop dimensions and its original-image
    offset. Padding/letterboxing must be removed explicitly by the caller.
    """
    gh, gw = map(int, grid_hw)
    oh, ow = map(float, original_hw)
    if min(gh, gw, oh, ow) <= 0:
        raise ValueError("Grid/image dimensions must be positive")
    if processed_hw is not None and min(processed_hw) <= 0:
        raise ValueError("Processed dimensions must be positive")
    yy, xx = np.meshgrid((np.arange(gh) + .5) * oh / gh,
                         (np.arange(gw) + .5) * ow / gw, indexing="ij")
    return np.stack([xx, yy], -1).reshape(-1, 2) + np.asarray(offset_xy)


def voxel_coordinates(indices_zyx, spacing_xyz_um, origin_xyz_um=(0, 0, 0)):
    """Convert sampled volume voxel centers to physical XYZ micrometers."""
    indices = _points(indices_zyx, 3)
    spacing = np.asarray(spacing_xyz_um, dtype=float)
    origin = np.asarray(origin_xyz_um, dtype=float)
    if spacing.shape != (3,) or origin.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Three explicit positive XYZ spacings in micrometers required")
    if not np.isfinite(origin).all():
        raise ValueError("Invalid volume origin")
    return (indices[:, ::-1] + .5) * spacing + origin


def match_descriptors(source, target, ratio=.9, mutual=True, backend="scipy", device="cpu"):
    """L2-normalized descriptor NN matching, using scipy's exact KD tree.

    Ratio is the Euclidean Lowe ratio, NOT cosine-distance ratio. Invalid/zero
    descriptors are excluded and returned indices refer to the original arrays.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Descriptor arrays must be (N,D) with equal D")
    if not 0 < ratio <= 1:
        raise ValueError("Ratio must be in (0,1]")
    sn, tn = np.linalg.norm(source, axis=1), np.linalg.norm(target, axis=1)
    si = np.flatnonzero(np.isfinite(source).all(1) & (sn > 0))
    ti = np.flatnonzero(np.isfinite(target).all(1) & (tn > 0))
    if not len(si) or len(ti) < 2:
        return np.empty((0, 2), dtype=np.int64)
    s, t = source[si] / sn[si, None], target[ti] / tn[ti, None]
    if backend == "torch":
        import torch
        source_tensor = torch.as_tensor(s, dtype=torch.float32, device=device)
        target_tensor = torch.as_tensor(t, dtype=torch.float32, device=device)
        # Exact normalized Euclidean distances, evaluated by blocked GEMM.
        # This avoids high-dimensional KD-tree scans for 4-even descriptors.
        best_distances, best_indices = [], []
        reverse_distances = torch.full((len(t),), float("inf"), device=device)
        reverse_indices = torch.zeros(len(t), dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for start in range(0, len(s), 256):
                distances_squared = (2. - 2. * source_tensor[start:start + 256] @ target_tensor.T).clamp_min(0)
                values, idx = distances_squared.topk(2, largest=False, dim=1)
                best_distances.append(values.sqrt().cpu().numpy())
                best_indices.append(idx.cpu().numpy())
                block_values, block_indices = distances_squared.min(0)
                improve = block_values < reverse_distances
                reverse_distances[improve] = block_values[improve]
                reverse_indices[improve] = block_indices[improve] + start
        distances, indices = np.concatenate(best_distances), np.concatenate(best_indices)
        reverse = reverse_indices.cpu().numpy()
    elif backend == "scipy":
        distances, indices = cKDTree(t).query(s, k=2, workers=1)
        reverse = cKDTree(s).query(t, k=1, workers=1)[1] if mutual else None
    else:
        raise ValueError("Matching backend must be scipy or torch")
    keep = (distances[:, 0] < ratio * distances[:, 1]) & (distances[:, 1] > 0)
    if mutual:
        keep &= reverse[indices[:, 0]] == np.arange(len(s))
    return np.column_stack([si[keep], ti[indices[keep, 0]]])


def apply_affine(points, matrix):
    points = _points(points)
    matrix = np.asarray(matrix, dtype=float)
    dim = points.shape[1]
    if matrix.shape != (dim, dim + 1) or not np.isfinite(matrix).all():
        raise ValueError("Invalid affine matrix")
    return points @ matrix[:, :dim].T + matrix[:, dim]


def fit_correspondences(source_xy, target_xy, threshold=16., seed=0, confidence=.99,
                        max_iterations=2000):
    """Estimate genuine 2D/3D affine transform; identity on failed fitting.

    threshold is in TARGET coordinate units (pixels for 2D, um for 3D).
    OpenCV's 3D RANSAC implementation does not expose a max-iterations option.
    """
    source = _points(source_xy)
    target = _points(target_xy, source.shape[1])
    if source.shape != target.shape:
        raise ValueError("Matched coordinate counts disagree")
    if threshold <= 0 or not 0 < confidence < 1 or max_iterations < 1:
        raise ValueError("Invalid RANSAC configuration")
    dim, count = source.shape[1], len(source)
    identity = np.eye(dim, dim + 1)
    if count < dim + 1:
        return RegistrationResult(identity, False, count, 0, "insufficient_matches")
    if np.linalg.matrix_rank(source - source.mean(0)) < dim:
        return RegistrationResult(identity, False, count, 0, "degenerate_source_geometry")
    cv2.setRNGSeed(int(seed))
    if dim == 2:
        matrix, inliers = cv2.estimateAffine2D(source, target, method=cv2.RANSAC,
            ransacReprojThreshold=float(threshold), maxIters=int(max_iterations),
            confidence=float(confidence), refineIters=10)
    else:
        success, matrix, inliers = cv2.estimateAffine3D(source, target,
            ransacThreshold=float(threshold), confidence=float(confidence))
        if not success:
            matrix = None
    n_inliers = int(np.asarray(inliers).sum()) if inliers is not None else 0
    if matrix is None or not np.isfinite(matrix).all() or n_inliers < dim + 1:
        return RegistrationResult(identity, False, count, n_inliers, "ransac_failed")
    if np.linalg.matrix_rank(matrix[:, :dim]) < dim:
        return RegistrationResult(identity, False, count, n_inliers, "singular_transform")
    return RegistrationResult(np.asarray(matrix), True, count, n_inliers)


def register_descriptors(source_features, target_features, source_coords, target_coords,
                         ratio=.9, mutual=True, matching_backend="scipy", device="cpu", **ransac_options):
    source, target = _points(source_coords), _points(target_coords)
    if len(source) != len(source_features) or len(target) != len(target_features):
        raise ValueError("Feature/coordinate counts disagree")
    matches = match_descriptors(source_features, target_features, ratio, mutual, matching_backend, device)
    return fit_correspondences(source[matches[:, 0]], target[matches[:, 1]], **ransac_options)


def registration_metrics(source_landmarks, target_landmarks, warped_landmarks,
                         target_diagonal, coordinate_unit="pixels"):
    """BIRL TRE/rTRE and robustness, retaining EVERY correspondence.

    Relative TRE divides by target-image diagonal, never source/resized-image
    diagonal. ANHIR ranking across methods must be computed separately; mean
    median rTRE here is a standalone score, NOT official mean rank.
    """
    target = _points(target_landmarks)
    source = _points(source_landmarks, target.shape[1])
    warped = _points(warped_landmarks, target.shape[1])
    if not len(target) or source.shape != target.shape or warped.shape != target.shape:
        raise ValueError("Scoring requires all matched landmark IDs, no truncation")
    if not np.isfinite(target_diagonal) or target_diagonal <= 0:
        raise ValueError("Positive target diagonal required")
    errors = np.linalg.norm(warped - target, axis=1)
    initial = np.linalg.norm(source - target, axis=1)
    return {"landmarks": len(target), "coordinate_unit": coordinate_unit,
            "median_tre": float(np.median(errors)), "mean_tre": float(errors.mean()),
            "p90_tre": float(np.percentile(errors, 90)),
            "median_rtre": float(np.median(errors) / target_diagonal),
            "mean_rtre": float(errors.mean() / target_diagonal),
            "robustness": float((errors < initial).mean())}
