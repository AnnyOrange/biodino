"""ACROBAT 2022 native WSI registration and challenge-landmark export.

The public source-point file contains no H&E targets. It permits genuine
registration predictions/submission preparation, NOT local accuracy claims.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import cv2

from .core import apply_affine
from .core import register_descriptors


@dataclass(frozen=True)
class AcrobatPair:
    case_id: str
    source_image: Path
    target_image: Path
    point_ids: tuple
    source_points_xy_um: np.ndarray
    source_mpp: float
    target_mpp: float


def acrobat_pairs(root):
    root = Path(root)
    annotation_path = root / "metadata/acrobat_validation_points_public_1_of_1.csv"
    with annotation_path.open(newline="") as handle:
        annotations = list(csv.DictReader(handle))
    grouped = {}
    for row in annotations:
        grouped.setdefault(row["anon_id"], []).append(row)
    pairs = []
    for case, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        source = root / "extracted/valid" / Path(rows[0]["anon_filename_ihc"]).with_suffix(".tif").name
        target = root / "extracted/valid" / Path(rows[0]["anon_filename_he"]).with_suffix(".tif").name
        point_ids = tuple(row["point_id"] for row in rows)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError(f"Duplicate ACROBAT point IDs: {case}")
        # Public CSV ihc_x/y are native 10X PIXELS; challenge docker files and
        # submission predictions instead use MICROMETERS. Convert once here.
        mpp_s, mpp_t = float(rows[0]["mpp_ihc_10X"]), float(rows[0]["mpp_he_10X"])
        if mpp_s <= 0 or mpp_t <= 0:
            raise ValueError("Positive native WSI microns-per-pixel required")
        coordinates = np.asarray([[float(row["ihc_x"]), float(row["ihc_y"])] for row in rows]) * mpp_s
        pairs.append(AcrobatPair(case, source, target, point_ids, coordinates, mpp_s, mpp_t))
    return pairs


def read_wsi_overview(path, max_side=2048):
    """Read a TIFF pyramid level, retaining native level-0 coordinate scale."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        # ACROBAT stores reductions as separate TIFF pages, not necessarily
        # SubIFDs. tifffile.series[0].levels can misleadingly contain only level0.
        levels = list(series.levels) + [page for page in handle.pages if page.is_reduced]
        def hw(level):
            return level.shape[level.axes.index("Y")], level.shape[level.axes.index("X")]
        native_hw = hw(levels[0])
        eligible = [level for level in levels if max(hw(level)) <= max_side]
        if not eligible:
            raise ValueError("No bounded-size WSI overview available; refuse level-0 decode")
        level = max(eligible, key=lambda level: max(hw(level)))
        pixels = level.asarray()
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
        raise ValueError("Expected RGB WSI overview")
    return pixels[..., :3], native_hw


def export_registered_landmarks(pair, source_to_target_um, output):
    """Native submission output x_target/y_target in micrometers."""
    points = apply_affine(pair.source_points_xy_um, source_to_target_um)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_target", "y_target"])
        writer.writerows(points.tolist())
    return points


def acrobat_pair_score(predicted_xy_um, target_annotators_xy_um):
    """Official 2022 pair scoring: average annotator distances, then 90th pct."""
    prediction = np.asarray(predicted_xy_um, dtype=float)
    annotations = np.asarray(target_annotators_xy_um, dtype=float)
    if prediction.ndim != 2 or prediction.shape[1] != 2 or annotations.ndim != 3 or annotations.shape[1:] != prediction.shape:
        raise ValueError("Expected predictions (N,2), target annotators (A,N,2)")
    if annotations.shape[0] < 2 or not np.isfinite(prediction).all() or not np.isfinite(annotations).all():
        raise ValueError("At least two complete finite target annotations required")
    distances = np.linalg.norm(annotations - prediction[None], axis=-1).mean(axis=0)
    return float(np.percentile(distances, 90))


def acrobat_2022_aggregate(pair_scores):
    scores = np.asarray(pair_scores, dtype=float)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("Complete finite pair scores required")
    return float(np.median(scores))


def acrobat_preflight(root):
    pairs = acrobat_pairs(root)
    failures = []
    native_sizes = []
    for pair in pairs:
        for image in (pair.source_image, pair.target_image):
            try:
                with tifffile.TiffFile(image) as handle:
                    series = handle.series[0]
                    native_sizes.append({"image": str(image), "shape": list(series.shape), "axes": series.axes,
                                         "pyramid_levels": 1 + sum(page.is_reduced for page in handle.pages),
                                         "tiff_series_levels": len(series.levels)})
            except (OSError, ValueError) as error:
                failures.append({"image": str(image), "error": str(error)})
    return {"pairs": len(pairs), "patient_cases": len({p.case_id for p in pairs}),
            "source_landmarks": sum(len(p.point_ids) for p in pairs),
            "source_unit": "CSV native10Xpixels converted to micrometers",
            "target_ground_truth": "HIDDEN_CHALLENGE_SERVER", "local_accuracy_scorable": False,
            "native_images": native_sizes, "failures": failures, "success": not failures}


def acrobat_cpu_smoke(root, output, case_index=0):
    """Real native WSI registration/submission smoke, with no fabricated score."""
    pair = acrobat_pairs(root)[case_index]
    descriptors, coordinates = [], []
    for path, mpp in [(pair.source_image, pair.source_mpp), (pair.target_image, pair.target_mpp)]:
        image, native_hw = read_wsi_overview(path, max_side=2048)
        keypoints, features = cv2.SIFT_create(nfeatures=2000).detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        if features is None:
            features = np.empty((0, 128), np.float32)
        xy = np.asarray([point.pt for point in keypoints], dtype=float).reshape(-1, 2)
        xy *= np.asarray([native_hw[1] / image.shape[1], native_hw[0] / image.shape[0]]) * mpp
        descriptors.append(features)
        coordinates.append(xy)
    result = register_descriptors(*descriptors, *coordinates, threshold=200., seed=0)
    points = export_registered_landmarks(pair, result.matrix, output)
    return {"case": pair.case_id, "success": result.success, "reason": result.reason,
        "matches": result.matches, "inliers": result.inliers, "matrix": result.matrix.tolist(),
        "submission_landmarks": len(points), "all_predictions_finite": bool(np.isfinite(points).all()),
        "output": str(output), "coordinate_unit": "micrometers",
        "model": "OpenCV-SIFT-CPU-SMOKE-NOT-HS6", "accuracy_metric": None,
        "accuracy_unavailable_reason": "Official target landmarks private on challenge server"}
