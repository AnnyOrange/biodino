#!/usr/bin/env python3
"""Propagate sparse provenance labels along calibrated source-ID intervals.

The legacy database assigned integer ``original_images_all.id`` values during
dataset ingestion.  Same-source images therefore often occupy local ID
intervals.  This script validates that structure with alternating, per-class
source-ID folds and only propagates labels inside class-specific distance
thresholds that meet a requested held-out precision.  Everything else remains
unresolved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scripts.calibrate_bioclip_semantic_routing import (
        canonical_acquisition,
        canonical_organism,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ first on sys.path.
    from calibrate_bioclip_semantic_routing import canonical_acquisition, canonical_organism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--min-class-cv-accepted", type=int, default=3)
    parser.add_argument("--max-distance", type=int, default=1_000_000)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def nearest_predictions(
    query_ids: np.ndarray,
    anchor_ids: np.ndarray,
    anchor_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_ids = np.asarray(query_ids, dtype=np.int64)
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    anchor_labels = np.asarray(anchor_labels).astype(str)
    if len(anchor_ids) == 0:
        return np.full(len(query_ids), ""), np.full(len(query_ids), np.iinfo(np.int64).max)
    order = np.argsort(anchor_ids)
    anchor_ids = anchor_ids[order]
    anchor_labels = anchor_labels[order]
    positions = np.searchsorted(anchor_ids, query_ids)
    left = np.clip(positions - 1, 0, len(anchor_ids) - 1)
    right = np.clip(positions, 0, len(anchor_ids) - 1)
    left_distance = np.abs(query_ids - anchor_ids[left])
    right_distance = np.abs(query_ids - anchor_ids[right])
    choose_left = left_distance <= right_distance
    chosen = np.where(choose_left, left, right)
    return anchor_labels[chosen], np.minimum(left_distance, right_distance)


def alternating_cv_predictions(
    source_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_ids = np.asarray(source_ids, dtype=np.int64)
    labels = np.asarray(labels).astype(str)
    fold = np.full(len(source_ids), -1, dtype=np.int8)
    for label in sorted(set(labels.tolist()) - {""}):
        indices = np.flatnonzero(labels == label)
        indices = indices[np.argsort(source_ids[indices])]
        if len(indices) < 2:
            continue
        fold[indices[::2]] = 0
        fold[indices[1::2]] = 1

    evaluated_indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for test_fold in (0, 1):
        test = np.flatnonzero(fold == test_fold)
        train = np.flatnonzero((fold >= 0) & (fold != test_fold))
        if not len(test) or not len(train):
            continue
        predicted, distance = nearest_predictions(
            source_ids[test], source_ids[train], labels[train]
        )
        evaluated_indices.append(test)
        predictions.append(predicted)
        distances.append(distance)
    if not evaluated_indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=str),
            np.empty(0, dtype=np.int64),
        )
    indices = np.concatenate(evaluated_indices)
    order = np.argsort(indices)
    return indices[order], np.concatenate(predictions)[order], np.concatenate(distances)[order]


def fit_class_distance_thresholds(
    truth: np.ndarray,
    predicted: np.ndarray,
    distances: np.ndarray,
    *,
    target_precision: float,
    min_accepted: int,
    max_distance: int,
) -> dict[str, int | None]:
    truth = np.asarray(truth).astype(str)
    predicted = np.asarray(predicted).astype(str)
    distances = np.asarray(distances, dtype=np.int64)
    thresholds: dict[str, int | None] = {}
    for label in sorted(set(predicted.tolist()) - {""}):
        rows = np.flatnonzero(predicted == label)
        candidates = sorted(set(int(value) for value in distances[rows] if value <= max_distance))
        best = None
        for threshold in candidates:
            accepted = rows[distances[rows] <= threshold]
            if len(accepted) < min_accepted:
                continue
            precision = float(np.mean(truth[accepted] == label))
            if precision >= target_precision:
                best = threshold
        thresholds[label] = best
    return thresholds


def threshold_acceptance(
    predicted: np.ndarray,
    distances: np.ndarray,
    thresholds: dict[str, int | None],
) -> np.ndarray:
    return np.asarray(
        [
            thresholds.get(str(label)) is not None
            and int(distance) <= int(thresholds[str(label)])
            for label, distance in zip(predicted, distances)
        ],
        dtype=np.bool_,
    )


def calibrate_task(
    name: str,
    source_ids: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    min_accepted: int,
    max_distance: int,
) -> tuple[dict[str, int | None], dict[str, object]]:
    cv_indices, cv_prediction, cv_distance = alternating_cv_predictions(source_ids, labels)
    cv_truth = labels[cv_indices]
    thresholds = fit_class_distance_thresholds(
        cv_truth,
        cv_prediction,
        cv_distance,
        target_precision=target_precision,
        min_accepted=min_accepted,
        max_distance=max_distance,
    )
    accepted = threshold_acceptance(cv_prediction, cv_distance, thresholds)
    selected = np.flatnonzero(accepted)
    precision = (
        float(np.mean(cv_prediction[selected] == cv_truth[selected])) if len(selected) else 0.0
    )
    per_class = {}
    for label in sorted(thresholds):
        rows = selected[cv_prediction[selected] == label]
        if len(rows):
            per_class[label] = {
                "accepted": int(len(rows)),
                "precision": float(np.mean(cv_truth[rows] == label)),
                "threshold": thresholds[label],
            }
    validated_classes = sum(value is not None for value in thresholds.values())
    report = {
        "name": name,
        "anchor_count": int(np.sum(labels != "")),
        "anchor_class_counts": dict(Counter(value for value in labels if value)),
        "cv_evaluated": int(len(cv_indices)),
        "cv_raw_accuracy": (
            float(np.mean(cv_prediction == cv_truth)) if len(cv_indices) else 0.0
        ),
        "cv_accepted": int(len(selected)),
        "cv_coverage": float(len(selected) / len(cv_indices)) if len(cv_indices) else 0.0,
        "cv_precision": precision,
        "validated_classes": int(validated_classes),
        "thresholds": thresholds,
        "per_predicted_class": per_class,
        "gate_pass": bool(
            len(selected) >= min_accepted
            and precision >= target_precision
            and validated_classes >= 2
        ),
    }
    return thresholds, report


def propagate_task(
    all_source_ids: np.ndarray,
    anchor_source_ids: np.ndarray,
    anchor_labels: np.ndarray,
    thresholds: dict[str, int | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known = anchor_labels != ""
    prediction, distance = nearest_predictions(
        all_source_ids, anchor_source_ids[known], anchor_labels[known]
    )
    accepted = threshold_acceptance(prediction, distance, thresholds)
    return np.where(accepted, prediction, ""), distance, accepted


def main() -> None:
    args = parse_args()
    if not 0 < args.target_precision <= 1:
        raise ValueError("--target-precision must be in (0, 1]")
    if args.min_class_cv_accepted <= 0 or args.max_distance <= 0:
        raise ValueError("Minimum accepted count and maximum distance must be positive")

    with np.load(args.truth_overlay, allow_pickle=False) as overlay:
        keys = np.asarray(overlay["keys"]).astype(str)
        sample_source_ids = np.asarray(overlay["source_id"], dtype=np.int64)
        recovered = np.asarray(overlay["recovered"], dtype=np.bool_)
        raw_domain = np.asarray(overlay["domain"]).astype(str)
        raw_organism = np.asarray(overlay["organism"]).astype(str)
        raw_acquisition = np.asarray(overlay["acquisition_family"]).astype(str)

    anchor_rows: dict[int, tuple[str, str, str]] = {}
    for row in np.flatnonzero(recovered):
        source_id = int(sample_source_ids[row])
        values = (
            raw_domain[row] if raw_domain[row] != "unresolved" else "",
            canonical_organism(raw_organism[row]),
            canonical_acquisition(raw_acquisition[row]),
        )
        previous = anchor_rows.get(source_id)
        if previous is not None and previous != values:
            raise ValueError(f"Conflicting recovered metadata for source ID {source_id}")
        anchor_rows[source_id] = values
    anchor_source_ids = np.asarray(sorted(anchor_rows), dtype=np.int64)
    anchor_domain = np.asarray([anchor_rows[int(value)][0] for value in anchor_source_ids])
    anchor_organism = np.asarray([anchor_rows[int(value)][1] for value in anchor_source_ids])
    anchor_acquisition = np.asarray([anchor_rows[int(value)][2] for value in anchor_source_ids])

    task_inputs = {
        "domain": anchor_domain,
        "organism": anchor_organism,
        "acquisition": anchor_acquisition,
    }
    thresholds = {}
    reports = {}
    for name, labels in task_inputs.items():
        thresholds[name], reports[name] = calibrate_task(
            name,
            anchor_source_ids,
            labels,
            target_precision=args.target_precision,
            min_accepted=args.min_class_cv_accepted,
            max_distance=args.max_distance,
        )

    unique_source_ids, sample_to_source = np.unique(sample_source_ids, return_inverse=True)
    propagated = {}
    distances = {}
    accepted = {}
    for name, labels in task_inputs.items():
        propagated[name], distances[name], accepted[name] = propagate_task(
            unique_source_ids,
            anchor_source_ids,
            labels,
            thresholds[name],
        )
    # Exact truth is retained even for rare classes that cannot calibrate a
    # propagation distance on their own.
    source_to_row = {int(value): index for index, value in enumerate(unique_source_ids)}
    for source_id, values in anchor_rows.items():
        row = source_to_row[source_id]
        for task_index, name in enumerate(("domain", "organism", "acquisition")):
            if values[task_index]:
                propagated[name][row] = values[task_index]
                distances[name][row] = 0
                accepted[name][row] = True

    sample_domain = propagated["domain"][sample_to_source]
    sample_organism = propagated["organism"][sample_to_source]
    sample_acquisition = propagated["acquisition"][sample_to_source]
    sample_type = np.where(
        np.isin(sample_acquisition, ["histopathology", "imaging_mass_cytometry"]),
        "tissue",
        np.where(sample_acquisition != "", "cell", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=keys,
        source_id=sample_source_ids,
        recovered=recovered,
        domain=sample_domain,
        organism=sample_organism,
        acquisition_family=sample_acquisition,
        sample_type=sample_type,
        domain_distance=distances["domain"][sample_to_source],
        organism_distance=distances["organism"][sample_to_source],
        acquisition_distance=distances["acquisition"][sample_to_source],
        domain_propagated=accepted["domain"][sample_to_source] & ~recovered,
        organism_propagated=accepted["organism"][sample_to_source] & ~recovered,
        acquisition_propagated=accepted["acquisition"][sample_to_source] & ~recovered,
    )

    report = {
        "truth_overlay": str(args.truth_overlay),
        "output": str(args.output),
        "samples": int(len(keys)),
        "source_images": int(len(unique_source_ids)),
        "recovered_source_images": int(len(anchor_source_ids)),
        "target_precision": args.target_precision,
        "min_class_cv_accepted": args.min_class_cv_accepted,
        "max_distance": args.max_distance,
        **reports,
        "routed_source_fraction": {
            name: float(np.mean(values != "")) for name, values in propagated.items()
        },
        "routed_sample_fraction": {
            "domain": float(np.mean(sample_domain != "")),
            "organism": float(np.mean(sample_organism != "")),
            "acquisition": float(np.mean(sample_acquisition != "")),
        },
        "routed_sample_organisms": dict(Counter(value for value in sample_organism if value)),
        "routed_sample_acquisitions": dict(
            Counter(value for value in sample_acquisition if value)
        ),
        "gates_pass": bool(reports["organism"]["gate_pass"] and reports["acquisition"]["gate_pass"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.enforce_gates and not report["gates_pass"]:
        raise SystemExit("OID interval metadata propagation failed one or more gates")


if __name__ == "__main__":
    main()
