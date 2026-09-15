"""Exact requested-only CTC scoring for large fragmented tracking outputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def create_edge_mapping_indexed(
    tracks: np.ndarray,
    labels: list,
    vertex_true_positive: np.ndarray,
    cumulative_indices: np.ndarray,
) -> np.ndarray:
    """Return py-ctcmetrics' edge table using indexed parent lookups."""
    if tracks is None:
        return np.zeros((0, 9))

    all_edges = []
    vertex_offset = 0
    current_frame = 0
    for labels_first, labels_second in zip(labels[:-1], labels[1:]):
        labels_first = np.asarray(labels_first)
        labels_second = np.asarray(labels_second)
        mapping = labels_first[:, None] == labels_second[None, :]
        index_first, index_second = np.where(mapping)
        id_first = labels_first[index_first]
        id_second = labels_second[index_second]
        frame_first = np.ones_like(id_first) * current_frame
        frame_second = frame_first + 1
        index_first = index_first + vertex_offset
        index_second = index_second + vertex_offset + len(labels_first)
        all_edges.append(
            np.stack(
                [
                    index_first,
                    id_first,
                    vertex_true_positive[index_first],
                    frame_first,
                    index_second,
                    id_second,
                    vertex_true_positive[index_second],
                    frame_second,
                    np.zeros_like(id_first),
                ],
                axis=1,
            )
        )
        vertex_offset += len(labels_first)
        current_frame += 1

    tracks_by_id = {int(track[0]): track for track in tracks}
    positions_by_frame = [
        {int(label): index for index, label in enumerate(frame_labels)}
        for frame_labels in labels
    ]
    for track in tracks:
        label_second, birth_second, _, parent_second = (int(value) for value in track)
        if parent_second == 0:
            continue
        label_first, _, end_first, _ = (
            int(value) for value in tracks_by_id[parent_second]
        )
        index_first = positions_by_frame[end_first][label_first] + int(cumulative_indices[end_first])
        index_second = (
            positions_by_frame[birth_second][label_second]
            + int(cumulative_indices[birth_second])
        )
        all_edges.append(
            np.asarray(
                [
                    index_first,
                    label_first,
                    int(vertex_true_positive[index_first]),
                    end_first,
                    index_second,
                    label_second,
                    int(vertex_true_positive[index_second]),
                    birth_second,
                    1,
                ]
            )[None, :]
        )
    return np.concatenate(all_edges, axis=0).astype(int)


def calculate_requested_metrics(
    comp_tracks: np.ndarray,
    ref_tracks: np.ndarray,
    trajectory: dict[str, Any],
    segmentation: dict[str, Any],
    is_valid: bool,
) -> tuple[dict[str, Any], float]:
    """Calculate Valid/DET/SEG/TRA from the pinned package primitives."""
    from ctc_metrics.metrics import det, seg, tra
    from ctc_metrics.utils import representations

    if not is_valid:
        return {"Valid": 0, "DET": None, "SEG": None, "TRA": None}, 0.0

    started = time.perf_counter()
    original_edge_mapping = representations.create_edge_mapping
    try:
        representations.create_edge_mapping = create_edge_mapping_indexed
        operations = representations.count_acyclic_graph_correction_operations(
            ref_tracks,
            comp_tracks,
            trajectory["labels_ref"],
            trajectory["labels_comp"],
            trajectory["mapped_ref"],
            trajectory["mapped_comp"],
        )
    finally:
        representations.create_edge_mapping = original_edge_mapping

    tra_value, aogm, aogm0 = tra(**operations)
    metrics = {
        "Valid": 1,
        "DET": _scalar(det(**operations)),
        "SEG": _scalar(seg(segmentation["labels_ref"], segmentation["ious"])),
        "TRA": _scalar(tra_value),
        "AOGM": _scalar(aogm),
        "AOGM_0": _scalar(aogm0),
    }
    for key in ("NS", "FN", "FP", "ED", "EA", "EC"):
        metrics[f"AOGM_{key}"] = _scalar(operations[key])
    return metrics, time.perf_counter() - started


def score_requested_metrics(
    result_dir: Path,
    ground_truth_dir: Path,
    vendor: Path,
    *,
    threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one sequence and score only the four protocol-requested metrics."""
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from ctc_metrics.scripts.evaluate import load_data

    started = time.perf_counter()
    comp_tracks, ref_tracks, trajectory, segmentation, _, is_valid = load_data(
        str(result_dir),
        str(ground_truth_dir),
        trajectory_data=True,
        segmentation_data=True,
        threads=threads,
    )
    loaded = time.perf_counter()
    metrics, graph_seconds = calculate_requested_metrics(
        comp_tracks,
        ref_tracks,
        trajectory,
        segmentation,
        is_valid,
    )
    diagnostics = {
        "implementation": "pinned_primitives_requested_only_indexed_parent_lookup_v1",
        "threads": threads,
        "load_data_seconds": loaded - started,
        "graph_operations_seconds": graph_seconds,
        "seconds": time.perf_counter() - started,
        "unused_merged_track_products_computed": False,
    }
    return metrics, diagnostics
