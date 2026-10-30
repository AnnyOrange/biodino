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
        positions = {int(label): i for i, label in enumerate(labels_second)}
        index_first = np.asarray([i for i, label in enumerate(labels_first)
                                  if int(label) in positions], dtype=np.int64)
        index_second = np.asarray([positions[int(labels_first[i])] for i in index_first], dtype=np.int64)
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


def count_graph_operations_indexed(ref_tracks, comp_tracks, labels_ref, labels_comp, mapped_ref, mapped_comp):
    """Pinned AOGM operations without sparse full-row zeroing's dense allocation.

    Vertex membership is identical to the pinned binary detection matrix. Edges
    are matched by endpoint pairs, retaining the exact original semantics.
    """
    cumulative_r = np.cumsum([0] + [len(x) for x in labels_ref])
    cumulative_c = np.cumsum([0] + [len(x) for x in labels_comp])
    nr, nc = int(cumulative_r[-1]), int(cumulative_c[-1])
    pairs = set()
    for t, (lr, lc, mr, mc) in enumerate(zip(labels_ref, labels_comp, mapped_ref, mapped_comp)):
        rp = {int(v): i + int(cumulative_r[t]) for i, v in enumerate(lr)}
        cp = {int(v): i + int(cumulative_c[t]) for i, v in enumerate(lc)}
        pairs.update((cp[int(c)], rp[int(r)]) for r, c in zip(mr, mc))
    pairs = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    ac = np.bincount(pairs[:, 0], minlength=nc)
    ar = np.bincount(pairs[:, 1], minlength=nr)
    if np.any(ar > 1):
        raise ValueError('A reference vertex cannot map to multiple computed vertices')
    tp_r, tp_c = ar == 1, ac == 1
    stats = dict(num_vertices_R=nr, num_vertices_C=nc, num_vertices=nr,
        TP=int(tp_r.sum()), FN=int((~tp_r).sum()), FP=int((ac == 0).sum()), VS=int((ac > 1).sum()))
    stats['NS'] = stats['TP'] - int(tp_c.sum()) - stats['VS']
    comp_to_ref = np.full(nc, -1, dtype=np.int64)
    unique = pairs[tp_c[pairs[:, 0]]]
    comp_to_ref[unique[:, 0]] = unique[:, 1]
    er = create_edge_mapping_indexed(ref_tracks, labels_ref, tp_r, cumulative_r)
    ec = create_edge_mapping_indexed(comp_tracks, labels_comp, tp_c, cumulative_c)
    if not len(ec):
        stats.update(ED=0, EA=len(er), EC=0, num_edges=len(er))
        return stats
    ec = ec[(ec[:, 2] * ec[:, 6]) == 1]
    ref_edges = {(int(e[0]), int(e[4])): int(e[8]) for e in er}
    comp_edges = {(int(comp_to_ref[e[0]]), int(comp_to_ref[e[4]])): int(e[8]) for e in ec}
    if len(ref_edges) != len(er) or len(comp_edges) != len(ec):
        raise ValueError('Duplicate endpoint pairs violate pinned graph assumptions')
    shared = ref_edges.keys() & comp_edges.keys()
    stats.update(ED=len(comp_edges.keys() - ref_edges.keys()), EA=len(ref_edges.keys() - comp_edges.keys()),
        EC=sum(ref_edges[k] != comp_edges[k] for k in shared), num_edges=len(er))
    return stats


def calculate_requested_metrics(
    comp_tracks: np.ndarray,
    ref_tracks: np.ndarray,
    trajectory: dict[str, Any],
    segmentation: dict[str, Any],
    is_valid: bool,
) -> tuple[dict[str, Any], float]:
    """Calculate Valid/DET/SEG/TRA from the pinned package primitives."""
    from ctc_metrics.metrics import det, seg, tra

    if not is_valid:
        return {"Valid": 0, "DET": None, "SEG": None, "TRA": None}, 0.0

    started = time.perf_counter()
    operations = count_graph_operations_indexed(
        ref_tracks, comp_tracks, trajectory['labels_ref'], trajectory['labels_comp'],
        trajectory['mapped_ref'], trajectory['mapped_comp'],
    )

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
    from importlib import import_module
    from .ctc_mask_matching import match_paths
    from .ctc_scoring_resources import scoring_budget
    evaluator = import_module('ctc_metrics.scripts.evaluate')

    started = time.perf_counter()
    requested_threads = threads
    with scoring_budget():
        original_match = evaluator.match_tracks
        try:
            evaluator.match_tracks = match_paths
            # This evaluator is called synchronously by each queue child.
            # Keep vendor parsing/validation, but never create its 16-worker pool.
            comp_tracks, ref_tracks, trajectory, segmentation, _, is_valid = evaluator.load_data(
                str(result_dir), str(ground_truth_dir), trajectory_data=True,
                segmentation_data=True, threads=1)
        finally:
            evaluator.match_tracks = original_match
        loaded = time.perf_counter()
        metrics, graph_seconds = calculate_requested_metrics(
            comp_tracks, ref_tracks, trajectory, segmentation, is_valid)
    diagnostics = {
        "implementation": "pinned_primitives_requested_only_sparse_masks_indexed_graph_v3",
        "threads": 1,
        "requested_threads": requested_threads,
        "host_scoring_slots": 2,
        "load_data_seconds": loaded - started,
        "graph_operations_seconds": graph_seconds,
        "seconds": time.perf_counter() - started,
        "unused_merged_track_products_computed": False,
    }
    return metrics, diagnostics
