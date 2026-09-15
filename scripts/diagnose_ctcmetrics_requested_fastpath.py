#!/usr/bin/env python3
"""Score requested CTC metrics while skipping unused merged-track products."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fast-edge-mapping", action="store_true")
    return parser.parse_args()


def create_edge_mapping_indexed(
    tracks: np.ndarray,
    labels: list,
    vertex_true_positive: np.ndarray,
    cumulative_indices: np.ndarray,
) -> np.ndarray:
    """Return the pinned implementation's edge table using indexed lookups."""
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


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    from ctc_metrics.metrics import det, seg, tra
    from ctc_metrics.scripts.evaluate import load_data
    from ctc_metrics.utils import representations

    if args.fast_edge_mapping:
        representations.create_edge_mapping = create_edge_mapping_indexed

    started = time.perf_counter()
    print(f"[fastpath] load_data threads={args.threads}", flush=True)
    comp_tracks, ref_tracks, trajectory, segmentation, _, is_valid = load_data(
        args.result_dir,
        args.ground_truth_dir,
        trajectory_data=True,
        segmentation_data=True,
        threads=args.threads,
    )
    loaded = time.perf_counter()
    print(f"[fastpath] load_data seconds={loaded - started:.3f}", flush=True)
    graph_operations = representations.count_acyclic_graph_correction_operations(
        ref_tracks,
        comp_tracks,
        trajectory["labels_ref"],
        trajectory["labels_comp"],
        trajectory["mapped_ref"],
        trajectory["mapped_comp"],
    )
    graphed = time.perf_counter()
    print(f"[fastpath] graph_operations seconds={graphed - loaded:.3f}", flush=True)
    tra_value, aogm, aogm0 = tra(**graph_operations)
    result = {
        "Valid": int(is_valid),
        "DET": scalar(det(**graph_operations)),
        "SEG": scalar(seg(segmentation["labels_ref"], segmentation["ious"])),
        "TRA": scalar(tra_value),
        "AOGM": scalar(aogm),
        "AOGM_0": scalar(aogm0),
        "seconds": time.perf_counter() - started,
        "load_data_seconds": loaded - started,
        "graph_operations_seconds": graphed - loaded,
        "threads": args.threads,
        "indexed_edge_mapping": args.fast_edge_mapping,
        "requested_metric_path": "pinned py-ctcmetrics primitives; unused merge_tracks skipped",
    }
    for key in ("NS", "FN", "FP", "ED", "EA", "EC"):
        result[f"AOGM_{key}"] = scalar(graph_operations[key])
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
