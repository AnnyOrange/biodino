#!/usr/bin/env python3
"""Construct label-free ISRD graphs, freeze hashes, then audit labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dinov3.eval.bio_frozen_eval.intervention_stable_relations import (
    ALL_VIEWS,
    build_relation_graphs,
    graph_label_precision,
    graph_metrics,
    graph_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--min-survival", type=float, default=0.50)
    parser.add_argument("--min-safe-cosine", type=float, default=0.85)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args()


def payload_hash(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        array = np.ascontiguousarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_graph.exists() or args.output_json.exists():
        raise FileExistsError("Refusing to overwrite preflight outputs")
    with np.load(args.bank, allow_pickle=False) as bank:
        view_names = tuple(np.asarray(bank["view_names"]).astype(str).tolist())
        if view_names != ALL_VIEWS:
            raise ValueError(f"Expected views {ALL_VIEWS}, got {view_names}")
        labels = np.asarray(bank["labels"])
        paths = np.asarray(bank["paths"]).astype(str)
        readout_arrays = {
            readout: np.asarray(bank[f"features_{readout}"])
            for readout in ("nlb2_avg", "nlb2_cls")
        }

    combined_payload: dict[str, np.ndarray] = {
        "paths": paths,
        "view_names": np.asarray(view_names),
        "n_samples": np.asarray(len(paths), dtype=np.int64),
    }
    report: dict[str, object] = {
        "bank": str(args.bank),
        "samples": len(paths),
        "labels_used_for_graph": False,
        "protocol": {
            "k": args.k,
            "max_neighbors": args.max_neighbors,
            "min_survival": args.min_survival,
            "min_safe_cosine": args.min_safe_cosine,
            "shuffle_seed": args.shuffle_seed,
            "construction_views": list(ALL_VIEWS[:-1]),
            "heldout_view": ALL_VIEWS[-1],
        },
        "readouts": {},
    }
    graph_objects = {}
    diagnostics_by_readout = {}
    for readout, features in readout_arrays.items():
        graphs, diagnostics = build_relation_graphs(
            features,
            view_names=view_names,
            k=args.k,
            max_neighbors=args.max_neighbors,
            min_survival=args.min_survival,
            min_safe_cosine=args.min_safe_cosine,
            shuffle_seed=args.shuffle_seed,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        graph_objects[readout] = graphs
        diagnostics_by_readout[readout] = diagnostics
        for name, array in graph_payload(graphs).items():
            combined_payload[f"{readout}_{name}"] = array

    graph_hash = payload_hash(combined_payload)
    combined_payload["graph_sha256"] = np.asarray(graph_hash)
    combined_payload["source_bank"] = np.asarray(str(args.bank))
    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_graph, **combined_payload)

    # Labels are accessed only after the graph artifact and its hash are fixed.
    for readout, graphs in graph_objects.items():
        diagnostics = diagnostics_by_readout[readout]
        heldout_codes = np.asarray(diagnostics["heldout_codes"])
        metrics = {
            name: {
                **graph_metrics(graph, heldout_codes=heldout_codes),
                "label_edge_precision_eval_only": graph_label_precision(graph, labels),
            }
            for name, graph in graphs.items()
        }
        view_medians = diagnostics["view_cosine_median"]
        unsafe = diagnostics["view_unsafe_fraction"]
        safe_views = all(
            0.85 <= float(view_medians[name]) <= 0.9995 and float(unsafe[name]) <= 0.05
            for name in ALL_VIEWS[1:-1]
        )
        isrd_metrics = metrics["isrd"]
        gates = {
            "interventions_meaningful_and_safe": safe_views,
            "coverage": isrd_metrics["coverage"] >= 0.30 and isrd_metrics["mean_outdegree"] >= 1.0,
            "no_hub_collapse": (
                isrd_metrics["max_indegree_fraction"] <= 0.01
                and isrd_metrics["effective_destination_fraction"] >= 0.20
            ),
            "heldout_vs_single": (
                isrd_metrics["heldout_survival"] - metrics["single"]["heldout_survival"] >= 0.05
            ),
            "heldout_vs_mean": (
                isrd_metrics["heldout_survival"] - metrics["mean"]["heldout_survival"] >= 0.02
            ),
            "heldout_vs_view_shuffled": (
                isrd_metrics["heldout_survival"]
                - metrics["view_shuffled"]["heldout_survival"]
                >= 0.10
            ),
            "label_noninferiority_eval_only": all(
                metrics["isrd"]["label_edge_precision_eval_only"]
                >= metrics[control]["label_edge_precision_eval_only"] - 0.005
                for control in ("single", "mean")
            ),
        }
        gates["unlabeled_pass"] = all(
            value for name, value in gates.items() if name != "label_noninferiority_eval_only"
        )
        report["readouts"][readout] = {
            "view_cosine_median": view_medians,
            "view_unsafe_fraction": unsafe,
            "graphs": metrics,
            "gates": gates,
        }
    report["graph_file"] = str(args.output_graph)
    report["graph_sha256"] = graph_hash
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
