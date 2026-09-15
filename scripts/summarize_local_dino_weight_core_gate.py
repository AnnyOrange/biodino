#!/usr/bin/env python3
"""Summarize the preregistered non-dense local-DINO-weight gate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ARMS = ("w1", "w025", "w0")
EXPECTED_DATASETS = {
    "classification": {"bloodmnist", "bbbc048-cellcycle", "cyclops-protein-loc"},
    "regression": {"bbbc005"},
    "retrieval": {"hpa-subcellular", "rxrx1-cross"},
    "clustering": {"hpa-subcellular", "rxrx1-cross"},
}
ROW_SELECTORS = {
    ("retrieval", "hpa-subcellular"): {
        "task": "retrieval",
        "aggregation": "global",
        "protocol_contains": "same-gene-query-gallery",
        "metric": "recall_at_1",
    },
    ("clustering", "hpa-subcellular"): {
        "task": "clustering",
        "aggregation": "location",
        "protocol_contains": "single-location-ge10",
        "metric": "nmi",
    },
    ("retrieval", "rxrx1-cross"): {
        "task": "retrieval",
        "aggregation": "macro-cell-type",
        "protocol_contains": "official-cross-experiment-core",
        "metric": "recall_at_1",
    },
    ("clustering", "rxrx1-cross"): {
        "task": "clustering",
        "aggregation": "global-perturbation",
        "protocol_contains": "official-cross-experiment-core",
        "metric": "nmi",
    },
}
GATE = {
    "minimum_positive_families": 3,
    "minimum_winning_cells": 5,
    "maximum_allowed_family_mean_regression": -0.01,
}


def _payload_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if payload.get("error"):
        raise ValueError(f"Failed result at {path}: {payload['error']}")
    rows = payload.get("rows", [payload])
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Malformed result rows at {path}")
    dataset = payload.get("dataset")
    normalized = []
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Failed result row at {path}: {row['error']}")
        item = dict(row)
        if dataset and not item.get("dataset"):
            item["dataset"] = dataset
        normalized.append(item)
    return normalized


def _load_cells(root: Path) -> dict[str, dict[str, float]]:
    cells: dict[str, dict[str, float]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("last_result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_payload_rows(payload, path))

    for row in rows:
        dataset = str(row.get("dataset", ""))
        task = str(row.get("task", ""))
        if task == "classification" and dataset in EXPECTED_DATASETS["classification"]:
            if dataset in cells["classification"]:
                raise ValueError(f"Duplicate classification result for {dataset} at {root}")
            cells["classification"][dataset] = float(row["balanced_accuracy"])
        elif task == "regression" and dataset in EXPECTED_DATASETS["regression"]:
            if dataset in cells["regression"]:
                raise ValueError(f"Duplicate regression result for {dataset} at {root}")
            cells["regression"][dataset] = float(row["r2"])

    for (family, dataset), selector in ROW_SELECTORS.items():
        matches = [
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("task") == selector["task"]
            and row.get("aggregation") == selector["aggregation"]
            and selector["protocol_contains"] in str(row.get("protocol", ""))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {family}/{dataset} row at {root}, found {len(matches)}"
            )
        cells[family][dataset] = float(matches[0][selector["metric"]])
    missing = {
        family: sorted(datasets - set(cells.get(family, {})))
        for family, datasets in EXPECTED_DATASETS.items()
        if datasets - set(cells.get(family, {}))
    }
    if missing:
        raise ValueError(f"Incomplete core gate at {root}: {missing}")
    return dict(cells)


def _flat(cells: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{family}/{dataset}": value
        for family, datasets in cells.items()
        for dataset, value in datasets.items()
    }


def _bootstrap_family_macro(
    family_deltas: dict[str, list[float]], *, seed: int, reps: int = 10000
) -> dict[str, float]:
    # Each family contributes one mean, so classification cannot dominate merely
    # because the sentinel contains more classification datasets.
    values = torch.tensor([statistics.fmean(items) for items in family_deltas.values()])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.numel(), (reps, values.numel()), generator=generator)
    boot = values[indices].mean(dim=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(torch.quantile(boot, 0.025)),
        "ci95_high": float(torch.quantile(boot, 0.975)),
    }


def _comparison(
    candidate: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = []
    family_deltas: dict[str, list[float]] = defaultdict(list)
    for family in EXPECTED_DATASETS:
        for dataset in sorted(EXPECTED_DATASETS[family]):
            delta = candidate[family][dataset] - reference[family][dataset]
            rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "candidate": candidate[family][dataset],
                    "reference": reference[family][dataset],
                    "delta": delta,
                }
            )
            family_deltas[family].append(delta)
    family_means = {family: statistics.fmean(items) for family, items in family_deltas.items()}
    deltas = [row["delta"] for row in rows]
    positive_families = sum(value > 0 for value in family_means.values())
    winning_cells = sum(value > 0 for value in deltas)
    no_clear_family_regression = min(family_means.values()) >= GATE["maximum_allowed_family_mean_regression"]
    passed = (
        positive_families >= GATE["minimum_positive_families"]
        and winning_cells >= GATE["minimum_winning_cells"]
        and no_clear_family_regression
    )
    return {
        "cells": rows,
        "n_cells": len(rows),
        "winning_cells": winning_cells,
        "median_cell_delta": statistics.median(deltas),
        "family_mean_deltas": family_means,
        "positive_families": positive_families,
        "family_macro_bootstrap": _bootstrap_family_macro(family_deltas, seed=seed),
        "no_clear_family_regression": no_clear_family_regression,
        "pass_for_dense_followup": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iter", type=int, default=255)
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_root = args.eval_root / f"u{args.checkpoint_iter}"
    cells = {arm: _load_cells(checkpoint_root / arm) for arm in ARMS}
    anchor_dir = args.anchor_dir or args.eval_root / "u0_anchor"
    anchor = _load_cells(anchor_dir)

    comparisons = {
        "w025_minus_w1": _comparison(cells["w025"], cells["w1"], seed=20260910),
        "w0_minus_w1": _comparison(cells["w0"], cells["w1"], seed=20260911),
        "w1_minus_u0": _comparison(cells["w1"], anchor, seed=20260912),
        "w025_minus_u0": _comparison(cells["w025"], anchor, seed=20260913),
        "w0_minus_u0": _comparison(cells["w0"], anchor, seed=20260914),
    }
    eligible = [
        arm
        for arm in ("w025", "w0")
        if comparisons[f"{arm}_minus_w1"]["pass_for_dense_followup"]
    ]
    report = {
        "summary": "local_dino_weight_core_gate_v1",
        "checkpoint_iter": args.checkpoint_iter,
        "preregistered_gate": GATE,
        "primary_metrics": {
            "classification": "balanced_accuracy",
            "regression": "r2",
            "retrieval": "recall_at_1",
            "clustering": "nmi",
        },
        "retrieval_clustering_row_selectors": {
            f"{family}/{dataset}": selector
            for (family, dataset), selector in ROW_SELECTORS.items()
        },
        "raw_cells": {arm: _flat(value) for arm, value in cells.items()},
        "anchor_cells": _flat(anchor),
        "comparisons": comparisons,
        "eligible_for_dense_followup": eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_iter": args.checkpoint_iter,
                "eligible_for_dense_followup": eligible,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
