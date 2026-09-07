#!/usr/bin/env python3
"""Build the fair cross-foundation-model matrix used by Fig. 3 panel C.

The current BioDINO H+ checkpoint and official DINOv3 H+/16 baseline are read
from the completed H100 alpha sweep. The 14 external models are read from the
existing frozen-probe campaign. Scores are averaged only after intersecting
the exact datasets shared by every model in a task family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_HPLUS_DETAILS = DEFAULT_OUTPUT_DIR / "sources/hplus_s6_alpha1_full_details.json"
EXTERNAL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_fair_protocol_20260721"
GAPFILL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_hplus_protocol_gapfill_20260811"

DISPLAY = {
    "bioclip": "BioCLIP",
    "conch": "CONCH",
    "cytoimagenet": "CytoImageNet",
    "cytoself": "CytoSelf",
    "dinov2": "DINOv2",
    "gigapath": "GigaPath",
    "hoptimus0": "H-optimus-0",
    "jump_cp": "JUMP-CP",
    "mae": "MAE",
    "pe": "PE",
    "phikon2": "Phikon-v2",
    "siglip2": "SigLIP2",
    "uni": "UNI",
    "virchow2": "Virchow2",
}

TASKS = {
    "classification": {"metric": "macro_f1", "label": "Class. macro F1"},
    "regression": {"metric": "r2", "label": "Reg. R2"},
    "retrieval": {"metric": "map_at_5", "label": "Retr. mAP@5"},
    "clustering": {"metric": "nmi", "label": "Clust. NMI"},
    "detection": {"metric": "test_patch_f1", "label": "Detect. F1"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_score(value: Any, path: Path, key: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key} in {path}: {value!r}") from exc
    if not np.isfinite(score):
        raise ValueError(f"Non-finite {key} in {path}: {value!r}")
    return score


def load_hplus_rows(
    details_path: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    payload = json.loads(details_path.read_text())
    by_label = {str(row["label"]): row for row in payload}
    required = {"alpha_1.00", "alpha_0.00"}
    missing = sorted(required - set(by_label))
    if missing:
        raise ValueError(f"H+ details missing {missing}: {details_path}")

    models = {
        "biodino_hplus": by_label["alpha_1.00"],
        "dinov3_hplus_official": by_label["alpha_0.00"],
    }
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for model_key, entry in models.items():
        task_rows: dict[str, dict[str, float]] = defaultdict(dict)
        for row in entry["dataset_rows"]:
            task = str(row["task"])
            if task in TASKS:
                task_rows[task][str(row["dataset"])] = finite_score(
                    row["value"], details_path, f"{model_key}.{task}"
                )
        scores[model_key] = dict(task_rows)

    provenance = {
        "path": str(details_path),
        "sha256": source_sha256(details_path),
        "biodino_hplus": {
            "alpha_label": "alpha_1.00",
            "checkpoint": str(models["biodino_hplus"]["checkpoint"]),
            "core_root": models["biodino_hplus"]["core_root"],
            "dense_root": models["biodino_hplus"]["dense_root"],
        },
        "dinov3_hplus_official": {
            "alpha_label": "alpha_0.00",
            "checkpoint": str(models["dinov3_hplus_official"]["checkpoint"]),
            "core_root": models["dinov3_hplus_official"]["core_root"],
            "dense_root": models["dinov3_hplus_official"]["dense_root"],
        },
    }
    return scores, provenance


def load_external_rows(
    model_keys: list[str],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, str]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    sources: dict[str, str] = {}
    for model_key in model_keys:
        task_rows: dict[str, dict[str, float]] = defaultdict(dict)

        scalar_path = EXTERNAL_ROOT / "classification" / model_key / "summary.csv"
        if not scalar_path.is_file():
            raise FileNotFoundError(scalar_path)
        for row in read_csv(scalar_path):
            task = row.get("task", "")
            if task in {"classification", "multilabel_classification"}:
                task_rows["classification"][row["dataset"]] = finite_score(
                    row.get("macro_f1"), scalar_path, "macro_f1"
                )
            elif task == "regression":
                task_rows["regression"][row["dataset"]] = finite_score(
                    row.get("r2"), scalar_path, "r2"
                )
        sources[f"{model_key}.classification_regression"] = str(scalar_path)

        retrieval_path = GAPFILL_ROOT / "retrieval_clustering" / model_key / "summary.csv"
        if not retrieval_path.is_file():
            raise FileNotFoundError(retrieval_path)
        for row in read_csv(retrieval_path):
            if row.get("error"):
                raise ValueError(f"External retrieval error in {retrieval_path}: {row['error']}")
            dataset = row["dataset"]
            task_rows["retrieval"][dataset] = finite_score(
                row.get("map_at_5"), retrieval_path, "map_at_5"
            )
            task_rows["clustering"][dataset] = finite_score(
                row.get("nmi"), retrieval_path, "nmi"
            )
        sources[f"{model_key}.retrieval_clustering"] = str(retrieval_path)

        detection_path = GAPFILL_ROOT / "detection" / model_key / "results_bio_detection.json"
        if not detection_path.is_file():
            raise FileNotFoundError(detection_path)
        detection = json.loads(detection_path.read_text())
        score = finite_score(detection.get("test_patch_f1"), detection_path, "test_patch_f1")
        task_rows["detection"]["livecell"] = score / 100.0 if score > 1.0 else score
        sources[f"{model_key}.detection"] = str(detection_path)
        scores[model_key] = dict(task_rows)
    return scores, sources


def common_datasets(
    all_scores: dict[str, dict[str, dict[str, float]]], task: str
) -> list[str]:
    datasets: set[str] | None = None
    for model_key, task_scores in all_scores.items():
        model_datasets = set(task_scores.get(task, {}))
        if not model_datasets:
            raise ValueError(f"{model_key} has no {task} values")
        datasets = model_datasets if datasets is None else datasets & model_datasets
    if not datasets:
        raise ValueError(f"No common datasets for {task}")
    return sorted(datasets)


def mean_rank_percentile(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    ranks = values.rank(ascending=False, method="average")
    denominator = max(len(values) - 1, 1)
    percentile = 1.0 - (ranks - 1.0) / denominator
    return ranks, percentile


def build_matrix(details_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hplus_scores, hplus_provenance = load_hplus_rows(details_path)
    external_keys = sorted(DISPLAY)
    external_scores, external_sources = load_external_rows(external_keys)
    all_scores = {**hplus_scores, **external_scores}

    model_metadata = {
        "biodino_hplus": {
            "model": "BioDINO H+/16 (840M)",
            "group": "ours",
            "display_order": 0,
        },
        "dinov3_hplus_official": {
            "model": "DINOv3 H+/16",
            "group": "official",
            "display_order": 1,
        },
        **{
            key: {"model": DISPLAY[key], "group": "external", "display_order": 2}
            for key in external_keys
        },
    }

    shared = {task: common_datasets(all_scores, task) for task in TASKS}
    long_rows: list[dict[str, Any]] = []
    for model_key, task_scores in all_scores.items():
        for task, spec in TASKS.items():
            for dataset in shared[task]:
                long_rows.append(
                    {
                        "model_key": model_key,
                        "model": model_metadata[model_key]["model"],
                        "group": model_metadata[model_key]["group"],
                        "task": task,
                        "metric": spec["metric"],
                        "dataset": dataset,
                        "score": task_scores[task][dataset],
                    }
                )
    long_df = pd.DataFrame(long_rows)

    # Normalize separately per dataset before averaging task-family performance.
    long_df["dataset_rank"] = np.nan
    long_df["dataset_percentile"] = np.nan
    for _, index in long_df.groupby(["task", "dataset"], sort=False).groups.items():
        ranks, percentiles = mean_rank_percentile(long_df.loc[index, "score"])
        long_df.loc[index, "dataset_rank"] = ranks
        long_df.loc[index, "dataset_percentile"] = percentiles

    summary_rows: list[dict[str, Any]] = []
    for model_key, metadata in model_metadata.items():
        row: dict[str, Any] = {"model_key": model_key, **metadata}
        for task in TASKS:
            subset = long_df[(long_df["model_key"] == model_key) & (long_df["task"] == task)]
            row[f"{task}_score"] = float(subset["score"].mean())
            row[f"{task}_rank"] = float(subset["dataset_rank"].mean())
            row[f"{task}_percentile"] = float(subset["dataset_percentile"].mean())
            row[f"{task}_n_datasets"] = int(len(subset))
        row["mean_percentile"] = float(
            mean(row[f"{task}_percentile"] for task in TASKS)
        )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    external_order = (
        summary_df[summary_df["group"] == "external"]
        .sort_values(["mean_percentile", "model"], ascending=[False, True])
        .index
    )
    summary_df.loc[external_order, "display_order"] = np.arange(2, 2 + len(external_order))
    summary_df = summary_df.sort_values("display_order").reset_index(drop=True)

    provenance = {
        "description": (
            "Panel C compares current alpha=1 BioDINO H+/16 (840M), alpha=0 official "
            "DINOv3 H+/16, and all 14 external foundation models with complete "
            "outputs under matching frozen-probe task protocols."
        ),
        "models": int(len(summary_df)),
        "model_keys": summary_df["model_key"].tolist(),
        "tasks": {
            task: {
                "metric": spec["metric"],
                "n_datasets": len(shared[task]),
                "datasets": shared[task],
            }
            for task, spec in TASKS.items()
        },
        "ranking": (
            "Scores are ranked per dataset (higher is better); tie-aware percentile "
            "is 1.0 for rank 1 and 0.0 for last rank, then averaged within each task."
        ),
        "excluded": {
            "segmentation": (
                "Excluded because H+ used 50 dense-probe epochs while external "
                "foundation models used 20 epochs."
            )
        },
        "hplus_alpha_sweep": hplus_provenance,
        "external_sources": external_sources,
    }
    return summary_df, long_df, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hplus-details", type=Path, default=DEFAULT_HPLUS_DETAILS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hplus_details.is_file():
        raise FileNotFoundError(
            f"Missing H+ detailed alpha sweep: {args.hplus_details}. "
            "Copy full_summary/details.json from the H100 campaign first."
        )
    summary, long, provenance = build_matrix(args.hplus_details)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "panel_c_foundation_matrix.csv", index=False)
    long.to_csv(args.output_dir / "panel_c_foundation_dataset_scores.csv", index=False)
    (args.output_dir / "panel_c_foundation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
