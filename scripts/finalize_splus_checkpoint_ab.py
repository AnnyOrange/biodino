#!/usr/bin/env python3
"""Build a strict, report-ready comparison of finalized S+ checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


CLASSIFICATION_DATASETS = [
    "bloodmnist",
    "pathmnist",
    "tissuemnist",
    "breastmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "chestmnist",
    "bbbc048-cellcycle",
    "cyclops-protein-loc",
    "midog25-atypical",
    "pcam",
    "nct-crc-he",
    "lc25000",
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-cp-task3",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
]
RETRIEVAL_DATASETS = ["lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"]
SEGMENTATION_DATASETS = [
    "bbbc038",
    "conic",
    "monuseg",
    "pannuke",
    "tissuenet",
    "livecell",
    "multimodal_cellseg",
    "cellpose",
]
SUMMARY_METRICS = [
    "c25_macro_f1",
    "bbbc005_r2",
    "retrieval4_map_at_5",
    "clustering4_nmi",
    "segmentation8_mdice",
    "livecell_detection_f1",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError(f"Invalid result: {path}")
    return data


def require_datasets(actual: set[str], expected: list[str], task: str, root: Path) -> None:
    missing = sorted(set(expected) - actual)
    if missing:
        raise ValueError(f"{task} is incomplete under {root}; missing={missing}")


def collect_candidate(label: str, core_root: Path, dense_root: Path, ckpt: str) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []

    classification: dict[str, float] = {}
    for dataset in CLASSIFICATION_DATASETS:
        path = core_root / "bio_classification" / dataset / ckpt / "last_result.json"
        result = load_json(path)
        classification[dataset] = float(result["macro_f1"])
        dataset_rows.append(
            {
                "candidate": label,
                "task": "classification",
                "dataset": dataset,
                "metric": "macro_f1",
                "value": classification[dataset],
                "result_path": str(path),
            }
        )

    regression_path = core_root / "bio_regression" / "bbbc005" / ckpt / "last_result.json"
    regression = float(load_json(regression_path)["r2"])
    dataset_rows.append(
        {
            "candidate": label,
            "task": "regression",
            "dataset": "bbbc005",
            "metric": "r2",
            "value": regression,
            "result_path": str(regression_path),
        }
    )

    retrieval: dict[str, dict[str, float]] = {}
    for dataset in RETRIEVAL_DATASETS:
        path = core_root / "bio_retrieval" / dataset / ckpt / "last_result.json"
        result = load_json(path)
        retrieval[dataset] = {
            "map_at_5": float(result["map_at_5"]),
            "nmi": float(result["nmi"]),
        }
        for metric, value in retrieval[dataset].items():
            dataset_rows.append(
                {
                    "candidate": label,
                    "task": "retrieval" if metric == "map_at_5" else "clustering",
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "result_path": str(path),
                }
            )

    segmentation_paths: dict[str, Path] = {}
    for path in dense_root.glob(f"bio_segmentation/**/{ckpt}/results.json"):
        dataset = path.parents[1].name
        if dataset in segmentation_paths:
            raise ValueError(f"Duplicate segmentation result for {dataset}: {path}")
        segmentation_paths[dataset] = path
    require_datasets(set(segmentation_paths), SEGMENTATION_DATASETS, "segmentation", dense_root)
    segmentation: dict[str, float] = {}
    for dataset in SEGMENTATION_DATASETS:
        path = segmentation_paths[dataset]
        segmentation[dataset] = float(load_json(path)["test"]["mDice"])
        dataset_rows.append(
            {
                "candidate": label,
                "task": "segmentation",
                "dataset": dataset,
                "metric": "test_mDice",
                "value": segmentation[dataset],
                "result_path": str(path),
            }
        )

    detection_path = dense_root / "bio_detection" / "livecell" / ckpt / "results_bio_detection.json"
    detection = float(load_json(detection_path)["test_patch_f1"])
    if detection > 1.0:
        detection /= 100.0
    dataset_rows.append(
        {
            "candidate": label,
            "task": "detection",
            "dataset": "livecell",
            "metric": "test_patch_f1",
            "value": detection,
            "result_path": str(detection_path),
        }
    )

    summary = {
        "c25_macro_f1": mean(classification.values()),
        "bbbc005_r2": regression,
        "retrieval4_map_at_5": mean(row["map_at_5"] for row in retrieval.values()),
        "clustering4_nmi": mean(row["nmi"] for row in retrieval.values()),
        "segmentation8_mdice": mean(segmentation.values()),
        "livecell_detection_f1": detection,
    }
    return {
        "label": label,
        "checkpoint": ckpt,
        "core_root": str(core_root),
        "dense_root": str(dense_root),
        "coverage": {
            "classification": len(classification),
            "regression": 1,
            "retrieval": len(retrieval),
            "segmentation": len(segmentation),
            "detection": 1,
        },
        "summary": summary,
        "dataset_rows": dataset_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, candidates: list[dict[str, Any]], reference: str) -> None:
    reference_row = next(row for row in candidates if row["label"] == reference)
    headers = ["candidate", *SUMMARY_METRICS]
    lines = [
        "# S+ checkpoint interpolation A/B",
        "",
        f"Reference for deltas: `{reference}`.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for candidate in candidates:
        cells = [candidate["label"]]
        for metric in SUMMARY_METRICS:
            value = candidate["summary"][metric]
            if candidate["label"] == reference:
                cells.append(f"{value:.6f}")
            else:
                delta = value - reference_row["summary"][metric]
                cells.append(f"{value:.6f} ({delta:+.6f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Coverage", "", "| candidate | C | R | Ret | Seg | Det |", "|---|---:|---:|---:|---:|---:|"])
    for candidate in candidates:
        coverage = candidate["coverage"]
        lines.append(
            f"| {candidate['label']} | {coverage['classification']}/25 | "
            f"{coverage['regression']}/1 | {coverage['retrieval']}/4 | "
            f"{coverage['segmentation']}/8 | {coverage['detection']}/1 |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "CORE_ROOT", "DENSE_ROOT"),
        required=True,
        help="Add a candidate; use the same root twice when core and dense outputs are unified.",
    )
    parser.add_argument("--checkpoint", default="75")
    parser.add_argument("--reference", help="Candidate label used for deltas; defaults to the first candidate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = [spec[0] for spec in args.candidate]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Candidate labels must be unique: {labels}")
    reference = args.reference or labels[0]
    if reference not in labels:
        raise ValueError(f"Reference {reference!r} is not one of {labels}")

    candidates = [
        collect_candidate(label, Path(core_root), Path(dense_root), args.checkpoint)
        for label, core_root, dense_root in args.candidate
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    serializable = []
    dataset_rows = []
    summary_rows = []
    for candidate in candidates:
        dataset_rows.extend(candidate["dataset_rows"])
        clean = {key: value for key, value in candidate.items() if key != "dataset_rows"}
        serializable.append(clean)
        summary_rows.append({"candidate": candidate["label"], **candidate["summary"]})

    (args.output_dir / "metrics.json").write_text(json.dumps(serializable, indent=2) + "\n")
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "per_dataset.csv", dataset_rows)
    write_markdown(args.output_dir / "README.md", candidates, reference)
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
