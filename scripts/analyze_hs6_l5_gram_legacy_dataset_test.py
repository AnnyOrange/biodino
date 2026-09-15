#!/usr/bin/env python3
"""Aggregate the fixed-endpoint HS6-L5 Gram legacy dataset-test campaign.

The historical report selected a best checkpoint independently for every
dataset and metric.  This script deliberately does not repeat that selection:
it compares the fixed ck20007 start with matched ck20495 endpoints, while
reusing the historical common-dataset scopes and metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs/02_eval_runs/hs6_l5_gram_u488_legacy_dataset_test_20260912"
DEFAULT_REFERENCE = (
    ROOT / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908" / "point_20007"
)
DEFAULT_HISTORICAL_SCOPE = (
    ROOT
    / "outputs/03_comparisons/2026-7-1-test-overall/compare_1tb_vs_5tb_hplus"
    / "task_metric_summary_common_datasets.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/03_comparisons/hs6_l5_gram_legacy_dataset_test_20260912"
EXPECTED_LANES = {
    "classification_a",
    "classification_b",
    "classification_c",
    "classification_d",
    "regression",
    "retrieval",
    "detection",
    "segmentation_a",
    "segmentation_b",
    "segmentation_c",
    "segmentation_d",
    "ood",
}
ENDPOINT_CHECKPOINT_ID = 20495


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    group: str
    higher_is_better: bool = True
    native_percent: bool = False


METRICS = {
    metric.key: metric
    for metric in (
        Metric("classification_accuracy", "Classification Acc", "Classification"),
        Metric(
            "classification_balanced_accuracy",
            "Classification BA",
            "Classification",
        ),
        Metric("classification_macro_f1", "Classification macro F1", "Classification"),
        Metric("multilabel_macro_auc", "Multi-label macro AUC", "Classification"),
        Metric("multilabel_micro_auc", "Multi-label micro AUC", "Classification"),
        Metric("regression_mae", "Regression MAE", "Regression", False),
        Metric("regression_r2", "Regression R2", "Regression"),
        Metric("regression_spearman", "Regression Spearman", "Regression"),
        Metric("retrieval_recall_at_1", "Retrieval R@1", "Retrieval"),
        Metric("retrieval_recall_at_5", "Retrieval R@5", "Retrieval"),
        Metric("retrieval_recall_at_10", "Retrieval R@10", "Retrieval"),
        Metric("retrieval_map_at_10", "Retrieval mAP@10", "Retrieval"),
        Metric("retrieval_mrr", "Retrieval MRR", "Retrieval"),
        Metric("clustering_nmi", "Clustering NMI", "Clustering"),
        Metric("clustering_ari", "Clustering ARI", "Clustering"),
        Metric("clustering_cluster_accuracy", "Clustering Acc", "Clustering"),
        Metric(
            "clustering_silhouette_cosine",
            "Clustering silhouette cosine",
            "Clustering",
        ),
        Metric(
            "detection_livecell_patch_f1",
            "Detection LIVECell patch F1",
            "Detection",
            native_percent=True,
        ),
        Metric(
            "detection_patch_f1",
            "Detection patch F1",
            "Detection",
            native_percent=True,
        ),
        Metric(
            "detection_patch_accuracy",
            "Detection patch accuracy",
            "Detection",
            native_percent=True,
        ),
        Metric("segmentation_mDice", "Segmentation mDice", "Segmentation"),
        Metric("segmentation_mIoU", "Segmentation mIoU", "Segmentation"),
        Metric("segmentation_AJI", "Segmentation AJI", "Segmentation"),
        Metric("segmentation_AP50", "Segmentation AP50", "Segmentation"),
        Metric("ood_composite", "OOD composite", "OOD"),
        Metric("ood_xray_pair_recall_at_1", "OOD X-ray pair R@1", "OOD"),
        Metric("ood_xray_dose_r2", "OOD X-ray dose R2", "OOD"),
        Metric("ood_cryo_class_accuracy", "OOD cryo class acc", "OOD"),
        Metric("ood_cryo_quality_auroc", "OOD cryo quality AUROC", "OOD"),
        Metric("ood_cryo_retrieval_map_at_10", "OOD cryo mAP@10", "OOD"),
        Metric("ood_cryo_cluster_nmi", "OOD cryo cluster NMI", "OOD"),
    )
}


def finite(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_value(
    values: dict[tuple[str, str], dict[str, Any]],
    *,
    metric: str,
    dataset: str,
    value: Any,
    source: Path,
) -> None:
    number = finite(value)
    if number is None or metric not in METRICS:
        return
    key = (metric, dataset)
    record = {"metric_key": metric, "dataset": dataset, "value": number, "source": str(source)}
    previous = values.get(key)
    if previous is not None and not math.isclose(previous["value"], number, abs_tol=1e-12):
        raise RuntimeError(f"Conflicting fixed-point cells for {key}: {previous['value']} vs {number}")
    values[key] = record


def collect_point(point: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not point.is_dir():
        raise FileNotFoundError(point)
    values: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(point.rglob("summary.csv")):
        if "02_eval_inputs" in path.parts:
            continue
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[-1]
        if "ood" in path.parts:
            ood_columns = {
                "xray_pair_recall_at_1": "ood_xray_pair_recall_at_1",
                "xray_dose_r2": "ood_xray_dose_r2",
                "cryo_class_accuracy": "ood_cryo_class_accuracy",
                "cryo_quality_auroc": "ood_cryo_quality_auroc",
                "cryo_retrieval_map_at_10": "ood_cryo_retrieval_map_at_10",
                "cryo_cluster_nmi": "ood_cryo_cluster_nmi",
            }
            for column, metric in ood_columns.items():
                add_value(
                    values,
                    metric=metric,
                    dataset=column,
                    value=row.get(column),
                    source=path,
                )
            continue

        if row.get("error"):
            continue
        dataset = row.get("dataset", "").strip()
        task = row.get("task", "").strip()
        if not dataset:
            continue
        columns: dict[str, str]
        if task == "classification":
            columns = {
                "accuracy": "classification_accuracy",
                "balanced_accuracy": "classification_balanced_accuracy",
                "macro_f1": "classification_macro_f1",
            }
        elif task == "multilabel_classification":
            columns = {
                "macro_auc": "multilabel_macro_auc",
                "micro_auc": "multilabel_micro_auc",
            }
        elif task == "regression":
            columns = {
                "mae": "regression_mae",
                "r2": "regression_r2",
                "spearman": "regression_spearman",
            }
        elif task == "retrieval_clustering":
            columns = {
                "recall_at_1": "retrieval_recall_at_1",
                "recall_at_5": "retrieval_recall_at_5",
                "recall_at_10": "retrieval_recall_at_10",
                "map_at_10": "retrieval_map_at_10",
                "mrr": "retrieval_mrr",
                "nmi": "clustering_nmi",
                "ari": "clustering_ari",
                "cluster_accuracy": "clustering_cluster_accuracy",
                "silhouette_cosine": "clustering_silhouette_cosine",
            }
        else:
            continue
        for column, metric in columns.items():
            add_value(
                values,
                metric=metric,
                dataset=dataset,
                value=row.get(column),
                source=path,
            )

    for path in sorted(point.rglob("results_bio_detection.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset") or path.parts[-3])
        add_value(
            values,
            metric="detection_patch_f1",
            dataset=dataset,
            value=payload.get("test_patch_f1"),
            source=path,
        )
        add_value(
            values,
            metric="detection_patch_accuracy",
            dataset=dataset,
            value=payload.get("test_patch_accuracy"),
            source=path,
        )
        if dataset == "livecell":
            add_value(
                values,
                metric="detection_livecell_patch_f1",
                dataset=dataset,
                value=payload.get("test_patch_f1"),
                source=path,
            )

    for path in sorted(point.rglob("results.json")):
        if "bio_segmentation" not in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test")
        if not isinstance(test, dict):
            continue
        dataset = path.parts[-3]
        for column, metric in {
            "mDice": "segmentation_mDice",
            "mIoU": "segmentation_mIoU",
            "AJI": "segmentation_AJI",
            "AP50": "segmentation_AP50",
        }.items():
            add_value(
                values,
                metric=metric,
                dataset=dataset,
                value=test.get(column),
                source=path,
            )

    composite_keys = (
        "ood_xray_pair_recall_at_1",
        "ood_xray_dose_r2",
        "ood_cryo_class_accuracy",
        "ood_cryo_quality_auroc",
        "ood_cryo_retrieval_map_at_10",
    )
    components = [
        values[(metric, metric.removeprefix("ood_"))]["value"]
        for metric in composite_keys
        if (metric, metric.removeprefix("ood_")) in values
    ]
    if len(components) == len(composite_keys):
        add_value(
            values,
            metric="ood_composite",
            dataset="ood_composite",
            value=sum(components) / len(components),
            source=point / "ood",
        )
    return values


def historical_scopes(path: Path) -> dict[str, set[str]]:
    scopes: dict[str, set[str]] = {}
    for row in read_csv(path):
        key = row["metric_key"]
        if key not in METRICS:
            continue
        scopes[key] = {item for item in row["datasets"].split(";") if item}
    return scopes


def lane_audit_from_state(state: Path, checkpoint_id: int = ENDPOINT_CHECKPOINT_ID) -> dict[str, Any]:
    """Audit durable per-lane state; online status files are intentionally ignored."""
    state = state.resolve()
    prefix = f"ckpt_{checkpoint_id}__"
    done = {path.stem.removeprefix(prefix) for path in (state / "done").glob(f"{prefix}*.json")}
    terminal = sorted(path.name for path in (state / "terminal").glob("*"))
    failures = sorted(path.name for path in (state / "failures").glob("*"))
    return {
        "state_root": str(state),
        "done_lanes": sorted(done),
        "missing_lanes": sorted(EXPECTED_LANES - done),
        "unexpected_done_lanes": sorted(done - EXPECTED_LANES),
        "terminal": terminal,
        "failure_attempts": failures,
        "complete": done == EXPECTED_LANES and not terminal,
    }


def lane_audit(campaign: Path, arm: str) -> dict[str, Any]:
    return lane_audit_from_state(campaign / arm / "_state")


def parse_extra_arm(raw: str) -> tuple[str, Path]:
    try:
        name, path = raw.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("extra arm must be NAME=POINT_ROOT") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("extra arm must be NAME=POINT_ROOT")
    return name, Path(path).resolve()


def parse_extra_arm_state(raw: str) -> tuple[str, Path]:
    try:
        name, path = raw.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("extra arm state must be NAME=STATE_ROOT") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("extra arm state must be NAME=STATE_ROOT")
    return name, Path(path).resolve()


def infer_extra_arm_state_root(point_root: Path, explicit_state: Path | None = None) -> Path:
    """Resolve the durable state root for an extra fixed-endpoint arm."""
    point_root = point_root.resolve()
    if explicit_state is not None:
        state = explicit_state.resolve()
    elif point_root.name == f"point_{ENDPOINT_CHECKPOINT_ID}":
        state = point_root.parent / "_state"
    else:
        raise ValueError(
            f"cannot infer extra-arm state from {point_root}; expected a "
            f"point_{ENDPOINT_CHECKPOINT_ID} directory or "
            "--extra-arm-state NAME=STATE_ROOT"
        )
    if not state.is_dir():
        raise ValueError(
            f"extra-arm state root does not exist: {state}; provide "
            "--extra-arm-state NAME=STATE_ROOT if it is stored elsewhere"
        )
    return state


def display_delta(metric: Metric, delta: float) -> float:
    return delta if metric.native_percent else 100.0 * delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--historical-scope", type=Path, default=DEFAULT_HISTORICAL_SCOPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-arm", action="append", type=parse_extra_arm, default=[])
    parser.add_argument(
        "--extra-arm-state",
        action="append",
        type=parse_extra_arm_state,
        default=[],
        help=(
            "Optional NAME=STATE_ROOT override for an extra arm. By default, a "
            ".../point_20495 arm uses its sibling _state directory."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()

    arm_paths = {
        "start_ck20007": args.reference.resolve(),
        "control_ck20495": campaign / "control/point_20495",
        "gram_a7807_ck20495": campaign / "anchor7807/point_20495",
        "gram_a17079_ck20495": campaign / "anchor17079/point_20495",
    }
    extra_arm_paths: dict[str, Path] = {}
    for name, path in args.extra_arm:
        if name in arm_paths or name in extra_arm_paths:
            parser.error(f"duplicate or reserved extra-arm name: {name}")
        extra_arm_paths[name] = path
    arm_paths.update(extra_arm_paths)

    explicit_extra_states: dict[str, Path] = {}
    for name, path in args.extra_arm_state:
        if name in explicit_extra_states:
            parser.error(f"duplicate extra-arm-state name: {name}")
        explicit_extra_states[name] = path
    unknown_state_names = sorted(set(explicit_extra_states) - set(extra_arm_paths))
    if unknown_state_names:
        parser.error("--extra-arm-state names without matching --extra-arm: " + ", ".join(unknown_state_names))
    try:
        extra_state_roots = {
            name: infer_extra_arm_state_root(path, explicit_extra_states.get(name))
            for name, path in extra_arm_paths.items()
        }
    except ValueError as error:
        parser.error(str(error))

    scopes = historical_scopes(args.historical_scope.resolve())
    collected = {arm: collect_point(path) for arm, path in arm_paths.items()}

    audits: dict[str, Any] = {}
    for arm, path in arm_paths.items():
        missing = [
            f"{metric}/{dataset}"
            for metric, datasets in scopes.items()
            for dataset in sorted(datasets)
            if (metric, dataset) not in collected[arm]
        ]
        audit: dict[str, Any] = {
            "point_root": str(path),
            "cells_collected": len(collected[arm]),
            "missing_historical_cells": missing,
        }
        campaign_arm = {
            "control_ck20495": "control",
            "gram_a7807_ck20495": "anchor7807",
            "gram_a17079_ck20495": "anchor17079",
        }.get(arm)
        if campaign_arm:
            audit["lanes"] = lane_audit(campaign, campaign_arm)
        elif arm in extra_state_roots:
            audit["lanes"] = lane_audit_from_state(extra_state_roots[arm])
        audit["complete"] = not missing and ("lanes" not in audit or audit["lanes"]["complete"])
        audits[arm] = audit

    if not args.allow_incomplete:
        incomplete = [arm for arm, audit in audits.items() if not audit["complete"]]
        if incomplete:
            raise RuntimeError(f"incomplete arms: {incomplete}; rerun with --allow-incomplete to preview")

    metric_rows: list[dict[str, Any]] = []
    for arm, values in collected.items():
        for (metric_key, dataset), record in sorted(values.items()):
            metric = METRICS[metric_key]
            metric_rows.append(
                {
                    "arm": arm,
                    "metric_group": metric.group,
                    "metric_key": metric_key,
                    "metric": metric.label,
                    "dataset": dataset,
                    "value": record["value"],
                    "higher_is_better": metric.higher_is_better,
                    "source": record["source"],
                }
            )
    write_csv(
        output / "fixed_endpoint_metric_rows.csv",
        metric_rows,
        [
            "arm",
            "metric_group",
            "metric_key",
            "metric",
            "dataset",
            "value",
            "higher_is_better",
            "source",
        ],
    )

    historical_rows: list[dict[str, Any]] = []
    for metric_key, datasets in scopes.items():
        metric = METRICS[metric_key]
        for arm in arm_paths:
            available = [
                collected[arm][(metric_key, dataset)]["value"]
                for dataset in sorted(datasets)
                if (metric_key, dataset) in collected[arm]
            ]
            if not available:
                continue
            value = sum(available) / len(available)
            start_values = [
                collected["start_ck20007"][(metric_key, dataset)]["value"]
                for dataset in sorted(datasets)
                if (metric_key, dataset) in collected[arm] and (metric_key, dataset) in collected["start_ck20007"]
            ]
            control_values = [
                collected["control_ck20495"][(metric_key, dataset)]["value"]
                for dataset in sorted(datasets)
                if (metric_key, dataset) in collected[arm] and (metric_key, dataset) in collected["control_ck20495"]
            ]
            start_mean = sum(start_values) / len(start_values) if start_values else math.nan
            control_mean = sum(control_values) / len(control_values) if control_values else math.nan
            historical_rows.append(
                {
                    "metric_group": metric.group,
                    "metric_key": metric_key,
                    "metric": metric.label,
                    "arm": arm,
                    "n_historical_datasets": len(available),
                    "expected_historical_datasets": len(datasets),
                    "mean": value,
                    "delta_vs_start": value - start_mean,
                    "delta_vs_control": value - control_mean,
                    "oriented_delta_vs_control": (
                        value - control_mean if metric.higher_is_better else control_mean - value
                    ),
                    "delta_vs_control_pp": display_delta(metric, value - control_mean),
                    "datasets": ";".join(sorted(datasets)),
                    "fixed_endpoint_not_best_over_curve": True,
                }
            )
    write_csv(
        output / "historical_scope_fixed_endpoint_summary.csv",
        historical_rows,
        [
            "metric_group",
            "metric_key",
            "metric",
            "arm",
            "n_historical_datasets",
            "expected_historical_datasets",
            "mean",
            "delta_vs_start",
            "delta_vs_control",
            "oriented_delta_vs_control",
            "delta_vs_control_pp",
            "datasets",
            "fixed_endpoint_not_best_over_curve",
        ],
    )

    common_cells = set.intersection(*(set(values) for values in collected.values()))
    expanded: list[dict[str, Any]] = []
    by_metric: dict[str, list[str]] = defaultdict(list)
    for metric_key, dataset in sorted(common_cells):
        by_metric[metric_key].append(dataset)
    for metric_key, datasets in sorted(by_metric.items()):
        metric = METRICS[metric_key]
        for arm in arm_paths:
            values = [collected[arm][(metric_key, dataset)]["value"] for dataset in datasets]
            control = [collected["control_ck20495"][(metric_key, dataset)]["value"] for dataset in datasets]
            mean, control_mean = sum(values) / len(values), sum(control) / len(control)
            expanded.append(
                {
                    "metric_group": metric.group,
                    "metric_key": metric_key,
                    "metric": metric.label,
                    "arm": arm,
                    "n_common_datasets": len(datasets),
                    "mean": mean,
                    "delta_vs_control": mean - control_mean,
                    "oriented_delta_vs_control": (
                        mean - control_mean if metric.higher_is_better else control_mean - mean
                    ),
                    "delta_vs_control_pp": display_delta(metric, mean - control_mean),
                    "datasets": ";".join(datasets),
                }
            )
    write_csv(
        output / "expanded_common_fixed_endpoint_summary.csv",
        expanded,
        [
            "metric_group",
            "metric_key",
            "metric",
            "arm",
            "n_common_datasets",
            "mean",
            "delta_vs_control",
            "oriented_delta_vs_control",
            "delta_vs_control_pp",
            "datasets",
        ],
    )

    per_dataset: list[dict[str, Any]] = []
    for metric_key, dataset in sorted(common_cells):
        metric = METRICS[metric_key]
        control = collected["control_ck20495"][(metric_key, dataset)]["value"]
        for arm in arm_paths:
            value = collected[arm][(metric_key, dataset)]["value"]
            per_dataset.append(
                {
                    "metric_group": metric.group,
                    "metric_key": metric_key,
                    "metric": metric.label,
                    "dataset": dataset,
                    "arm": arm,
                    "value": value,
                    "delta_vs_control": value - control,
                    "oriented_delta_vs_control": (value - control if metric.higher_is_better else control - value),
                    "delta_vs_control_pp": display_delta(metric, value - control),
                }
            )
    write_csv(
        output / "per_dataset_fixed_endpoint_comparison.csv",
        per_dataset,
        [
            "metric_group",
            "metric_key",
            "metric",
            "dataset",
            "arm",
            "value",
            "delta_vs_control",
            "oriented_delta_vs_control",
            "delta_vs_control_pp",
        ],
    )

    input_files = {
        str(path): sha256(path)
        for values in collected.values()
        for record in values.values()
        if (path := Path(record["source"])).is_file()
    }
    audit_payload = {
        "status": (
            "VALID_COMPLETE_FIXED_ENDPOINT_OBSERVATIONAL"
            if all(audit["complete"] for audit in audits.values())
            else "INCOMPLETE_PREVIEW"
        ),
        "admission": "OBSERVATIONAL_LEGACY_PROTOCOL_REPRODUCTION",
        "historical_semantics": (
            "Dataset scopes and metric definitions match the historical common-dataset report; "
            "the new values are fixed endpoints and are not best-over-curve selections."
        ),
        "no_global_overall_scalar_in_historical_report": True,
        "arms": audits,
        "historical_scope_file": str(args.historical_scope.resolve()),
        "historical_scope_sha256": sha256(args.historical_scope.resolve()),
        "input_sha256": input_files,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    focus = {
        "retrieval_recall_at_1",
        "clustering_nmi",
        "clustering_ari",
        "clustering_cluster_accuracy",
    }
    markdown = [
        "# HS6-L5 corrected Gram fixed-endpoint legacy dataset test",
        "",
        f"Status: `{audit_payload['status']}`.",
        "",
        "This report is observational. It reuses the historical common-dataset scopes, but "
        "compares fixed ck20007/ck20495 points rather than selecting a best checkpoint from a curve.",
        "The historical report did not define one scalar overall score; its `overall` artifact was "
        "the collection of per-metric dataset-equal means.",
        "",
        "## Retrieval and clustering",
        "",
        "| metric | arm | n | mean | delta vs matched control (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in historical_rows:
        if row["metric_key"] in focus:
            markdown.append(
                f"| {row['metric']} | {row['arm']} | {row['n_historical_datasets']} | "
                f"{row['mean']:.6f} | {row['delta_vs_control_pp']:+.4f} |"
            )
    markdown.extend(
        [
            "",
            "## All historical metrics",
            "",
            "See `historical_scope_fixed_endpoint_summary.csv`. Expanded current-registry common "
            "datasets are reported separately in `expanded_common_fixed_endpoint_summary.csv`.",
            "",
            "No downstream label selected the student start, either Gram anchor, loss weight, "
            "teacher refresh, or sample stream. Frozen probes are evaluation only.",
        ]
    )
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": audit_payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
