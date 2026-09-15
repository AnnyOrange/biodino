#!/usr/bin/env python3
"""Audit and summarize the matched HS6-L5 official-Gram causal screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/01_training_runs"
CONTROL_RUN = "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
OFFICIAL_RUN = "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
SPATIAL_ROOT = ROOT / "outputs/00_reports/hs6_l5_official_gram_spatial_u488_20260911"
RXRX_ROOT = ROOT / "outputs/02_eval_runs/hs6_l5_official_gram_u488_rxrx3_formal_20260911"
CLASS_ROOT = ROOT / "outputs/02_eval_runs/hs6_l5_official_gram_u488_peak_tasks_20260911"
CURVE_ROOT = ROOT / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
OUTPUT_ROOT = ROOT / "outputs/00_reports/hs6_l5_official_gram_causal_screen_20260911"

EXPECTED_CLASSIFICATION = (
    "retinamnist",
    "chammi-cp-task3",
    "bbbc048-cellcycle",
    "nct-crc-he",
    "dermamnist",
    "pneumoniamnist",
    "pathmnist",
    "midog25-atypical",
    "octmnist",
    "chammi-allen-task1",
)
CLASS_GROUPS = ("nct", "cp3", "peak7", "bbbc048")
TRAIN_METRICS = (
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "ibot_loss",
    "koleo_loss",
    "backbone_grad_norm",
    "total_loss",
)
SPATIAL_METRICS = ("true_minus_shifted", "true_minus_cross", "hit1", "mrr")
RXRX_RECALL = {"recall_at_1": 1, "recall_at_5": 5, "recall_at_10": 10}
BOOTSTRAP_SAMPLES = 50_000
SEED = 20260911


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(child, path))
    return output


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    return True


def paired_ci(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 1000):
        stop = min(start + 1000, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def exact_paired_binary_p(control: np.ndarray, official: np.ndarray) -> dict[str, float | int]:
    control = control.astype(bool)
    official = official.astype(bool)
    control_only = int(np.sum(control & ~official))
    official_only = int(np.sum(~control & official))
    discordant = control_only + official_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(control_only, official_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "control_only_successes": control_only,
        "official_only_successes": official_only,
        "discordant_queries": discordant,
        "two_sided_exact_p": p_value,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "first_100_mean": float(array[:100].mean()),
        "last_100_mean": float(array[-100:].mean()),
        "last": float(array[-1]),
    }


def analyze_training() -> dict[str, Any]:
    run_names = {"control": CONTROL_RUN, "official": OFFICIAL_RUN}
    configs: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for arm, run_name in run_names.items():
        run = RUN_ROOT / run_name
        configs[arm] = yaml.safe_load((run / "config.yaml").read_text())
        rows[arm] = [
            json.loads(line) for line in (run / "raw_loss_metrics.jsonl").read_text().splitlines()
        ]
        if len(rows[arm]) != 488:
            raise RuntimeError(f"{arm} has {len(rows[arm])} updates, expected 488")
        if (rows[arm][0]["optimizer_update"], rows[arm][-1]["optimizer_update"]) != (
            20008,
            20495,
        ):
            raise RuntimeError(f"{arm} optimizer-update range is invalid")
        if not finite_tree(rows[arm]):
            raise RuntimeError(f"{arm} contains a non-finite training value")

    digest_matches = [
        control["batch_sample_key_digest"] == official["batch_sample_key_digest"]
        for control, official in zip(rows["control"], rows["official"], strict=True)
    ]
    if not all(digest_matches):
        raise RuntimeError(f"only {sum(digest_matches)}/488 sample digests match")

    flat_configs = {arm: flatten(config) for arm, config in configs.items()}
    differing_config_keys = sorted(
        key
        for key in set(flat_configs["control"]) | set(flat_configs["official"])
        if flat_configs["control"].get(key) != flat_configs["official"].get(key)
    )
    expected_differences = ["gram.ckpt", "gram.use_loss", "train.output_dir"]
    if differing_config_keys != expected_differences:
        raise RuntimeError(f"unexpected config differences: {differing_config_keys}")

    arms = {
        arm: {
            "updates": len(arm_rows),
            "optimizer_update_first": int(arm_rows[0]["optimizer_update"]),
            "optimizer_update_last": int(arm_rows[-1]["optimizer_update"]),
            "effective_global_batch": int(arm_rows[0]["effective_global_batch_size"]),
            "metrics": {
                metric: summarize_values([float(row[metric]) for row in arm_rows])
                for metric in TRAIN_METRICS
            },
        }
        for arm, arm_rows in rows.items()
    }
    official_rows = rows["official"]
    arms["official"]["gram_loss"] = summarize_values(
        [float(row["gram_loss"]) for row in official_rows]
    )
    segments = []
    for low, high in ((20008, 20199), (20200, 20399), (20400, 20495)):
        values = [
            float(row["gram_loss"])
            for row in official_rows
            if low <= int(row["optimizer_update"]) <= high
        ]
        segments.append(
            {
                "optimizer_update_first": low,
                "optimizer_update_last": high,
                "updates": len(values),
                "first": values[0],
                "last": values[-1],
                "mean": float(np.mean(values)),
            }
        )
    arms["official"]["gram_segments"] = segments

    base_loss_deltas = {}
    for metric in TRAIN_METRICS[:-1]:
        delta = np.asarray(
            [
                float(official[metric]) - float(control[metric])
                for control, official in zip(rows["control"], rows["official"], strict=True)
            ]
        )
        base_loss_deltas[metric] = {
            "mean": float(delta.mean()),
            "last_100_mean": float(delta[-100:].mean()),
            "last": float(delta[-1]),
        }
    return {
        "status": "VALID_MATCHED_CAUSAL_SCREEN",
        "config_differences": differing_config_keys,
        "sample_digest_matches": int(sum(digest_matches)),
        "sample_digest_total": len(digest_matches),
        "nonfinite_values": 0,
        "arms": arms,
        "official_minus_control_base_loss": base_loss_deltas,
    }


def analyze_spatial() -> dict[str, Any]:
    summaries = {
        arm: json.loads((SPATIAL_ROOT / f"{arm}.json").read_text())
        for arm in ("control", "official")
    }
    records = {
        arm: json.loads((SPATIAL_ROOT / f"{arm}_records.json").read_text())
        for arm in ("control", "official")
    }
    if records["control"]["keys"] != records["official"]["keys"]:
        raise RuntimeError("spatial diagnostic image keys differ")
    if records["control"]["local_area"] != records["official"]["local_area"]:
        raise RuntimeError("spatial diagnostic crop areas differ")
    if any(summary["n_images"] != 128 for summary in summaries.values()):
        raise RuntimeError("spatial diagnostic does not contain 128 images per arm")

    layers: dict[str, Any] = {}
    for layer_index, layer in enumerate(("block_6", "block_12", "block_18", "block_24")):
        metrics = {}
        for metric_index, metric in enumerate(SPATIAL_METRICS):
            control = np.asarray(records["control"]["layers"][layer][metric], dtype=np.float64)
            official = np.asarray(records["official"]["layers"][layer][metric], dtype=np.float64)
            delta = official - control
            metrics[metric] = {
                "control_mean": float(control.mean()),
                "official_mean": float(official.mean()),
                "official_minus_control": float(delta.mean()),
                "paired_bootstrap_ci95": paired_ci(
                    delta, SEED + 100 * layer_index + metric_index
                ),
                "positive_images": int(np.sum(delta > 0)),
                "negative_images": int(np.sum(delta < 0)),
                "zero_images": int(np.sum(delta == 0)),
            }
        layers[layer] = metrics
    return {
        "status": "VALID_PAIRED_LABEL_FREE_DIAGNOSTIC",
        "paired_images": 128,
        "ordered_keys_exact_match": True,
        "local_areas_exact_match": True,
        "decision_gate": {arm: summary["decision_gate"] for arm, summary in summaries.items()},
        "layers": layers,
        "input_sha256": {
            path.name: sha256(path)
            for path in (
                SPATIAL_ROOT / "control.json",
                SPATIAL_ROOT / "control_records.json",
                SPATIAL_ROOT / "official.json",
                SPATIAL_ROOT / "official_records.json",
            )
        },
    }


def query_ranks(features: np.ndarray, labels: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    gallery_features = features[gallery].astype(np.float32)
    query_features = features[~gallery].astype(np.float32)
    gallery_features /= np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12
    query_features /= np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12
    order = np.argsort(-(query_features @ gallery_features.T), axis=1)
    gallery_labels = labels[gallery]
    query_labels = labels[~gallery]
    return np.asarray(
        [
            np.flatnonzero(gallery_labels[indices] == query_labels[index])[0] + 1
            for index, indices in enumerate(order)
        ],
        dtype=np.int32,
    )


def analyze_rxrx3() -> dict[str, Any]:
    comparison_path = RXRX_ROOT / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    if comparison.get("status") != "VALID_COMPLETE":
        raise RuntimeError("RxRx3 formal comparison is incomplete")
    data_root = ROOT / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
    metadata = json.loads((data_root / "metadata.json").read_text())
    labels = np.load(data_root / "labels.npy")
    gallery = np.asarray([row["split"] == "gallery" for row in metadata["rows"]])
    if (int(gallery.sum()), int((~gallery).sum())) != (734, 734):
        raise RuntimeError("RxRx3 split is not the locked 734/734 protocol")

    ranks = {}
    for arm in ("control", "official"):
        feature_path = RXRX_ROOT / "models" / f"hs6_l5_{arm}_gram_u488" / "rxrx3_features.npy"
        ranks[arm] = query_ranks(np.load(feature_path, mmap_mode="r"), labels, gallery)

    metrics: dict[str, Any] = {}
    for metric_index, metric in enumerate((*RXRX_RECALL, "mrr")):
        if metric in RXRX_RECALL:
            control = ranks["control"] <= RXRX_RECALL[metric]
            official = ranks["official"] <= RXRX_RECALL[metric]
        else:
            control = 1.0 / ranks["control"].astype(np.float64)
            official = 1.0 / ranks["official"].astype(np.float64)
        delta = official.astype(np.float64) - control.astype(np.float64)
        row: dict[str, Any] = {
            "control": float(np.mean(control)),
            "official": float(np.mean(official)),
            "official_minus_control": float(delta.mean()),
            "paired_bootstrap_ci95": paired_ci(delta, SEED + 1000 + metric_index),
        }
        if metric in RXRX_RECALL:
            row.update(exact_paired_binary_p(control, official))
        formal_delta = float(comparison["official_minus_control"][metric])
        if abs(row["official_minus_control"] - formal_delta) > (1e-12 if metric != "mrr" else 1e-6):
            raise RuntimeError(f"cached RxRx3 {metric} does not reproduce the formal result")
        metrics[metric] = row
    return {
        "status": "VALID_FORMAL_WITH_PAIRED_QUERY_ANALYSIS",
        "protocol_id": comparison["protocol_id"],
        "paired_queries": 734,
        "metrics": metrics,
        "nmi": {
            "control": comparison["results"]["control"]["nmi"],
            "official": comparison["results"]["official"]["nmi"],
            "official_minus_control": comparison["official_minus_control"]["nmi"],
            "warning": "Raw NMI is not used for inference in this two-samples-per-class protocol.",
        },
        "campaign_manifest_sha256": comparison["manifest_sha256"],
        "comparison_sha256": sha256(comparison_path),
    }


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def historical_classification(step: int) -> dict[str, float]:
    output: dict[str, float] = {}
    point = CURVE_ROOT / f"point_{step}"
    for path in point.glob("classification_*/bio_classification/*/*/summary.csv"):
        for row in read_summary_rows(path):
            dataset = row.get("dataset", "")
            if dataset in EXPECTED_CLASSIFICATION and not row.get("error"):
                output[dataset] = float(row["balanced_accuracy"])
    if set(output) != set(EXPECTED_CLASSIFICATION):
        missing = sorted(set(EXPECTED_CLASSIFICATION) - set(output))
        raise RuntimeError(f"historical ck{step} classification rows are missing: {missing}")
    return output


def analyze_classification() -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, str]]] = {"control": {}, "official": {}}
    summary_hashes: dict[str, str] = {}
    for group in CLASS_GROUPS:
        for arm in rows:
            path = CLASS_ROOT / group / arm / "summary.csv"
            if not path.is_file():
                raise RuntimeError(f"classification summary is missing: {path}")
            summary_hashes[str(path.relative_to(ROOT))] = sha256(path)
            for row in read_summary_rows(path):
                if row.get("error"):
                    raise RuntimeError(f"classification failed for {arm}/{row.get('dataset')}: {row['error']}")
                dataset = row["dataset"]
                if dataset in rows[arm]:
                    raise RuntimeError(f"duplicate classification row for {arm}/{dataset}")
                rows[arm][dataset] = row
    for arm, arm_rows in rows.items():
        if set(arm_rows) != set(EXPECTED_CLASSIFICATION):
            missing = sorted(set(EXPECTED_CLASSIFICATION) - set(arm_rows))
            extra = sorted(set(arm_rows) - set(EXPECTED_CLASSIFICATION))
            raise RuntimeError(f"{arm} classification coverage mismatch: missing={missing}, extra={extra}")

    historical = {step: historical_classification(step) for step in (20007, 20495)}
    datasets = {}
    for dataset in EXPECTED_CLASSIFICATION:
        control = float(rows["control"][dataset]["balanced_accuracy"])
        official = float(rows["official"][dataset]["balanced_accuracy"])
        datasets[dataset] = {
            "historical_ck20007": historical[20007][dataset],
            "historical_ck20495": historical[20495][dataset],
            "matched_control": control,
            "official_gram": official,
            "official_minus_control": official - control,
            "control_minus_ck20007": control - historical[20007][dataset],
            "official_minus_ck20007": official - historical[20007][dataset],
            "official_minus_historical_ck20495": official - historical[20495][dataset],
            "n_train": int(rows["official"][dataset]["n_train"]),
            "n_test": int(rows["official"][dataset]["n_test"]),
            "image_size": int(rows["official"][dataset]["image_size"]),
            "split": rows["official"][dataset]["split"],
        }
    deltas = np.asarray([row["official_minus_control"] for row in datasets.values()])
    return {
        "status": "VALID_POSTHOC_PEAK_TASK_STRESS_TEST",
        "datasets": datasets,
        "aggregate": {
            "datasets": len(datasets),
            "official_minus_control_mean_balanced_accuracy": float(deltas.mean()),
            "official_better": int(np.sum(deltas > 0)),
            "official_worse": int(np.sum(deltas < 0)),
            "exact_ties": int(np.sum(deltas == 0)),
        },
        "selection_warning": (
            "Nine datasets were selected after observing their checkpoint peaks; "
            "chammi-allen-task1 is an endpoint-peaking negative control."
        ),
        "summary_sha256": summary_hashes,
    }


def build_markdown(report: dict[str, Any]) -> str:
    training = report["training"]
    spatial = report["spatial"]["layers"]["block_24"]
    rxrx = report["rxrx3"]["metrics"]
    classification = report["classification"]
    lines = [
        "# HS6-L5 official Gram causal screen",
        "",
        "Status: `VALID_COMPLETE`; matched 488-update short screen from full ck20007.",
        "",
        "## Integrity",
        "",
        (
            f"Control and official Gram have {training['sample_digest_matches']}/"
            f"{training['sample_digest_total']} identical per-update sample digests. "
            "Their resolved configs differ only in `gram.use_loss`, `gram.ckpt`, and "
            "`train.output_dir`; all recorded values are finite."
        ),
        "",
        "This official-structure short arm uses ck7807, clean 512-pixel teacher crops, "
        "normalized within-image patch Gram, mature-stage weight 2, and proportionally "
        "adapted teacher refreshes after completed updates 20200 and 20400.",
        "",
        "## Results",
        "",
        "| Readout | Control | Official Gram | Delta | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in (("Spatial block24 true-shift", "true_minus_shifted"), ("Spatial block24 true-cross", "true_minus_cross"), ("Spatial block24 Hit@1", "hit1"), ("Spatial block24 MRR", "mrr")):
        row = spatial[metric]
        low, high = row["paired_bootstrap_ci95"]
        lines.append(
            f"| {label} | {row['control_mean']:.6f} | {row['official_mean']:.6f} | "
            f"{row['official_minus_control']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    for label, metric in (("RxRx3 R@1", "recall_at_1"), ("RxRx3 R@5", "recall_at_5"), ("RxRx3 R@10", "recall_at_10"), ("RxRx3 MRR", "mrr")):
        row = rxrx[metric]
        low, high = row["paired_bootstrap_ci95"]
        lines.append(
            f"| {label} | {row['control']:.6f} | {row['official']:.6f} | "
            f"{row['official_minus_control']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "| Frozen classification | ck20007 | Historical ck20495 | Matched control | Official Gram | Gram-control |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, row in classification["datasets"].items():
        lines.append(
            f"| {dataset} | {row['historical_ck20007']:.6f} | "
            f"{row['historical_ck20495']:.6f} | {row['matched_control']:.6f} | "
            f"{row['official_gram']:.6f} | {row['official_minus_control']:+.6f} |"
        )
    aggregate = classification["aggregate"]
    gram = training["arms"]["official"]["gram_loss"]
    lines.extend(
        [
            "",
            (
                f"Across the ten selected classification tasks, official Gram is better/worse/tied on "
                f"{aggregate['official_better']}/{aggregate['official_worse']}/{aggregate['exact_ties']}; "
                f"mean balanced-accuracy delta is "
                f"{aggregate['official_minus_control_mean_balanced_accuracy']:+.6f}."
            ),
            "",
            "## Interpretation",
            "",
            (
                f"The Gram loss falls from {gram['first_100_mean']:.6f} in the first 100 updates "
                f"to {gram['last_100_mean']:.6f} in the last 100, so optimization is active. "
                "The primary question is whether its within-image spatial gain survives without "
                "damaging cross-image and downstream discrimination."
            ),
            "",
            report["decision"],
            "",
            "This is a kill/advance screen, not a production-scale estimate: effective global batch "
            "is 64 and only 31,232 images are consumed per arm. The classification stress tasks are "
            "post-hoc selected, and there is one training seed. The SSL update itself uses no labels; "
            "labels appear only in frozen downstream evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output = args.output.resolve()

    report = {
        "status": "VALID_COMPLETE",
        "screen": {
            "student_resume_checkpoint": 20007,
            "final_checkpoint": 20495,
            "optimizer_updates": 488,
            "effective_global_batch": 64,
            "images_per_arm": 488 * 64,
            "gram_anchor_checkpoint": 7807,
            "official_gram_structure": {
                "student_crop": 256,
                "teacher_crop": 512,
                "teacher_crop_distorted": False,
                "normalized": True,
                "img_level": True,
                "loss_weight": 2.0,
                "teacher_refresh_completed_updates": [20200, 20400],
            },
        },
        "training": analyze_training(),
        "spatial": analyze_spatial(),
        "rxrx3": analyze_rxrx3(),
        "classification": analyze_classification(),
    }
    spatial_margin = report["spatial"]["layers"]["block_24"]["true_minus_shifted"]
    spatial_cross = report["spatial"]["layers"]["block_24"]["true_minus_cross"]
    rxrx_mrr = report["rxrx3"]["metrics"]["mrr"]
    class_delta = report["classification"]["aggregate"][
        "official_minus_control_mean_balanced_accuracy"
    ]
    if (
        spatial_margin["paired_bootstrap_ci95"][0] > 0
        and spatial_cross["paired_bootstrap_ci95"][0] >= 0
        and rxrx_mrr["official_minus_control"] >= 0
        and class_delta >= 0
    ):
        report["decision"] = "Advance the official Gram arm to a production-global-batch confirmation."
        report["decision_code"] = "ADVANCE_OFFICIAL_GRAM"
    else:
        report["decision"] = (
            "Do not scale the official Gram arm as-is. Treat any within-image spatial gain together "
            "with the measured cross-image/retrieval cost, then test a separately constrained "
            "fully self-supervised relation-preservation arm before committing a large run."
        )
        report["decision_code"] = "DO_NOT_SCALE_OFFICIAL_GRAM_AS_IS"

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "analysis.json", report)
    (args.output / "README.md").write_text(build_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
