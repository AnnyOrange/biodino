#!/usr/bin/env python3
"""Audit the matched HS6-L5 official-Gram anchor ablation."""

from __future__ import annotations

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
RUNS = {
    "control": "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
    "anchor7807": "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
    "anchor17079": "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911",
}
SPATIAL_ROOT = ROOT / "outputs/00_reports/hs6_l5_official_gram_spatial_u488_20260911"
SPATIAL_FILES = {
    "control": "control",
    "anchor7807": "official",
    "anchor17079": "anchor17079",
}
PRIMARY_RXRX_ROOT = ROOT / "outputs/02_eval_runs/hs6_l5_official_gram_u488_rxrx3_formal_20260911"
ANCHOR_RXRX_ROOT = ROOT / (
    "outputs/02_eval_runs/"
    "hs6_l5_official_gram_anchor7807_vs_17079_u488_rxrx3_formal_20260911"
)
PRIMARY_CLASS_ROOT = ROOT / (
    "outputs/02_eval_runs/hs6_l5_official_gram_u488_peak_tasks_20260911"
)
ANCHOR_CLASS_ROOT = ROOT / (
    "outputs/02_eval_runs/hs6_l5_official_gram_a17079_u488_peak_tasks_20260911"
)
DATA_ROOT = ROOT / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
OUTPUT_ROOT = ROOT / "outputs/00_reports/hs6_l5_gram_anchor_ablation_20260911"

CLASS_GROUPS = ("nct", "cp3", "peak7", "bbbc048")
CLASS_DATASETS = (
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
SPATIAL_METRICS = (
    "true_cosine",
    "shifted_cosine",
    "cross_cosine",
    "true_minus_shifted",
    "true_minus_cross",
    "hit1",
    "mrr",
)
RXRX_METRICS = {"recall_at_1": 1, "recall_at_5": 5, "recall_at_10": 10}
COMPARISONS = (
    ("control", "anchor7807"),
    ("control", "anchor17079"),
    ("anchor7807", "anchor17079"),
)
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


def exact_paired_binary_p(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    baseline = baseline.astype(bool)
    candidate = candidate.astype(bool)
    baseline_only = int(np.sum(baseline & ~candidate))
    candidate_only = int(np.sum(~baseline & candidate))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(baseline_only, candidate_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_only_successes": baseline_only,
        "candidate_only_successes": candidate_only,
        "discordant_queries": discordant,
        "two_sided_exact_p": p_value,
    }


def comparison_key(baseline: str, candidate: str) -> str:
    return f"{candidate}_minus_{baseline}"


def analyze_training() -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for arm, run_name in RUNS.items():
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
            raise RuntimeError(f"{arm} has the wrong optimizer-update range")
        if not finite_tree(rows[arm]):
            raise RuntimeError(f"{arm} contains a non-finite value")

    digest_matches = {
        arm: sum(
            control["batch_sample_key_digest"] == candidate["batch_sample_key_digest"]
            for control, candidate in zip(rows["control"], rows[arm], strict=True)
        )
        for arm in ("anchor7807", "anchor17079")
    }
    if set(digest_matches.values()) != {488}:
        raise RuntimeError(f"training streams are not exactly matched: {digest_matches}")

    flat = {arm: flatten(config) for arm, config in configs.items()}
    config_differences = {}
    for baseline, candidate in COMPARISONS:
        keys = sorted(
            key
            for key in set(flat[baseline]) | set(flat[candidate])
            if flat[baseline].get(key) != flat[candidate].get(key)
        )
        config_differences[comparison_key(baseline, candidate)] = keys
    if config_differences["anchor17079_minus_anchor7807"] != [
        "gram.ckpt",
        "train.output_dir",
    ]:
        raise RuntimeError("anchor arms differ in more than the Gram checkpoint and output path")

    summaries = {}
    for arm, arm_rows in rows.items():
        summary: dict[str, Any] = {
            "updates": len(arm_rows),
            "optimizer_update_first": int(arm_rows[0]["optimizer_update"]),
            "optimizer_update_last": int(arm_rows[-1]["optimizer_update"]),
            "effective_global_batch": int(arm_rows[0]["effective_global_batch_size"]),
        }
        if arm != "control":
            gram = np.asarray([float(row["gram_loss"]) for row in arm_rows])
            summary["gram_loss"] = {
                "mean": float(gram.mean()),
                "first": float(gram[0]),
                "first_100_mean": float(gram[:100].mean()),
                "last_100_mean": float(gram[-100:].mean()),
                "last": float(gram[-1]),
            }
        summaries[arm] = summary
    return {
        "status": "VALID_THREE_ARM_MATCHED_TRAINING",
        "sample_digest_matches_vs_control": digest_matches,
        "config_differences": config_differences,
        "arms": summaries,
    }


def analyze_spatial() -> dict[str, Any]:
    records = {
        arm: json.loads((SPATIAL_ROOT / f"{stem}_records.json").read_text())
        for arm, stem in SPATIAL_FILES.items()
    }
    reference = records["control"]
    if len(reference["keys"]) != 128:
        raise RuntimeError(f"spatial diagnostic has {len(reference['keys'])} images, expected 128")
    if any(record["keys"] != reference["keys"] for record in records.values()):
        raise RuntimeError("spatial image keys differ between arms")
    if any(record["local_area"] != reference["local_area"] for record in records.values()):
        raise RuntimeError("spatial crop areas differ between arms")

    layers: dict[str, Any] = {}
    for layer_index, layer in enumerate(("block_6", "block_12", "block_18", "block_24")):
        arms = {
            arm: {
                metric: float(np.mean(record["layers"][layer][metric]))
                for metric in SPATIAL_METRICS
            }
            for arm, record in records.items()
        }
        comparisons: dict[str, Any] = {}
        for comparison_index, (baseline, candidate) in enumerate(COMPARISONS):
            metrics = {}
            for metric_index, metric in enumerate(SPATIAL_METRICS):
                baseline_values = np.asarray(
                    records[baseline]["layers"][layer][metric], dtype=np.float64
                )
                candidate_values = np.asarray(
                    records[candidate]["layers"][layer][metric], dtype=np.float64
                )
                delta = candidate_values - baseline_values
                metrics[metric] = {
                    "mean": float(delta.mean()),
                    "paired_bootstrap_ci95": paired_ci(
                        delta,
                        SEED + 10_000 * layer_index + 1000 * comparison_index + metric_index,
                    ),
                    "positive_images": int(np.sum(delta > 0)),
                    "negative_images": int(np.sum(delta < 0)),
                }
            comparisons[comparison_key(baseline, candidate)] = metrics
        layers[layer] = {"arms": arms, "comparisons": comparisons}
    return {
        "status": "VALID_PAIRED_LABEL_FREE_DIAGNOSTIC",
        "paired_images": len(reference["keys"]),
        "ordered_keys_exact_match": True,
        "local_areas_exact_match": True,
        "layers": layers,
        "input_sha256": {
            path.name: sha256(path)
            for stem in SPATIAL_FILES.values()
            for path in (
                SPATIAL_ROOT / f"{stem}.json",
                SPATIAL_ROOT / f"{stem}_records.json",
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
    primary = json.loads((PRIMARY_RXRX_ROOT / "comparison.json").read_text())
    anchor = json.loads((ANCHOR_RXRX_ROOT / "comparison.json").read_text())
    if primary.get("status") != "VALID_COMPLETE" or anchor.get("status") != "VALID_COMPLETE":
        raise RuntimeError("RxRx3 formal comparisons are incomplete")

    metadata = json.loads((DATA_ROOT / "metadata.json").read_text())
    labels = np.load(DATA_ROOT / "labels.npy")
    gallery = np.asarray([row["split"] == "gallery" for row in metadata["rows"]])
    feature_paths = {
        "control": PRIMARY_RXRX_ROOT / "models/hs6_l5_control_gram_u488/rxrx3_features.npy",
        "anchor7807": ANCHOR_RXRX_ROOT
        / "models/hs6_l5_anchor7807_gram_u488/rxrx3_features.npy",
        "anchor17079": ANCHOR_RXRX_ROOT
        / "models/hs6_l5_anchor17079_gram_u488/rxrx3_features.npy",
    }
    ranks = {
        arm: query_ranks(np.load(path, mmap_mode="r"), labels, gallery)
        for arm, path in feature_paths.items()
    }
    arms = {
        "control": primary["results"]["control"],
        "anchor7807": anchor["results"]["anchor7807"],
        "anchor17079": anchor["results"]["anchor17079"],
    }
    comparisons: dict[str, Any] = {}
    for comparison_index, (baseline, candidate) in enumerate(COMPARISONS):
        metrics = {}
        for metric_index, metric in enumerate((*RXRX_METRICS, "mrr")):
            if metric in RXRX_METRICS:
                baseline_values = ranks[baseline] <= RXRX_METRICS[metric]
                candidate_values = ranks[candidate] <= RXRX_METRICS[metric]
            else:
                baseline_values = 1.0 / ranks[baseline].astype(np.float64)
                candidate_values = 1.0 / ranks[candidate].astype(np.float64)
            delta = candidate_values.astype(np.float64) - baseline_values.astype(np.float64)
            row: dict[str, Any] = {
                "baseline": float(np.mean(baseline_values)),
                "candidate": float(np.mean(candidate_values)),
                "mean": float(delta.mean()),
                "paired_bootstrap_ci95": paired_ci(
                    delta, SEED + 50_000 + 1000 * comparison_index + metric_index
                ),
            }
            if metric in RXRX_METRICS:
                row.update(exact_paired_binary_p(baseline_values, candidate_values))
            metrics[metric] = row
        metrics["nmi"] = {
            "baseline": float(arms[baseline]["nmi"]),
            "candidate": float(arms[candidate]["nmi"]),
            "mean": float(arms[candidate]["nmi"] - arms[baseline]["nmi"]),
            "warning": "NMI has no paired-query confidence interval.",
        }
        comparisons[comparison_key(baseline, candidate)] = metrics
    return {
        "status": "VALID_FORMAL_WITH_PAIRED_QUERY_ANALYSIS",
        "protocol_id": primary["protocol_id"],
        "paired_queries": int((~gallery).sum()),
        "arms": arms,
        "comparisons": comparisons,
        "input_sha256": {
            "primary_comparison": sha256(PRIMARY_RXRX_ROOT / "comparison.json"),
            "anchor_comparison": sha256(ANCHOR_RXRX_ROOT / "comparison.json"),
        },
    }


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_classification() -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, str]]] = {arm: {} for arm in RUNS}
    hashes = {}
    expected_inputs = {
        arm: {
            "checkpoint": (
                RUN_ROOT / run_name / "eval/training_20495/teacher_checkpoint.pth"
            ).resolve(),
            "train_config": (RUN_ROOT / run_name / "config.yaml").resolve(),
        }
        for arm, run_name in RUNS.items()
    }
    for group in CLASS_GROUPS:
        paths = {
            "control": PRIMARY_CLASS_ROOT / group / "control/summary.csv",
            "anchor7807": PRIMARY_CLASS_ROOT / group / "official/summary.csv",
            "anchor17079": ANCHOR_CLASS_ROOT / group / "anchor17079/summary.csv",
        }
        for arm, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"classification summary is missing: {path}")
            hashes[str(path.relative_to(ROOT))] = sha256(path)
            for row in read_summary(path):
                if row.get("error"):
                    raise RuntimeError(f"classification failed for {arm}/{row.get('dataset')}")
                checkpoint = Path(row["checkpoint"])
                if not checkpoint.samefile(expected_inputs[arm]["checkpoint"]):
                    raise RuntimeError(f"classification checkpoint mismatch for {arm}")
                if Path(row["train_config"]).resolve() != expected_inputs[arm]["train_config"]:
                    raise RuntimeError(f"classification config mismatch for {arm}")
                protocol = {
                    "resolution_protocol": row["resolution_protocol"],
                    "batch_size": row["batch_size"],
                    "seed": row["seed"],
                    "channel_policy": row["channel_policy"],
                    "channel_tta_samples": row["channel_tta_samples"],
                    "channel_policy_seed": row["channel_policy_seed"],
                }
                expected_protocol = {
                    "resolution_protocol": "best",
                    "batch_size": "64",
                    "seed": "0",
                    "channel_policy": "auto",
                    "channel_tta_samples": "8",
                    "channel_policy_seed": "0",
                }
                if protocol != expected_protocol:
                    raise RuntimeError(f"classification protocol mismatch for {arm}: {protocol}")
                dataset = row["dataset"]
                if dataset in rows[arm]:
                    raise RuntimeError(f"duplicate classification row for {arm}/{dataset}")
                rows[arm][dataset] = row
    for arm, arm_rows in rows.items():
        if set(arm_rows) != set(CLASS_DATASETS):
            missing = sorted(set(CLASS_DATASETS) - set(arm_rows))
            extra = sorted(set(arm_rows) - set(CLASS_DATASETS))
            raise RuntimeError(f"{arm} classification mismatch: missing={missing}, extra={extra}")

    datasets = {}
    for dataset in CLASS_DATASETS:
        values = {
            arm: float(rows[arm][dataset]["balanced_accuracy"])
            for arm in RUNS
        }
        matched_fields = ("n_train", "n_test", "split", "image_size", "resize_size")
        for field in matched_fields:
            observed = {rows[arm][dataset][field] for arm in RUNS}
            if len(observed) != 1:
                raise RuntimeError(f"{dataset} differs across arms in {field}: {observed}")
        datasets[dataset] = {
            "balanced_accuracy": values,
            "deltas": {
                comparison_key(baseline, candidate): values[candidate] - values[baseline]
                for baseline, candidate in COMPARISONS
            },
            "n_train": int(rows["control"][dataset]["n_train"]),
            "n_test": int(rows["control"][dataset]["n_test"]),
            "split": rows["control"][dataset]["split"],
            "image_size": int(rows["control"][dataset]["image_size"]),
        }
    aggregates = {}
    for baseline, candidate in COMPARISONS:
        key = comparison_key(baseline, candidate)
        deltas = np.asarray([row["deltas"][key] for row in datasets.values()])
        aggregates[key] = {
            "mean_balanced_accuracy_delta": float(deltas.mean()),
            "median_balanced_accuracy_delta": float(np.median(deltas)),
            "better": int(np.sum(deltas > 0)),
            "worse": int(np.sum(deltas < 0)),
            "ties": int(np.sum(deltas == 0)),
        }
    return {
        "status": "VALID_POSTHOC_TEN_TASK_STRESS_TEST",
        "datasets": datasets,
        "aggregates": aggregates,
        "selection_warning": (
            "Nine tasks were selected after observing historical checkpoint peaks; "
            "chammi-allen-task1 is an endpoint-peaking negative control."
        ),
        "summary_sha256": hashes,
    }


def format_ci(row: dict[str, Any]) -> str:
    low, high = row["paired_bootstrap_ci95"]
    return f"{row['mean']:+.6f} [{low:+.6f}, {high:+.6f}]"


def build_markdown(report: dict[str, Any]) -> str:
    spatial = report["spatial"]["layers"]["block_24"]
    rxrx = report["rxrx3"]
    classification = report["classification"]
    lines = [
        "# HS6-L5 official Gram anchor ablation",
        "",
        "Status: `VALID_COMPLETE`; three matched 488-update arms from full ck20007.",
        "",
        "## Integrity",
        "",
        (
            "All arms contain 488 finite updates (20008--20495) and have 488/488 "
            "identical per-update sample digests. The two Gram arms differ only in "
            "`gram.ckpt` and `train.output_dir`."
        ),
        "",
        "Both Gram arms use the published structure adapted to this short refinement window: "
        "student 256, clean teacher 512, normalized within-image patch Gram, weight 2, "
        "and teacher refreshes after completed updates 20200 and 20400. No inter-image "
        "experimental loss is active.",
        "",
        "## Formal RxRx3",
        "",
        "| Arm | R@1 | R@5 | R@10 | MRR | NMI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in RUNS:
        row = rxrx["arms"][arm]
        lines.append(
            f"| {arm} | {row['recall_at_1']:.6f} | {row['recall_at_5']:.6f} | "
            f"{row['recall_at_10']:.6f} | {row['mrr']:.6f} | {row['nmi']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Paired comparison | R@1 delta [95% CI] | R@5 delta [95% CI] | R@10 delta [95% CI] | MRR delta [95% CI] | NMI delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, metrics in rxrx["comparisons"].items():
        lines.append(
            f"| {key} | {format_ci(metrics['recall_at_1'])} | "
            f"{format_ci(metrics['recall_at_5'])} | {format_ci(metrics['recall_at_10'])} | "
            f"{format_ci(metrics['mrr'])} | {metrics['nmi']['mean']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Label-free spatial diagnostic",
            "",
            "| Arm | True-shift | True-cross | Hit@1 | MRR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, metrics in spatial["arms"].items():
        lines.append(
            f"| {arm} | {metrics['true_minus_shifted']:.6f} | "
            f"{metrics['true_minus_cross']:.6f} | {metrics['hit1']:.6f} | {metrics['mrr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Paired comparison | True-shift delta [95% CI] | True-cross delta [95% CI] | Hit@1 delta [95% CI] | MRR delta [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, metrics in spatial["comparisons"].items():
        lines.append(
            f"| {key} | {format_ci(metrics['true_minus_shifted'])} | "
            f"{format_ci(metrics['true_minus_cross'])} | {format_ci(metrics['hit1'])} | "
            f"{format_ci(metrics['mrr'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen classification",
            "",
            "| Dataset | Control | Anchor 7807 | Anchor 17079 | 7807-control | 17079-control | 17079-7807 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, row in classification["datasets"].items():
        values = row["balanced_accuracy"]
        deltas = row["deltas"]
        lines.append(
            f"| {dataset} | {values['control']:.6f} | {values['anchor7807']:.6f} | "
            f"{values['anchor17079']:.6f} | {deltas['anchor7807_minus_control']:+.6f} | "
            f"{deltas['anchor17079_minus_control']:+.6f} | "
            f"{deltas['anchor17079_minus_anchor7807']:+.6f} |"
        )
    lines.extend(["", "Classification macro over these post-hoc stress tasks:"])
    for key, row in classification["aggregates"].items():
        lines.append(
            f"- `{key}`: mean {row['mean_balanced_accuracy_delta']:+.6f}, "
            f"median {row['median_balanced_accuracy_delta']:+.6f}, "
            f"better/worse/tied {row['better']}/{row['worse']}/{row['ties']}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "This is a kill/advance screen at effective global batch 64 (31,232 images per arm), "
            "not a production-scale estimate. Classification tasks are post-hoc selected and there "
            "is one training seed. SSL training is fully self-supervised; labels are used only by "
            "the frozen downstream probes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = {
        "status": "VALID_COMPLETE",
        "screen": {
            "student_resume_checkpoint": 20007,
            "final_checkpoint": 20495,
            "optimizer_updates": 488,
            "effective_global_batch": 64,
            "images_per_arm": 31_232,
            "gram_anchors": [7807, 17079],
            "inter_image_loss_weight": 0.0,
        },
        "training": analyze_training(),
        "spatial": analyze_spatial(),
        "rxrx3": analyze_rxrx3(),
        "classification": analyze_classification(),
    }
    spatial_vs_control = report["spatial"]["layers"]["block_24"]["comparisons"][
        "anchor17079_minus_control"
    ]
    rxrx_anchor_delta = report["rxrx3"]["comparisons"][
        "anchor17079_minus_anchor7807"
    ]["mrr"]["mean"]
    class_anchor_delta = report["classification"]["aggregates"][
        "anchor17079_minus_anchor7807"
    ]["mean_balanced_accuracy_delta"]
    if spatial_vs_control["true_minus_cross"]["paired_bootstrap_ci95"][1] < 0:
        report["decision_code"] = "DO_NOT_SCALE_OFFICIAL_GRAM_AS_IS"
        report["decision"] = (
            "Do not scale either official-Gram arm as-is: the newer ck17079 anchor may recover "
            "global metrics relative to ck7807, but its final-layer cross-image spatial margin "
            "remains significantly below the matched no-Gram control. Use the ck17079 result to "
            "motivate a separately matched, fully self-supervised cross-image relation-preservation "
            "ablation before any production-global-batch run."
        )
    elif rxrx_anchor_delta >= 0 and class_anchor_delta >= 0:
        report["decision_code"] = "ADVANCE_ANCHOR17079_CONFIRMATION"
        report["decision"] = (
            "Advance ck17079, not ck7807, to a production-global-batch confirmation; retain the "
            "no-Gram arm and all paired diagnostics."
        )
    else:
        report["decision_code"] = "DO_NOT_SCALE_ANCHOR_SCREEN"
        report["decision"] = (
            "Do not scale the anchor screen: ck17079 does not consistently improve the global and "
            "spatial evidence relative to ck7807 and the no-Gram control."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "analysis.json", report)
    (OUTPUT_ROOT / "README.md").write_text(build_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
