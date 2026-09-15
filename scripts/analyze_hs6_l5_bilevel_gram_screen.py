#!/usr/bin/env python3
"""Audit the exploratory HS6-L5 inter-image relation Gram screen."""

from __future__ import annotations

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
    "anchor17079": "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911",
    "bilevel17079": "HS6_L5_ck20007_bilevel_gram_a17079_u488_gb64_4x3090qi_bilevel_20260911",
}
SPATIAL_ROOT = ROOT / "outputs/00_reports/hs6_l5_official_gram_spatial_u488_20260911"
RXRX_ROOT = ROOT / (
    "outputs/02_eval_runs/hs6_l5_gram_anchor17079_vs_bilevel_u488_rxrx3_formal_20260911"
)
DATA_ROOT = ROOT / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
OUTPUT_ROOT = ROOT / "outputs/00_reports/hs6_l5_bilevel_gram_screen_20260911"
METRICS = ("true_minus_shifted", "true_minus_cross", "hit1", "mrr")
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


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "first_100_mean": float(array[:100].mean()),
        "last_100_mean": float(array[-100:].mean()),
        "last": float(array[-1]),
    }


def analyze_training() -> dict[str, Any]:
    rows = {}
    configs = {}
    for arm, run_name in RUNS.items():
        run = RUN_ROOT / run_name
        rows[arm] = [
            json.loads(line) for line in (run / "raw_loss_metrics.jsonl").read_text().splitlines()
        ]
        configs[arm] = yaml.safe_load((run / "config.yaml").read_text())
        if len(rows[arm]) != 488 or not finite_tree(rows[arm]):
            raise RuntimeError(f"invalid training rows for {arm}")
        if (rows[arm][0]["optimizer_update"], rows[arm][-1]["optimizer_update"]) != (
            20008,
            20495,
        ):
            raise RuntimeError(f"wrong optimizer-update range for {arm}")

    digest_matches = sum(
        baseline["batch_sample_key_digest"] == candidate["batch_sample_key_digest"]
        for baseline, candidate in zip(rows["anchor17079"], rows["bilevel17079"], strict=True)
    )
    if digest_matches != 488:
        raise RuntimeError(f"only {digest_matches}/488 sample digests match")
    flat = {arm: flatten(config) for arm, config in configs.items()}
    differences = sorted(
        key
        for key in set(flat["anchor17079"]) | set(flat["bilevel17079"])
        if flat["anchor17079"].get(key) != flat["bilevel17079"].get(key)
    )
    if differences != ["gram.inter_image_loss_weight", "train.output_dir"]:
        raise RuntimeError(f"unexpected config differences: {differences}")

    arms = {}
    for arm, arm_rows in rows.items():
        arms[arm] = {
            "gram_loss": summarize([float(row["gram_loss"]) for row in arm_rows]),
            "base_losses": {
                metric: summarize([float(row[metric]) for row in arm_rows])
                for metric in (
                    "dino_local_crops_loss",
                    "dino_global_crops_loss",
                    "ibot_loss",
                    "koleo_loss",
                    "backbone_grad_norm",
                )
            },
        }
    arms["bilevel17079"]["inter_image_loss"] = summarize(
        [float(row["gram_inter_image_loss"]) for row in rows["bilevel17079"]]
    )
    return {
        "status": "VALID_MATCHED_TRAINING",
        "updates": 488,
        "sample_digest_matches": digest_matches,
        "config_differences": differences,
        "arms": arms,
    }


def analyze_spatial() -> dict[str, Any]:
    paths = {
        "control": SPATIAL_ROOT / "control_records.json",
        "anchor17079": SPATIAL_ROOT / "anchor17079_records.json",
        "bilevel17079": SPATIAL_ROOT / "bilevel17079_records.json",
    }
    records = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    reference = records["control"]
    if len(reference["keys"]) != 128:
        raise RuntimeError("spatial reference does not have 128 images")
    if any(record["keys"] != reference["keys"] for record in records.values()):
        raise RuntimeError("spatial keys differ")
    if any(record["local_area"] != reference["local_area"] for record in records.values()):
        raise RuntimeError("spatial crop areas differ")

    layer = "block_24"
    arms = {
        arm: {
            metric: float(np.mean(record["layers"][layer][metric]))
            for metric in METRICS
        }
        for arm, record in records.items()
    }
    comparisons = {}
    for comparison_index, baseline in enumerate(("anchor17079", "control")):
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            baseline_values = np.asarray(records[baseline]["layers"][layer][metric])
            candidate_values = np.asarray(records["bilevel17079"]["layers"][layer][metric])
            delta = candidate_values - baseline_values
            metrics[metric] = {
                "mean": float(delta.mean()),
                "paired_bootstrap_ci95": paired_ci(
                    delta, SEED + 1000 * comparison_index + metric_index
                ),
            }
        comparisons[f"bilevel17079_minus_{baseline}"] = metrics
    return {
        "status": "VALID_PAIRED_LABEL_FREE_DIAGNOSTIC",
        "paired_images": 128,
        "arms": arms,
        "comparisons": comparisons,
        "input_sha256": {path.name: sha256(path) for path in paths.values()},
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
        raise RuntimeError("RxRx3 comparison is incomplete")
    metadata = json.loads((DATA_ROOT / "metadata.json").read_text())
    labels = np.load(DATA_ROOT / "labels.npy")
    gallery = np.asarray([row["split"] == "gallery" for row in metadata["rows"]])
    feature_paths = {
        "anchor17079": RXRX_ROOT
        / "models/hs6_l5_anchor17079_gram_u488/rxrx3_features.npy",
        "bilevel17079": RXRX_ROOT
        / "models/hs6_l5_bilevel17079_gram_u488/rxrx3_features.npy",
    }
    ranks = {
        arm: query_ranks(np.load(path, mmap_mode="r"), labels, gallery)
        for arm, path in feature_paths.items()
    }
    metrics = {}
    for metric_index, metric in enumerate((*RXRX_RECALL, "mrr")):
        if metric in RXRX_RECALL:
            baseline = ranks["anchor17079"] <= RXRX_RECALL[metric]
            candidate = ranks["bilevel17079"] <= RXRX_RECALL[metric]
        else:
            baseline = 1.0 / ranks["anchor17079"].astype(np.float64)
            candidate = 1.0 / ranks["bilevel17079"].astype(np.float64)
        delta = candidate.astype(np.float64) - baseline.astype(np.float64)
        row: dict[str, Any] = {
            "baseline": float(np.mean(baseline)),
            "candidate": float(np.mean(candidate)),
            "mean": float(delta.mean()),
            "paired_bootstrap_ci95": paired_ci(delta, SEED + 10_000 + metric_index),
        }
        if metric in RXRX_RECALL:
            row.update(exact_paired_binary_p(baseline, candidate))
        metrics[metric] = row
    formal = comparison["results"]
    metrics["nmi"] = {
        "baseline": float(formal["anchor17079"]["nmi"]),
        "candidate": float(formal["bilevel17079"]["nmi"]),
        "mean": float(formal["bilevel17079"]["nmi"] - formal["anchor17079"]["nmi"]),
        "warning": "NMI has no paired-query confidence interval.",
    }
    return {
        "status": "VALID_FORMAL_WITH_PAIRED_QUERY_ANALYSIS",
        "paired_queries": int((~gallery).sum()),
        "protocol_id": comparison["protocol_id"],
        "metrics": metrics,
        "comparison_sha256": sha256(comparison_path),
    }


def format_ci(row: dict[str, Any]) -> str:
    low, high = row["paired_bootstrap_ci95"]
    return f"{row['mean']:+.6f} [{low:+.6f}, {high:+.6f}]"


def build_markdown(report: dict[str, Any]) -> str:
    training = report["training"]
    spatial = report["spatial"]
    rxrx = report["rxrx3"]["metrics"]
    anchor = training["arms"]["anchor17079"]
    bilevel = training["arms"]["bilevel17079"]
    spatial_delta = spatial["comparisons"]["bilevel17079_minus_anchor17079"]
    lines = [
        "# HS6-L5 inter-image relation Gram screen",
        "",
        "Status: `VALID_COMPLETE_EXPLORATORY_KILL_SCREEN`.",
        "",
        "The official ck17079 arm and the bi-level arm have 488/488 identical sample "
        "digests. Resolved configs differ only in `gram.inter_image_loss_weight` (0 versus "
        "0.25) and `train.output_dir`. All values are finite.",
        "",
        "The candidate adds a per-global-crop, per-rank CLS sample-relation Gram to the "
        "same ck17079 teacher used by the official within-image patch Gram. It remains fully "
        "self-supervised and is not mixed into the official baseline.",
        "",
        "## Optimization",
        "",
        "| Loss | Official first/last 100 | Bi-level first/last 100 |",
        "|---|---:|---:|",
        (
            f"| Patch Gram | {anchor['gram_loss']['first_100_mean']:.6f} / "
            f"{anchor['gram_loss']['last_100_mean']:.6f} | "
            f"{bilevel['gram_loss']['first_100_mean']:.6f} / "
            f"{bilevel['gram_loss']['last_100_mean']:.6f} |"
        ),
        (
            "| Inter-image relation | n/a | "
            f"{bilevel['inter_image_loss']['first_100_mean']:.6f} / "
            f"{bilevel['inter_image_loss']['last_100_mean']:.6f} |"
        ),
        "",
        "## Direct gates",
        "",
        "| RxRx3 metric | Official ck17079 | Bi-level | Delta [paired 95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for metric in (*RXRX_RECALL, "mrr"):
        row = rxrx[metric]
        lines.append(
            f"| {metric} | {row['baseline']:.6f} | {row['candidate']:.6f} | {format_ci(row)} |"
        )
    lines.append(
        f"| nmi | {rxrx['nmi']['baseline']:.6f} | {rxrx['nmi']['candidate']:.6f} | "
        f"{rxrx['nmi']['mean']:+.6f} (no paired CI) |"
    )
    lines.extend(
        [
            "",
            "| Spatial block24 metric | Official ck17079 | Bi-level | Delta [paired 95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        lines.append(
            f"| {metric} | {spatial['arms']['anchor17079'][metric]:.6f} | "
            f"{spatial['arms']['bilevel17079'][metric]:.6f} | "
            f"{format_ci(spatial_delta[metric])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "No frozen classification matrix was launched for this candidate because both "
            "predeclared direct gates failed. The result does not reject a dual-anchor method: "
            "this implementation uses one teacher for both levels, refreshes that teacher with "
            "the patch schedule, and forms relations only within each rank's 16-image microbatch.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = {
        "status": "VALID_COMPLETE_EXPLORATORY_KILL_SCREEN",
        "method": {
            "patch_anchor": 17079,
            "relation_anchor": 17079,
            "patch_gram_weight": 2.0,
            "inter_image_relative_weight": 0.25,
            "relation_scope": "per_crop_per_rank_batch16",
            "teacher_refresh_completed_updates": [20200, 20400],
        },
        "training": analyze_training(),
        "spatial": analyze_spatial(),
        "rxrx3": analyze_rxrx3(),
        "classification": {
            "status": "NOT_RUN_AFTER_DIRECT_GATE_FAILURE",
            "reason": "RxRx3 regressed and spatial improvements versus official ck17079 were inconclusive.",
        },
    }
    report["decision_code"] = "REJECT_SAME_TEACHER_BILEVEL_W025"
    report["decision"] = (
        "Reject this same-teacher, weight-0.25 bi-level formulation. The inter-image loss rises "
        "rather than falls, formal RxRx3 regresses, and the small spatial improvements over the "
        "official ck17079 arm have intervals crossing zero. A next experiment must decouple a "
        "frozen global anchor from the refreshable patch teacher and should gather relation targets "
        "across ranks; do not tune the weight on downstream labels."
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "analysis.json", report)
    (OUTPUT_ROOT / "README.md").write_text(build_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
