#!/usr/bin/env python3
"""Paired curve analysis for the locked HS6-L RxRx3-core all-49 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "mrr")
RECALL_K = {"recall_at_1": 1, "recall_at_5": 5, "recall_at_10": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_paired_binary_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign/McNemar test on discordant binary pairs."""
    a_wins = int(np.sum(a & ~b))
    b_wins = int(np.sum(~a & b))
    n = a_wins + b_wins
    if n == 0:
        return a_wins, b_wins, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_wins, b_wins) + 1)) / (2**n)
    return a_wins, b_wins, min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    delta: np.ndarray, *, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    means = np.empty(samples, dtype=np.float64)
    width = 1000
    for start in range(0, samples, width):
        stop = min(start + width, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def max_stat_signflip_p(
    deltas: np.ndarray, *, rng: np.random.Generator, samples: int
) -> float:
    """One-sided max-statistic p-value over all non-endpoint checkpoints."""
    observed = float(np.max(deltas.mean(axis=1)))
    exceed = 0
    width = 1000
    for start in range(0, samples, width):
        count = min(width, samples - start)
        signs = rng.integers(0, 2, size=(count, deltas.shape[1]), dtype=np.int8)
        signs = signs.astype(np.float32) * 2.0 - 1.0
        null_max = np.max((signs @ deltas.T) / deltas.shape[1], axis=1)
        exceed += int(np.sum(null_max >= observed - 1e-15))
    return (exceed + 1.0) / (samples + 1.0)


def query_ranks(features: np.ndarray, labels: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    g = features[gallery].astype(np.float32)
    q = features[~gallery].astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    order = np.argsort(-(q @ g.T), axis=1)
    gy, qy = labels[gallery], labels[~gallery]
    return np.asarray(
        [np.flatnonzero(gy[indices] == qy[index])[0] + 1 for index, indices in enumerate(order)],
        dtype=np.int32,
    )


def per_query_metric(ranks: np.ndarray, metric: str) -> np.ndarray:
    if metric in RECALL_K:
        return ranks <= RECALL_K[metric]
    reciprocal = 1.0 / ranks.astype(np.float64)
    if metric == "mrr_at_10":
        reciprocal[ranks > 10] = 0.0
    return reciprocal


def nmi_chance_diagnostic(
    feature_path: Path,
    labels: np.ndarray,
    *,
    formal_nmi: float,
    rng: np.random.Generator,
    permutations: int = 100,
) -> dict[str, float | int]:
    features = np.load(feature_path, mmap_mode="r").astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    predicted = MiniBatchKMeans(
        n_clusters=len(np.unique(labels)),
        n_init=5,
        random_state=0,
        batch_size=min(1024, len(features)),
        max_iter=200,
    ).fit_predict(features)
    shuffled_nmi = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled_nmi[index] = normalized_mutual_info_score(rng.permutation(labels), predicted)
    return {
        "formal_float32_nmi": formal_nmi,
        "cached_float16_nmi": float(normalized_mutual_info_score(labels, predicted)),
        "shuffled_label_nmi_mean": float(shuffled_nmi.mean()),
        "shuffled_label_nmi_std": float(shuffled_nmi.std(ddof=1)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(labels, predicted)),
        "adjusted_rand_index": float(adjusted_rand_score(labels, predicted)),
        "permutations": permutations,
    }


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=repo / "outputs/02_eval_runs/rxrx3_core_formal_l5_all49_v3_single3090_20260911",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo / "outputs/02_eval_inputs/formal_v3/rxrx3-core",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--signflip-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_path = args.campaign / "validation_report.json"
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "VALID_COMPLETE" or validation.get("valid_checkpoints") != 49:
        raise RuntimeError("the locked campaign is not independently valid at 49/49 checkpoints")

    manifest_path = args.campaign / "campaign_manifest.json"
    if sha256(manifest_path) != validation.get("campaign_manifest_sha256"):
        raise RuntimeError("campaign manifest hash no longer matches validation_report.json")
    manifest = json.loads(manifest_path.read_text())

    metadata = json.loads((args.dataset_root / "metadata.json").read_text())
    labels = np.load(args.dataset_root / "labels.npy")
    gallery = np.asarray([row["split"] == "gallery" for row in metadata["rows"]])
    if int(gallery.sum()) != 734 or int((~gallery).sum()) != 734:
        raise RuntimeError("unexpected RxRx3 gallery/query counts")

    formal_rows: dict[int, dict[str, str]] = {}
    with (args.campaign / "checkpoint_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            formal_rows[int(row["checkpoint"])] = row
    checkpoints = [int(model["checkpoint_step"]) for model in manifest["models"]]
    if len(checkpoints) != 49 or sorted(formal_rows) != sorted(checkpoints):
        raise RuntimeError("checkpoint_metrics.csv does not cover the locked 49-checkpoint manifest")

    ranks: dict[int, np.ndarray] = {}
    for index, checkpoint in enumerate(checkpoints, start=1):
        feature_path = args.campaign / "models" / f"hs6_l_5tb_ck{checkpoint}" / "rxrx3_features.npy"
        ranks[checkpoint] = query_ranks(np.load(feature_path, mmap_mode="r"), labels, gallery)
        print(f"[ranks] {index:02d}/49 ck{checkpoint}", flush=True)

    endpoint = checkpoints[-1]
    rng = np.random.default_rng(args.seed)
    analyses = []
    cache_checks = []
    for metric in METRICS:
        values = np.stack([per_query_metric(ranks[checkpoint], metric) for checkpoint in checkpoints])
        cache_means = values.mean(axis=1)
        formal_means = np.asarray([float(formal_rows[checkpoint][metric]) for checkpoint in checkpoints])
        max_abs_error = float(np.max(np.abs(cache_means - formal_means)))
        cache_checks.append({"metric": metric, "max_abs_error": max_abs_error})
        tolerance = 1e-12 if metric in RECALL_K else 1e-6
        if max_abs_error > tolerance:
            raise RuntimeError(f"cached-feature {metric} differs from formal result by {max_abs_error}")

        peak_indices = np.flatnonzero(formal_means == formal_means.max())
        # The latest exact tie is the most conservative mature anchor candidate.
        peak_index = int(peak_indices[-1])
        peak = checkpoints[peak_index]
        delta = values[peak_index].astype(np.float64) - values[-1].astype(np.float64)
        ci_low, ci_high = paired_bootstrap_ci(delta, rng=rng, samples=args.bootstrap_samples)
        all_deltas = values[:-1].astype(np.float32) - values[-1].astype(np.float32)
        adjusted_p = max_stat_signflip_p(all_deltas, rng=rng, samples=args.signflip_samples)
        row = {
            "metric": metric,
            "peak_checkpoint": peak,
            "peak_checkpoints": [checkpoints[int(index)] for index in peak_indices],
            "peak_tie_policy": "latest checkpoint among exact formal ties",
            "peak_value": float(formal_means[peak_index]),
            "endpoint_checkpoint": endpoint,
            "endpoint_value": float(formal_means[-1]),
            "paired_delta": float(delta.mean()),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "peak_selection_max_stat_p": adjusted_p,
        }
        if metric in RECALL_K:
            peak_success = values[peak_index].astype(bool)
            endpoint_success = values[-1].astype(bool)
            a_wins, b_wins, raw_p = exact_paired_binary_p(peak_success, endpoint_success)
            row.update(
                {
                    "peak_successes": int(peak_success.sum()),
                    "endpoint_successes": int(endpoint_success.sum()),
                    "peak_only_successes": a_wins,
                    "endpoint_only_successes": b_wins,
                    "paired_exact_unadjusted_p": raw_p,
                }
            )
        analyses.append(row)

    metric_by_name = {row["metric"]: row for row in analyses}
    nmi_peak = max(checkpoints, key=lambda checkpoint: (float(formal_rows[checkpoint]["nmi"]), -checkpoint))
    diagnostic_checkpoints = sorted({nmi_peak, int(metric_by_name["mrr"]["peak_checkpoint"]), endpoint})
    nmi_rng = np.random.default_rng(args.seed + 1)
    nmi_diagnostics = {}
    for checkpoint in diagnostic_checkpoints:
        feature_path = args.campaign / "models" / f"hs6_l_5tb_ck{checkpoint}" / "rxrx3_features.npy"
        nmi_diagnostics[str(checkpoint)] = nmi_chance_diagnostic(
            feature_path,
            labels,
            formal_nmi=float(formal_rows[checkpoint]["nmi"]),
            rng=nmi_rng,
        )

    output = {
        "status": "VALID_COMPLETE",
        "analysis_scope": "paired query-level post-hoc curve analysis",
        "selection_warning": "Peak-versus-endpoint CIs and raw p-values are selection-naive; use max-stat p for 49-checkpoint peak selection.",
        "campaign_manifest_sha256": sha256(manifest_path),
        "checkpoint_metrics_sha256": sha256(args.campaign / "checkpoint_metrics.csv"),
        "feature_cache_precision": "float16; recall ranks exactly reproduce formal in-memory results",
        "n_checkpoints": len(checkpoints),
        "n_query": int((~gallery).sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "signflip_samples": args.signflip_samples,
        "seed": args.seed,
        "cache_reproduction": cache_checks,
        "metrics": analyses,
        "nmi_chance_diagnostics": nmi_diagnostics,
        "nmi_decision": "Do not interpret raw NMI: 734 clusters for 1468 samples yields a similarly high shuffled-label baseline.",
    }
    json_path = args.campaign / "paired_curve_statistics.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    csv_rows = []
    for row in analyses:
        csv_row = dict(row)
        csv_row["peak_checkpoints"] = ";".join(str(value) for value in row["peak_checkpoints"])
        csv_rows.append(csv_row)
    fields = sorted({key for row in csv_rows for key in row})
    csv_path = args.campaign / "paired_curve_statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# RxRx3-core HS6-L all-49 paired curve analysis",
        "",
        f"Status: `VALID_COMPLETE`; 49 checkpoints, 734 paired queries; campaign manifest `{output['campaign_manifest_sha256']}`.",
        "",
        "| Metric | Peak ck | Peak | Endpoint | Delta | Paired 95% CI | Raw paired p | 49-ck max-stat p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analyses:
        raw = row.get("paired_exact_unadjusted_p")
        raw_text = f"{raw:.6g}" if raw is not None else "n/a"
        lines.append(
            f"| {row['metric']} | {row['peak_checkpoint']} | {row['peak_value']:.6f} | "
            f"{row['endpoint_value']:.6f} | {row['paired_delta']:+.6f} | "
            f"[{row['bootstrap_ci95_low']:+.6f}, {row['bootstrap_ci95_high']:+.6f}] | "
            f"{raw_text} | {row['peak_selection_max_stat_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The raw paired p-value and bootstrap CI compare the retrospectively selected peak with the endpoint and therefore do not correct checkpoint selection. The max-statistic sign-flip p-value controls selection over all 48 non-endpoint checkpoints for each metric.",
            "",
            "Raw NMI is excluded from inference: with 734 classes and only two samples per class, shuffled cluster assignments have nearly the same high NMI. Retrieval recall and rank metrics are the usable evidence.",
            "",
            "| Checkpoint | Formal NMI | Cached NMI | Shuffled-label NMI | AMI | ARI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, row in nmi_diagnostics.items():
        lines.append(
            f"| {checkpoint} | {row['formal_float32_nmi']:.6f} | {row['cached_float16_nmi']:.6f} | "
            f"{row['shuffled_label_nmi_mean']:.6f} | {row['adjusted_mutual_info']:.6f} | "
            f"{row['adjusted_rand_index']:.6f} |"
        )
    (args.campaign / "paired_curve_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
