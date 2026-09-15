#!/usr/bin/env python3
"""Analyze paired native-2D CTC results across the precommitted L5 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


METRICS = {
    "DET": ("ctc_metrics", "DET"),
    "SEG": ("ctc_metrics", "SEG"),
    "TRA": ("ctc_metrics", "TRA"),
    "mean_foreground_dice": ("extra_segmentation_metrics", "mean_foreground_dice"),
    "AP": ("extra_segmentation_metrics", "AP"),
    "AP50": ("extra_segmentation_metrics", "AP50"),
    "AP75": ("extra_segmentation_metrics", "AP75"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(row: dict, metric: str) -> float:
    section, key = METRICS[metric]
    return float(row[section][key])


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_statistics(
    matrix: np.ndarray,
    steps: list[int],
    domains: list[str],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    comparator_index = len(steps) - 1
    candidate_indices = list(range(comparator_index))
    means = matrix.mean(axis=1)
    overall_peak_value = float(means.max())
    overall_tied = [
        index
        for index, value in enumerate(means)
        if math.isclose(value, overall_peak_value, abs_tol=1e-15)
    ]
    overall_peak_index = overall_tied[-1]
    differences = matrix - matrix[comparator_index]
    earlier_mean_differences = differences[candidate_indices].mean(axis=1)
    observed_max = float(earlier_mean_differences.max())
    selected_earlier_indices = [
        index
        for index in candidate_indices
        if math.isclose(
            float(differences[index].mean()), observed_max, abs_tol=1e-15
        )
    ]
    selected_earlier_index = selected_earlier_indices[-1]

    null_maxima = []
    selected_earlier_null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(domains)):
        signed = differences[candidate_indices] * np.asarray(signs)[None, :]
        null_maxima.append(float(signed.mean(axis=1).max()))
        selected_earlier_null.append(
            float((differences[selected_earlier_index] * np.asarray(signs)).mean())
        )
    tolerance = 1e-15
    max_stat_p = float(np.mean(np.asarray(null_maxima) >= observed_max - tolerance))
    raw_p = float(
        np.mean(np.asarray(selected_earlier_null) >= observed_max - tolerance)
    )

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(domains),
        size=(bootstrap_samples, len(domains)),
    )
    selected_difference = differences[selected_earlier_index]
    bootstrap = selected_difference[sample_indices].mean(axis=1)
    return {
        "steps": steps,
        "domains": domains,
        "macro_by_step": {str(step): float(value) for step, value in zip(steps, means)},
        "overall_peak_step_latest_exact_tie": steps[overall_peak_index],
        "overall_peak_tied_steps": [steps[index] for index in overall_tied],
        "overall_peak": float(means[overall_peak_index]),
        "late_comparator_step": steps[comparator_index],
        "late_comparator": float(means[comparator_index]),
        "best_earlier_step_latest_exact_tie": steps[selected_earlier_index],
        "best_earlier_tied_steps": [steps[index] for index in selected_earlier_indices],
        "best_earlier": float(means[selected_earlier_index]),
        "best_earlier_paired_delta": observed_max,
        "best_earlier_paired_bootstrap_95_ci": percentile_interval(bootstrap),
        "best_earlier_raw_one_sided_exact_signflip_p": raw_p,
        "earlier_candidate_selection_adjusted_max_stat_p": max_stat_p,
        "best_earlier_domain_differences": {
            domain: float(value)
            for domain, value in zip(domains, differences[selected_earlier_index])
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "exact_sign_patterns": 2 ** len(domains),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    args = parser.parse_args()

    manifest_path = args.campaign / "campaign_manifest.json"
    validation_path = args.campaign / "validation_report.json"
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "VALID_COMPLETE":
        raise RuntimeError(f"campaign is not complete: {validation}")
    manifest_sha = sha256(manifest_path)
    if validation.get("campaign_manifest_sha256") != manifest_sha:
        raise RuntimeError("validation report does not match the campaign manifest")

    steps = sorted(int(model["checkpoint_step"]) for model in manifest["models"])
    results = {}
    long_rows = []
    for step in steps:
        path = args.campaign / "models" / f"hs6_l_5tb_ck{step}" / "results.json"
        payload = json.loads(path.read_text())
        if payload.get("status") != "VALID_COMPLETE" or payload.get(
            "campaign_manifest_sha256"
        ) != manifest_sha:
            raise RuntimeError(f"invalid candidate result: {path}")
        results[step] = {row["domain"]: row for row in payload["domain_rows"]}
        if len(results[step]) != 10:
            raise RuntimeError(f"expected 10 unique domains for checkpoint {step}")
        for domain, row in sorted(results[step].items()):
            long_rows.append(
                {
                    "checkpoint": step,
                    "domain": domain,
                    **{metric: metric_value(row, metric) for metric in METRICS},
                }
            )

    domain_sets = [set(results[step]) for step in steps]
    if any(domains != domain_sets[0] for domains in domain_sets[1:]):
        raise RuntimeError("candidate domain sets differ")
    domains = sorted(domain_sets[0])
    analyses = {}
    for offset, metric in enumerate(METRICS):
        matrix = np.asarray(
            [[metric_value(results[step][domain], metric) for domain in domains] for step in steps]
        )
        analyses[metric] = paired_statistics(
            matrix,
            steps,
            domains,
            seed=args.seed + offset,
            bootstrap_samples=args.bootstrap_samples,
        )

    report = {
        "status": "VALID_COMPLETE",
        "admission": "OBSERVATIONAL_NATIVE_2D",
        "campaign_manifest_sha256": manifest_sha,
        "checkpoint_count": len(steps),
        "domain_count": len(domains),
        "late_comparator_definition": "latest checkpoint in the precommitted candidate set",
        "inference_warning": (
            "Domains are a fixed heterogeneous suite; paired resampling quantifies suite stability, "
            "not population-level significance."
        ),
        "metrics": analyses,
    }
    json_path = args.campaign / "paired_candidate_statistics.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    csv_path = args.campaign / "per_domain_candidate_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    lines = [
        "# CTC native 2-D L5 candidate analysis",
        "",
        f"Status: `VALID_COMPLETE`; manifest `{manifest_sha}`.",
        "",
        "This is an observational 10-domain 2-D screen, not the formal 20-domain CTC result.",
        "The late comparator is ck21959, the latest checkpoint in the precommitted candidate set.",
        "",
        (
            "| Metric | Overall best ck | Best | ck21959 | Best earlier ck | "
            "Earlier delta | Paired 95% CI | Raw p | Max-stat p |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, result in analyses.items():
        lower, upper = result["best_earlier_paired_bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {result['overall_peak_step_latest_exact_tie']} | "
            f"{result['overall_peak']:.6f} | {result['late_comparator']:.6f} | "
            f"{result['best_earlier_step_latest_exact_tie']} | "
            f"{result['best_earlier_paired_delta']:+.6f} | "
            f"[{lower:+.6f}, {upper:+.6f}] | "
            f"{result['best_earlier_raw_one_sided_exact_signflip_p']:.6f} | "
            f"{result['earlier_candidate_selection_adjusted_max_stat_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "The delta and paired interval always compare the best of the three earlier "
                "candidates with ck21959."
            ),
            "The max-statistic p-value controls that earlier-candidate selection within each metric.",
            "The 10 domains are fixed and heterogeneous, so these resampling values measure suite stability only.",
        ]
    )
    (args.campaign / "paired_candidate_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
