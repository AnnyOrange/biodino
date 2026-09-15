#!/usr/bin/env python3
"""Analyze the post-hoc paired ck7807 versus ck23911 spatial diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911/paired_endpoint"
)
CURVE_ROOT = ROOT / "outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911"
EARLY_STEP = 7807
LATE_STEP = 23911
BOOTSTRAP_SAMPLES = 50_000
SIGNFLIP_SAMPLES = 100_000
SEED = 20260911


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root = args.root.resolve()

    summaries = {
        step: load_json(args.root / f"ck{step}.summary.json")
        for step in (EARLY_STEP, LATE_STEP)
    }
    records = {
        step: load_json(args.root / f"ck{step}.records.json")
        for step in (EARLY_STEP, LATE_STEP)
    }
    for step in (EARLY_STEP, LATE_STEP):
        original = load_json(CURVE_ROOT / "checkpoints" / f"ck{step}.json")
        if summaries[step]["layers"] != original["layers"]:
            raise RuntimeError(f"ck{step} repeat does not exactly match the all-49 curve")
        if summaries[step]["n_images"] != 128 or summaries[step]["n_unique_keys"] != 128:
            raise RuntimeError(f"ck{step} has an invalid sample count")

    early_records, late_records = records[EARLY_STEP], records[LATE_STEP]
    if early_records["keys"] != late_records["keys"]:
        raise RuntimeError("ordered image keys differ between checkpoints")
    if early_records["local_area"] != late_records["local_area"]:
        raise RuntimeError("local crop areas differ between checkpoints")
    early = np.asarray(
        early_records["layers"]["block_24"]["true_minus_shifted"], dtype=np.float64
    )
    late = np.asarray(
        late_records["layers"]["block_24"]["true_minus_shifted"], dtype=np.float64
    )
    if early.shape != (128,) or late.shape != (128,):
        raise RuntimeError(f"unexpected per-image shapes: {early.shape}, {late.shape}")
    differences = early - late
    observed = float(differences.mean())

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    sign_rng = np.random.default_rng(SEED + 1)
    exceedances = 0
    remaining = SIGNFLIP_SAMPLES
    while remaining:
        current = min(10_000, remaining)
        signs = sign_rng.integers(0, 2, size=(current, len(differences)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_values = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(null_values) >= abs(observed) - 1.0e-15))
        remaining -= current
    signflip_p = (exceedances + 1.0) / (SIGNFLIP_SAMPLES + 1.0)

    result = {
        "status": "VALID_COMPLETE",
        "admission": "POSTHOC_LABEL_FREE_PAIRED_DIAGNOSTIC",
        "metric": "block24 mean true-minus-shifted cosine",
        "early_checkpoint": EARLY_STEP,
        "late_checkpoint": LATE_STEP,
        "paired_images": len(differences),
        "ordered_keys_exact_match": True,
        "local_areas_exact_match": True,
        "all49_summary_exact_match": True,
        "early_mean": float(early.mean()),
        "late_mean": float(late.mean()),
        "early_minus_late": observed,
        "relative_drop_from_early": float(observed / early.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "positive_image_differences": int(np.count_nonzero(differences > 0)),
        "negative_image_differences": int(np.count_nonzero(differences < 0)),
        "zero_image_differences": int(np.count_nonzero(differences == 0)),
        "posthoc_two_sided_monte_carlo_signflip_p": float(signflip_p),
        "signflip_samples": SIGNFLIP_SAMPLES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "inputs": {
            f"ck{step}": {
                "summary_sha256": sha256(args.root / f"ck{step}.summary.json"),
                "records_sha256": sha256(args.root / f"ck{step}.records.json"),
            }
            for step in (EARLY_STEP, LATE_STEP)
        },
        "warning": (
            "ck7807 was selected after viewing the 49-checkpoint curve. The interval describes "
            "the paired sample; the p-value is not corrected for checkpoint selection."
        ),
    }
    if not all(math.isfinite(value) for value in (observed, signflip_p)):
        raise RuntimeError("non-finite paired result")
    (args.root / "paired_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lower, upper = result["paired_bootstrap_95_ci"]
    lines = [
        "# Post-hoc spatial peak-to-endpoint pair",
        "",
        "Status: `VALID_COMPLETE`; label-free, 128 exactly paired images.",
        "",
        f"ck{EARLY_STEP}: {result['early_mean']:.6f}; ck{LATE_STEP}: {result['late_mean']:.6f}.",
        (
            f"Paired difference: {observed:+.6f} "
            f"({100.0 * result['relative_drop_from_early']:.2f}% of ck{EARLY_STEP}), "
            f"bootstrap 95% CI [{lower:+.6f}, {upper:+.6f}]."
        ),
        (
            f"Positive/negative/zero image differences: "
            f"{result['positive_image_differences']}/"
            f"{result['negative_image_differences']}/"
            f"{result['zero_image_differences']}."
        ),
        (
            f"Post-hoc two-sided Monte Carlo sign-flip p={signflip_p:.6g} "
            f"({SIGNFLIP_SAMPLES} samples)."
        ),
        "",
        (
            "ck7807 was selected after viewing the 49-checkpoint curve. This p-value is not "
            "selection-corrected and must not be presented as confirmatory evidence."
        ),
    ]
    (args.root / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
