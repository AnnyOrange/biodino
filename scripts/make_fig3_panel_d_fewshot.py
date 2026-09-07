#!/usr/bin/env python3
"""Run the leakage-safe BBBC048 annotation-efficiency evaluation for Fig. 3.

The evaluation keeps the committed BBBC048 source-group test fold fixed and
subsamples only its training side.  It applies the repository's canonical
StandardScaler + class-balanced LogisticRegression frozen-feature probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
EXTERNAL_CACHE = (
    REPO_ROOT
    / "outputs/02_eval_runs/external_fm_fair_protocol_20260721/feature_cache/bbbc048-cellcycle"
)
HPLUS_FEATURE = (
    REPO_ROOT
    / "outputs/02_eval_runs/bio_eval_hplus_all_ckpts_current_nonseg_20260701_cpu7_tail/"
    "bio_classification/bbbc048-cellcycle/8199/features/bbbc048-cellcycle/dinov3-8199_s512_r585.npz"
)
BASELINE_CACHE = DEFAULT_OUT_DIR / "panel_d_feature_cache/bbbc048-cellcycle"
EXTERNAL_SUMMARY_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_fair_protocol_20260721/classification"
HPLUS_FULLSHOT_SUMMARY = (
    REPO_ROOT
    / "outputs/02_eval_runs/bio_eval_hplus_all_ckpts_current_nonseg_20260701_cpu7_tail/"
    "bio_classification/bbbc048-cellcycle/8199/summary.csv"
)


MODEL_SPECS: dict[str, dict[str, str]] = {
    "biodino_hplus_s6_alpha1": {
        "label": "Biodino H+ (S6)",
        "path": str(HPLUS_FEATURE),
        "family": "ours",
    },
    "dinov3_official_vitl16": {
        "label": "DINOv3-L/16 (original)",
        "path": str(BASELINE_CACHE / "dinov3_official_vitl16.npz"),
        "family": "general-fm",
    },
    "dinov3_official_vit7b16": {
        "label": "DINOv3-7B/16 (original)",
        "path": str(BASELINE_CACHE / "dinov3_official_vit7b16.npz"),
        "family": "general-fm",
    },
    "imagenet_resnet50": {
        "label": "ImageNet ResNet50",
        "path": str(BASELINE_CACHE / "imagenet_resnet50.npz"),
        "family": "imagenet",
    },
    "dinov2": {
        "label": "DINOv2 ViT-B/14",
        "path": str(EXTERNAL_CACHE / "dinov2.npz"),
        "family": "general-fm",
    },
    "hoptimus0": {
        "label": "H-optimus-0",
        "path": str(EXTERNAL_CACHE / "hoptimus0.npz"),
        "family": "pathology-fm",
    },
    "gigapath": {
        "label": "GigaPath",
        "path": str(EXTERNAL_CACHE / "gigapath.npz"),
        "family": "pathology-fm",
    },
    "bioclip": {
        "label": "BioCLIP",
        "path": str(EXTERNAL_CACHE / "bioclip.npz"),
        "family": "biomedical-fm",
    },
}


def parse_csv_floats(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(x <= 0 or x > 1 for x in values):
        raise ValueError("Fractions must be in (0, 1].")
    return values


def parse_csv_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def sampled_train_indices(labels: np.ndarray, train_idx: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Take a deterministic class-stratified subset from the fixed train fold."""
    if fraction >= 1.0:
        return train_idx.copy()
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(labels[train_idx]):
        candidates = train_idx[labels[train_idx] == label]
        # Retain every represented class for a meaningful macro-F1 probe.
        n_keep = min(len(candidates), max(1, int(round(len(candidates) * fraction))))
        selected.append(np.sort(rng.choice(candidates, size=n_keep, replace=False)))
    return np.sort(np.concatenate(selected))


def load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        if "features" not in pack or "labels" not in pack:
            raise ValueError(f"{path} must contain features and labels arrays.")
        features = np.asarray(pack["features"], dtype=np.float32)
        labels = np.asarray(pack["labels"], dtype=np.int64).reshape(-1)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError(f"Invalid feature cache shapes in {path}: {features.shape} / {labels.shape}")
    return features, labels


def known_fullshot(model_key: str) -> dict[str, float] | None:
    """Read the existing canonical full-label result without refitting it."""
    path = HPLUS_FULLSHOT_SUMMARY if model_key == "biodino_hplus_s6_alpha1" else EXTERNAL_SUMMARY_ROOT / model_key / "summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    hit = table[
        table["dataset"].astype(str).eq("bbbc048-cellcycle")
        & table["split"].astype(str).eq("group-split")
        & table["error"].fillna("").eq("")
    ]
    if len(hit) != 1:
        return None
    row = hit.iloc[0]
    return {
        "macro_f1": float(row["macro_f1"]),
        "accuracy": float(row["accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "n_train": int(row["n_train"]),
        "n_test": int(row["n_test"]),
        "summary_file": str(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument(
        "--models",
        default="biodino_hplus_s6_alpha1,dinov3_official_vitl16,imagenet_resnet50,dinov2,hoptimus0,gigapath,bioclip",
        help="Comma-separated MODEL_SPECS keys.",
    )
    parser.add_argument("--fractions", default="0.01,0.05,0.10,0.25,1.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument(
        "--reuse-known-fullshot",
        action="store_true",
        help="Reuse canonical 100%-label BBBC048 rows when an identical encoder/split result exists.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping a missing feature cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
    from dinov3.eval.bio_frozen_eval.probes import run_classification_probe_split
    from dinov3.eval.bio_frozen_eval.registry import build_dataset

    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in model_keys if key not in MODEL_SPECS]
    if unknown:
        raise KeyError(f"Unknown models: {unknown}; choices: {sorted(MODEL_SPECS)}")
    fractions = parse_csv_floats(args.fractions)
    seeds = parse_csv_ints(args.seeds)
    raw_path = args.output_dir / "panel_d_fewshot_curves.csv"
    summary_path = args.output_dir / "panel_d_fewshot_summary.csv"
    metadata_path = args.output_dir / "panel_d_fewshot_metadata.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, task = build_dataset(
        "bbbc048-cellcycle", "train", None, None, benchmark_root=args.benchmark_root
    )
    if task != "classification":
        raise RuntimeError(f"Expected classification task, got {task!r}")
    train_idx, test_idx = group_split_indices("bbbc048-cellcycle", dataset, args.benchmark_root)
    expected_labels = np.asarray([int(label) for _, label in dataset.samples], dtype=np.int64)

    existing = pd.DataFrame()
    if raw_path.exists() and not args.overwrite:
        existing = pd.read_csv(raw_path)
    rows: list[dict[str, Any]] = existing.to_dict(orient="records")
    # CSV readers materialize an empty ``error`` cell as NaN. Treat it as a
    # completed run so an interrupted invocation resumes without duplication.
    completed = {
        (str(row["model_key"]), float(row["label_fraction"]), int(row["seed"]))
        for row in rows
        if not str(row.get("error", "") if pd.notna(row.get("error", "")) else "").strip()
    }

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        feature_path = Path(spec["path"])
        if not feature_path.exists():
            message = f"missing feature cache: {feature_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[panel-d] skip {model_key}: {message}", flush=True)
            continue
        features, labels = load_feature_cache(feature_path)
        if len(labels) != len(expected_labels):
            raise ValueError(f"{model_key}: cache has {len(labels)} labels, expected {len(expected_labels)}")
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"{model_key}: feature-cache labels do not match the committed dataset ordering")
        print(f"[panel-d] {model_key}: features={features.shape}, train/test={len(train_idx)}/{len(test_idx)}", flush=True)

        for fraction in fractions:
            run_seeds = [seeds[0]] if fraction >= 1.0 else seeds
            for seed in run_seeds:
                key = (model_key, float(fraction), int(seed))
                if key in completed:
                    continue
                subset = sampled_train_indices(labels, train_idx, fraction, seed)
                try:
                    known = known_fullshot(model_key) if args.reuse_known_fullshot and fraction >= 1.0 else None
                    if known is None:
                        result = run_classification_probe_split(
                            features[subset], labels[subset], features[test_idx], labels[test_idx], max_iter=args.max_iter
                        )
                        macro_f1 = float(result.metrics["macro_f1"])
                        accuracy = float(result.metrics["accuracy"])
                        balanced_accuracy = float(result.metrics["balanced_accuracy"])
                        provenance = "new_fewshot_probe"
                    else:
                        macro_f1 = known["macro_f1"]
                        accuracy = known["accuracy"]
                        balanced_accuracy = known["balanced_accuracy"]
                        provenance = f"canonical_fullshot:{known['summary_file']}"
                    row = {
                        "dataset": "bbbc048-cellcycle",
                        "split": "committed-source-group-test",
                        "metric": "macro_f1",
                        "model_key": model_key,
                        "model": spec["label"],
                        "model_family": spec["family"],
                        "feature_file": str(feature_path),
                        "label_fraction": float(fraction),
                        "label_percent": float(fraction * 100.0),
                        "seed": int(seed),
                        "n_train": int(known["n_train"] if known is not None else len(subset)),
                        "n_test": int(known["n_test"] if known is not None else len(test_idx)),
                        "score": macro_f1,
                        "macro_f1": macro_f1,
                        "accuracy": accuracy,
                        "balanced_accuracy": balanced_accuracy,
                        "provenance": provenance,
                        "error": "",
                    }
                    print(
                        f"[panel-d] {model_key} {fraction * 100:g}% seed={seed}: "
                        f"macro_f1={row['macro_f1']:.4f} n={len(subset)}",
                        flush=True,
                    )
                except Exception as exc:
                    row = {
                        "dataset": "bbbc048-cellcycle",
                        "split": "committed-source-group-test",
                        "metric": "macro_f1",
                        "model_key": model_key,
                        "model": spec["label"],
                        "model_family": spec["family"],
                        "feature_file": str(feature_path),
                        "label_fraction": float(fraction),
                        "label_percent": float(fraction * 100.0),
                        "seed": int(seed),
                        "n_train": int(len(subset)),
                        "n_test": int(len(test_idx)),
                        "score": float("nan"),
                        "macro_f1": float("nan"),
                        "accuracy": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "provenance": "failed_fewshot_probe",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[panel-d] failed {model_key} {fraction * 100:g}% seed={seed}: {row['error']}", flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_path, index=False)

    frame = pd.DataFrame(rows)
    frame = frame[frame["error"].fillna("").eq("")].copy()
    if frame.empty:
        raise RuntimeError("No successful few-shot runs were written.")
    summary = (
        frame.groupby(
            ["dataset", "split", "metric", "model_key", "model", "model_family", "label_fraction", "label_percent"],
            as_index=False,
        )
        .agg(
            score=("score", "mean"),
            score_std=("score", "std"),
            n_seeds=("seed", "nunique"),
            n_train=("n_train", "mean"),
            n_test=("n_test", "first"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .sort_values(["model", "label_fraction"])
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    summary["n_train"] = summary["n_train"].round().astype(int)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "dataset": "bbbc048-cellcycle",
        "metric": "macro_f1",
        "split": "committed source-group split, fixed test fold",
        "n_total": int(len(dataset)),
        "n_train_full": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "fractions": fractions,
        "seeds": seeds,
        "max_iter": int(args.max_iter),
        "models_requested": model_keys,
        "models_completed": sorted(frame["model_key"].unique().tolist()),
        "raw_results": str(raw_path),
        "summary": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[panel-d] wrote {raw_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
