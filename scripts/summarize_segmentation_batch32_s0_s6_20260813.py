#!/usr/bin/env python3
"""Audit the strict batch-32 segmentation results for S0 and current-best S6."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "outputs/00_reports/segmentation_batch32_s0s6_audit_20260813"
DATASETS = (
    "bbbc038",
    "cellpose",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
)

MODELS = {
    "S+": {
        "s0_dir": "sp",
        "s6_recipe": "SIGReg=0.05, alpha=0.75, ck8199",
        "machine": "local (existing completed result)",
        "s6_root": REPO_ROOT
        / "outputs/02_eval_runs/S6interp_official_sigreg005repl8199_20260719__full_dense_local_bf16_20260722/bio_segmentation",
        "tag": "75",
        "source_cache": "existing strict dense-feature evaluation",
    },
    "B": {
        "s0_dir": "b",
        "s6_recipe": "SIGReg=0.05, alpha=0.60, ck14349",
        "machine": "3090-qi",
        "s6_root": REPO_ROOT
        / "outputs/02_eval_runs/segmentation_batch32_s0s6_audit_20260813/B_s6_sigreg005_alpha060",
        "tag": None,
        "source_cache": "outputs/01_training_runs/B_s6recipe_sigreg005_alpha_tune_20260804/eval_dense/cache/bio_segmentation (tag 60)",
    },
    "L": {
        "s0_dir": "l",
        "s6_recipe": "no SIGReg, alpha=1.00, ck6149",
        "machine": "local",
        "s6_root": REPO_ROOT
        / "outputs/02_eval_runs/segmentation_batch32_s0s6_audit_20260813/L_s6_nosigreg_alpha100",
        "tag": None,
        "source_cache": "outputs/01_training_runs/L_s6recipe_nosigreg_alpha_tune_20260804/eval_dense/cache/bio_segmentation (tag 100)",
    },
    "H+": {
        "s0_dir": "hplus",
        "s6_recipe": "no SIGReg, alpha=1.00, ck8199",
        "machine": "suxin-8H100-1 (copied back to local)",
        "s6_root": REPO_ROOT
        / "outputs/02_eval_runs/segmentation_batch32_s0s6_audit_20260813/Hplus_s6_nosigreg_alpha100",
        "tag": None,
        "source_cache": "/data_2/suxin/runs/h100_hplus_7b_sigreg_ab_tuning_20260725/hplus_nosigreg/alpha_tune_e15_2gpu/eval_dense/cache/bio_segmentation (tag 100)",
    },
}

PROTOCOL = {
    "bbbc038": {"input": 512, "resize": "pad", "features": "even4"},
    "cellpose": {"input": 512, "resize": "pad", "features": "last1"},
    "conic": {
        "input": 256,
        "resize": "stretch",
        "features": "even4",
        "class_weight": "sqrt_inverse",
    },
    "livecell": {"input": 512, "resize": "pad", "features": "even4"},
    "monuseg": {"input": 768, "resize": "pad", "features": "last1"},
    "pannuke": {"input": 256, "resize": "stretch", "features": "even4"},
    "tissuenet": {"input": 256, "resize": "stretch", "features": "last1"},
}


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def unique_result(root: Path, dataset: str, tag: str | None) -> Path:
    if tag is None:
        candidate = root / dataset / "results.json"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    matches = list(root.glob(f"**/{dataset}/{tag}/results.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {dataset}/{tag} result under {root}, got {matches}")
    return matches[0]


def s0_result(model_dir: str, dataset: str) -> Path:
    root = (
        REPO_ROOT
        / "outputs/02_eval_runs/bioseg_best_7models_20260622"
        / model_dir
        / "bio_segmentation_best"
    )
    matches = list(root.glob(f"**/{dataset}/8199/results.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one S0 {model_dir}/{dataset}/8199 result, got {matches}")
    return matches[0]


def load_result(path: Path) -> dict:
    result = json.loads(path.read_text())
    for metric in ("mDice", "mIoU"):
        if metric not in result.get("test", {}):
            raise RuntimeError(f"Missing test.{metric}: {path}")
    return result


def validate_s6(result: dict, dataset: str, path: Path) -> None:
    meta = result.get("_meta", {})
    expected = {
        "probe_batch_size": 32,
        "probe_epochs": 50,
        "seed": 0,
        "probe_rng_seeded": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"Unexpected {key} in {path}: {meta.get(key)!r} != {value!r}")
    expected_weight = "sqrt_inverse" if dataset == "conic" else "none"
    if meta.get("class_weight_mode") != expected_weight:
        raise RuntimeError(
            f"Unexpected class_weight_mode in {path}: "
            f"{meta.get('class_weight_mode')!r} != {expected_weight!r}"
        )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_rows = []
    model_rows = []
    manifest = {
        "comparison_scope": "earliest microscopy S0 ck8199 vs current-best S6",
        "warning": (
            "This is a tuned current-best comparison, not a controlled N-scaling curve; "
            "the S6 objective, checkpoint, and alpha differ by model size."
        ),
        "probe_protocol": {
            "batch_size": 32,
            "epochs": 50,
            "seed": 0,
            "datasets": PROTOCOL,
        },
        "models": {},
    }

    for model, config in MODELS.items():
        mdice_before = []
        mdice_after = []
        miou_before = []
        miou_after = []
        current_paths = {}
        baseline_paths = {}

        for dataset in DATASETS:
            before_path = s0_result(config["s0_dir"], dataset)
            after_path = unique_result(config["s6_root"], dataset, config["tag"])
            before = load_result(before_path)
            after = load_result(after_path)
            validate_s6(after, dataset, after_path)

            b_dice = float(before["test"]["mDice"])
            a_dice = float(after["test"]["mDice"])
            b_iou = float(before["test"]["mIoU"])
            a_iou = float(after["test"]["mIoU"])
            mdice_before.append(b_dice)
            mdice_after.append(a_dice)
            miou_before.append(b_iou)
            miou_after.append(a_iou)
            baseline_paths[dataset] = relative(before_path)
            current_paths[dataset] = relative(after_path)

            dataset_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "s0_mDice": b_dice,
                    "s6_mDice": a_dice,
                    "delta_mDice": a_dice - b_dice,
                    "s0_mIoU": b_iou,
                    "s6_mIoU": a_iou,
                    "delta_mIoU": a_iou - b_iou,
                    "s6_recipe": config["s6_recipe"],
                    "evaluation_machine": config["machine"],
                    "s0_result_path": relative(before_path),
                    "s6_result_path": relative(after_path),
                }
            )

        row = {
            "model": model,
            "s6_recipe": config["s6_recipe"],
            "s0_mean_mDice": statistics.fmean(mdice_before),
            "s6_mean_mDice": statistics.fmean(mdice_after),
            "delta_mean_mDice": statistics.fmean(mdice_after)
            - statistics.fmean(mdice_before),
            "mDice_wins": sum(a > b for a, b in zip(mdice_after, mdice_before)),
            "s0_mean_mIoU": statistics.fmean(miou_before),
            "s6_mean_mIoU": statistics.fmean(miou_after),
            "delta_mean_mIoU": statistics.fmean(miou_after)
            - statistics.fmean(miou_before),
            "mIoU_wins": sum(a > b for a, b in zip(miou_after, miou_before)),
        }
        model_rows.append(row)
        manifest["models"][model] = {
            **row,
            "evaluation_machine": config["machine"],
            "source_cache": config["source_cache"],
            "s0_result_paths": baseline_paths,
            "s6_result_paths": current_paths,
        }

    write_csv(
        REPORT_ROOT / "dataset_results.csv",
        dataset_rows,
        [
            "model",
            "dataset",
            "s0_mDice",
            "s6_mDice",
            "delta_mDice",
            "s0_mIoU",
            "s6_mIoU",
            "delta_mIoU",
            "s6_recipe",
            "evaluation_machine",
            "s0_result_path",
            "s6_result_path",
        ],
    )
    write_csv(
        REPORT_ROOT / "model_summary.csv",
        model_rows,
        [
            "model",
            "s6_recipe",
            "s0_mean_mDice",
            "s6_mean_mDice",
            "delta_mean_mDice",
            "mDice_wins",
            "s0_mean_mIoU",
            "s6_mean_mIoU",
            "delta_mean_mIoU",
            "mIoU_wins",
        ],
    )
    (REPORT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Strict batch-32 segmentation: earliest S0 vs current-best S6",
        "",
        "All values are test-split, seven-dataset equal means. The probe uses batch 32, "
        "50 epochs, and seed 0. CoNIC uses sqrt-inverse class weights.",
        "",
        "Important: this is a tuned current-best comparison, not a controlled N-scaling "
        "curve. S6 objective, checkpoint, and interpolation alpha differ by model size.",
        "",
        "| Model | Current S6 recipe | S0 mDice | S6 mDice | Delta | Wins | S0 mIoU | S6 mIoU | Delta | Wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['s6_recipe']} | "
            f"{row['s0_mean_mDice']:.6f} | {row['s6_mean_mDice']:.6f} | "
            f"{row['delta_mean_mDice']:+.6f} | {row['mDice_wins']}/7 | "
            f"{row['s0_mean_mIoU']:.6f} | {row['s6_mean_mIoU']:.6f} | "
            f"{row['delta_mean_mIoU']:+.6f} | {row['mIoU_wins']}/7 |"
        )
    lines.extend(
        [
            "",
            "The S0 baseline is the earliest microscopy-trained model family at shared "
            "ck8199 (8.3958M image visits), not an official DINOv3 checkpoint.",
            "",
            "Machine-readable per-dataset values and source paths are in "
            "`dataset_results.csv`; full protocol and cache provenance are in `manifest.json`.",
            "",
        ]
    )
    (REPORT_ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
