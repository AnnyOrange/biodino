#!/usr/bin/env python3
"""Validate and summarize the strict current-best S6 N-scaling evaluation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs/02_eval_runs/s6_best_n_protocol_20260813"
REPORT_ROOT = ROOT / "outputs/00_reports/s6_best_n_protocol_20260813"

CLASSIFICATION = (
    "bbbc048-cellcycle",
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "cyclops-protein-loc",
    "dermamnist",
    "midog25-atypical",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
)
CLASSIFICATION_SIZE = {
    "bloodmnist": 384,
    "bbbc048-cellcycle": 512,
    "cyclops-protein-loc": 224,
    "midog25-atypical": 384,
    "chestmnist": 512,
}
RETRIEVAL = ("lc25000", "nct-crc-he-1k", "crc-val-he-7k")
SEGMENTATION = (
    "bbbc038",
    "cellpose",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
)
SEGMENTATION_PROTOCOL = {
    "bbbc038": (512, "pad", "even4", "none"),
    "cellpose": (512, "pad", "last1", "none"),
    "conic": (256, "stretch", "even4", "sqrt_inverse"),
    "livecell": (512, "pad", "even4", "none"),
    "monuseg": (768, "pad", "last1", "none"),
    "pannuke": (256, "stretch", "even4", "none"),
    "tissuenet": (256, "stretch", "last1", "none"),
}
EVEN4_LAYERS = {
    "S+": "custom_2_5_8_11",
    "B": "custom_2_5_8_11",
    "L": "custom_4_11_17_23",
    "H+": "custom_7_15_23_31",
}

MODELS = {
    "S+": {
        "directory": "Splus_sigreg005_alpha075",
        "checkpoint": "75",
        "recipe": "SigReg=0.05, ck8199, alpha=0.75",
        "checkpoint_fragment": "S6interp_official_sigreg005repl8199_20260719/ckpt/75/checkpoint.pth",
        "config_fragment": "S6interp_official_sigreg005repl8199_20260719/config.yaml",
    },
    "B": {
        "directory": "B_sigreg005_alpha060",
        "checkpoint": "60",
        "recipe": "SigReg=0.05, ck14349, alpha=0.60",
        "checkpoint_fragment": "B_s6recipe_sigreg005_alpha_tune_20260804/ckpt/60/checkpoint.pth",
        "config_fragment": "B_s6recipe_sigreg005_alpha_tune_20260804/config.yaml",
    },
    "L": {
        "directory": "L_nosigreg_raw6149_equiv_alpha100",
        "checkpoint": "100",
        "recipe": "no-SigReg, raw ck6149 (alpha100 tensor-identical export)",
        "checkpoint_fragment": "L_s6recipe_nosigreg_alpha_tune_20260804/ckpt/100/checkpoint.pth",
        "config_fragment": "L_s6recipe_nosigreg_alpha_tune_20260804/config.yaml",
    },
    "H+": {
        "directory": "Hplus_nosigreg_alpha100",
        "checkpoint": "100",
        "recipe": "no-SigReg, ck8199, alpha=1.00",
        "checkpoint_fragment": "hplus_nosigreg/alpha_tune_e15_2gpu/ckpt/100/checkpoint.pth",
        "config_fragment": "hplus_nosigreg/alpha_tune_e15_2gpu/config.yaml",
        "remote_source": "/data_2/suxin/runs/s6_best_n_protocol_20260813/Hplus_nosigreg_alpha100",
    },
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def unique(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern} under {root}, got {matches}")
    return matches[0]


def scalar_result(root: Path, family: str, dataset: str, checkpoint: str) -> Path:
    """Accept both one-dataset and grouped-dataset scheduler layouts."""
    candidates = list(root.glob(f"{family}/{dataset}/{checkpoint}/last_result.json"))
    candidates += list(root.glob(f"{family}/**/{checkpoint}/{dataset}/last_result.json"))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {family}/{dataset} result at checkpoint {checkpoint} "
            f"under {root}, got {candidates}"
        )
    return candidates[0]


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_rows: list[dict] = []
    summary_rows: list[dict] = []
    manifest: dict = {
        "scope": "S6 current-best tuned N-scaling; ID tasks only",
        "protocol_document": relative(REPORT_ROOT / "PROTOCOL.md"),
        "models": {},
    }

    hplus_root = EVAL_ROOT / MODELS["H+"]["directory"]
    if not hplus_root.exists():
        raise RuntimeError(
            "Copy the completed H+ remote result tree into "
            f"{hplus_root} before summarizing. Source: {MODELS['H+']['remote_source']}"
        )

    for model, spec in MODELS.items():
        root = EVAL_ROOT / spec["directory"]
        if not (root / "ALL_DONE").is_file():
            raise RuntimeError(f"Evaluation is not complete: {root}")
        scalar_root = root / "scalar_detection"
        seg_root = root / "segmentation"
        scalar_log = (root / "logs/scalar_detection.log").read_text()
        seg_log = (root / "logs/segmentation.log").read_text()

        required_scalar_tokens = (
            "--frozen-batch-size 32",
            "--classification-resolution-protocol best",
            "--det-batch-size 8",
            "--det-epochs 5",
            "--seed 0",
        )
        required_seg_tokens = (
            "--seg-feature-batch-size 32",
            "--seg-probe-batch-size 32",
            "--seg-probe-epochs 50",
            "--segmentation-protocol best",
            "--seed 0",
        )
        for token in required_scalar_tokens:
            if token not in scalar_log:
                raise RuntimeError(f"Missing scalar protocol token {token!r}: {root}")
        for token in required_seg_tokens:
            if token not in seg_log:
                raise RuntimeError(f"Missing segmentation protocol token {token!r}: {root}")

        classification_values = []
        source_paths: dict[str, dict[str, str] | str] = {"classification": {}}
        for dataset in CLASSIFICATION:
            path = scalar_result(
                scalar_root, "bio_classification", dataset, spec["checkpoint"]
            )
            result = load(path)
            expected_size = CLASSIFICATION_SIZE.get(dataset, 224)
            if result.get("dataset") != dataset:
                raise RuntimeError(f"Dataset mismatch: {path}")
            if result.get("resolution_protocol") != "best":
                raise RuntimeError(f"Not best resolution protocol: {path}")
            if int(result.get("image_size", -1)) != expected_size:
                raise RuntimeError(f"Wrong classification size for {dataset}: {path}")
            if result.get("channel_policy") != "auto":
                raise RuntimeError(f"Wrong channel policy: {path}")
            if result.get("split") not in {"official-test", "group-split"}:
                raise RuntimeError(f"Wrong classification split: {path}")
            if spec["checkpoint_fragment"] not in result.get("checkpoint", ""):
                raise RuntimeError(f"Wrong classification checkpoint: {path}")
            if spec["config_fragment"] not in result.get("train_config", ""):
                raise RuntimeError(f"Wrong classification config: {path}")
            metric = "macro_auc" if dataset == "chestmnist" else "balanced_accuracy"
            value = float(result[metric])
            classification_values.append(value)
            dataset_rows.append(
                {
                    "model": model,
                    "family": "classification",
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "result_path": relative(path),
                }
            )
            source_paths["classification"][dataset] = relative(path)

        regression_path = scalar_result(
            scalar_root, "bio_regression", "bbbc005", spec["checkpoint"]
        )
        regression_result = load(regression_path)
        if regression_result.get("dataset") != "bbbc005":
            raise RuntimeError(f"Wrong regression dataset: {regression_path}")
        if regression_result.get("channel_policy") != "auto":
            raise RuntimeError(f"Wrong regression channel policy: {regression_path}")
        if spec["checkpoint_fragment"] not in regression_result.get("checkpoint", ""):
            raise RuntimeError(f"Wrong regression checkpoint: {regression_path}")
        if spec["config_fragment"] not in regression_result.get("train_config", ""):
            raise RuntimeError(f"Wrong regression config: {regression_path}")
        regression_value = float(regression_result["r2"])
        dataset_rows.append(
            {
                "model": model,
                "family": "regression",
                "dataset": "bbbc005",
                "metric": "r2",
                "value": regression_value,
                "result_path": relative(regression_path),
            }
        )
        source_paths["regression"] = relative(regression_path)

        retrieval_values = []
        clustering_values = []
        source_paths["retrieval_clustering"] = {}
        for dataset in RETRIEVAL:
            path = scalar_result(
                scalar_root, "bio_retrieval", dataset, spec["checkpoint"]
            )
            result = load(path)
            if result.get("dataset") != dataset:
                raise RuntimeError(f"Wrong retrieval dataset: {path}")
            if result.get("channel_policy") != "auto":
                raise RuntimeError(f"Wrong retrieval channel policy: {path}")
            if spec["checkpoint_fragment"] not in result.get("checkpoint", ""):
                raise RuntimeError(f"Wrong retrieval checkpoint: {path}")
            if spec["config_fragment"] not in result.get("train_config", ""):
                raise RuntimeError(f"Wrong retrieval config: {path}")
            retrieval_value = float(result["recall_at_1"])
            clustering_value = float(result["nmi"])
            retrieval_values.append(retrieval_value)
            clustering_values.append(clustering_value)
            for family, metric, value in (
                ("retrieval", "recall_at_1", retrieval_value),
                ("clustering", "nmi", clustering_value),
            ):
                dataset_rows.append(
                    {
                        "model": model,
                        "family": family,
                        "dataset": dataset,
                        "metric": metric,
                        "value": value,
                        "result_path": relative(path),
                    }
                )
            source_paths["retrieval_clustering"][dataset] = relative(path)

        segmentation_values = []
        source_paths["segmentation"] = {}
        for dataset in SEGMENTATION:
            path = unique(
                seg_root,
                f"bio_segmentation/**/{dataset}/{spec['checkpoint']}/results.json",
            )
            result = load(path)
            meta = result.get("_meta", {})
            size, resize, feature_policy, expected_weight = SEGMENTATION_PROTOCOL[dataset]
            expected_layer = "last1" if feature_policy == "last1" else EVEN4_LAYERS[model]
            path_text = str(path)
            required_path_tokens = [f"__{expected_layer}", f"__s{size}"]
            if resize == "pad":
                required_path_tokens.append("__pad")
            elif "__pad" in path_text:
                raise RuntimeError(f"Unexpected pad resize in segmentation path: {path}")
            if expected_weight == "sqrt_inverse":
                required_path_tokens.append("__cw_sqrt_inverse")
            for token in required_path_tokens:
                if token not in path_text:
                    raise RuntimeError(
                        f"Missing segmentation protocol token {token!r}: {path}"
                    )
            expected_meta = {
                "probe_batch_size": 32,
                "probe_epochs": 50,
                "seed": 0,
                "class_weight_mode": expected_weight,
            }
            for key, expected in expected_meta.items():
                if meta.get(key) != expected:
                    raise RuntimeError(
                        f"Wrong segmentation {key}: {path}: {meta.get(key)!r} != {expected!r}"
                    )
            value = float(result["test"]["mDice"])
            segmentation_values.append(value)
            dataset_rows.append(
                {
                    "model": model,
                    "family": "segmentation",
                    "dataset": dataset,
                    "metric": "test_mDice",
                    "value": value,
                    "result_path": relative(path),
                }
            )
            source_paths["segmentation"][dataset] = relative(path)

        detection_path = unique(
            scalar_root,
            f"bio_detection/livecell/{spec['checkpoint']}/results_bio_detection.json",
        )
        detection_result = load(detection_path)
        expected_detection = {
            "dataset": "livecell",
            "batch_size": 8,
            "epochs": 5,
            "image_size": 224,
            "seed": 0,
        }
        for key, expected in expected_detection.items():
            if detection_result.get(key) != expected:
                raise RuntimeError(f"Wrong detection {key}: {detection_path}")
        detection_value = float(detection_result["test_patch_f1"])
        if detection_value > 1:
            detection_value /= 100.0
        dataset_rows.append(
            {
                "model": model,
                "family": "detection",
                "dataset": "livecell",
                "metric": "test_patch_f1",
                "value": detection_value,
                "result_path": relative(detection_path),
            }
        )
        source_paths["detection"] = relative(detection_path)

        row = {
            "model": model,
            "recipe": spec["recipe"],
            "classification_c15_primary": statistics.fmean(classification_values),
            "regression_bbbc005": regression_value,
            "retrieval_ret3": statistics.fmean(retrieval_values),
            "clustering_ret3": statistics.fmean(clustering_values),
            "segmentation_seg7": statistics.fmean(segmentation_values),
            "detection_livecell": detection_value,
        }
        summary_rows.append(row)
        manifest["models"][model] = {**row, "sources": source_paths}

    write_csv(
        REPORT_ROOT / "per_dataset.csv",
        dataset_rows,
        ["model", "family", "dataset", "metric", "value", "result_path"],
    )
    write_csv(
        REPORT_ROOT / "task_family_summary.csv",
        summary_rows,
        [
            "model",
            "recipe",
            "classification_c15_primary",
            "regression_bbbc005",
            "retrieval_ret3",
            "clustering_ret3",
            "segmentation_seg7",
            "detection_livecell",
        ],
    )
    (REPORT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
