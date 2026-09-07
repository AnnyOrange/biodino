#!/usr/bin/env python3
"""Create a provenance-first Fig. 3 data ledger without rendering figures.

The Fig. 3 work accumulated results from several evaluation campaigns.  This
script deliberately keeps incompatible protocols in separate cohorts instead
of silently merging their aggregate scores.  It writes raw/aggregate result
rows, cohort definitions, issues, and a source-hash manifest for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
HPLUS_DETAILS = FIG_DIR / "sources/hplus_s6_alpha1_full_details.json"
HPLUS_SCALARS = FIG_DIR / "sources/hplus_s6_scalar_metrics_complete.json"
EXTERNAL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_fair_protocol_20260721"
GAPFILL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_hplus_protocol_gapfill_20260811"
LAST_LAYER_ROOT = REPO_ROOT / "outputs/02_eval_runs/fig3_hplus_external_dense_protocol_20260813_v2"
HISTORICAL_ROOT = REPO_ROOT / "outputs/00_reports/20260708_taskwise_fm_figures_vertical_white"

DISPLAY = {
    "bioclip": "BioCLIP", "conch": "CONCH", "cytoimagenet": "CytoImageNet",
    "cytoself": "CytoSelf", "dinov2": "DINOv2", "gigapath": "GigaPath",
    "hoptimus0": "H-optimus-0", "jump_cp": "JUMP-CP", "mae": "MAE", "pe": "PE",
    "phikon2": "Phikon-v2", "siglip2": "SigLIP2", "uni": "UNI", "virchow2": "Virchow2",
    "biodino_hplus": "BioDINO H+/16", "dinov3_hplus_official": "DINOv3 H+/16",
    "imagenet_resnet50": "ImageNet ResNet-50", "dinov2_local": "DINOv2",
    "dinov3_official_vitl16": "DINOv3 H+/16", "dinov3_official_vit7b16": "DINOv3 7B/16",
    "biodino_hplus_s6_alpha1": "BioDINO H+/16",
}
EXTERNAL_MODELS = [
    "bioclip", "conch", "cytoimagenet", "cytoself", "dinov2", "gigapath",
    "hoptimus0", "jump_cp", "mae", "pe", "phikon2", "siglip2", "uni", "virchow2",
]
CURRENT_SEGMENTATION_ID_DATASETS = {
    "bbbc038", "cellpose", "conic", "livecell", "monuseg", "pannuke", "tissuenet",
}
LAST_LAYER_SEGMENTATION_DATASETS = {
    "bbbc038", "conic", "livecell", "monuseg", "pannuke", "tissuenet",
}

LEDGER_FIELDS = [
    "record_id", "panel", "cohort_id", "cohort_status", "eligible_for_main_figure",
    "model_key", "model", "checkpoint_or_variant", "task", "dataset", "split_or_fold",
    "metric", "value", "aggregation", "feature_protocol", "input_resolution",
    "probe_protocol", "source_path", "source_type", "source_sha256", "notes",
]
COHORT_FIELDS = [
    "cohort_id", "panel", "cohort_status", "eligible_for_main_figure", "models",
    "datasets", "metrics", "split_or_fold", "feature_protocol", "input_resolution",
    "probe_protocol", "aggregation", "source_paths", "comparability_statement",
]
ISSUE_FIELDS = [
    "issue_id", "severity", "panel", "cohort_id", "description", "impact", "required_action",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any, source: Path, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {source}: {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite {field} in {source}")
    return result


class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.cohorts: list[dict[str, str]] = []
        self.issues: list[dict[str, str]] = []
        self.sources: set[Path] = set()
        self._row_index = 0

    def add_row(self, **values: Any) -> None:
        self._row_index += 1
        row = {field: "" for field in LEDGER_FIELDS}
        row.update({key: self.format(value) for key, value in values.items() if key in row})
        row["record_id"] = f"R{self._row_index:05d}"
        source = Path(row["source_path"]) if row["source_path"] else None
        if source and source.is_file():
            self.sources.add(source)
            row["source_sha256"] = sha256(source)
        self.rows.append(row)

    def add_cohort(self, **values: Any) -> None:
        row = {field: "" for field in COHORT_FIELDS}
        row.update({key: self.format(value) for key, value in values.items() if key in row})
        self.cohorts.append(row)

    def add_issue(self, **values: Any) -> None:
        row = {field: "" for field in ISSUE_FIELDS}
        row.update({key: self.format(value) for key, value in values.items() if key in row})
        row["issue_id"] = f"I{len(self.issues) + 1:03d}"
        self.issues.append(row)

    @staticmethod
    def format(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:.12g}"
        if isinstance(value, (list, tuple, set)):
            return ";".join(str(item) for item in value)
        return str(value)


def add_cohorts(ledger: Ledger) -> None:
    ledger.add_cohort(
        cohort_id="C_current_scalar_hplus_alpha_sweep", panel="C", cohort_status="candidate_current",
        eligible_for_main_figure=True, models="BioDINO H+/16 (alpha=1.00);DINOv3 H+/16 (alpha=0.00)",
        datasets="25 classification;BBBC005 regression;4 retrieval/clustering;LIVECell detection",
        metrics="macro_f1;r2;mae;recall_at_1;map_at_5;mrr;cluster_accuracy;ari;nmi;test_patch_f1",
        split_or_fold="ID evaluation snapshots; fold IDs absent from copied summary",
        feature_protocol="frozen encoder; task-specific prior scalar-evaluation protocol",
        input_resolution="task-specific; not recorded in copied summary",
        probe_protocol="task-specific; not recorded in copied summary",
        aggregation="raw per-dataset values; no cross-model aggregate", source_paths=[HPLUS_DETAILS, HPLUS_SCALARS],
        comparability_statement="Comparable only to a cohort with identical per-task evaluation configuration and fold.",
    )
    ledger.add_cohort(
        cohort_id="C_external_scalar_gapfill_20260811", panel="C", cohort_status="candidate_external",
        eligible_for_main_figure="conditional", models=[DISPLAY[item] for item in EXTERNAL_MODELS],
        datasets="25 classification;BBBC005 regression;4 retrieval/clustering;LIVECell detection",
        metrics="macro_f1;r2;mae;recall_at_1;map_at_5;mrr;cluster_accuracy;ari;nmi;test_patch_f1",
        split_or_fold="reported as ID by campaign, fold IDs must be audited from original run configs",
        feature_protocol="frozen encoder; model-specific feature extraction",
        input_resolution="not represented in summary CSVs", probe_protocol="campaign-specific frozen probes",
        aggregation="raw per-dataset values; no cross-model aggregate", source_paths=[EXTERNAL_ROOT, GAPFILL_ROOT],
        comparability_statement="Candidate comparison cohort; source run configs must confirm fold and probe parity before use.",
    )
    ledger.add_cohort(
        cohort_id="C_current_selected_spatial_hplus_alpha_sweep", panel="C", cohort_status="candidate_current",
        eligible_for_main_figure=True, models="BioDINO H+/16 (alpha=1.00);DINOv3 H+/16 (alpha=0.00)",
        datasets=sorted(CURRENT_SEGMENTATION_ID_DATASETS), metrics="test_mDice", split_or_fold="ID test results; fold IDs absent from copied summary",
        feature_protocol="dataset-selected last1 or custom_7_15_23_31 spatial features",
        input_resolution="dataset-selected: 256, 512, or 768; see source_path",
        probe_protocol="selected dense-probe configuration from alpha sweep", aggregation="mean over seven listed ID datasets only",
        source_paths=[HPLUS_DETAILS],
        comparability_statement="Valid current H+ vs original DINOv3 comparison. It cannot be combined with the six-dataset 224 last-layer ablation.",
    )
    ledger.add_cohort(
        cohort_id="C_historical_taskwise_fm_20260708", panel="C", cohort_status="legacy_reference",
        eligible_for_main_figure="conditional", models="BioDINOv3 historical reference;11 external FMs",
        datasets="7 segmentation;3 retrieval/clustering;BBBC005 regression;task-specific classification",
        metrics="historical taskwise metrics exported separately", split_or_fold="not encoded in aggregate CSV",
        feature_protocol="historical selected protocol; per-row architecture/layer configuration not preserved in aggregate CSV",
        input_resolution="not encoded in aggregate CSV", probe_protocol="historical taskwise frozen-probe campaign",
        aggregation="aggregate-only exported values", source_paths=[HISTORICAL_ROOT],
        comparability_statement="Use as a legacy reference only. Do not relabel historical BioDINOv3 as the current H+ checkpoint or merge it with a different dataset set.",
    )
    ledger.add_cohort(
        cohort_id="C_last_layer_224_ablation_v2", panel="C", cohort_status="ablation_not_for_main",
        eligible_for_main_figure=False, models="BioDINO H+/16;DINOv3 H+/16;14 external FMs",
        datasets=sorted(LAST_LAYER_SEGMENTATION_DATASETS), metrics="test mDice", split_or_fold="ID test results",
        feature_protocol="final-layer patch tokens only", input_resolution="224", probe_protocol="frozen linear probe;20 epochs;seed 0",
        aggregation="mean over six datasets", source_paths=[LAST_LAYER_ROOT],
        comparability_statement="A valid explicitly labelled ablation at most; not comparable to the selected multi-layer spatial results used for the foundation-model conclusion.",
    )
    ledger.add_cohort(
        cohort_id="A_landscape_cytoimagenet_subset", panel="A", cohort_status="exploratory_unvalidated",
        eligible_for_main_figure=False, models="BioDINO H+/16", datasets="137 held-out samples;8 biology labels;4 modalities",
        metrics="nearest-neighbor accuracy;cosine silhouette", split_or_fold="deterministic test mask; sample-level manifest preserved",
        feature_protocol="global embedding; exact encoder extraction parameters not stored in metrics JSON",
        input_resolution="not recorded in metadata", probe_protocol="none", aggregation="sample-level summary",
        source_paths=[FIG_DIR / "panel_a_test_mask.csv", FIG_DIR / "panel_a_crossmodality_metrics.json"],
        comparability_statement="Exploratory representation visualization; current summary does not establish modality invariance.",
    )
    ledger.add_cohort(
        cohort_id="B_crossmodal_pair_alignment", panel="B", cohort_status="exploratory_unvalidated",
        eligible_for_main_figure=False, models="BioDINO H+/16;ImageNet ResNet-50;DINOv2;DINOv3 H+/16",
        datasets="1,200 fixed pairs from the CytoImageNet subset", metrics="cosine similarity;alignment score;bootstrap CI",
        split_or_fold="deterministic held-out pair mask", feature_protocol="global embeddings; extraction settings not fully recorded",
        input_resolution="not recorded in metadata", probe_protocol="none", aggregation="mean pair similarity and bootstrap interval",
        source_paths=[FIG_DIR / "panel_b_test_pairs.csv", FIG_DIR / "panel_b_model_pair_scores.csv", FIG_DIR / "panel_b_model_alignment.csv"],
        comparability_statement="Exploratory pair analysis. Current confidence intervals cross zero, so it is not sufficient for a superiority claim.",
    )
    ledger.add_cohort(
        cohort_id="D_bbbc048_fewshot_partial", panel="D", cohort_status="incomplete_not_for_main",
        eligible_for_main_figure="conditional", models="BioDINO H+/16;DINOv2;DINOv3 H+/16;DINOv3 7B/16;H-optimus-0;ImageNet ResNet-50",
        datasets="BBBC048-cellcycle", metrics="macro_f1", split_or_fold="committed source-group split; fixed test fold",
        feature_protocol="frozen global features", input_resolution="model-specific cached features", probe_protocol="class-balanced logistic regression;seed 0",
        aggregation="summary CSV contains 1/5/10/25/100% rows; metadata is stale and claims 10% only", source_paths=[FIG_DIR / "panel_d_fewshot_curves.csv", FIG_DIR / "panel_d_fewshot_summary.csv", FIG_DIR / "panel_d_fewshot_metadata.json"],
        comparability_statement="Candidate few-shot cohort only after matching model/fraction coverage and metadata are reconciled.",
    )
    ledger.add_cohort(
        cohort_id="E_last_layer_dense_transfer_v2", panel="E", cohort_status="ablation_not_for_main",
        eligible_for_main_figure=False, models="BioDINO H+/16;DINOv3 H+/16",
        datasets=sorted(LAST_LAYER_SEGMENTATION_DATASETS), metrics="test mDice;mIoU", split_or_fold="ID test",
        feature_protocol="final-layer patch tokens", input_resolution="224", probe_protocol="BatchNorm plus 1x1 convolution;20 epochs;seed 0",
        aggregation="paired per-dataset results", source_paths=[FIG_DIR / "panel_e_dense_transfer_metrics.csv", FIG_DIR / "panel_e_dense_transfer_metadata.json"],
        comparability_statement="Dense-transfer ablation only; it does not represent the selected multi-layer segmentation protocol.",
    )
    ledger.add_cohort(
        cohort_id="E_spatial_feature_visualization", panel="E", cohort_status="exploratory_visual_only",
        eligible_for_main_figure=False, models="BioDINO H+/16", datasets="one CIMA H&E image", metrics="none",
        split_or_fold="single illustrative image", feature_protocol="patch tokens;PCA/clustering", input_resolution="448",
        probe_protocol="none", aggregation="none", source_paths=[FIG_DIR / "panel_e_spatial_metadata.json"],
        comparability_statement="Qualitative visualization only; no numerical spatial-representation claim.",
    )


def add_current_hplus_scalar_rows(ledger: Ledger) -> None:
    details = json.loads(HPLUS_DETAILS.read_text())
    entries = {entry["label"]: entry for entry in details}
    labels = {"alpha_1.00": "biodino_hplus", "alpha_0.00": "dinov3_hplus_official"}
    for label, model_key in labels.items():
        entry = entries[label]
        for row in entry["dataset_rows"]:
            task = str(row["task"])
            if task == "segmentation":
                continue
            ledger.add_row(
                panel="C", cohort_id="C_current_scalar_hplus_alpha_sweep", cohort_status="candidate_current",
                eligible_for_main_figure=True, model_key=model_key, model=DISPLAY[model_key],
                checkpoint_or_variant=f"{label}; checkpoint={entry['checkpoint']}", task=task, dataset=row["dataset"],
                split_or_fold="ID evaluation snapshot; fold ID absent", metric=row["metric"], value=finite(row["value"], HPLUS_DETAILS, "value"),
                aggregation="per-dataset raw result", feature_protocol="task-specific frozen encoder", input_resolution="not recorded",
                probe_protocol="not recorded", source_path=HPLUS_DETAILS, source_type="local snapshot of remote result", notes=row["result_path"],
            )


def add_current_selected_segmentation_rows(ledger: Ledger) -> None:
    details = json.loads(HPLUS_DETAILS.read_text())
    entries = {entry["label"]: entry for entry in details}
    labels = {"alpha_1.00": "biodino_hplus", "alpha_0.00": "dinov3_hplus_official"}
    for label, model_key in labels.items():
        entry = entries[label]
        selected: list[float] = []
        for row in entry["dataset_rows"]:
            if row["task"] != "segmentation" or row["dataset"] not in CURRENT_SEGMENTATION_ID_DATASETS:
                continue
            score = finite(row["value"], HPLUS_DETAILS, "value")
            selected.append(score)
            path_text = str(row["result_path"])
            config = path_text.split("/bio_segmentation/")[-1].split("/")[0]
            parts = config.split("__")
            feature_protocol = next((part for part in parts if part == "last1" or part.startswith("custom_")), config)
            input_resolution = next((part for part in parts if part.startswith("s") and part[1:].isdigit()), "dataset-selected")
            ledger.add_row(
                panel="C", cohort_id="C_current_selected_spatial_hplus_alpha_sweep", cohort_status="candidate_current",
                eligible_for_main_figure=True, model_key=model_key, model=DISPLAY[model_key],
                checkpoint_or_variant=f"{label}; checkpoint={entry['checkpoint']}", task="segmentation", dataset=row["dataset"],
                split_or_fold="ID test; fold ID absent", metric=row["metric"], value=score, aggregation="per-dataset raw result",
                feature_protocol=feature_protocol, input_resolution=input_resolution, probe_protocol=config,
                source_path=HPLUS_DETAILS, source_type="local snapshot of remote result", notes=path_text,
            )
        if set(CURRENT_SEGMENTATION_ID_DATASETS) != {
            row["dataset"] for row in entry["dataset_rows"] if row["task"] == "segmentation"
        } & CURRENT_SEGMENTATION_ID_DATASETS:
            raise ValueError(f"Missing current selected segmentation datasets for {label}")
        ledger.add_row(
            panel="C", cohort_id="C_current_selected_spatial_hplus_alpha_sweep", cohort_status="candidate_current",
            eligible_for_main_figure=True, model_key=model_key, model=DISPLAY[model_key],
            checkpoint_or_variant=f"{label}; checkpoint={entry['checkpoint']}", task="segmentation", dataset="__mean_7_id_datasets__",
            split_or_fold="ID test; seven-dataset fixed set", metric="test_mDice", value=sum(selected) / len(selected),
            aggregation="unweighted arithmetic mean over seven per-dataset values", feature_protocol="dataset-selected spatial features",
            input_resolution="dataset-selected", probe_protocol="dataset-selected dense probes", source_path=HPLUS_DETAILS,
            source_type="derived from local snapshot", notes="bbbc038;cellpose;conic;livecell;monuseg;pannuke;tissuenet",
        )


def add_current_hplus_extra_scalar_metrics(ledger: Ledger) -> None:
    """Add metrics absent from the compact alpha-sweep detail snapshot.

    ``hplus_s6_alpha1_full_details.json`` retains macro F1, mAP@5, NMI and
    R2.  The companion scalar snapshot has the remaining raw metrics and the
    extraction metadata, so retain it rather than deriving values from a plot.
    """
    payload = json.loads(HPLUS_SCALARS.read_text())
    for model_key, item in payload["models"].items():
        checkpoint = item.get("checkpoint", "not recorded")
        for dataset, values in item["retrieval_clustering"].items():
            for task, metrics in {
                "retrieval": ("recall_at_1", "mrr"),
                "clustering": ("cluster_accuracy", "ari"),
            }.items():
                for metric in metrics:
                    ledger.add_row(
                        panel="C", cohort_id="C_current_scalar_hplus_alpha_sweep", cohort_status="candidate_current",
                        eligible_for_main_figure=True, model_key=model_key, model=DISPLAY[model_key],
                        checkpoint_or_variant=f"alpha sweep checkpoint={checkpoint}", task=task, dataset=dataset,
                        split_or_fold="frozen ID scalar evaluation; fold ID absent", metric=metric,
                        value=finite(values[metric], HPLUS_SCALARS, metric), aggregation="per-dataset raw result",
                        feature_protocol="global frozen features", input_resolution=str(values.get("image_size", "not recorded")),
                        probe_protocol=f"channel_policy={values.get('channel_policy', 'not recorded')}; channel_tta_samples={values.get('channel_tta_samples', 'not recorded')}",
                        source_path=HPLUS_SCALARS, source_type="local snapshot of remote result", notes=f"n_samples={values.get('n_samples', 'not recorded')}; {values.get('feature_file', '')}",
                    )
        regression = item["regression"]
        ledger.add_row(
            panel="C", cohort_id="C_current_scalar_hplus_alpha_sweep", cohort_status="candidate_current",
            eligible_for_main_figure=True, model_key=model_key, model=DISPLAY[model_key],
            checkpoint_or_variant=f"alpha sweep checkpoint={checkpoint}", task="regression", dataset="bbbc005",
            split_or_fold=str(regression.get("split", "not recorded")), metric="mae",
            value=finite(regression["mae"], HPLUS_SCALARS, "mae"), aggregation="per-dataset raw result",
            feature_protocol="global frozen features", input_resolution=str(regression.get("image_size", "not recorded")),
            probe_protocol=f"channel_policy={regression.get('channel_policy', 'not recorded')}; channel_tta_samples={regression.get('channel_tta_samples', 'not recorded')}",
            source_path=HPLUS_SCALARS, source_type="local snapshot of remote result", notes=f"n_train={regression.get('n_train', 'not recorded')}; n_test={regression.get('n_test', 'not recorded')}",
        )


def add_external_scalar_rows(ledger: Ledger) -> None:
    for model_key in EXTERNAL_MODELS:
        scalar_path = EXTERNAL_ROOT / "classification" / model_key / "summary.csv"
        for row in read_csv(scalar_path):
            task = row.get("task", "")
            if task in {"classification", "multilabel_classification"}:
                metric = "macro_f1"
            elif task == "regression" and row.get("dataset") == "bbbc005":
                for metric in ("r2", "mae"):
                    ledger.add_row(
                        panel="C", cohort_id="C_external_scalar_gapfill_20260811", cohort_status="candidate_external",
                        eligible_for_main_figure="conditional", model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="external checkpoint; ID unknown",
                        task="regression", dataset="bbbc005", split_or_fold=row.get("split", "not recorded"), metric=metric,
                        value=finite(row.get(metric), scalar_path, metric), aggregation="per-dataset raw result",
                        feature_protocol="model-specific frozen features", input_resolution="not recorded", probe_protocol="campaign frozen probe",
                        source_path=scalar_path, source_type="campaign summary CSV", notes="",
                    )
                continue
            else:
                continue
            ledger.add_row(
                panel="C", cohort_id="C_external_scalar_gapfill_20260811", cohort_status="candidate_external",
                eligible_for_main_figure="conditional", model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="external checkpoint; ID unknown",
                task="classification", dataset=row["dataset"], split_or_fold=row.get("split", "not recorded"), metric=metric,
                value=finite(row.get(metric), scalar_path, metric), aggregation="per-dataset raw result",
                feature_protocol="model-specific frozen features", input_resolution="not recorded", probe_protocol="campaign frozen probe",
                source_path=scalar_path, source_type="campaign summary CSV", notes="",
            )
        retrieval_path = GAPFILL_ROOT / "retrieval_clustering" / model_key / "summary.csv"
        for row in read_csv(retrieval_path):
            if row.get("error"):
                raise ValueError(f"Gapfill error in {retrieval_path}: {row['error']}")
            for task, metrics in {
                "retrieval": ("recall_at_1", "map_at_5", "mrr"),
                "clustering": ("cluster_accuracy", "ari", "nmi"),
            }.items():
                for metric in metrics:
                    ledger.add_row(
                        panel="C", cohort_id="C_external_scalar_gapfill_20260811", cohort_status="candidate_external",
                        eligible_for_main_figure="conditional", model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="external checkpoint; ID unknown",
                        task=task, dataset=row["dataset"], split_or_fold="reported ID; fold ID absent", metric=metric,
                        value=finite(row.get(metric), retrieval_path, metric), aggregation="per-dataset raw result",
                        feature_protocol="model-specific frozen features", input_resolution="not recorded", probe_protocol="gapfill campaign",
                        source_path=retrieval_path, source_type="gapfill campaign CSV", notes="",
                    )
        detection_path = GAPFILL_ROOT / "detection" / model_key / "results_bio_detection.json"
        detection = json.loads(detection_path.read_text())
        score = finite(detection.get("test_patch_f1"), detection_path, "test_patch_f1")
        ledger.add_row(
            panel="C", cohort_id="C_external_scalar_gapfill_20260811", cohort_status="candidate_external",
            eligible_for_main_figure="conditional", model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="external checkpoint; ID unknown",
            task="detection", dataset="livecell", split_or_fold="reported ID; fold ID absent", metric="test_patch_f1",
            value=score / 100.0 if score > 1 else score, aggregation="per-dataset raw result", feature_protocol="model-specific frozen features",
            input_resolution="not recorded", probe_protocol="gapfill campaign", source_path=detection_path,
            source_type="gapfill campaign JSON", notes="Converted percent to fraction when needed.",
        )


def add_last_layer_segmentation_rows(ledger: Ledger) -> None:
    models = ["biodino_hplus", "dinov3_hplus_official", *EXTERNAL_MODELS]
    for model_key in models:
        values: list[float] = []
        for dataset in sorted(LAST_LAYER_SEGMENTATION_DATASETS):
            root = LAST_LAYER_ROOT / model_key if model_key in {"biodino_hplus", "dinov3_hplus_official"} else EXTERNAL_ROOT / "segmentation"
            result = root / "linear_probe" / dataset / model_key / "results.json"
            if not result.is_file():
                raise FileNotFoundError(result)
            payload = json.loads(result.read_text())
            score = finite(payload["test"].get("mDice"), result, "test.mDice")
            values.append(score)
            ledger.add_row(
                panel="C", cohort_id="C_last_layer_224_ablation_v2", cohort_status="ablation_not_for_main",
                eligible_for_main_figure=False, model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="see source result",
                task="segmentation", dataset=dataset, split_or_fold="ID test", metric="test_mDice", value=score,
                aggregation="per-dataset raw result", feature_protocol="final-layer patch tokens only", input_resolution="224",
                probe_protocol="frozen linear probe;20 epochs;seed 0", source_path=result,
                source_type="temporary ablation JSON" if model_key in {"biodino_hplus", "dinov3_hplus_official"} else "external fair-protocol JSON",
                notes="Do not mix with selected multi-layer segmentation results. External protocol metadata still requires audit.",
            )
        ledger.add_row(
            panel="C", cohort_id="C_last_layer_224_ablation_v2", cohort_status="ablation_not_for_main",
            eligible_for_main_figure=False, model_key=model_key, model=DISPLAY[model_key], checkpoint_or_variant="see source results",
            task="segmentation", dataset="__mean_6_id_datasets__", split_or_fold="ID test", metric="test_mDice", value=sum(values) / len(values),
            aggregation="unweighted arithmetic mean over six per-dataset values", feature_protocol="final-layer patch tokens only", input_resolution="224",
            probe_protocol="frozen linear probe;20 epochs;seed 0", source_path="", source_type="derived from temporary ablation",
            notes="bbbc038;conic;livecell;monuseg;pannuke;tissuenet",
        )


def add_historical_rows(ledger: Ledger) -> None:
    files = {
        "classification": "id_classification_balanced_accuracy.csv",
        "segmentation": "id_segmentation_mdice.csv",
        "retrieval": "id_retrieval_recall_at_1.csv",
        "retrieval_mrr": "id_retrieval_mrr.csv",
        "retrieval_map_at_10": "id_retrieval_map_at_10.csv",
        "clustering": "id_clustering_nmi.csv",
        "clustering_cluster_accuracy": "id_clustering_cluster_accuracy.csv",
        "clustering_ari": "id_clustering_ari.csv",
        "regression": "id_regression_r2.csv",
        "regression_mae": "id_regression_mae.csv",
    }
    for task_key, filename in files.items():
        path = HISTORICAL_ROOT / filename
        if not path.is_file():
            continue
        task, metric = task_key, "score"
        if task_key == "segmentation": task, metric = "segmentation", "mDice"
        elif task_key == "retrieval": task, metric = "retrieval", "recall_at_1"
        elif task_key == "retrieval_mrr": task, metric = "retrieval", "mrr"
        elif task_key == "retrieval_map_at_10": task, metric = "retrieval", "map_at_10"
        elif task_key == "clustering": task, metric = "clustering", "nmi"
        elif task_key == "clustering_cluster_accuracy": task, metric = "clustering", "cluster_accuracy"
        elif task_key == "clustering_ari": task, metric = "clustering", "ari"
        elif task_key == "regression": task, metric = "regression", "r2"
        elif task_key == "regression_mae": task, metric = "regression", "mae"
        elif task_key == "classification": task, metric = "classification", "balanced_accuracy"
        for row in read_csv(path):
            model = row["model"]
            ledger.add_row(
                panel="C", cohort_id="C_historical_taskwise_fm_20260708", cohort_status="legacy_reference",
                eligible_for_main_figure="conditional", model_key=model.lower().replace("-", "_").replace(" ", "_"), model=model,
                checkpoint_or_variant="historical; exact checkpoint not stored in aggregate CSV", task=task, dataset="__historical_aggregate__",
                split_or_fold="not encoded", metric=metric, value=finite(row.get("score"), path, "score"),
                aggregation="historical aggregate-only score", feature_protocol="not encoded in aggregate CSV", input_resolution="not encoded",
                probe_protocol="historical taskwise frozen-probe campaign", source_path=path, source_type="legacy aggregate CSV",
                notes=f"Historical group={row.get('group', '')}; not directly mergeable with current checkpoint.",
            )


def add_panel_a_b_d_e_rows(ledger: Ledger) -> None:
    panel_a = FIG_DIR / "panel_a_crossmodality_metrics.json"
    metrics = json.loads(panel_a.read_text())
    for metric in ("modality_nn_acc", "biology_nn_acc", "modality_silhouette_cosine", "biology_silhouette_cosine"):
        ledger.add_row(
            panel="A", cohort_id="A_landscape_cytoimagenet_subset", cohort_status="exploratory_unvalidated",
            eligible_for_main_figure=False, model_key="biodino_hplus", model="BioDINO H+/16", checkpoint_or_variant="current H+ expected; not stored in metrics JSON",
            task="representation_landscape", dataset="cytoimagenet_subset", split_or_fold="deterministic test mask", metric=metric,
            value=finite(metrics[metric], panel_a, metric), aggregation=f"n={metrics['n']} samples", feature_protocol="global embeddings",
            input_resolution="not recorded", probe_protocol="none", source_path=panel_a, source_type="panel summary JSON", notes="",
        )
    panel_b = FIG_DIR / "panel_b_model_alignment.csv"
    for row in read_csv(panel_b):
        for metric in ("same_biology_cross_modality_mean", "different_biology_same_modality_mean", "alignment_score", "ci_low", "ci_high"):
            ledger.add_row(
                panel="B", cohort_id="B_crossmodal_pair_alignment", cohort_status="exploratory_unvalidated",
                eligible_for_main_figure=False, model_key=row["model"], model=DISPLAY.get(row["model"], row["model"]), checkpoint_or_variant="not recorded",
                task="crossmodal_alignment", dataset="cytoimagenet_subset", split_or_fold="fixed 1,200-pair mask", metric=metric,
                value=finite(row[metric], panel_b, metric), aggregation=f"n_pairs={row['n_pairs']}", feature_protocol="global embeddings",
                input_resolution="not recorded", probe_protocol="none", source_path=panel_b, source_type="panel summary CSV", notes="",
            )
    panel_d = FIG_DIR / "panel_d_fewshot_summary.csv"
    for row in read_csv(panel_d):
        ledger.add_row(
            panel="D", cohort_id="D_bbbc048_fewshot_partial", cohort_status="metadata_conflict",
            eligible_for_main_figure="conditional", model_key=row.get("model_key", row["model"]), model=row["model"], checkpoint_or_variant="not recorded",
            task="classification_fewshot", dataset=row["dataset"], split_or_fold=row["split"], metric=row["metric"], value=finite(row["score"], panel_d, "score"),
            aggregation=f"label_percent={row['label_percent']}; n_seeds={row['n_seeds']}; score_std={row['score_std']}", feature_protocol="frozen features",
            input_resolution="not recorded", probe_protocol="class-balanced logistic regression", source_path=panel_d, source_type="few-shot summary CSV", notes="Only completed label fraction is eligible for recording.",
        )
    panel_e = FIG_DIR / "panel_e_dense_transfer_metrics.csv"
    for row in read_csv(panel_e):
        for metric in ("mDice", "mIoU"):
            ledger.add_row(
                panel="E", cohort_id="E_last_layer_dense_transfer_v2", cohort_status="ablation_not_for_main",
                eligible_for_main_figure=False, model_key=row["model"], model=row.get("model_label", DISPLAY.get(row["model"], row["model"])), checkpoint_or_variant="see result JSON",
                task="segmentation_dense_transfer", dataset=row["dataset"], split_or_fold="ID test", metric=metric, value=finite(row[metric], panel_e, metric),
                aggregation="per-dataset paired result", feature_protocol="final-layer patch tokens", input_resolution="224",
                probe_protocol="BatchNorm plus 1x1 convolution;20 epochs;seed 0", source_path=panel_e, source_type="panel metrics CSV", notes=row["results_json"],
            )


def add_issues(ledger: Ledger) -> None:
    ledger.add_issue(
        severity="critical", panel="C", cohort_id="C_last_layer_224_ablation_v2",
        description="The rendered C segmentation values came from a six-dataset, 224, final-layer, 20-epoch ablation.",
        impact="They contradict the project's selected multi-layer segmentation protocol and invert the established FM conclusion.",
        required_action="Exclude this cohort from all main-result CSVs and figures; retain it only as an explicitly labelled ablation.",
    )
    ledger.add_issue(
        severity="critical", panel="C", cohort_id="C_historical_taskwise_fm_20260708",
        description="Historical taskwise aggregates use a different model coverage and task dataset set than the new C matrix.",
        impact="Historical values, especially BioDINOv3, cannot be relabelled as the current H+ S6 alpha=1 checkpoint or averaged with new values.",
        required_action="Keep historical CSVs immutable; build a new common-protocol cohort only from raw per-dataset runs with checkpoint identifiers.",
    )
    ledger.add_issue(
        severity="high", panel="C", cohort_id="C_external_scalar_gapfill_20260811",
        description="External scalar/gapfill CSVs lack explicit fold IDs, input resolution, and feature-layer configuration.",
        impact="Their numeric values cannot yet be certified as protocol-matched to the current H+ snapshots.",
        required_action="Extract run config and split manifests for every source campaign before declaring a cross-model ranking.",
    )
    ledger.add_issue(
        severity="high", panel="C", cohort_id="C_current_scalar_hplus_alpha_sweep",
        description="Current H+ scalar and selected spatial values are local snapshots whose original result paths are on the H100 filesystem.",
        impact="The values are traceable to a path string but fold/config artifacts are not locally materialized.",
        required_action="Copy the original result JSON/config/split manifests or generate a verified local snapshot manifest before final use.",
    )
    ledger.add_issue(
        severity="high", panel="D", cohort_id="D_bbbc048_fewshot_partial",
        description="The few-shot summary CSV contains 1/5/10/25/100% rows, while the metadata claims only 10% was completed.",
        impact="The curve's provenance is internally inconsistent until source-row coverage and generation metadata are reconciled.",
        required_action="Regenerate metadata from the final CSV and verify every model has all five fractions on the same fixed fold.",
    )
    ledger.add_issue(
        severity="high", panel="B", cohort_id="B_crossmodal_pair_alignment",
        description="All current alignment-score bootstrap intervals cross zero and embedding extraction metadata is incomplete.",
        impact="The data do not support a claim that BioDINO removes modality bias better than baselines.",
        required_action="Freeze a biologically valid paired benchmark and rerun with documented encoder extraction and stratified uncertainty analysis.",
    )
    ledger.add_issue(
        severity="medium", panel="A", cohort_id="A_landscape_cytoimagenet_subset",
        description="The only numeric landscape summary reports modality NN accuracy 0.927 and biology NN accuracy 0.869.",
        impact="This does not demonstrate removal of modality bias; visual UMAP separation is not sufficient evidence.",
        required_action="Do not use Panel A as a modality-invariance claim until the labelled benchmark and controls are redesigned.",
    )
    ledger.add_issue(
        severity="medium", panel="E", cohort_id="E_last_layer_dense_transfer_v2",
        description="Panel E uses the same last-layer 224 ablation as the erroneous C segmentation panel.",
        impact="It is informative only as a narrow ablation, not evidence for the selected multi-layer spatial representation.",
        required_action="Label it as an ablation or rebuild it from the selected spatial protocol.",
    )


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR / "data_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger()
    add_cohorts(ledger)
    add_current_hplus_scalar_rows(ledger)
    add_current_hplus_extra_scalar_metrics(ledger)
    add_current_selected_segmentation_rows(ledger)
    add_external_scalar_rows(ledger)
    add_last_layer_segmentation_rows(ledger)
    add_historical_rows(ledger)
    add_panel_a_b_d_e_rows(ledger)
    add_issues(ledger)
    write_csv(args.output_dir / "fig3_data_ledger.csv", ledger.rows, LEDGER_FIELDS)
    write_csv(args.output_dir / "fig3_data_cohorts.csv", ledger.cohorts, COHORT_FIELDS)
    write_csv(args.output_dir / "fig3_data_issues.csv", ledger.issues, ISSUE_FIELDS)
    manifest = {
        "description": "Fig. 3 audit ledger. Cohorts are intentionally not merged across protocol differences.",
        "ledger": str(args.output_dir / "fig3_data_ledger.csv"),
        "cohorts": str(args.output_dir / "fig3_data_cohorts.csv"),
        "issues": str(args.output_dir / "fig3_data_issues.csv"),
        "record_count": len(ledger.rows),
        "cohort_count": len(ledger.cohorts),
        "issue_count": len(ledger.issues),
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sorted(ledger.sources)],
    }
    (args.output_dir / "fig3_data_audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("record_count", "cohort_count", "issue_count")}, indent=2))


if __name__ == "__main__":
    main()
