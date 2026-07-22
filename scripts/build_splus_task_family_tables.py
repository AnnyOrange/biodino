#!/usr/bin/env python3
"""Build report-ready S+ tables with one table per downstream task family."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/00_reports/splus_sweetspot_live_results_summary.csv"
BBBC013 = ROOT / "outputs/00_reports/bbbc013_compound_oof_20260716/results.csv"
OUT = ROOT / "outputs/00_reports/splus_task_family_tables_20260716"
SEG_AUDIT = ROOT / "outputs/00_reports/splus_segmentation_audit_20260716"

MODELS = OrderedDict(
    [
        ("Official S+", ("S7z_splus_official_reference", "0", "Official")),
        ("H-S0 packwds GB1024", ("bio_continue_vits16_ep15_1025", "15374", "H_S0_packwds")),
        ("R-S0 packwds GB4096", ("S0b_packwds_dino256_b4096_lr2e-4_wu2_e15", "3899", "R_S0_packwds")),
        ("S1 robust+DINO", ("S1b_robust_dino256_b4096_lr2e-4_wu2_e15", "3899", "S1_robust")),
        ("S2 robust+BioAug wu2", ("S2b_robust_biosafe256_b4096_lr2e-4_wu2_e15", "3899", "S2_biosafe_wu2")),
        ("S2 robust+BioAug wu5", ("S2b_robust_biosafe256_b4096_lr2e-4_wu5_e15", "3899", "S2_biosafe_wu5")),
        ("S3 BioAug crop224", ("S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15", "3899", "S3_crop224")),
        ("S6 BioAug ck5199 (~20p)", ("S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30", "5199", "S6_ck5199")),
    ]
)

COMMON15 = [
    "bbbc048-cellcycle", "bloodmnist", "breastmnist", "chestmnist", "cyclops-protein-loc",
    "dermamnist", "midog25-atypical", "octmnist", "organamnist", "organcmnist",
    "organsmnist", "pathmnist", "pneumoniamnist", "retinamnist", "tissuemnist",
]
BIO3 = ["bloodmnist", "bbbc048-cellcycle", "cyclops-protein-loc"]
EXTRA3 = ["lc25000", "nct-crc-he", "pcam"]
CHAMMI7 = [
    "chammi-allen-task1", "chammi-allen-task2", "chammi-cp-task1", "chammi-cp-task2",
    "chammi-cp-task3", "chammi-hpa-task1", "chammi-hpa-task2",
]
CLASS25 = COMMON15 + CHAMMI7 + EXTRA3
RETRIEVAL4 = ["crc-val-he-7k", "lc25000", "nct-crc-he-100", "nct-crc-he-1k"]
SEG7 = ["bbbc038", "cellpose", "livecell", "tissuenet", "monuseg", "pannuke", "conic"]


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nonseg_priority(label: str, task: str, eval_run: str) -> int:
    if label == "Official S+":
        return 0 if eval_run == "S7z_splus_official_reference__Dscale_core_first3_ck0_qi8" else 50
    if label == "H-S0 packwds GB1024":
        return 0 if eval_run == "bio_eval_sp_all_ckpts_current_nonseg_20260630" else 50
    if label.startswith("S6"):
        if task == "classification" and "__Dscale_core_first3_ck3899_qi" in eval_run:
            return 0
        if task == "classification" and "__ckpt3899_cls_" in eval_run:
            return 1
        if task == "classification" and any(
            tag in eval_run
            for tag in ("__ckpt5199_cls_med", "__ckpt5199_cls_bio", "__ckpt5199_cls_chammi_cp", "__ckpt5199_cls_chammi_hpa")
        ):
            return 0
        if task == "classification" and "S6_ck5199_class_core_fill3090_singlecard_20260712_1616" in eval_run:
            return 1
        if task in {"regression", "retrieval"} and "__ckpt5199_regret" in eval_run:
            return 0
        if task in {"regression", "retrieval"} and "S6_ck5199_regret_fill3090_singlecard_20260712_1616" in eval_run:
            return 1
        return 20
    if task == "classification":
        if "__3899_cls_" in eval_run:
            return 0
        if "class_core_fill3090_singlecard_20260712_1616" in eval_run:
            return 1
    if task in {"regression", "retrieval"}:
        if "__3899_regret" in eval_run:
            return 0
        if "regret_fill3090_singlecard_20260712_1616" in eval_run:
            return 1
    return 20


def pick_nonseg(rows, label: str, task: str, dataset: str, ckpt_override: str | None = None):
    train_run, default_ckpt, _ = MODELS[label]
    ckpt = ckpt_override or default_ckpt
    if label == "S3 BioAug crop224" and task == "classification" and dataset in {"nct-crc-he", "pcam"}:
        path = (
            ROOT
            / "outputs/02_eval_runs/S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15__C25_missing_first3_20260716"
            / "bio_classification"
            / dataset
            / ckpt
            / "last_result.json"
        )
        if path.exists():
            result = json.loads(path.read_text())
            if not result.get("error"):
                return {
                    "primary_metric": "balanced_accuracy",
                    "metric_value": result["balanced_accuracy"],
                    "macro_f1": result["macro_f1"],
                    "eval_run": path.parents[3].name,
                    "result_path": str(path.relative_to(ROOT)),
                }
    candidates = [
        row for row in rows
        if row["train_run"] == train_run and row["ckpt"] == ckpt
        and row["task"] == task and row["dataset"] == dataset and row["metric_value"] != ""
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (nonseg_priority(label, task, row["eval_run"]), row["result_path"]))
    best = candidates[0]
    if nonseg_priority(label, task, best["eval_run"]) >= 20:
        return None
    return best


def pick_segmentation(rows, label: str, dataset: str):
    _, ckpt, _ = MODELS[label]
    candidates = [row for row in rows if row["task"] == "segmentation" and row["dataset"] == dataset and row["ckpt"] == ckpt]
    if label == "Official S+":
        candidates = [row for row in candidates if row["eval_run"] == "splus_random_data_scaling_seg_ood" and row["subrun"].startswith("official_0/")]
    elif label == "H-S0 packwds GB1024":
        candidates = [row for row in candidates if row["eval_run"] == "bioseg_best_7models_20260622" and row["subrun"] == "sp"]
    else:
        tag = {
            "R-S0 packwds GB4096": "S0_e15_seg_all_fill3090_singlecard_20260712_1616",
            "S1 robust+DINO": "S1_e15_seg_all_fill3090_singlecard_20260712_1616",
            "S2 robust+BioAug wu2": "S2WU2_e15_seg_all_fill3090_singlecard_20260712_1616",
            "S2 robust+BioAug wu5": "S2WU5_e15_seg_all_fill3090_singlecard_20260712_1616",
            "S3 BioAug crop224": "S3_e15_seg_all_fill3090_singlecard_20260712_1616",
            "S6 BioAug ck5199 (~20p)": "S6_ck5199_seg_all_fill3090_singlecard_20260712_1616",
        }.get(label)
        candidates = [row for row in candidates if tag and tag in row["eval_run"]]
    return sorted(candidates, key=lambda row: row["result_path"])[0] if candidates else None


def pick_detection(rows, label: str):
    if label == "Official S+":
        return None
    if label == "H-S0 packwds GB1024":
        path = ROOT / "outputs/02_eval_runs/bio_eval_sp_15374/bio_detection/livecell/15374/results_bio_detection.json"
        return json.loads(path.read_text())
    _, ckpt, _ = MODELS[label]
    tag = {
        "R-S0 packwds GB4096": "S0_e15_detection_fill3090_singlecard_20260712_1616",
        "S1 robust+DINO": "S1_e15_detection_fill3090_singlecard_20260712_1616",
        "S2 robust+BioAug wu2": "S2WU2_e15_detection_fill3090_singlecard_20260712_1616",
        "S2 robust+BioAug wu5": "S2WU5_e15_detection_fill3090_singlecard_20260712_1616",
        "S3 BioAug crop224": "S3_e15_detection_fill3090_singlecard_20260712_1616",
        "S6 BioAug ck5199 (~20p)": "S6_ck5199_detection_fill3090_singlecard_20260712_1616",
    }.get(label)
    candidates = [
        row for row in rows if row["task"] == "detection" and row["dataset"] == "livecell"
        and row["ckpt"] == ckpt and tag and tag in row["eval_run"]
    ]
    if not candidates:
        return None
    return json.loads(Path(candidates[0]["result_path"]).read_text())


def avg(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def complete_avg(values):
    values = list(values)
    return mean(values) if values and all(value is not None for value in values) else None


def write_csv(name: str, rows: list[dict]):
    path = OUT / name
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=5):
    return "N/A" if value is None else f"{value:.{digits}f}"


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits=5):
    lines = ["| " + " | ".join(title for _, title in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            cells.append(str(value) if isinstance(value, str) else fmt(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open() as f:
        rows = list(csv.DictReader(f))
    with BBBC013.open() as f:
        bbbc013 = {row["candidate"]: row for row in csv.DictReader(f)}

    classification = []
    retrieval = []
    clustering = []
    regression = []
    segmentation = []
    detection = []
    provenance = []

    for label, (_, _, bbbc_key) in MODELS.items():
        class_ckpt = "3899" if label.startswith("S6") else MODELS[label][1]
        class_primary, class_f1 = {}, {}
        for dataset in CLASS25:
            row = pick_nonseg(rows, label, "classification", dataset, ckpt_override=class_ckpt)
            class_primary[dataset] = fnum(row["metric_value"]) if row else None
            class_f1[dataset] = fnum(row["macro_f1"]) if row else None
            if row:
                for metric, value in ((row["primary_metric"], row["metric_value"]), ("macro_f1", row["macro_f1"])):
                    provenance.append(
                        {
                            "model": label,
                            "task": "classification",
                            "dataset": dataset,
                            "metric": metric,
                            "value": value,
                            "eval_run": row["eval_run"],
                            "result_path": row["result_path"],
                        }
                    )
        c25_present = sum(class_f1[d] is not None for d in CLASS25)
        classification.append(
            {
                "model": "S6 BioAug" if label.startswith("S6") else label,
                "checkpoint": class_ckpt,
                "c25_coverage": f"{c25_present}/25",
                "c25_macro_f1": complete_avg(class_f1[d] for d in CLASS25),
                "c25_primary_mean": complete_avg(class_primary[d] for d in CLASS25),
                "c15_macro_f1": complete_avg(class_f1[d] for d in COMMON15),
                "c15_primary_mean": complete_avg(class_primary[d] for d in COMMON15),
                "bio3_macro_f1": complete_avg(class_f1[d] for d in BIO3),
                "chammi7_macro_f1": complete_avg(class_f1[d] for d in CHAMMI7),
                **{f"macro_f1_{d}": class_f1[d] for d in CLASS25},
                **{f"primary_{d}": class_primary[d] for d in CLASS25},
            }
        )

        ret_values, nmi_values = {}, {}
        for dataset in RETRIEVAL4:
            row = pick_nonseg(rows, label, "retrieval", dataset)
            ret_values[dataset] = fnum(row["recall_at_1"]) if row else None
            nmi_values[dataset] = fnum(row["nmi"]) if row else None
            if row:
                for metric, value in (("recall_at_1", row["recall_at_1"]), ("nmi", row["nmi"])):
                    provenance.append(
                        {
                            "model": label,
                            "task": "retrieval" if metric == "recall_at_1" else "clustering",
                            "dataset": dataset,
                            "metric": metric,
                            "value": value,
                            "eval_run": row["eval_run"],
                            "result_path": row["result_path"],
                        }
                    )
        retrieval.append({"model": label, "macro_r1": complete_avg(ret_values.values()), **ret_values})
        clustering.append({"model": label, "macro_nmi": complete_avg(nmi_values.values()), **nmi_values})

        b5 = pick_nonseg(rows, label, "regression", "bbbc005")
        b13 = bbbc013.get(bbbc_key)
        b5_r2 = fnum(b5["r2"]) if b5 else None
        b13_r2 = fnum(b13["r2"]) if b13 else None
        if b5:
            provenance.append(
                {
                    "model": label,
                    "task": "regression",
                    "dataset": "bbbc005",
                    "metric": "r2",
                    "value": b5["r2"],
                    "eval_run": b5["eval_run"],
                    "result_path": b5["result_path"],
                }
            )
        if b13:
            provenance.append(
                {
                    "model": label,
                    "task": "regression",
                    "dataset": "bbbc013",
                    "metric": "compound_oof_macro_r2",
                    "value": b13["r2"],
                    "eval_run": "bbbc013_compound_oof_20260716",
                    "result_path": str(BBBC013.relative_to(ROOT)),
                }
            )
        regression.append(
            {
                "model": label,
                "regression2_r2_mean": complete_avg([b5_r2, b13_r2]),
                "bbbc005_r2": b5_r2,
                "bbbc013_macro_r2": b13_r2,
                "bbbc013_macro_spearman": fnum(b13["spearman"]) if b13 else None,
                "bbbc013_macro_mae": fnum(b13["mae"]) if b13 else None,
                "wortmannin_r2": fnum(b13["wortmannin_r2"]) if b13 else None,
                "ly294002_r2": fnum(b13["ly294002_r2"]) if b13 else None,
            }
        )

        seg_iou, seg_dice = {}, {}
        for dataset in SEG7:
            row = pick_segmentation(rows, label, dataset)
            seg_iou[dataset] = fnum(row["seg_mIoU"]) if row else None
            seg_dice[dataset] = fnum(row["seg_mDice"]) if row else None
            if row:
                for metric, value in (("mIoU", row["seg_mIoU"]), ("mDice", row["seg_mDice"])):
                    provenance.append(
                        {
                            "model": label,
                            "task": "segmentation",
                            "dataset": dataset,
                            "metric": metric,
                            "value": value,
                            "eval_run": row["eval_run"],
                            "result_path": row["result_path"],
                        }
                    )
        segmentation.append(
            {
                "model": label,
                "macro_mIoU": complete_avg(seg_iou.values()),
                "macro_mDice": complete_avg(seg_dice.values()),
                **{f"miou_{key}": value for key, value in seg_iou.items()},
                **{f"mdice_{key}": value for key, value in seg_dice.items()},
            }
        )

        det = pick_detection(rows, label)
        if det:
            provenance.append(
                {
                    "model": label,
                    "task": "detection",
                    "dataset": "livecell",
                    "metric": "test_patch_f1",
                    "value": det.get("test_patch_f1"),
                    "eval_run": "protocol-selected LiveCell detection",
                    "result_path": "embedded in detection source JSON",
                }
            )
        detection.append(
            {
                "model": label,
                "livecell_precision": fnum(det.get("test_patch_precision")) if det else None,
                "livecell_recall": fnum(det.get("test_patch_recall")) if det else None,
                "livecell_f1": fnum(det.get("test_patch_f1")) if det else None,
            }
        )

    # The original segmentation rows mixed batch-16 and batch-32 heads and
    # came from an evaluator whose seed did not control probe-side RNG. Replace
    # them with the corrected deterministic three-seed audit when available.
    seg_common_path = SEG_AUDIT / "segmentation_common7.csv"
    seg_dataset_path = SEG_AUDIT / "segmentation_per_dataset.csv"
    if seg_common_path.exists() and seg_dataset_path.exists():
        with seg_common_path.open() as f:
            seg_common = {row["model_key"]: row for row in csv.DictReader(f)}
        with seg_dataset_path.open() as f:
            seg_dataset = {
                (row["model_key"], row["dataset"]): row for row in csv.DictReader(f)
            }
        label_to_key = {
            "Official S+": "official",
            "H-S0 packwds GB1024": "h_s0",
            "R-S0 packwds GB4096": "r_s0",
            "S1 robust+DINO": "s1",
            "S2 robust+BioAug wu2": "s2_wu2",
            "S2 robust+BioAug wu5": "s2_wu5",
            "S3 BioAug crop224": "s3",
            "S6 BioAug ck5199 (~20p)": "s6_ck5199",
        }
        segmentation = []
        provenance = [row for row in provenance if row["task"] != "segmentation"]
        for label in MODELS:
            model_key = label_to_key[label]
            summary = seg_common[model_key]
            seg_iou = {
                dataset: fnum(seg_dataset[(model_key, dataset)]["mIoU_mean"])
                for dataset in SEG7
            }
            seg_dice = {
                dataset: fnum(seg_dataset[(model_key, dataset)]["mDice_mean"])
                for dataset in SEG7
            }
            segmentation.append(
                {
                    "model": label,
                    "macro_mIoU": fnum(summary["macro_mIoU_mean"]),
                    "macro_mIoU_std": fnum(summary["macro_mIoU_std"]),
                    "macro_mDice": fnum(summary["macro_mDice_mean"]),
                    "macro_mDice_std": fnum(summary["macro_mDice_std"]),
                    **{f"miou_{key}": value for key, value in seg_iou.items()},
                    **{f"mdice_{key}": value for key, value in seg_dice.items()},
                }
            )
            for dataset in SEG7:
                for metric, value in (("mIoU", seg_iou[dataset]), ("mDice", seg_dice[dataset])):
                    provenance.append(
                        {
                            "model": label,
                            "task": "segmentation",
                            "dataset": dataset,
                            "metric": metric,
                            "value": value,
                            "eval_run": "deterministic batch32 seeds 0/1/2",
                            "result_path": str((SEG_AUDIT / "segmentation_provenance.csv").relative_to(ROOT)),
                        }
                    )

    write_csv("classification.csv", classification)
    write_csv("regression.csv", regression)
    write_csv("retrieval.csv", retrieval)
    write_csv("clustering.csv", clustering)
    write_csv("segmentation.csv", segmentation)
    write_csv("detection.csv", detection)
    write_csv("provenance.csv", provenance)

    report = [
        "# BioDINOv3 S+ task-family tables",
        "",
        "All tables keep task families separate. No raw metrics from different families are averaged together.",
        "",
        "## Classification",
        "",
        "The headline classification metric is the historical broad-scaling `C25 macro-F1`. `C25 primary` is a separate diagnostic that averages balanced accuracy (ChestMNIST uses macro-AUC); it must not be called macro-F1. `C15 macro-F1` is the strict common subset available for H-S0. S6 uses checkpoint 3899 here so the row exactly matches the earlier 15-pass scaling result.",
        "",
        markdown_table(classification, [("model", "Backbone"), ("checkpoint", "Ckpt"), ("c25_macro_f1", "C25 macro-F1"), ("c25_coverage", "Coverage"), ("c25_primary_mean", "C25 primary"), ("c15_macro_f1", "C15 macro-F1"), ("bio3_macro_f1", "Bio-3 macro-F1"), ("chammi7_macro_f1", "CHAMMI-7 macro-F1")]),
        "",
        "Protocol reconciliation:",
        "",
        "- The earlier `0.679588 -> 0.682174` result is reproduced exactly by Official versus S6 checkpoint 3899 on C25 macro-F1.",
        "- The earlier recipe-race values such as `0.71535` and `0.72161` are C25 primary-metric means, not macro-F1.",
        "- H-S0's legacy `0.7443` used a random 80/20 split of the official training set and is not comparable to Official's official-test result. The current fair C15 values are `0.71741` primary and `0.66182` macro-F1.",
        "- C25 gains are not uniform across datasets: S6 improves the full C25 macro-F1 and CHAMMI-7, while its C15 and Bio-3 macro-F1 remain below Official.",
        "",
        "## Regression",
        "",
        "BBBC013 uses the compound-aware log1p leave-one-replicate-row-out protocol. `Regression-2` is the mean of BBBC005 R2 and BBBC013 macro R2.",
        "",
        markdown_table(regression, [("model", "Backbone"), ("regression2_r2_mean", "Regression-2 R2"), ("bbbc005_r2", "BBBC005 R2"), ("bbbc013_macro_r2", "BBBC013 R2"), ("bbbc013_macro_spearman", "BBBC013 rho"), ("bbbc013_macro_mae", "BBBC013 MAE down"), ("wortmannin_r2", "Wort R2"), ("ly294002_r2", "LY R2")]),
        "",
        "## Retrieval",
        "",
        markdown_table(retrieval, [("model", "Backbone"), ("macro_r1", "Macro R@1"), ("crc-val-he-7k", "CRC7K"), ("lc25000", "LC25000"), ("nct-crc-he-100", "NCT100"), ("nct-crc-he-1k", "NCT1K")]),
        "",
        "## Clustering",
        "",
        markdown_table(clustering, [("model", "Backbone"), ("macro_nmi", "Macro NMI"), ("crc-val-he-7k", "CRC7K"), ("lc25000", "LC25000"), ("nct-crc-he-100", "NCT100"), ("nct-crc-he-1k", "NCT1K")]),
        "",
        "## Segmentation",
        "",
        "Macro values use the seven datasets shared by Official, H-S0, and the recipe race. All heads use the corrected deterministic batch-32 protocol and report the mean across probe seeds 0/1/2.",
        "",
        markdown_table(segmentation, [("model", "Backbone"), ("macro_mDice", "Macro mDice"), ("macro_mDice_std", "mDice sd"), ("macro_mIoU", "Macro mIoU"), ("macro_mIoU_std", "mIoU sd"), ("miou_bbbc038", "BBBC038 mIoU"), ("miou_cellpose", "Cellpose mIoU"), ("miou_livecell", "LiveCell mIoU"), ("miou_tissuenet", "TissueNet mIoU"), ("miou_monuseg", "MoNuSeg mIoU"), ("miou_pannuke", "PanNuke mIoU"), ("miou_conic", "CoNIC mIoU")]),
        "",
        f"Detailed paired deltas, Dense-8 results, and per-dataset mDice are in `{SEG_AUDIT.relative_to(ROOT)}/README.md`.",
        "",
        "## Detection",
        "",
        "Official S+ does not yet have a protocol-matched LiveCell detection result.",
        "",
        markdown_table(detection, [("model", "Backbone"), ("livecell_precision", "Precision"), ("livecell_recall", "Recall"), ("livecell_f1", "F1")]),
        "",
        "## Protocol notes",
        "",
        "- Official non-segmentation uses the fixed `Dscale_core_first3` evaluation.",
        "- H-S0 non-segmentation uses the current group/official split evaluator; legacy split-less values are excluded.",
        "- Recipe-race tables use checkpoint 3899; S6 ck5199 is shown separately and has about 20 passes.",
        "- Classification is the exception: S6 uses checkpoint 3899 to reproduce the exact historical C25 scaling comparison; later task-family tables use checkpoint 5199.",
        "- Segmentation uses dataset-specific `best` frozen features, deterministic batch-32 heads, three probe seeds, and the seven-dataset common set.",
        "- OOD is omitted because the recipe-race OOD runs contain feature caches but no complete comparable result files.",
    ]
    (OUT / "README.md").write_text("\n".join(report) + "\n")
    print(OUT / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
