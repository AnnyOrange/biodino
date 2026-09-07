#!/usr/bin/env python3
"""Consolidate completed ViT-L/16 instance-segmentation screening results.

This script only reads completed CSV artifacts and rewrites summary artifacts. It
does not import model code, allocate a GPU, or launch training/evaluation jobs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
SCREEN_RUN = "vitl16_backbone_screen_lr1e5_decay075_20260813_183025"
SCREEN_DATASETS = ("monuseg", "livecell", "cellpose")
SCREEN_MODES = ("frozen", "last4", "last8", "lora8", "lora16", "adapter", "finetune")
DATASET_ORDER = ("monuseg", "bbbc038", "cellpose", "tissuenet", "livecell", "pannuke", "conic")

DISPLAY_NAME = {
    "monuseg": "MoNuSeg",
    "bbbc038": "BBBC038",
    "cellpose": "Cellpose",
    "tissuenet": "TissueNet",
    "livecell": "LIVECell",
    "pannuke": "PanNuke",
    "conic": "CoNIC",
}

MODE_NAME = {
    "frozen": "Frozen",
    "last4": "Last-4",
    "last8": "Last-8",
    "lora8": "LoRA-8",
    "lora16": "LoRA-16",
    "adapter": "Adapter",
    "finetune": "Full FT",
}

PRIMARY_METRIC = {
    "monuseg": "AJI",
    "bbbc038": "CellposeStyleAP",
    "cellpose": "CellposeStyleAP",
    "tissuenet": "CellposeStyleAP",
    "livecell": "SEG",
    "pannuke": "bPQ",
    "conic": "local mPQ",
}

METRIC_DEFINITION = {
    "AJI": "Aggregated Jaccard Index",
    "CellposeStyleAP": "Mean TP/(TP+FP+FN) over IoU thresholds 0.50:0.05:0.95; not COCO AP",
    "SEG": "Cell Tracking Challenge SEG; not official LIVECell COCO mask AP",
    "bPQ": "Local binary Panoptic Quality; PanNuke validation probe, not official 3-fold evaluation",
    "local mPQ": "Local class-aware mean Panoptic Quality at IoU 0.50; not official CoNIC mPQ+",
}

THRESHOLD_TAG = {
    "monuseg": "local_fg_0.54_energy_0.45",
    "bbbc038": "local_fg_0.46_energy_0.58",
    "cellpose": "local_fg_0.46_energy_0.45",
    "tissuenet": "local_fg_0.57_energy_0.40",
    "livecell": "local_fg_0.28_energy_0.42",
    "pannuke": "local_fg_0.46_energy_0.45",
    "conic": "local_fg_0.46_energy_0.43",
}

PRE_BACKBONE_BEST = {
    "monuseg": ("Frozen adaptation", 0.5272155732919562),
    "livecell": ("Local threshold search", 0.6189078066904272),
    "cellpose": ("Focal+Tversky frozen A/B", 0.2769550562692866),
}

LEDGER_FIELDS = [
    "timestamp",
    "dataset",
    "optimization_method",
    "metric",
    "baseline_metric",
    "optimized_metric",
    "absolute_gain",
    "config",
    "train_infer_cost",
    "adopted",
    "gain_type",
    "metric_definition",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite(value: object) -> bool:
    return math.isfinite(number(value))


def fmt(value: object, digits: int = 6) -> str:
    parsed = number(value)
    return "" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def signed(value: object, digits: int = 6) -> str:
    parsed = number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:+.{digits}f}"


def clean_json_value(value: object) -> object:
    parsed = number(value)
    if isinstance(value, str) and value == "":
        return ""
    if math.isfinite(parsed):
        return parsed
    return value


def load_inputs():
    adaptation = read_csv(OUT / "adaptation_results.csv")
    screen_rows = [
        row
        for row in adaptation
        if SCREEN_RUN in row.get("results_json", "")
        and row.get("dataset") in SCREEN_DATASETS
        and row.get("mode") in SCREEN_MODES
        and not row.get("max_train_batches")
        and not row.get("max_eval_images")
        and row.get("exit_code") == "0"
    ]
    keys = {(row["dataset"], row["mode"]) for row in screen_rows}
    expected = {(dataset, mode) for dataset in SCREEN_DATASETS for mode in SCREEN_MODES}
    if len(screen_rows) != 21 or keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(
            f"Expected exactly 21 completed screen rows; got {len(screen_rows)}; "
            f"missing={missing}; extra={extra}"
        )

    screen = {(row["dataset"], row["mode"]): row for row in screen_rows}
    baseline_rows = read_csv(OUT / "full_val_baseline.csv")
    baseline = {
        row["dataset"]: row
        for row in baseline_rows
        if row.get("tag") == "default_full_val" and row.get("exit_code") == "0"
    }
    eval_rows = read_csv(OUT / "eval_only_results.csv")

    threshold: Dict[str, Dict[str, str]] = {}
    for dataset, tag in THRESHOLD_TAG.items():
        matches = [row for row in eval_rows if row.get("dataset") == dataset and row.get("tag") == tag]
        if not matches:
            raise RuntimeError(f"Missing eval-only result for {dataset}/{tag}")
        threshold[dataset] = matches[-1]
    return adaptation, screen_rows, screen, baseline, eval_rows, threshold


def corrected_baseline(dataset: str, row: Mapping[str, str]) -> float:
    if dataset == "pannuke":
        return number(row["primary_value"])
    return number(row["primary_value"])


def corrected_threshold_value(dataset: str, row: Mapping[str, str]) -> float:
    if dataset in {"bbbc038", "cellpose", "tissuenet"}:
        return number(row["ObjectAP"])
    if dataset == "pannuke":
        return number(row["mPQ"])
    if dataset == "conic":
        return number(row["mPQ"])
    return number(row[PRIMARY_METRIC[dataset]])


def screen_best(screen: Mapping[tuple, Mapping[str, str]], dataset: str) -> Mapping[str, str]:
    return max(
        (screen[(dataset, mode)] for mode in SCREEN_MODES),
        key=lambda row: number(row["primary_value"]),
    )


def update_ledger(
    screen_rows: Sequence[Mapping[str, str]],
    screen: Mapping[tuple, Mapping[str, str]],
    baseline: Mapping[str, Mapping[str, str]],
) -> None:
    path = OUT / "method_gain_ledger.csv"
    existing = read_csv(path)
    rows: List[MutableMapping[str, object]] = []

    for old in existing:
        if old.get("optimization_method", "").startswith("backbone_screen_"):
            continue
        row: MutableMapping[str, object] = dict(old)
        if row.get("metric") == "ObjectAP":
            row["metric"] = "CellposeStyleAP"
            row["metric_definition"] = METRIC_DEFINITION["CellposeStyleAP"]
        if row.get("dataset") == "pannuke" and row.get("metric") == "mPQ":
            row["metric"] = "bPQ"
            row["metric_definition"] = METRIC_DEFINITION["bPQ"]
            try:
                config = json.loads(str(row.get("config") or "{}"))
            except json.JSONDecodeError:
                config = {}
            if row.get("optimization_method") == "threshold_local_2d":
                config["local_mPQ"] = 0.6927968116061269
            config["metric_label_correction"] = "0.731334 is bPQ, not mPQ"
            row["config"] = json.dumps(config, sort_keys=True)
        if row.get("dataset") == "conic" and row.get("metric") == "mPQ":
            row["metric"] = "local mPQ"
            row["metric_definition"] = METRIC_DEFINITION["local mPQ"]
        if row.get("optimization_method") == "vitl16_screen_focal_tversky_ab":
            try:
                config = json.loads(str(row.get("config") or "{}"))
            except json.JSONDecodeError:
                config = {}
            config.update(
                {
                    "interpretation": "inconclusive point estimate; delta=+0.000861",
                    "multi_seed_assessment": "unavailable; seed 0 only",
                }
            )
            row["config"] = json.dumps(config, sort_keys=True)
            row["adopted"] = 0
            row["gain_type"] = "point_estimate_inconclusive"
            row["metric_definition"] = METRIC_DEFINITION["CellposeStyleAP"]
        rows.append(row)

    frozen = {
        dataset: number(screen[(dataset, "frozen")]["primary_value"])
        for dataset in SCREEN_DATASETS
    }
    best_mode = {
        dataset: screen_best(screen, dataset)["mode"]
        for dataset in SCREEN_DATASETS
    }

    for source in sorted(screen_rows, key=lambda row: (SCREEN_DATASETS.index(row["dataset"]), SCREEN_MODES.index(row["mode"]))):
        dataset = source["dataset"]
        mode = source["mode"]
        metric = PRIMARY_METRIC[dataset]
        original = corrected_baseline(dataset, baseline[dataset])
        optimized = number(source["primary_value"])
        independent = optimized - frozen[dataset]
        pre_name, pre_value = PRE_BACKBONE_BEST[dataset]
        point_best = mode == best_mode[dataset]
        interpretation = "single-seed point estimate; multi-seed significance unknown"
        adopted = int(point_best)
        gain_type = "backbone_screen_vs_original_baseline"
        if dataset == "monuseg" and mode == "adapter":
            interpretation = "inconclusive vs Frozen; point delta=+0.000635; seed 0 only"
            adopted = 0
            gain_type = "point_estimate_inconclusive"

        config = {
            "run_family": SCREEN_RUN,
            "mode": mode,
            "layers": source.get("layers", ""),
            "fusion_mode": source.get("fusion_mode", ""),
            "epochs": int(number(source.get("epochs"))),
            "seed": int(number(source.get("seed"))),
            "decoder_lr": clean_json_value(source.get("decoder_lr", "")),
            "backbone_lr": clean_json_value(source.get("backbone_lr", "")),
            "layer_wise_lr_decay": clean_json_value(source.get("layer_wise_lr_decay", "")),
            "np_loss_mode": source.get("np_loss_mode", "ce_dice"),
            "same_screen_frozen_baseline": frozen[dataset],
            "backbone_independent_gain_vs_frozen": independent,
            "pre_backbone_best_name": pre_name,
            "pre_backbone_best": pre_value,
            "marginal_gain_if_point_best": optimized - pre_value if point_best else None,
            "point_estimate_best": point_best,
            "multi_seed_assessment": "unavailable; all screen rows use seed 0",
            "interpretation": interpretation,
            "results_json": source.get("results_json", ""),
        }
        rows.append(
            {
                "timestamp": source["timestamp"],
                "dataset": dataset,
                "optimization_method": f"backbone_screen_{mode}",
                "metric": metric,
                "baseline_metric": original,
                "optimized_metric": optimized,
                "absolute_gain": optimized - original,
                "config": json.dumps(config, sort_keys=True),
                "train_infer_cost": (
                    f"ViT-L/16 strategy screening; 50ep full validation; "
                    f"wall_seconds={source.get('wall_seconds', '')}; "
                    f"effective_batch={source.get('effective_batch_size', '')}; "
                    f"oom_retry={source.get('oom_retry', '')}; no test tuning"
                ),
                "adopted": adopted,
                "gain_type": gain_type,
                "metric_definition": METRIC_DEFINITION[metric],
            }
        )

    write_csv(path, LEDGER_FIELDS, rows)


def metric_bundle(dataset: str, row: Mapping[str, str], source: str) -> Dict[str, object]:
    object_ap = row.get("CellposeStyleAP") if finite(row.get("CellposeStyleAP")) else row.get("ObjectAP")
    object_ap50 = row.get("CellposeStyleAP50") if finite(row.get("CellposeStyleAP50")) else row.get("ObjectAP50")
    object_ap75 = row.get("CellposeStyleAP75") if finite(row.get("CellposeStyleAP75")) else row.get("ObjectAP75")
    values: Dict[str, object] = {
        "AJI": fmt(row.get("AJI")),
        "Dice": fmt(row.get("Dice")),
        "COCOProxyAP@[0.50:0.95]": fmt(row.get("AP")),
        "COCOProxyAP50": fmt(row.get("AP50")),
        "COCOProxyAP75": fmt(row.get("AP75")),
        "CellposeStyleAP@[0.50:0.95]": fmt(object_ap),
        "CellposeStyleAP50": fmt(object_ap50),
        "CellposeStyleAP75": fmt(object_ap75),
        "bPQ": fmt(row.get("bPQ")),
        "mPQ": fmt(row.get("mPQ")),
        "SEG": fmt(row.get("SEG")),
    }
    if dataset == "pannuke":
        values["bPQ"] = fmt(row.get("mPQ"))
        values["mPQ"] = fmt(row.get("SEG"))
        values["SEG"] = ""
    if dataset == "conic":
        values["mPQ"] = fmt(row.get("mPQ"))
    return values


def update_unified_metrics(
    screen: Mapping[tuple, Mapping[str, str]],
    threshold: Mapping[str, Mapping[str, str]],
) -> None:
    fields = [
        "Dataset", "Split", "Primary Official Metric", "Protocol", "Best Strategy",
        "AJI", "Dice", "COCOProxyAP@[0.50:0.95]", "COCOProxyAP50", "COCOProxyAP75",
        "CellposeStyleAP@[0.50:0.95]", "CellposeStyleAP50", "CellposeStyleAP75",
        "bPQ", "mPQ", "SEG", "Metric Availability Note",
    ]
    best_source = {
        "monuseg": screen_best(screen, "monuseg"),
        "bbbc038": threshold["bbbc038"],
        "cellpose": screen_best(screen, "cellpose"),
        "tissuenet": threshold["tissuenet"],
        "livecell": screen_best(screen, "livecell"),
        "pannuke": threshold["pannuke"],
        "conic": threshold["conic"],
    }
    strategy = {
        "monuseg": "Adapter (point estimate; inconclusive vs Frozen)",
        "bbbc038": "Local threshold search",
        "cellpose": "Full fine-tuning (single seed)",
        "tissuenet": "Local threshold search",
        "livecell": "Full fine-tuning (single seed)",
        "pannuke": "Local threshold search",
        "conic": "Local threshold search",
    }
    primary = {
        "monuseg": "AJI",
        "bbbc038": "AJI; AP50; CellposeStyleAP@[0.50:0.95]",
        "cellpose": "CellposeStyleAP@[0.50:0.95]",
        "tissuenet": "CellposeStyleAP@[0.50:0.95]",
        "livecell": "Official COCO mask AP unavailable; SEG reported",
        "pannuke": "bPQ and local mPQ; official 3-fold protocol unavailable",
        "conic": "Official mPQ+ unavailable; local mPQ reported",
    }
    protocol = {
        dataset: f"ViT-L/16 validation probe; {DISPLAY_NAME[dataset]} local evaluator; no test tuning"
        for dataset in DATASET_ORDER
    }
    notes = {
        "monuseg": "Adapter-Frozen AJI delta is +0.000635 and is inconclusive; all backbone rows are seed 0",
        "bbbc038": "CellposeStyleAP is object-count AP, not COCO AP; official DSB2018 evaluator unavailable",
        "cellpose": "CellposeStyleAP is TP/(TP+FP+FN) averaged over IoU 0.50:0.95; single-seed training result",
        "tissuenet": "Local CellposeStyleAP must not be compared numerically with external COCO-style mean AP",
        "livecell": "0.626138 is SEG, not official COCO mask AP; confidence-based COCO output is unavailable",
        "pannuke": "0.731334 is bPQ and 0.692797 is local mPQ; neither is an official 3-fold result",
        "conic": "0.603464 is local mPQ, not official CoNIC mPQ+",
    }
    rows = []
    for dataset in DATASET_ORDER:
        row = {
            "Dataset": DISPLAY_NAME[dataset],
            "Split": "val",
            "Primary Official Metric": primary[dataset],
            "Protocol": protocol[dataset],
            "Best Strategy": strategy[dataset],
            "Metric Availability Note": notes[dataset],
        }
        row.update(metric_bundle(dataset, best_source[dataset], "screen" if dataset in SCREEN_DATASETS else "eval"))
        rows.append(row)
    write_csv(OUT / "unified_metrics_results.csv", fields, rows)


def update_backbone_status(screen: Mapping[tuple, Mapping[str, str]]) -> None:
    fields = [
        "Dataset", "Strategy", "Backbone LR", "Layer-wise LR Decay", "Epochs", "Seed", "AMP",
        "Decoder/Loss/Postprocess Protocol", "Status", "Result", "Notes",
    ]
    rows = []
    for dataset in SCREEN_DATASETS:
        frozen = number(screen[(dataset, "frozen")]["primary_value"])
        best = screen_best(screen, dataset)
        for mode in SCREEN_MODES:
            source = screen[(dataset, mode)]
            value = number(source["primary_value"])
            metric = PRIMARY_METRIC[dataset]
            delta = value - frozen
            note = f"Completed seed-0 point estimate; delta vs same-screen Frozen={delta:+.6f}"
            if dataset == "monuseg" and mode == "adapter":
                note += "; +0.000635 is inconclusive and not evidence of an effective gain"
            if mode == best["mode"] and not (dataset == "monuseg" and mode == "adapter"):
                note += "; point-estimate winner, multi-seed significance unavailable"
            rows.append(
                {
                    "Dataset": DISPLAY_NAME[dataset],
                    "Strategy": MODE_NAME[mode],
                    "Backbone LR": source.get("backbone_lr") or "N/A (frozen)",
                    "Layer-wise LR Decay": source.get("layer_wise_lr_decay") or "N/A (frozen/adapter)",
                    "Epochs": source.get("epochs", ""),
                    "Seed": source.get("seed", ""),
                    "AMP": source.get("amp_dtype", ""),
                    "Decoder/Loss/Postprocess Protocol": "fixed ViT-L/16 decoder, CE+Dice, val split and postprocess",
                    "Status": "Completed",
                    "Result": f"{metric} {value:.6f}",
                    "Notes": note,
                }
            )
    write_csv(OUT / "backbone_screening_status.csv", fields, rows)


def update_eval_summary(eval_rows: Sequence[Mapping[str, str]], threshold: Mapping[str, Mapping[str, str]]) -> None:
    best_tags = {
        "pannuke": THRESHOLD_TAG["pannuke"],
        "conic": THRESHOLD_TAG["conic"],
        "monuseg": "combined_threshold_tiling_flip4",
        "livecell": THRESHOLD_TAG["livecell"],
        "bbbc038": THRESHOLD_TAG["bbbc038"],
        "tissuenet": THRESHOLD_TAG["tissuenet"],
        "cellpose": THRESHOLD_TAG["cellpose"],
    }
    rows = {}
    for dataset, tag in best_tags.items():
        matches = [row for row in eval_rows if row.get("dataset") == dataset and row.get("tag") == tag]
        if not matches:
            raise RuntimeError(f"Missing eval-only summary row for {dataset}/{tag}")
        rows[dataset] = matches[-1]

    lines = [
        "# Eval-Only Optimization Summary",
        "",
        "All entries are ViT-L/16 validation probes. CellposeStyleAP is not COCO AP.",
        "",
        "| dataset | best eval-only tag | metric | value | supplementary corrected metric | crop | stride | fg | energy | tta |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("pannuke", "conic", "monuseg", "livecell", "bbbc038", "tissuenet", "cellpose"):
        row = rows[dataset]
        metric = PRIMARY_METRIC[dataset]
        value = corrected_threshold_value(dataset, row) if dataset != "monuseg" else number(row["AJI"])
        supplementary = ""
        if dataset == "pannuke":
            supplementary = f"local mPQ {number(row['SEG']):.6f}"
        lines.append(
            f"| {dataset} | {row.get('tag', '')} | {metric} | {value:.6f} | {supplementary} | "
            f"{row.get('crop_size', '')} | {row.get('stride', '')} | {row.get('fg_thresh', '')} | "
            f"{row.get('energy_thresh', '')} | {row.get('tta', '')} |"
        )
    lines.extend(
        [
            "",
            "Protocol limitations:",
            "",
            "- PanNuke 0.731334 is bPQ; local mPQ is 0.692797. These are not official 3-fold scores.",
            "- CoNIC 0.603464 is local mPQ, not official mPQ+.",
            "- LIVECell SEG is supplementary and must not be presented as COCO mask AP.",
        ]
    )
    write_text(OUT / "eval_only_summary.md", "\n".join(lines))


def update_adaptation_analysis(
    screen: Mapping[tuple, Mapping[str, str]],
    baseline: Mapping[str, Mapping[str, str]],
) -> None:
    lines = [
        "# ViT-L/16 Backbone Strategy Screening Analysis",
        "",
        "Exactly 21 completed rows are included: three datasets by seven strategies. All rows use seed 0, so point-estimate differences cannot be tested against multi-seed variability.",
        "",
        "| dataset | strategy | metric | value | gain vs original baseline | gain vs same-screen Frozen | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for dataset in SCREEN_DATASETS:
        original = corrected_baseline(dataset, baseline[dataset])
        frozen = number(screen[(dataset, "frozen")]["primary_value"])
        best = screen_best(screen, dataset)
        for mode in SCREEN_MODES:
            row = screen[(dataset, mode)]
            value = number(row["primary_value"])
            status = "Completed; seed 0"
            if mode == best["mode"]:
                status = "Point-estimate best; multi-seed significance unknown"
            if dataset == "monuseg" and mode == "adapter":
                status = "Point-estimate best; +0.000635 vs Frozen is inconclusive"
            lines.append(
                f"| {DISPLAY_NAME[dataset]} | {MODE_NAME[mode]} | {PRIMARY_METRIC[dataset]} | {value:.6f} | "
                f"{value - original:+.6f} | {value - frozen:+.6f} | {status} |"
            )

    lines.extend(
        [
            "",
            "## Gain Decomposition",
            "",
            "Cumulative gain is measured from the original full-validation baseline. Backbone-independent gain is measured from the Frozen row in the same 21-run screen. Marginal gain is measured from the best result available before the backbone screen.",
            "",
            "| dataset | original baseline | same-screen Frozen | best training strategy | final point estimate | cumulative gain | backbone gain vs Frozen | previous best | marginal gain | exceeds multi-seed variation? |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset in SCREEN_DATASETS:
        original = corrected_baseline(dataset, baseline[dataset])
        frozen = number(screen[(dataset, "frozen")]["primary_value"])
        best = screen_best(screen, dataset)
        value = number(best["primary_value"])
        pre_name, pre_value = PRE_BACKBONE_BEST[dataset]
        seed_status = "Unknown: only seed 0 is available"
        if dataset == "monuseg":
            seed_status = "No evidence; +0.000635 is inconclusive"
        lines.append(
            f"| {DISPLAY_NAME[dataset]} | {original:.6f} | {frozen:.6f} | {MODE_NAME[best['mode']]} | "
            f"{value:.6f} | {value - original:+.6f} | {value - frozen:+.6f} | "
            f"{pre_value:.6f} ({pre_name}) | {value - pre_value:+.6f} | {seed_status} |"
        )

    lines.extend(
        [
            "",
            "## Loss A/B",
            "",
            "Cellpose Frozen CE+Dice scored 0.276094 and Focal+Tversky scored 0.276955. The +0.000861 CellposeStyleAP point delta is **inconclusive** because only seed 0 is available; it is not evidence of an effective improvement.",
            "",
            "## Interpretation",
            "",
            "- MoNuSeg Adapter is the numerical best at AJI 0.527851, but its +0.000635 delta over Frozen is inconclusive.",
            "- LIVECell Full FT is the numerical best at SEG 0.626138. SEG is not official COCO mask AP.",
            "- Cellpose Full FT is the numerical best at CellposeStyleAP 0.277854. Its +0.000899 margin over the prior Focal+Tversky result is also unverified without multiple seeds.",
        ]
    )
    write_text(OUT / "adaptation_analysis.md", "\n".join(lines))


def final_rows(
    screen: Mapping[tuple, Mapping[str, str]],
    baseline: Mapping[str, Mapping[str, str]],
    threshold: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    protocol = {
        "monuseg": "ViT-L/16 val probe; AJI; single-seed backbone screen",
        "bbbc038": "ViT-L/16 val probe; local Cellpose-style object AP; not official DSB2018 evaluation",
        "cellpose": "ViT-L/16 val probe; CellposeStyleAP; single-seed backbone screen",
        "tissuenet": "ViT-L/16 val probe; nuclear target; local CellposeStyleAP, not COCO AP",
        "livecell": "ViT-L/16 val probe; SEG only; official COCO mask AP unavailable",
        "pannuke": "ViT-L/16 val probe; bPQ/local mPQ correction; not official 3-fold evaluation",
        "conic": "ViT-L/16 val probe; local mPQ; official mPQ+ unavailable",
    }
    for dataset in DATASET_ORDER:
        original = corrected_baseline(dataset, baseline[dataset])
        threshold_value = corrected_threshold_value(dataset, threshold[dataset])
        metric = PRIMARY_METRIC[dataset]
        training = "N/A (not screened)"
        final = threshold_value
        best_threshold = fmt(threshold_value)
        original_text = fmt(original)
        gain_text = signed(final - original)
        final_text = fmt(final)

        if dataset in SCREEN_DATASETS:
            best = screen_best(screen, dataset)
            training_value = number(best["primary_value"])
            qualifier = "point estimate; inconclusive vs Frozen" if dataset == "monuseg" else "single-seed point estimate"
            training = f"{MODE_NAME[best['mode']]} {training_value:.6f} ({qualifier})"
            final = max(threshold_value, training_value)
            final_text = fmt(final)
            gain_text = signed(final - original)
        if dataset == "pannuke":
            baseline_local_mpq = number(baseline[dataset]["SEG"])
            threshold_local_mpq = number(threshold[dataset]["SEG"])
            original_text = f"bPQ {original:.6f}; local mPQ {baseline_local_mpq:.6f}"
            best_threshold = f"bPQ {threshold_value:.6f}; local mPQ {threshold_local_mpq:.6f}"
            final_text = best_threshold
            gain_text = f"bPQ {threshold_value - original:+.6f}; local mPQ {threshold_local_mpq - baseline_local_mpq:+.6f}"
            metric = "bPQ; local mPQ"
        rows.append(
            {
                "Dataset": DISPLAY_NAME[dataset],
                "Metric": metric,
                "Protocol": protocol[dataset],
                "Original baseline": original_text,
                "Best threshold": best_threshold,
                "Best training strategy": training,
                "Final best": final_text,
                "Absolute gain": gain_text,
                "Comparable SOTA": "N/A (no protocol-identical result)",
            }
        )
    return rows


def update_final_summaries(
    screen: Mapping[tuple, Mapping[str, str]],
    baseline: Mapping[str, Mapping[str, str]],
    threshold: Mapping[str, Mapping[str, str]],
) -> None:
    rows = final_rows(screen, baseline, threshold)
    fields = [
        "Dataset", "Metric", "Protocol", "Original baseline", "Best threshold",
        "Best training strategy", "Final best", "Absolute gain", "Comparable SOTA",
    ]
    write_csv(OUT / "final_strategy_sota_comparison.csv", fields, rows)

    sota_rows = []
    for row in rows:
        sota_rows.append(
            {
                "Dataset": row["Dataset"],
                "Metric": row["Metric"],
                "Baseline / Reference": row["Original baseline"],
                "Best Current Result": row["Final best"],
                "Improvement": row["Absolute gain"],
                "Best Strategy": row["Best training strategy"] if not row["Best training strategy"].startswith("N/A") else "Local threshold search",
                "SOTA / External Reference": "N/A - no protocol-identical result",
                "Status": "Completed; comparison unavailable",
            }
        )
    write_csv(
        OUT / "vitl16_strategy_screening_sota_summary.csv",
        ["Dataset", "Metric", "Baseline / Reference", "Best Current Result", "Improvement", "Best Strategy", "SOTA / External Reference", "Status"],
        sota_rows,
    )

    unified_rows = []
    for row in rows:
        unified_rows.append(
            {
                "Dataset": row["Dataset"],
                "Metric Family": "Instance segmentation quality",
                "Our Metric": row["Metric"],
                "Our Score": row["Final best"],
                "Baseline Score": row["Original baseline"],
                "Best Strategy": row["Best training strategy"] if not row["Best training strategy"].startswith("N/A") else "Local threshold search",
                "External SOTA Method": "N/A",
                "SOTA Metric": "N/A",
                "SOTA Score": "N/A",
                "Metric Alignment": "No protocol-identical external result",
                "Comparison Status": "Unavailable",
                "Reason / Protocol Note": "No numerical SOTA value or gap is reported without identical split, target, metric and evaluator",
                "SOTA Source": "See sota_protocol_table.csv for non-comparable context only",
            }
        )
    write_csv(
        OUT / "vitl16_strategy_screening_unified_metrics.csv",
        [
            "Dataset", "Metric Family", "Our Metric", "Our Score", "Baseline Score", "Best Strategy",
            "External SOTA Method", "SOTA Metric", "SOTA Score", "Metric Alignment", "Comparison Status",
            "Reason / Protocol Note", "SOTA Source",
        ],
        unified_rows,
    )


def update_method_gain_summary() -> None:
    rows = read_csv(OUT / "method_gain_ledger.csv")
    lines = [
        "# Method Gain Summary",
        "",
        "Metrics use corrected display names. Point estimates marked inconclusive are not claims of effective improvement.",
        "",
        "| dataset | method | metric | baseline | optimized | gain | adopted | interpretation |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        interpretation = ""
        try:
            config = json.loads(row.get("config") or "{}")
            interpretation = config.get("interpretation", "")
        except json.JSONDecodeError:
            pass
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('optimization_method', '')} | {row.get('metric', '')} | "
            f"{fmt(row.get('baseline_metric'), 4)} | {fmt(row.get('optimized_metric'), 4)} | "
            f"{signed(row.get('absolute_gain'), 4)} | {row.get('adopted', '')} | {interpretation} |"
        )
    write_text(OUT / "method_gain_summary.md", "\n".join(lines))


def update_baseline_summary(baseline: Mapping[str, Mapping[str, str]]) -> None:
    lines = [
        "# Full Validation Baseline Summary",
        "",
        "These are ViT-L/16 validation probes. Corrected metric names are used; no row is an official test/challenge result.",
        "",
        "| dataset | n | primary metric | value | supplementary |",
        "|---|---:|---|---:|---|",
    ]
    for dataset in ("pannuke", "conic", "monuseg", "livecell", "bbbc038", "tissuenet", "cellpose"):
        row = baseline[dataset]
        metric = PRIMARY_METRIC[dataset]
        value = corrected_baseline(dataset, row)
        supplementary = ""
        if dataset == "pannuke":
            supplementary = f"local mPQ {number(row['SEG']):.6f}"
        lines.append(f"| {dataset} | {row.get('n_images', '')} | {metric} | {value:.6f} | {supplementary} |")
    write_text(OUT / "full_val_baseline_summary.md", "\n".join(lines))


def main() -> None:
    _, screen_rows, screen, baseline, eval_rows, threshold = load_inputs()
    update_ledger(screen_rows, screen, baseline)
    update_unified_metrics(screen, threshold)
    update_backbone_status(screen)
    update_eval_summary(eval_rows, threshold)
    update_adaptation_analysis(screen, baseline)
    update_final_summaries(screen, baseline, threshold)
    update_method_gain_summary()
    update_baseline_summary(baseline)
    print("Consolidated 21 ViT-L/16 backbone screening rows")


if __name__ == "__main__":
    main()
