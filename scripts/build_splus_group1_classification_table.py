#!/usr/bin/env python3
"""Build the protocol-matched first S+ classification group and S6 curve."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/00_reports/splus_sweetspot_live_results_summary.csv"
OUT = ROOT / "outputs/00_reports/splus_group1_classification_20260716"

COMMON15 = [
    "bbbc048-cellcycle", "bloodmnist", "breastmnist", "chestmnist", "cyclops-protein-loc",
    "dermamnist", "midog25-atypical", "octmnist", "organamnist", "organcmnist",
    "organsmnist", "pathmnist", "pneumoniamnist", "retinamnist", "tissuemnist",
]
BIO3 = ["bloodmnist", "bbbc048-cellcycle", "cyclops-protein-loc"]
CHAMMI7 = [
    "chammi-allen-task1", "chammi-allen-task2", "chammi-cp-task1", "chammi-cp-task2",
    "chammi-cp-task3", "chammi-hpa-task1", "chammi-hpa-task2",
]
EXTRA3 = ["lc25000", "nct-crc-he", "pcam"]
CLASS25 = COMMON15 + CHAMMI7 + EXTRA3
CLASS23 = [dataset for dataset in CLASS25 if dataset not in {"nct-crc-he", "pcam"}]

MODELS = OrderedDict(
    [
        ("Official S+", ("S7z_splus_official_reference", "0", 0, "reference")),
        (
            "S2-short robust+BioAug wu2",
            ("H0b_robust_biosafe256_b4096_lr2e-4_e8", "2079", 8, "8-pass schedule final; one C25 point"),
        ),
        (
            "R-S0 packwds GB4096",
            ("S0b_packwds_dino256_b4096_lr2e-4_wu2_e15", "3899", 15, "15-pass schedule final; one C25 point"),
        ),
        (
            "S1 robust+DINO",
            ("S1b_robust_dino256_b4096_lr2e-4_wu2_e15", "3899", 15, "15-pass schedule final; one C25 point"),
        ),
        (
            "S2 robust+BioAug wu2",
            ("S2b_robust_biosafe256_b4096_lr2e-4_wu2_e15", "3899", 15, "15-pass schedule final; one C25 point"),
        ),
        (
            "S2 robust+BioAug wu5",
            ("S2b_robust_biosafe256_b4096_lr2e-4_wu5_e15", "3899", 15, "15-pass schedule final; one C25 point"),
        ),
        (
            "S3 BioAug crop224",
            ("S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15", "3899", 15, "15-pass schedule final; one C25 point"),
        ),
        (
            "S6 robust+BioAug",
            (
                "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30",
                "3899",
                15,
                "best C25 macro-F1 among complete S6 curve points",
            ),
        ),
    ]
)

OVERLAYS = {
    ("H0b_robust_biosafe256_b4096_lr2e-4_e8", "2079"): ROOT
    / "outputs/02_eval_runs/H0b_robust_biosafe256_b4096_lr2e-4_e8__C25_missing_first3_20260716/bio_classification",
    ("S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15", "3899"): ROOT
    / "outputs/02_eval_runs/S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15__C25_missing_first3_20260716/bio_classification",
}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def complete_mean(values):
    values = list(values)
    return mean(values) if values and all(value is not None for value in values) else None


def source_priority(label: str, eval_run: str) -> int:
    if label == "Official S+":
        return 0 if eval_run == "S7z_splus_official_reference__Dscale_core_first3_ck0_qi8" else 50
    if label.startswith("S2-short"):
        return 0 if eval_run == "H0b_robust_biosafe256_b4096_lr2e-4_e8__cpu19_eval_2079" else 50
    if label.startswith("S6"):
        return 0 if eval_run.endswith("__Dscale_core_first3_ck3899_qi") else 50
    return 0 if "__3899_cls_" in eval_run else 50


def load_overlay(train_run: str, ckpt: str, dataset: str):
    root = OVERLAYS.get((train_run, ckpt))
    path = root / dataset / ckpt / "last_result.json" if root else None
    if not path or not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("error"):
        return None
    return {
        "primary_metric": "macro_auc" if dataset == "chestmnist" else "balanced_accuracy",
        "metric_value": result.get("macro_auc") if dataset == "chestmnist" else result.get("balanced_accuracy"),
        "macro_f1": result.get("macro_f1"),
        "eval_run": root.parent.name,
        "result_path": str(path.relative_to(ROOT)),
    }


def pick(rows, label: str, train_run: str, ckpt: str, dataset: str):
    overlay = load_overlay(train_run, ckpt, dataset)
    if overlay:
        return overlay
    candidates = [
        row
        for row in rows
        if row["train_run"] == train_run
        and row["ckpt"] == ckpt
        and row["task"] == "classification"
        and row["dataset"] == dataset
        and row["macro_f1"] != ""
    ]
    candidates.sort(key=lambda row: (source_priority(label, row["eval_run"]), row["result_path"]))
    if not candidates or source_priority(label, candidates[0]["eval_run"]) >= 50:
        return None
    return candidates[0]


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "N/A" if value is None else f"{value:.5f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open() as handle:
        source_rows = list(csv.DictReader(handle))

    table, provenance = [], []
    macro_f1_by_model = {}
    for label, (train_run, ckpt, passes, evidence) in MODELS.items():
        primary, macro_f1 = {}, {}
        for dataset in CLASS25:
            row = pick(source_rows, label, train_run, ckpt, dataset)
            primary[dataset] = number(row["metric_value"]) if row else None
            macro_f1[dataset] = number(row["macro_f1"]) if row else None
            if row:
                provenance.append(
                    {
                        "model": label,
                        "checkpoint": ckpt,
                        "dataset": dataset,
                        "macro_f1": macro_f1[dataset],
                        "primary_metric": row["primary_metric"],
                        "primary_value": primary[dataset],
                        "eval_run": row["eval_run"],
                        "result_path": row["result_path"],
                    }
                )
        coverage = sum(macro_f1[dataset] is not None for dataset in CLASS25)
        macro_f1_by_model[label] = macro_f1
        table.append(
            {
                "model": label,
                "checkpoint": ckpt,
                "passes": passes,
                "coverage": f"{coverage}/25",
                "c25_macro_f1": complete_mean(macro_f1[dataset] for dataset in CLASS25),
                "c25_primary_mean": complete_mean(primary[dataset] for dataset in CLASS25),
                "c15_macro_f1": complete_mean(macro_f1[dataset] for dataset in COMMON15),
                "bio3_macro_f1": complete_mean(macro_f1[dataset] for dataset in BIO3),
                "chammi7_macro_f1": complete_mean(macro_f1[dataset] for dataset in CHAMMI7),
                "chammi7_balanced_accuracy": complete_mean(primary[dataset] for dataset in CHAMMI7),
                "checkpoint_evidence": evidence,
            }
        )

    official_f1 = table[0]["c25_macro_f1"]
    official_dataset_f1 = macro_f1_by_model["Official S+"]
    official_c7 = table[0]["chammi7_balanced_accuracy"]
    for row in table:
        row["delta_vs_official"] = (
            row["c25_macro_f1"] - official_f1 if row["c25_macro_f1"] is not None else None
        )
        row["chammi7_delta_vs_official"] = (
            row["chammi7_balanced_accuracy"] - official_c7
            if row["chammi7_balanced_accuracy"] is not None
            else None
        )
        row["datasets_above_official"] = sum(
            macro_f1_by_model[row["model"]][dataset] > official_dataset_f1[dataset]
            for dataset in CLASS25
            if macro_f1_by_model[row["model"]][dataset] is not None
            and official_dataset_f1[dataset] is not None
        )

    dataset_deltas = []
    for label in MODELS:
        for dataset in CLASS25:
            value = macro_f1_by_model[label][dataset]
            reference = official_dataset_f1[dataset]
            dataset_deltas.append(
                {
                    "model": label,
                    "dataset": dataset,
                    "macro_f1": value,
                    "official_macro_f1": reference,
                    "delta_vs_official": value - reference if value is not None and reference is not None else None,
                }
            )

    curve = []
    s6_run = MODELS["S6 robust+BioAug"][0]
    for ckpt in ("1039", "2079", "3119", "3899", "5199", "6499", "7799"):
        values = {}
        for dataset in CLASS25:
            canonical_run = (
                lambda eval_run: eval_run.endswith("__Dscale_core_first3_ck3899_qi")
                if ckpt == "3899"
                else f"__ckpt{ckpt}_cls_" in eval_run
            )
            candidates = [
                row
                for row in source_rows
                if row["train_run"] == s6_run
                and row["ckpt"] == ckpt
                and row["task"] == "classification"
                and row["dataset"] == dataset
                and row["macro_f1"] != ""
                and canonical_run(row["eval_run"])
            ]
            candidates.sort(key=lambda row: row["result_path"])
            values[dataset] = number(candidates[0]["macro_f1"]) if candidates else None
        coverage = sum(values[dataset] is not None for dataset in CLASS25)
        curve.append(
            {
                "checkpoint": ckpt,
                "passes": (int(ckpt) + 1) / 260,
                "coverage": f"{coverage}/25",
                "c25_macro_f1": complete_mean(values[dataset] for dataset in CLASS25),
                "c23_macro_f1": complete_mean(values[dataset] for dataset in CLASS23),
                "bio3_macro_f1": complete_mean(values[dataset] for dataset in BIO3),
                "chammi7_macro_f1": complete_mean(values[dataset] for dataset in CHAMMI7),
            }
        )

    write_csv(OUT / "classification_group1.csv", table)
    write_csv(OUT / "classification_group1_provenance.csv", provenance)
    write_csv(OUT / "classification_group1_dataset_deltas.csv", dataset_deltas)
    write_csv(OUT / "s6_checkpoint_curve.csv", curve)

    lines = [
        "# S+ group 1: classification",
        "",
        "Headline metric: strict 25-dataset mean macro-F1. A macro is reported only at full coverage. `C25 primary` is a diagnostic mean of balanced accuracy, except ChestMNIST macro-AUC; it is not macro-F1. `C7 BA` is the historical CHAMMI-7 mean balanced accuracy and is the source of the older Official=0.6725 / S0~0.685x shorthand.",
        "For an RGB backbone, evaluator channel policy `auto` resolves exactly to `first3`.",
        "",
        "| Backbone | Ckpt | Passes | Coverage | C25 macro-F1 | Delta vs Official | Wins vs Official | C7 BA | C7 delta | C25 primary | C15 macro-F1 | Bio-3 macro-F1 | CHAMMI-7 macro-F1 | Checkpoint evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['checkpoint']} | {row['passes']} | {row['coverage']} | "
            f"{fmt(row['c25_macro_f1'])} | {fmt(row['delta_vs_official'])} | {row['datasets_above_official']}/25 | "
            f"{fmt(row['chammi7_balanced_accuracy'])} | {fmt(row['chammi7_delta_vs_official'])} | "
            f"{fmt(row['c25_primary_mean'])} | {fmt(row['c15_macro_f1'])} | "
            f"{fmt(row['bio3_macro_f1'])} | {fmt(row['chammi7_macro_f1'])} | {row['checkpoint_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Result reading",
            "",
            "- The historical `Official=0.6725 / S0~0.685x` classification statement uses C7 mean balanced accuracy, not C25 macro-F1. Under that historical protocol, every 15-pass GB4096 continuation row in this table is above Official.",
            "- On the strict C25 macro-F1 headline, S6 checkpoint 3899 is highest at 0.68217 (+0.00259 versus Official), followed by S1 at 0.68147 (+0.00188).",
            "- S2 warmup 2 is effectively tied with Official in point estimate (-0.00012); warmup 5, crop 224, the 8-pass schedule, and R-S0 are lower.",
            "- These are single-run point estimates without confidence intervals. The table supports descriptive ranking, not a statistical claim that the top two are significantly different.",
            "- Gains are heterogeneous: S6 is strongest on CHAMMI-7 but remains below Official on C15 and Bio-3 macro-F1.",
            "- `Wins vs Official` counts per-dataset macro-F1 signs; it prevents a positive microscopy subset from being mistaken for a uniform gain across the broader benchmark.",
            "",
            "## Why the historical GB1024 result is not a contradiction",
            "",
            "- H-S0 GB1024 does beat Official on the current protocol-matched C15: macro-F1 0.66182 versus 0.65587 (+0.00594). It has only 15/25 classification coverage, so it cannot supply a C25 mean or establish what happens on CHAMMI-7 plus the three added histology datasets.",
            "- H-S0 and R-S0 see a similar number of images but are not a batch-only comparison. H-S0 has about 15,375 optimizer updates, crop 224/96, optimizer warmup 3, teacher-temperature warmup 30, drop-path 0.30, fp32 LayerNorm, and its historical RGB statistics. R-S0 has about 3,900 updates, crop 256/112, warmups 2/5, drop-path 0.15, bf16 LayerNorm, and the newer microscopy RGB statistics.",
            "- Therefore H-S0 demonstrates that one mature packwds continuation improves the shared C15. It does not logically require the substantially different R-S0 GB4096 recipe to improve the broader C25.",
            "",
            "## R-S0 per-dataset sign audit",
            "",
            "R-S0 improves 10/25 datasets and declines on 15/25. Its largest changes are:",
            "",
            "| Dataset | R-S0 macro-F1 | Official macro-F1 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    r_s0_deltas = [row for row in dataset_deltas if row["model"] == "R-S0 packwds GB4096"]
    selected_r_s0_deltas = sorted(r_s0_deltas, key=lambda row: row["delta_vs_official"])[:5]
    selected_r_s0_deltas += sorted(r_s0_deltas, key=lambda row: row["delta_vs_official"], reverse=True)[:5]
    for row in selected_r_s0_deltas:
        lines.append(
            f"| {row['dataset']} | {fmt(row['macro_f1'])} | {fmt(row['official_macro_f1'])} | "
            f"{row['delta_vs_official']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "This mixed sign pattern is inconsistent with a simple global split, sample-count, or normalization shift. Full per-dataset values are in `classification_group1_dataset_deltas.csv`.",
            "",
            "## S6 checkpoint audit",
            "",
            "| Ckpt | Passes | Coverage | C25 macro-F1 | C23 macro-F1 | Bio-3 macro-F1 | CHAMMI-7 macro-F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in curve:
        lines.append(
            f"| {row['checkpoint']} | {row['passes']:.1f} | {row['coverage']} | {fmt(row['c25_macro_f1'])} | "
            f"{fmt(row['c23_macro_f1'])} | {fmt(row['bio3_macro_f1'])} | {fmt(row['chammi7_macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Selection rules",
            "",
            "- The fixed 15-pass ablation uses checkpoint 3899 for every GB4096 15-pass run.",
            "- The 8-pass row is a separate cosine schedule, not checkpoint 2079 from a 15-pass schedule.",
            "- S0-S3 have only one full C25 evaluation point, so checkpoint 3899 is the final assessed point, not a proven oracle optimum.",
            "- S6 checkpoint 3899 is selected for the headline because it has the best C25 macro-F1 among complete curve points. Checkpoint 3119 is slightly better on the common C23 subset and the historical three-task balanced-accuracy proxy.",
            "- Choosing a checkpoint by test-set maximum is descriptive oracle selection. A publishable deployment selection requires train/validation-only checkpoint selection followed by one official-test evaluation.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(OUT / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
