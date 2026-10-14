#!/usr/bin/env python3
"""Add the 1TB HS6-L selected-29 curve to the 5TB/20TB route2 diagnostic."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "plot/fig2/route2_20tb_v4_20261008"
SELECTED = ROOT / "plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv"
ONE_TB = ROOT / "outputs/00_reports/v4_subset_scaling_fm14_search_20260924/search_input_evidence.csv"
FIVE_TB = ROOT / "plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/v2_update_20261008/selected29_curve.csv"
FIVE_TB_MANIFEST = FIVE_TB.parent / "manifest.json"
FIVE_TB_EVIDENCE = SELECTED.with_name("per_dataset_source_evidence.csv")
TWENTY_TB = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/family_curve.csv"
STEM = "route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_diagnostic"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
METRICS = {"classification": {"macro_f1"}, "regression": {"spearman"},
           "retrieval": {"map_at_5"}, "clustering": {"nmi"},
           "segmentation": {"mDice_E20_primary_last"}}
LABELS = {"classification": "Classification", "regression": "Regression", "retrieval": "Retrieval",
          "clustering": "Clustering", "segmentation": "Segmentation"}
METRIC_LABELS = {"classification": "Macro-F1", "regression": "Spearman", "retrieval": "mAP@5",
                 "clustering": "NMI", "segmentation": "mDice (E20)"}
COLORS = {"classification": "#2563A6", "regression": "#9A4EAE", "retrieval": "#16807A",
          "clustering": "#D17A22", "segmentation": "#3A8F5C", "overall": "#C64B4B"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_selection():
    selected = defaultdict(set)
    for row in read_csv(SELECTED):
        selected[row["family"]].add(row["dataset"])
    expected_counts = {"classification": 16, "regression": 2, "retrieval": 4,
                       "clustering": 2, "segmentation": 5}
    if {family: len(selected[family]) for family in FAMILIES} != expected_counts:
        raise ValueError("The selected-29 dataset list changed")
    return selected


def one_tb_curve(selected):
    cells = defaultdict(dict)
    for row in read_csv(ONE_TB):
        if row["model"] != "hs6_l":
            continue
        family, dataset = row["family"], row["dataset"]
        if family not in selected or dataset not in selected[family]:
            continue
        if row["metric"] not in METRICS[family] or row["protocol"] != "v3":
            raise ValueError(f"Unexpected 1TB evidence: {row}")
        key = (int(row["checkpoint"]), family, dataset)
        if key in cells:
            raise ValueError(f"Duplicate 1TB selected-29 cell: {key}")
        cells[key] = float(row["value"])
    steps = sorted({step for step, _, _ in cells})
    expected_cells = {(family, dataset) for family in FAMILIES for dataset in selected[family]}
    if len(steps) != 15:
        raise ValueError(f"Expected 15 complete 1TB checkpoints, got {len(steps)}")
    result = []
    for step in steps:
        step_cells = {(family, dataset): value for (checkpoint, family, dataset), value in cells.items()
                      if checkpoint == step}
        if set(step_cells) != expected_cells:
            raise ValueError(f"Incomplete 1TB selected-29 checkpoint: {step}")
        means = {family: statistics.mean(step_cells[family, dataset]
                                         for dataset in sorted(selected[family]))
                 for family in FAMILIES}
        for family, value in means.items():
            result.append({"model": "1tb_hs6", "checkpoint": step, "family": family,
                           "value": value, "complete": True, "source_protocol": "validated_v3",
                           "source": str(ONE_TB)})
        result.append({"model": "1tb_hs6", "checkpoint": step, "family": "overall",
                       "value": statistics.mean(means.values()), "complete": True,
                       "source_protocol": "validated_v3", "source": str(ONE_TB)})
    return result


def five_tb_curve():
    manifest = json.loads(FIVE_TB_MANIFEST.read_text())
    selected = load_selection()
    if manifest["selected_datasets"] != {family: sorted(datasets)
                                         for family, datasets in selected.items()}:
        raise ValueError("5TB continuation selected a different dataset set")
    evidence_protocols = {row["protocol"] for row in read_csv(FIVE_TB_EVIDENCE)
                          if row["model"] == "5tb_no_gram"}
    if evidence_protocols != {"v3"}:
        raise ValueError(f"Unexpected 5TB historical provenance: {evidence_protocols}")
    rows = [row for row in read_csv(FIVE_TB) if row["model"] == "noGRAM"]
    if len(rows) != 85 * 6:
        raise ValueError(f"Expected 85 complete 5TB checkpoints, got {len(rows) // 6}")
    result = []
    for row in rows:
        if int(row["n"]) != int(row["required"]):
            raise ValueError(f"Incomplete 5TB selected-29 point: {row}")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite 5TB selected-29 value: {row}")
        result.append({"model": "5tb_no_gram", "checkpoint": int(row["checkpoint"]),
                       "family": "overall" if row["family"] == "overall" else row["family"],
                       "value": value, "complete": True,
                       "source_protocol": "v3_plus_selected29_continuation",
                       "source": str(FIVE_TB)})
    return result


def twenty_tb_curve():
    result = []
    for row in read_csv(TWENTY_TB):
        result.append({"model": "20tb_route2", "checkpoint": int(row["checkpoint"]),
                       "family": row["family"],
                       "value": float(row["value"]) if row["value"] else "",
                       "complete": row["complete"] == "True",
                       "source_protocol": "v4_shared_components", "source": str(TWENTY_TB)})
    return result


def plot(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    peak_rows = []
    for ax, family in zip(axes.flat, (*FAMILIES, "overall")):
        for model, color, marker, line in (
                ("1tb_hs6", "#D58166", "^", "-"),
                ("5tb_no_gram", "#58636D", "o", "--"),
                ("20tb_route2", COLORS[family], "s", "-")):
            points = sorted((row for row in rows if row["model"] == model
                             and row["family"] == family and row["complete"]),
                            key=lambda row: row["checkpoint"])
            xs = [row["checkpoint"] for row in points]
            ys = [float(row["value"]) for row in points]
            ax.plot(xs, ys, color=color, linestyle=line, linewidth=1.65,
                    marker=marker, markersize=2.8 if model == "5tb_no_gram" else 3.2,
                    alpha=0.92)
            peak = max(points, key=lambda row: (float(row["value"]), -row["checkpoint"]))
            ax.scatter([peak["checkpoint"]], [peak["value"]], marker="*", s=125,
                       color=color, edgecolor="white", linewidth=0.7, zorder=5)
            peak_rows.append({"model": model, "family": family,
                              "peak_checkpoint": peak["checkpoint"],
                              "peak_value": peak["value"], "complete_points": len(points)})
        ax.axvline(15374, color="#D58166", linestyle=":", linewidth=0.8, alpha=0.8)
        ax.axvline(41479, color="#AAB1B7", linestyle=":", linewidth=0.9)
        title = "Equal five-family mean" if family == "overall" else LABELS[family]
        count = "29 datasets" if family == "overall" else f"{next(row['datasets_expected'] for row in read_csv(TWENTY_TB) if row['family'] == family)} datasets"
        ax.set_title(f"{title} | {count}", fontsize=11)
        ax.set_ylabel("Equal-family mean" if family == "overall" else METRIC_LABELS[family])
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        ax.grid(axis="x", color="#EDF0F2", linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Checkpoint (optimizer steps)")
    max_step = max(row["checkpoint"] for row in rows)
    for ax in axes.flat:
        ax.set_xlim(0, max_step * 1.04)
    fig.suptitle("HS6-L | 1TB, 5TB no-GRAM and 20TB route2", fontsize=17,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.941, "Same fixed post hoc selected-29 datasets; five task means and equal-family mean; no smoothing",
             ha="center", fontsize=10.5, color="#4B5862")
    fig.legend(handles=[
        Line2D([0], [0], color="#D58166", marker="^", label="1TB HS6-L | validated v3"),
        Line2D([0], [0], color="#58636D", linestyle="--", marker="o",
               label="5TB no-GRAM | history + continuation"),
        Line2D([0], [0], color="#16807A", marker="s",
               label="20TB route2 | v4 shared components"),
        Line2D([0], [0], color="#D58166", linestyle=":", label="1TB endpoint ck15374"),
        Line2D([0], [0], color="#AAB1B7", linestyle=":", label="5TB endpoint ck41479"),
    ], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.008))
    fig.text(0.5, 0.053,
             "Exploratory cross-protocol diagnostic, not a controlled comparison or full v4 score. "
             "Dotted markers show 1TB ck15374 and 5TB ck41479 endpoints; incomplete 20TB points are gaps.",
             ha="center", fontsize=9, color="#68747D")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.89, hspace=0.30, wspace=0.25)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURE / f"{STEM}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    write_csv(FIGURE / f"{STEM}_curve.csv", rows)
    write_csv(FIGURE / f"{STEM}_peaks.csv", peak_rows)


def main():
    FIGURE.mkdir(parents=True, exist_ok=True)
    selected = load_selection()
    rows = one_tb_curve(selected) + five_tb_curve() + twenty_tb_curve()
    plot(rows)
    sources = (SELECTED, ONE_TB, FIVE_TB, FIVE_TB_MANIFEST, FIVE_TB_EVIDENCE, TWENTY_TB)
    manifest = {
        "selected29_sha256": hashlib.sha256(SELECTED.read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "checkpoints": {model: len({row["checkpoint"] for row in rows if row["model"] == model})
                        for model in ("1tb_hs6", "5tb_no_gram", "20tb_route2")},
        "selected_dataset_counts": {family: len(selected[family]) for family in FAMILIES},
        "one_tb_rule": "validated-v3 selected-29 cells; equal dataset mean within each family, then equal mean across five families",
        "twenty_tb_rule": "route2 v4 shared components; incomplete family and overall values remain gaps",
        "diagnostic_only": True,
        "full_v4_aggregate_allowed": False,
        "posthoc_test_selection": True,
    }
    (FIGURE / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {FIGURE / STEM}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
