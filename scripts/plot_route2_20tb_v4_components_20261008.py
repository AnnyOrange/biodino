#!/usr/bin/env python3
"""Build an auditable route2 component report and Fig2-style trajectory."""

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
CAMPAIGN = ROOT / "outputs/02_eval_runs/20tb_route2_union_v4_online_20260928"
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008"
FIGURE = ROOT / "plot/fig2/route2_20tb_v4_20261008"
PREFIX = "5090route2_ck"
SELECTION = ROOT / "plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv"
FIVE_TB_CURVE = SELECTION.with_name("task_family_curve.csv")
FIVE_TB_EVIDENCE = SELECTION.with_name("per_dataset_source_evidence.csv")
FIVE_TB_UPDATE = SELECTION.parent / "v2_update_20261008"
FIVE_TB_EXTENDED_CURVE = FIVE_TB_UPDATE / "selected29_curve.csv"
FIVE_TB_EXTENDED_EVIDENCE = FIVE_TB_UPDATE / "all_source_evidence.csv"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
METRICS = {"classification": "macro_f1", "regression": "spearman", "retrieval": "map_at_5", "clustering": "nmi", "segmentation": "mDice"}
LABELS = {"classification": "Classification", "regression": "Regression", "retrieval": "Retrieval", "clustering": "Clustering", "segmentation": "Segmentation"}
COLORS = {"classification": "#2563A6", "regression": "#9A4EAE", "retrieval": "#16807A", "clustering": "#D17A22", "segmentation": "#3A8F5C", "overall": "#C64B4B"}


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def task_value(task):
    key = task["key"]
    done = CAMPAIGN / "_state/done" / f"{key}.json"
    if not done.is_file():
        return None, None
    audit = json.loads(done.read_text())
    if audit["status"] != "VALID_COMPLETE":
        raise ValueError(f"Unexpected validation status: {done}")
    cell = CAMPAIGN / "cells" / key
    family = task["dataset"]["task"]
    if family == "segmentation":
        files = []
        for seed in range(3):
            matches = list(cell.glob(f"results/**/budget20/seed{seed}/**/results.json"))
            if len(matches) != 1:
                raise ValueError(f"Expected one E20 result for seed {seed}: {cell} ({len(matches)})")
            files.extend(matches)
        values = [float(json.loads(path.read_text())["test"]["mDice"]) for path in files]
        return statistics.mean(values), "|".join(str(path) for path in files)
    source = cell / "component_result.json"
    result = json.loads(source.read_text())
    if family == "retrieval" and "rows" in result:
        result = next(row for row in result["rows"] if row["task"] == "retrieval" and row["aggregation"] == "global")
    metric = METRICS[family]
    return float(result[metric]), str(source)


def build():
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    manifest_path = CAMPAIGN / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol_id"] == "bio-eval-union-v4"
    assert manifest["campaign_scope"] == "V4_SHARED_V3_COMPONENTS_ONLY"
    assert manifest["v4_aggregate_allowed"] is False
    tasks = [task for task in manifest["tasks"] if task["key"].startswith(PREFIX)]
    steps = sorted({int(task["asset"]["checkpoint_id"]) for task in tasks})
    expected = defaultdict(set)
    with SELECTION.open(newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["family"]].add(row["dataset"])
    assert {family: len(expected[family]) for family in FAMILIES} == {
        "classification": 16, "regression": 2, "retrieval": 4, "clustering": 2, "segmentation": 5
    }
    assert len(tasks) == 38 * len(steps)

    component_rows = []
    by_step = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task in sorted(tasks, key=lambda task: task["key"]):
        step = int(task["asset"]["checkpoint_id"])
        family = task["dataset"]["task"]
        dataset = task["dataset"]["dataset"]
        value, source = task_value(task)
        if value is not None:
            by_step[step][family][dataset].append(value)
        component_rows.append({
            "checkpoint": step, "task_key": task["key"], "family": family,
            "dataset": dataset, "split": task["dataset"].get("split", ""),
            "metric": METRICS[family], "value": "" if value is None else value,
            "status": "PENDING" if value is None else "VALID_COMPLETE",
            "source": source or "", "validation": str(CAMPAIGN / "_state/done" / f"{task['key']}.json") if source else "",
        })
        if family == "retrieval" and dataset in expected["clustering"]:
            cluster_value = None
            if source:
                cluster_value = float(json.loads(Path(source).read_text())["nmi"])
                by_step[step]["clustering"][dataset].append(cluster_value)
            component_rows.append({
                "checkpoint": step, "task_key": task["key"], "family": "clustering",
                "dataset": dataset, "split": task["dataset"].get("split", ""),
                "metric": "nmi", "value": "" if cluster_value is None else cluster_value,
                "status": "PENDING" if cluster_value is None else "VALID_COMPLETE",
                "source": source or "", "validation": str(CAMPAIGN / "_state/done" / f"{task['key']}.json") if source else "",
            })
    write_csv(REPORT / "components.csv", component_rows, list(component_rows[0]))

    curve = []
    for step in steps:
        values = {}
        for family in FAMILIES:
            observed = by_step[step][family]
            selected = {dataset: observed[dataset] for dataset in expected[family] if dataset in observed}
            complete = set(selected) == expected[family] and all(
                len(selected[dataset]) == (3 if dataset == "pannuke" else 1)
                for dataset in expected[family]
            )
            value = statistics.mean(statistics.mean(selected[dataset]) for dataset in sorted(expected[family])) if complete else None
            values[family] = value
            curve.append({"checkpoint": step, "family": family, "metric": METRICS[family],
                          "datasets_complete": len(selected), "datasets_expected": len(expected[family]),
                          "value": "" if value is None else value, "complete": complete})
        full = all(value is not None for value in values.values())
        curve.append({"checkpoint": step, "family": "overall", "metric": "equal_five_family_mean",
                      "datasets_complete": sum(value is not None for value in values.values()),
                      "datasets_expected": 5,
                      "value": statistics.mean(values.values()) if full else "", "complete": full})
    write_csv(REPORT / "family_curve.csv", curve, list(curve[0]))
    counts = defaultdict(int)
    for row in component_rows:
        if row["family"] != "clustering":
            counts[row["checkpoint"]] += row["status"] == "VALID_COMPLETE"
    metadata = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected29_manifest": str(SELECTION),
        "selected29_sha256": hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        "protocol_id": manifest["protocol_id"], "campaign_scope": manifest["campaign_scope"],
        "v4_aggregate_allowed": False, "old_results_relabelled": False,
        "checkpoints": steps, "components_per_checkpoint": 38,
        "completed_components": {str(step): counts[step] for step in steps},
        "family_dataset_counts": {family: len(expected[family]) for family in FAMILIES},
        "segmentation_rule": "E20 test mDice: mean of seeds 0/1/2; PanNuke mean of three rotations",
        "curve_rule": "Fixed posthoc selected-29 datasets; missing component leaves its family and overall blank; no smoothing",
        "overall_rule": "Descriptive equal mean of five complete family means, not an official v4 aggregate",
    }
    (REPORT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_plot(curve, steps, counts)
    make_comparison_plot(curve)
    latest = steps[-1]
    missing = [row["task_key"] for row in component_rows if row["checkpoint"] == latest and row["status"] != "VALID_COMPLETE"]
    (REPORT / "README.md").write_text(
        "# 20TB route2 v4 component evidence\n\n"
        f"Source: `{CAMPAIGN}`. Protocol `bio-eval-union-v4`, scope "
        "`V4_SHARED_V3_COMPONENTS_ONLY`. Each completed cell has a `VALID_COMPLETE` "
        "validator record. These are v4 component observations, not a complete v4 "
        "aggregate. The selected NCT/CRC clustering values come from nested "
        "NMI results in validated retrieval components; separate clustering tasks, "
        "detection, CTC and OOD are outside this campaign.\n\n"
        f"The campaign covers {len(steps)} checkpoints and 38 components per checkpoint "
        "(24 classification, 2 regression, 4 retrieval, 8 segmentation rotations). "
        f"The latest evaluated checkpoint, {latest}, has {counts[latest]}/38 validated components. "
        f"Pending: {', '.join(missing) if missing else 'none'}.\n\n"
        "`components.csv` gives every source and validator record, including derived "
        "NCT/CRC clustering NMI from their validated retrieval components. "
        "`family_curve.csv` uses the same fixed post hoc selected-29 datasets as "
        "the referenced 5TB figure (16/2/4/2/5). PanNuke's three rotations "
        "become one dataset; "
        "segmentation uses E20 test mDice averaged over seeds 0/1/2. A missing cell "
        "leaves that family and the descriptive five-family mean blank. No incomplete score is treated as a "
        "complete point.\n\n"
        f"Figures: `{FIGURE}` (PNG, SVG, PDF). The additional "
        "`route2_20tb_vs_5tb_nogram_selected29_diagnostic` overlays the 5TB no-GRAM "
        "historical v3 plus hxw continuation curve (85 checkpoints through ck41479) "
        "and 20TB v4 shared-component curve on the same fixed 29 datasets. "
        "The curves retain separate provenance and are descriptive, not a full-v4 "
        "or controlled head-to-head comparison. Regenerate with "
        "`python scripts/plot_route2_20tb_v4_components_20261008.py`.\n"
    )
    print(f"Report: {REPORT}\nFigure: {FIGURE}\nLatest: ck{latest}, {counts[latest]}/38")


def make_plot(curve, steps, counts):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    panels = (*FAMILIES, "overall")
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    for ax, family in zip(axes.flat, panels):
        rows = [row for row in curve if row["family"] == family and row["complete"]]
        xs = [row["checkpoint"] for row in rows]
        ys = [row["value"] for row in rows]
        color = COLORS[family]
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=3.2)
        if rows:
            peak = max(rows, key=lambda row: (row["value"], -row["checkpoint"]))
            ax.scatter([peak["checkpoint"]], [peak["value"]], marker="*", s=160,
                       color="#D44242", edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate(f"peak ck{peak['checkpoint']}", (peak["checkpoint"], peak["value"]),
                        xytext=(-8, -22), textcoords="offset points", ha="right", fontsize=8,
                        color="#9D2727")
        title = LABELS.get(family, "Five-family mean")
        ax.set_title(title if family == "overall" else f"{title} | {next(row['datasets_expected'] for row in rows)} datasets", fontsize=11)
        ax.set_ylabel("Equal-family mean" if family == "overall" else METRICS[family])
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        ax.grid(axis="x", color="#EDF0F2", linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("20TB checkpoint (optimizer steps)")
    for ax in axes.flat:
        ax.set_xlim(0, max(steps) * 1.04)
    fig.suptitle("HS6-L 20TB route2 | v4 component trajectory", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.942, "Fixed post hoc 29-dataset subset (16 / 2 / 4 / 2 / 5); E20 segmentation, 3 seeds; no smoothing",
             ha="center", fontsize=10.5, color="#4B5862")
    fig.legend(handles=[Line2D([0], [0], color="#5B6670", marker="o", label="Complete family means"),
                        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D44242",
                               markersize=12, label="Family peak")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.008))
    fig.text(0.5, 0.055, "Shared v3-compatible v4 components only. Post hoc test-selected subset; mean is descriptive, not a full v4 score. Incomplete points are gaps.",
             ha="center", fontsize=9, color="#68747D")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.89, hspace=0.30, wspace=0.25)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURE / f"route2_20tb_v4_components.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def make_comparison_plot(route2_curve):
    with FIVE_TB_CURVE.open(newline="") as handle:
        historical = [row for row in csv.DictReader(handle) if row["model"] == "5tb_no_gram"]
    with FIVE_TB_EVIDENCE.open(newline="") as handle:
        evidence_protocols = {row["protocol"] for row in csv.DictReader(handle) if row["model"] == "5tb_no_gram"}
    if evidence_protocols != {"v3"}:
        raise ValueError(f"5TB evidence provenance changed: {evidence_protocols}")
    update_manifest = json.loads((FIVE_TB_UPDATE / "manifest.json").read_text())
    selected = defaultdict(set)
    with SELECTION.open(newline="") as handle:
        for row in csv.DictReader(handle):
            selected[row["family"]].add(row["dataset"])
    if {family: sorted(names) for family, names in selected.items()} != update_manifest["selected_datasets"]:
        raise ValueError("5TB continuation uses a different selected-29 set")
    conflicts = json.loads((FIVE_TB_UPDATE / "conflicts.json").read_text())
    if any(conflict["key"][0] == "noGRAM" for conflict in conflicts):
        raise ValueError("5TB no-GRAM continuation has conflicting source cells")
    with FIVE_TB_EXTENDED_CURVE.open(newline="") as handle:
        five_tb = [row for row in csv.DictReader(handle) if row["model"] == "noGRAM"]
    family_alias = {"five_family_mean": "overall"}
    metric_alias = {"Macro-F1": "macro_f1", "Spearman rho": "spearman", "mAP@5": "map_at_5",
                    "NMI": "nmi", "mDice (E20, 3 seeds)": "mDice",
                    "Descriptive equal-family mean": "equal_five_family_mean"}
    historical_lookup = {(int(row["checkpoint"]), family_alias.get(row["task_family"], row["task_family"])):
                         float(row["raw_macro_mean"]) for row in historical}
    rows = []
    for row in five_tb:
        checkpoint = int(row["checkpoint"])
        family = row["family"]
        value = float(row["value"])
        if not math.isfinite(value) or int(row["n"]) != int(row["required"]):
            raise ValueError(f"Incomplete no-GRAM selected-29 point: {row}")
        if (checkpoint, family) in historical_lookup and abs(value - historical_lookup[checkpoint, family]) > 1e-10:
            raise ValueError(f"5TB continuation changed a historical point: {row}")
        if checkpoint <= 29279 and (checkpoint, family) not in historical_lookup:
            raise ValueError(f"Missing original 5TB point: {row}")
        rows.append({"model": "5tb_no_gram", "checkpoint": checkpoint,
                     "family": family, "value": value, "complete": True,
                     "source_protocol": "v3" if checkpoint <= 29279 else "hxw_selected29_continuation",
                     "source": str(FIVE_TB_EXTENDED_CURVE)})
    for row in historical:
        family = family_alias.get(row["task_family"], row["task_family"])
        if metric_alias[row["metric"]] != ("equal_five_family_mean" if family == "overall" else METRICS[family]):
            raise ValueError(f"Metric mismatch in 5TB curve: {row}")
    for row in route2_curve:
        rows.append({"model": "20tb_route2", "checkpoint": row["checkpoint"],
                     "family": row["family"], "value": row["value"],
                     "complete": row["complete"], "source_protocol": "v4_shared_components",
                     "source": str(REPORT / "family_curve.csv")})
    assert len(historical) == 60 * 6
    assert len(five_tb) == 85 * 6
    assert len(historical_lookup) == len(historical)
    assert len(route2_curve) % 6 == 0
    write_csv(FIGURE / "comparison_curve.csv", rows, list(rows[0]))
    latest_step = max(row["checkpoint"] for row in route2_curve)
    latest_complete = next(row["complete"] for row in route2_curve
                           if row["checkpoint"] == latest_step and row["family"] == "overall")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    peak_rows = []
    for ax, family in zip(axes.flat, (*FAMILIES, "overall")):
        for model, color, marker, line in (("5tb_no_gram", "#58636D", "o", "--"),
                                           ("20tb_route2", COLORS[family], "s", "-")):
            points = sorted((row for row in rows if row["model"] == model and
                             row["family"] == family and row["complete"]),
                            key=lambda row: row["checkpoint"])
            xs = [row["checkpoint"] for row in points]
            ys = [row["value"] for row in points]
            ax.plot(xs, ys, color=color, linestyle=line, linewidth=1.7,
                    marker=marker, markersize=2.9 if model == "5tb_no_gram" else 3.5,
                    alpha=0.92)
            peak = max(points, key=lambda row: (row["value"], -row["checkpoint"]))
            ax.scatter([peak["checkpoint"]], [peak["value"]], marker="*", s=140,
                       color=color, edgecolor="white", linewidth=0.7, zorder=5)
            peak_rows.append({"model": model, "family": family,
                              "peak_checkpoint": peak["checkpoint"], "peak_value": peak["value"],
                              "complete_points": len(points)})
        ax.axvline(41479, color="#AAB1B7", linestyle=":", linewidth=0.9)
        title = LABELS.get(family, "Five-family mean")
        count = "29 datasets" if family == "overall" else f"{next(row['datasets_expected'] for row in route2_curve if row['family'] == family)} datasets"
        ax.set_title(f"{title} | {count}", fontsize=11)
        ax.set_ylabel("Equal-family mean" if family == "overall" else METRICS[family])
        ax.set_xlim(0, max(row["checkpoint"] for row in route2_curve) * 1.04)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        ax.grid(axis="x", color="#EDF0F2", linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Checkpoint (optimizer steps)")
    fig.suptitle("HS6-L | 5TB no-GRAM and 20TB route2", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.941, "Same fixed post hoc 29-dataset subset; five task means and equal-family mean; no smoothing",
             ha="center", fontsize=10.5, color="#4B5862")
    fig.legend(handles=[Line2D([0], [0], color="#58636D", linestyle="--", marker="o", label="5TB no-GRAM | history + continuation"),
                        Line2D([0], [0], color="#16807A", marker="s", label="20TB route2 | v4 shared components"),
                        Line2D([0], [0], color="#AAB1B7", linestyle=":", label="5TB endpoint ck41479")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.008))
    latest_note = (f"20TB ck{latest_step} has all 29 selected datasets."
                   if latest_complete else f"20TB ck{latest_step} is incomplete; missing family values are gaps.")
    fig.text(0.5, 0.053, "Exploratory cross-protocol overlay, not a controlled comparison or full v4 score. " + latest_note,
             ha="center", fontsize=9, color="#68747D")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.89, hspace=0.30, wspace=0.25)
    stem = "route2_20tb_vs_5tb_nogram_selected29_diagnostic"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURE / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    write_csv(FIGURE / "comparison_peaks.csv", peak_rows, list(peak_rows[0]))
    campaign = json.loads((CAMPAIGN / "campaign_manifest.json").read_text())
    current_protocol = Path(campaign["protocol_rules_path"])
    current_protocol_sha256 = hashlib.sha256(current_protocol.read_bytes()).hexdigest()
    comparison_manifest = {
        "selection_sha256": hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        "five_tb_source": str(FIVE_TB_EXTENDED_CURVE),
        "five_tb_source_sha256": hashlib.sha256(FIVE_TB_EXTENDED_CURVE.read_bytes()).hexdigest(),
        "five_tb_original_source": str(FIVE_TB_CURVE),
        "five_tb_original_source_sha256": hashlib.sha256(FIVE_TB_CURVE.read_bytes()).hexdigest(),
        "five_tb_update_manifest": str(FIVE_TB_UPDATE / "manifest.json"),
        "five_tb_update_manifest_sha256": hashlib.sha256((FIVE_TB_UPDATE / "manifest.json").read_bytes()).hexdigest(),
        "five_tb_update_evidence": str(FIVE_TB_EXTENDED_EVIDENCE),
        "five_tb_update_evidence_sha256": hashlib.sha256(FIVE_TB_EXTENDED_EVIDENCE.read_bytes()).hexdigest(),
        "five_tb_tail_checkpoints": sorted({int(row["checkpoint"]) for row in five_tb if int(row["checkpoint"]) > 29279}),
        "five_tb_evidence_source": str(FIVE_TB_EVIDENCE),
        "five_tb_evidence_sha256": hashlib.sha256(FIVE_TB_EVIDENCE.read_bytes()).hexdigest(),
        "five_tb_evidence_protocols": sorted(evidence_protocols),
        "route2_source": str(REPORT / "family_curve.csv"),
        "route2_source_sha256": hashlib.sha256((REPORT / "family_curve.csv").read_bytes()).hexdigest(),
        "route2_campaign": str(CAMPAIGN / "campaign_manifest.json"),
        "route2_protocol_sha256_at_launch": campaign["protocol_sha256"],
        "current_protocol_sha256": current_protocol_sha256,
        "protocol_hash_matches_current": campaign["protocol_sha256"] == current_protocol_sha256,
        "models": {"5tb_no_gram": 85, "20tb_route2": len(route2_curve) // 6},
        "cross_protocol_diagnostic_only": True,
        "full_v4_aggregate_allowed": False,
        "missing_values_are_gaps": True,
    }
    (FIGURE / "comparison_manifest.json").write_text(json.dumps(comparison_manifest, indent=2) + "\n")


if __name__ == "__main__":
    build()
