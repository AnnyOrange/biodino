#!/usr/bin/env python3
"""Three honest 5TB early/middle plots: historical, validated v3, v4 inventory."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914"
OLD = ROOT / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
CAMPAIGNS = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs")
V3 = CAMPAIGNS / "retest_20260918"
UNION = CAMPAIGNS / "hs6_5tb_protocol_union_nonseg_20260921"
DETECTION = CAMPAIGNS / "hs6_5tb_union_detection_observation_20260921"
OUT = ROOT / "outputs/00_reports/hs6_l5_5tb_three_protocol_curves_20260921"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection")
COLORS = ("#2563a6", "#ae5895", "#168078", "#bd7725", "#32945f", "#b45c53")
METRIC = ("BA / AUC", "R²", "Recall@1", "NMI", "mDice", "patch F1")
TARGET = {
    "old": dict(zip(FAMILIES, (25, 4, 6, 6, 8, 3))),
    "v3": dict(zip(FAMILIES, (24, 2, 5, 5, 7, 0))),
    "v4": dict(zip(FAMILIES, (25, 4, 7, 7, 7, 3))),
}
ANCHORS = (12687, 20007, 28791)
MAX_STEP = 22000


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(family: str, obj: dict) -> float | None:
    if family == "classification":
        val = next((obj.get(k) for k in ("macro_auc", "macro_auroc", "auroc")
                    if isinstance(obj.get(k), (int, float))) , None) if obj.get("task") == "multilabel_classification" else obj.get("balanced_accuracy")
    elif family == "regression":
        val = obj.get("r2")
    elif family == "retrieval":
        val = obj.get("recall_at_1")
    elif family == "clustering":
        val = obj.get("nmi")
    elif family == "segmentation":
        val = obj.get("test", {}).get("mDice")
    else:
        val = obj.get("test_patch_f1")
        if isinstance(val, (int, float)) and val > 1:
            val /= 100
    return float(val) if isinstance(val, (int, float)) and math.isfinite(val) else None


def old_cells(step: int) -> dict[str, dict[str, float]]:
    root = OLD / f"point_{step}"
    cells: dict[str, dict[str, float]] = defaultdict(dict)
    for lane in root.glob("classification_*"):
        for p in lane.glob("bio_classification/*/*/last_result.json"):
            obj = load(p)
            val = metric("classification", obj)
            if val is not None and not obj.get("error"):
                cells["classification"][obj["dataset"]] = val
    for p in root.glob("regression/bio_regression/*/*/last_result.json"):
        obj = load(p)
        val = metric("regression", obj)
        if val is not None and not obj.get("error"):
            cells["regression"][obj["dataset"]] = val
    for p in root.glob("retrieval/bio_retrieval/*/*/last_result.json"):
        obj = load(p)
        if obj.get("error"):
            continue
        for family in ("retrieval", "clustering"):
            candidates = obj.get("rows", [obj])
            val = next((metric(family, row) for row in candidates
                        if row.get("task") == family and metric(family, row) is not None), None)
            if val is None and "rows" not in obj:
                val = metric(family, obj)
            if val is not None:
                cells[family][obj["dataset"]] = val
    for lane in root.glob("segmentation_*"):
        for p in lane.glob(f"**/{step}/results.json"):
            obj = load(p)
            val = metric("segmentation", obj)
            if val is not None:
                cells["segmentation"][p.parent.parent.name] = val
    for p in root.glob("detection/bio_detection/*/*/results_bio_detection.json"):
        obj = load(p)
        val = metric("detection", obj)
        if val is not None and not obj.get("error"):
            cells["detection"][obj["dataset"]] = val
    return cells


def verified_items(campaign: Path, wanted_steps: set[int]):
    manifest = load(campaign / "campaign_manifest.json")
    for task in manifest["tasks"]:
        asset = task["asset"]
        if asset.get("arm") != "5tb_no_gram":
            continue
        step = int(asset["checkpoint_id"])
        if step not in wanted_steps:
            continue
        spec = task["dataset"]
        if spec.get("status") != "PASS":
            continue
        folder = campaign / "cells" / task["key"]
        report = folder / "validation_report.json"
        if not report.is_file() or load(report).get("status") != "VALID_COMPLETE":
            continue
        yield step, spec, folder


def verified_cells(campaign: Path, steps: set[int], dense: bool = False):
    cells = defaultdict(lambda: defaultdict(dict))
    for step, spec, folder in verified_items(campaign, steps):
        family, dataset = spec["task"], spec["dataset"]
        if dense:
            if family != "segmentation":
                continue
            paths = sorted(folder.glob("results/**/budget50/seed*/**/results.json"))
            if len(paths) != 3:
                continue
            vals = [metric("segmentation", load(p)) for p in paths]
            if any(v is None for v in vals):
                continue
            fold = spec.get("split", "") if dataset == "pannuke" else ""
            cells[step]["segmentation"][dataset + str(fold)] = sum(vals) / 3
            continue
        result = folder / "component_result.json"
        if not result.is_file():
            continue
        obj = load(result)
        if family == "detection":
            val = metric("detection", obj)
            if val is not None:
                cells[step]["detection"][dataset] = val
        elif family == "retrieval":
            for name in ("retrieval", "clustering"):
                candidates = obj.get("rows", [obj])
                val = next((metric(name, row) for row in candidates
                            if row.get("task") == name and metric(name, row) is not None), None)
                if val is None and "rows" not in obj:
                    val = metric(name, obj)
                if val is not None:
                    cells[step][name][dataset] = val
        elif family in ("classification", "regression"):
            val = metric(family, obj)
            if val is not None:
                cells[step][family][dataset] = val
    if dense:
        for step in steps:
            scores = cells[step]["segmentation"]
            folds = {key: value for key, value in scores.items() if key.startswith("pannuke")}
            for key in folds:
                del scores[key]
            if len(folds) == 3:
                scores["pannuke"] = sum(folds.values()) / 3
    return cells


def add_cells(base, extension):
    for step, families in extension.items():
        for family, datasets in families.items():
            overlap = set(base[step][family]) & set(datasets)
            if overlap:
                raise ValueError(f"Overlapping {family} at ck{step}: {overlap}")
            base[step][family].update(datasets)


def summarize(cells, steps, protocol):
    rows = []
    for step in steps:
        for family in FAMILIES:
            scores = cells[step][family]
            expected = TARGET[protocol][family]
            n = len(scores)
            if n > expected:
                raise ValueError(f"{protocol}/{family}/ck{step}: {n} > {expected}")
            rows.append(dict(protocol=protocol, checkpoint=step, family=family,
                             present=n, expected=expected, complete=int(expected > 0 and n == expected),
                             macro_mean=(sum(scores.values()) / n if n else ""),
                             dataset_ids=";".join(sorted(scores))))
    return rows


def plot(rows, protocol, max_step=MAX_STEP, output_stem=None):
    title = {"old": "Old protocol | historical 25/4/6/8/3*",
             "v3": "v3 | validated components, not a full-suite score",
             "v4": "v4 inventory | matched v3 + validated legacy companions"}[protocol]
    if protocol == "old" and max_step > MAX_STEP:
        title = "Old protocol | 5TB trajectory through ck29279 (historical)"
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)
    for ax, family, color, unit in zip(axes.flat, FAMILIES, COLORS, METRIC):
        scope = [r for r in rows if r["protocol"] == protocol and r["family"] == family and r["checkpoint"] <= max_step]
        target = TARGET[protocol][family]
        ax.set_title(f"{family.capitalize()} | target {target}" if target else "Detection | no formal v3 cell")
        ax.set_xlabel("Training updates (k)")
        ax.set_ylabel(unit)
        ax.set_xlim(0, max_step / 1000 + .2)
        ax.grid(alpha=.23)
        if not target:
            ax.text(.5, .5, "No formal detection dataset", ha="center", va="center", transform=ax.transAxes)
            continue
        xs = [r["checkpoint"] / 1000 for r in scope]
        ys = [r["macro_mean"] if r["macro_mean"] != "" else math.nan for r in scope]
        line_ys = [value if protocol != "old" or r["complete"] else math.nan
                   for value, r in zip(ys, scope)]
        ax.plot(xs, line_ys, color=color, linewidth=1.4, linestyle="--", alpha=.85)
        full = [r for r in scope if r["complete"]]
        partial = [r for r in scope if not r["complete"] and r["present"]]
        ax.scatter([r["checkpoint"] / 1000 for r in full], [r["macro_mean"] for r in full],
                   s=18, c=color, label="all listed cells present", zorder=4)
        ax.scatter([r["checkpoint"] / 1000 for r in partial], [r["macro_mean"] for r in partial],
                   s=23, facecolors="white", edgecolors=color, label="partial only", zorder=4)
        markers = ((12687, "E"), (20007, "M"))
        if protocol == "old" and max_step > MAX_STEP:
            markers += ((29279, "L"),)
        for step, letter in markers:
            match = next((r for r in scope if r["checkpoint"] == step), None)
            if match and match["macro_mean"] != "":
                ax.annotate(f"{letter} {match['present']}/{target}",
                            (step / 1000, match["macro_mean"]),
                            xytext=(0, 11 if letter in ("E", "L") else -16), textcoords="offset points",
                            ha="center", color=color, fontsize=8, fontweight="bold")
        if full and partial:
            ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(title, fontsize=15)
    foot = {"old": "Historical frozen probes; segmentation uses old 8, detection is legacy B4 and NOT comparable to v4 B8.",
            "v3": "v3 target Ret5/Seg7; RxRx3 and official MoNuSeg not filled by legacy scores. E50/seed0-2 dense only.",
            "v4": "Target Ret7/Seg7/Det3*: RxRx3/MoNuSeg missing; LC25000 uses locked legacy random split; Det3* is proxy, often partial."}[protocol]
    fig.text(.5, -.055, foot + " Hollow = incomplete; changing coverage can produce artificial jumps.",
             fontsize=8, ha="center")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{output_stem or protocol + '_early_middle'}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def export_old_wide(rows, filename, max_step):
    """Exact six y-series in the requested old-protocol figure, plus coverage."""
    old_by_step = defaultdict(dict)
    for row in rows:
        if row["protocol"] == "old" and row["checkpoint"] <= max_step:
            old_by_step[row["checkpoint"]][row["family"]] = row
    records = []
    for step in sorted(old_by_step):
        by_family = old_by_step[step]
        if set(by_family) != set(FAMILIES):
            raise ValueError(f"Old-protocol family missing at ck{step}")
        record = {"checkpoint": step, "training_updates_k": step / 1000}
        for family in FAMILIES:
            record[f"{family}_macro"] = by_family[family]["macro_mean"]
            record[f"{family}_n"] = by_family[family]["present"]
        records.append(record)
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    with (AUDIT / "task_family_curve.csv").open(newline="") as handle:
        steps = sorted({int(r["checkpoint"]) for r in csv.DictReader(handle)})
    if len(steps) != 49 or not all(anchor in steps for anchor in ANCHORS):
        raise ValueError("Historical 49-point checkpoint support changed")
    step_set = set(steps)
    old = defaultdict(lambda: defaultdict(dict))
    for step in steps:
        old[step] = old_cells(step)
    v3 = verified_cells(V3, step_set)
    add_cells(v3, verified_cells(V3, step_set, dense=True))
    v4 = defaultdict(lambda: defaultdict(dict))
    add_cells(v4, v3)
    add_cells(v4, verified_cells(UNION, step_set))
    add_cells(v4, verified_cells(DETECTION, step_set))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for protocol, cells in (("old", old), ("v3", v3), ("v4", v4)):
        group = summarize(cells, steps, protocol)
        rows.extend(group)
        plot(group, protocol)
    with (OUT / "coverage_and_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export_old_wide(rows, "old_early_middle.csv", MAX_STEP)
    # The earlier plot intentionally zoomed to 22k; include every historical
    # point directory in a separate full-span plot without promoting it to v3/v4.
    all_old_steps = sorted(int(p.name.removeprefix("point_")) for p in OLD.glob("point_*")
                           if p.name.removeprefix("point_").isdigit())
    if len(all_old_steps) < len(steps) or all_old_steps[-1] != 29279:
        raise ValueError("Unexpected historical point inventory")
    for step in all_old_steps:
        if step not in old:
            old[step] = old_cells(step)
    full_old_rows = summarize(old, all_old_steps, "old")
    export_old_wide(full_old_rows, "old_full_trajectory.csv", 30000)
    plot(full_old_rows, "old", max_step=30000, output_stem="old_full_trajectory")
    (OUT / "README.txt").write_text(
        "DESCRIPTIVE / no checkpoint selection. Old: original 5TB point outputs; "
        "v3: independently validated retest_20260918 components only; v4 inventory: "
        "v3 + validated hs6_5tb_protocol_union_nonseg_20260921 and "
        "hs6_5tb_union_detection_observation_20260921. The last two campaigns "
        "predate the new protocol_v4.json and DO NOT certify a v4 aggregate. "
        "Only B50 mean over 3 seeds is plotted for v3/v4 segmentation; "
        "PanNuke three rotations count as separate required execution cells. "
        "CTC and OOD are not shown. Old detection B4 and v4 detection B8 "
        "cannot be compared as a controlled checkpoint effect. "
        "The LC25000 classification legacy stratified split is locked but "
        "not source-disjoint. Hollow markers show incomplete observed subsets. "
        "old_early_middle.csv covers plotted <=22k points; old_full_trajectory.csv "
        "covers all 60 historical point directories through ck29279 (including "
        "un-audited or partial point directories, identified by per-family n).\n"
    )
    print(OUT)


if __name__ == "__main__":
    main()
