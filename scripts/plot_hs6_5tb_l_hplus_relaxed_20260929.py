#!/usr/bin/env python3
"""Exploratory 5TB curves accepting historical B4 detection and MoNuSeg E50."""

import csv
import json
import math
import shlex
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_hs6_5tb_l_hplus_v4_20260929 as base


ROOT = base.ROOT
EARLY = base.EARLY
STRICT = base.OUT
INDEX = base.EVAL / "old_v3_protocol_union/results.csv"
OUT = ROOT / "outputs/00_reports/hs6_5tb_l_hplus_relaxed_curves_20260929"


def save(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = base.expected()
    cells = {}
    steps = defaultdict(set)
    for row in csv.DictReader((STRICT / "curves_and_coverage.csv").open()):
        steps[row["model"]].add(int(row["checkpoint"]))
    for row in csv.DictReader((STRICT / "cells.csv").open()):
        if row["family"] == "segmentation":
            continue
        key = row["model"], int(row["checkpoint"]), row["family"], row["dataset"]
        cells[key] = (float(row["value"]), row["source"], "fixed_v4_list")

    # Use E50 for every segmentation dataset. The old MoNuSeg observation
    # cannot be put into an E20 mean without changing the probe budget.
    for row in csv.DictReader(EARLY.open()):
        if row["method"] != "Vanilla (5TB no-GRAM)" or row["family"] != "segmentation" or row["budget"] != "50":
            continue
        key = "L no-GRAM", int(row["checkpoint"]), "segmentation", row["dataset"]
        cells[key] = (float(row["value"]), row["source"], "formal_e50")

    requests = []
    for model, path in (("L no-GRAM", base.LATE_L), ("H+", base.HPLUS)):
        inventory = json.loads(path.read_text())
        for step_s, families in inventory["checkpoints"].items():
            step = int(step_s)
            for dataset in targets["segmentation"]:
                names = ([dataset] if dataset != "pannuke" else
                         [f"pannuke/fold{i}" for i in (1, 2, 3)])
                for name in names:
                    entry = families["segmentation"].get(name)
                    if entry and entry["status"] == "VALID_COMPLETE":
                        requests.append((model, step, "segmentation", name, entry["evidence"], 50))
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "5090-hxw-xzj",
         "python3 -c " + shlex.quote(base.REMOTE)],
        input=json.dumps(requests), text=True, capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip())
    folds = defaultdict(list)
    for model, step, family, dataset, value, source in json.loads(process.stdout):
        if dataset.startswith("pannuke/fold"):
            folds[model, step].append((dataset, value, source))
        else:
            cells[model, step, family, dataset] = (value, source, "formal_e50")
    for (model, step), values in folds.items():
        if len(values) == 3 and {x[0] for x in values} == {f"pannuke/fold{i}" for i in (1, 2, 3)}:
            cells[model, step, "segmentation", "pannuke"] = (
                statistics.mean(x[1] for x in values), ";".join(x[2] for x in values), "formal_e50"
            )

    added = Counter()
    for row in csv.DictReader(INDEX.open()):
        if row["model"] != "5tb_no_gram":
            continue
        step = int(row["checkpoint"])
        if (row["family"], row["dataset"], row["protocol"]) == ("detection", "livecell", "old_observation"):
            key = "L no-GRAM", step, "detection_proxy", "livecell"
            if key in cells:
                continue
            obj = json.loads(Path(row["source"]).read_text())
            if (obj.get("batch_size"), obj.get("epochs"), obj.get("seed"), obj.get("image_size")) != (4, 5, 0, 224):
                raise ValueError(f"Unexpected legacy detection settings at ck{step}")
            value = obj.get("test_patch_f1")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            cells[key] = (value / 100, row["source"], "accepted_legacy_b4")
            added["detection_b4"] += 1
        if (row["family"], row["dataset"], row["protocol"]) == ("segmentation", "monuseg", "old"):
            key = "L no-GRAM", step, "segmentation", "monuseg"
            if key in cells:
                continue
            obj = json.loads(Path(row["source"]).read_text())
            meta = obj.get("_meta", {})
            if (meta.get("probe_epochs"), meta.get("probe_batch_size"), meta.get("seed")) != (50, 16, 0):
                raise ValueError(f"Unexpected legacy MoNuSeg settings at ck{step}")
            value = obj.get("test", {}).get("mDice")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            cells[key] = (value, row["source"], "legacy_monuseg_e50_b16_seed0_unverified_identity")
            added["monuseg_legacy"] += 1

    cell_rows = [dict(model=m, checkpoint=s, family=f, dataset=d, value=v,
                      provenance=kind, source=source)
                 for (m, s, f, d), (v, source, kind) in sorted(cells.items())]
    save(OUT / "cells_with_provenance.csv", cell_rows)
    curves = []
    for model in base.COLORS:
        for step in sorted(steps[model]):
            for family in base.FAMILIES:
                included = [cells[model, step, family, ds] for ds in targets[family]
                            if (model, step, family, ds) in cells]
                missing = [ds for ds in targets[family] if (model, step, family, ds) not in cells]
                curves.append(dict(model=model, checkpoint=step, family=family,
                                   observed=len(included), expected=len(targets[family]),
                                   mean=statistics.mean(x[0] for x in included) if not missing else "",
                                   legacy_cells=sum(x[2].startswith(("accepted_legacy", "legacy_monuseg")) for x in included),
                                   missing=";".join(missing)))
    save(OUT / "curves_and_coverage.csv", curves)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharex=True)
    labels = list(base.LABELS)
    labels[4] = "Segmentation E50*"
    for ax, family, label, metric in zip(axes.flat, base.FAMILIES, labels, base.METRICS):
        for model, color in base.COLORS.items():
            rr = [r for r in curves if r["model"] == model and r["family"] == family]
            xx = np.array([r["checkpoint"] for r in rr])
            yy = np.array([float(r["mean"]) if r["mean"] != "" else np.nan for r in rr])
            ax.plot(xx, yy, color=color, lw=1.5, marker="o", markersize=2.6,
                    label=model)
            legacy = [r for r in rr if r["mean"] != "" and r["legacy_cells"]]
            if legacy:
                ax.scatter([r["checkpoint"] for r in legacy], [float(r["mean"]) for r in legacy],
                           facecolors="white", edgecolors=color, s=25, linewidths=1, zorder=4)
            ax.text(.98, .05 if model == "L no-GRAM" else .14,
                    f"{model}: {np.isfinite(yy).sum()}/{len(rr)}", transform=ax.transAxes,
                    ha="right", color=color, fontsize=8)
        ax.set_title(f"{label}  |  {len(targets[family])} datasets", fontsize=11, weight="bold")
        ax.set_ylabel(metric)
        ax.grid(alpha=.22)
        ax.set_xlim(0, 42000)
    for ax in axes[1]:
        ax.set_xlabel("Optimizer updates")
        ax.set_xticks(np.arange(0, 40001, 10000), ["0", "10k", "20k", "30k", "40k"])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .946), frameon=False)
    fig.suptitle("5TB trajectories | exploratory mixed settings", fontsize=17, weight="bold", y=.995)
    fig.text(.5, .025, "Open circles: legacy B4 LIVECell detection or E50/B16/seed0 MoNuSeg. "
             "MoNuSeg identity unverified; not a formal v4 result.",
             ha="center", fontsize=9, color="#515a60")
    fig.subplots_adjust(left=.075, right=.985, top=.89, bottom=.10, wspace=.25, hspace=.28)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"5tb_l_hplus_relaxed_six_families.{ext}", dpi=190)
    plt.close(fig)

    counts = {model: {family: sum(r["mean"] != "" for r in curves if r["model"] == model and r["family"] == family)
                      for family in base.FAMILIES} for model in base.COLORS}
    (OUT / "README.md").write_text(
        "# Exploratory 5TB L / H+ comparison\n\n"
        "This is a mixed-settings observation, not a formal v4 aggregate. "
        "The six fixed v4 dataset lists are unchanged. L ck487–29279, L ck29767–41479, "
        "and H+ step0–34159 are included.\n\n"
        "Old L LIVECell detection B4/5 epochs/seed0 fills missing B8 results. "
        "All segmentation datasets use E50. Old L MoNuSeg E50/B16/seed0 fills "
        "missing formal E50 results, but its sample identity and train/val split "
        "were not independently admitted; open circles mark mixed-protocol means. "
        "At the eight overlapping points, old MoNuSeg mDice is 0.010–0.024 higher "
        "than formal E50, so the resulting segment trend must not be interpreted "
        "as a single-protocol v4 improvement.\n\n"
        "RxRx3 ck24399–28791 still lacks formal results. LC25000 and NCT100 "
        "admission remain provisional; CTC/OOD excluded.\n\n"
        f"Legacy cells added: {dict(added)}. Complete points: {counts}.\n\n"
        "`cells_with_provenance.csv` labels every cell. `curves_and_coverage.csv` "
        "shows complete-only family means and remaining gaps.\n"
    )
    print(json.dumps({"added": added, "complete": counts}, indent=2))


if __name__ == "__main__":
    main()
