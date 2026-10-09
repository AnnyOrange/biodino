#!/usr/bin/env python3
"""Plot fixed-list v4 task trajectories for 5TB L no-GRAM and H+."""

import csv
import json
import math
import shlex
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/02_eval_runs"
EARLY = ROOT / "outputs/00_reports/deepcad_method_20260927/v4_six_tasks_20260929/CELLS.csv"
LATE_L = EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260924/l_checkpoint_task_inventory.json"
HPLUS = EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260929/hplus_checkpoint_task_inventory.json"
OUT = ROOT / "outputs/00_reports/hs6_5tb_l_hplus_v4_curves_20260929"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection_proxy")
LABELS = ("Classification", "Regression", "Retrieval", "Clustering", "Segmentation E20", "Detection proxy")
METRICS = ("BA / chest AUC", "R²", "Recall@1", "NMI", "mDice", "Patch F1")
COLORS = {"L no-GRAM": "#187b79", "H+": "#c25338"}
STATUS = {"RESULT_PRESENT", "VALID_COMPLETE"}

# Run on HXW, where the inventory's evidence paths exist. Validation reports
# enumerate the exact dense result files belonging to each formal component.
REMOTE = r'''
import json, math
from pathlib import Path
import sys

items=json.load(sys.stdin)
out=[]
for item in items:
    model,step,family,dataset,path=item[:5]
    dense_budget=item[5] if len(item)>5 else 20
    try:
        obj=json.loads(Path(path).read_text())
        if family=='classification':
            value=obj.get('macro_auc') if dataset=='chestmnist' else obj.get('balanced_accuracy')
        elif family=='regression': value=obj.get('r2')
        elif family in ('retrieval','clustering'):
            if dataset=='rxrx3-core':
                value=obj.get('tests',{}).get('rxrx3',{}).get('recall_at_1' if family=='retrieval' else 'nmi')
            else:
                rows=obj.get('rows',[obj]); field='recall_at_1' if family=='retrieval' else 'nmi'
                if dataset=='hpa-subcellular':
                    rows=[r for r in rows if r.get('aggregation')==('global' if family=='retrieval' else 'location') and (family=='retrieval' or str(r.get('n_classes'))=='41')]
                elif dataset=='rxrx1-cross':
                    rows=[r for r in rows if r.get('aggregation')==('global' if family=='retrieval' else 'global-perturbation')]
                else: rows=[r for r in rows if r.get('aggregation','class') in ('class','global')]
                value=next((r[field] for r in rows if r.get(field) is not None),None)
        elif family=='segmentation':
            if obj.get('status')!='VALID_COMPLETE':continue
            vals=[]
            for relative in obj.get('result_sha256',{}):
                if '/budget%d/'%dense_budget not in relative:continue
                result=json.loads((Path(path).parent/relative).read_text())
                meta=result.get('_meta',{})
                if meta.get('probe_epochs')!=dense_budget or meta.get('probe_batch_size')!=32:continue
                vals.append((meta.get('seed'),result.get('test',{}).get('mDice')))
            value=sum(v for s,v in vals if s in (0,1,2) and isinstance(v,(int,float)))/3 if len(vals)==3 and {s for s,v in vals}=={0,1,2} else None
        else: value=obj.get('test_patch_f1')
        if isinstance(value,(int,float)) and math.isfinite(value):
            out.append([model,step,family,dataset,float(value)/100 if family=='detection_proxy' else float(value),path])
    except (OSError,ValueError,TypeError,KeyError,ZeroDivisionError): pass
print(json.dumps(out,separators=(',',':')))
'''


def expected():
    spec = json.loads((ROOT / "Evaluation Rules/protocol_v4.json").read_text())
    result = {}
    for family in ("classification", "regression", "retrieval", "segmentation"):
        result[family] = list(dict.fromkeys(
            ds for part in ("tier_a", "tier_b", "union_extension") for ds in spec[part].get(family, [])
        ))
    result["clustering"] = result["retrieval"][:]
    result["detection_proxy"] = spec["union_extension"]["detection_proxy"]
    return result


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = expected()
    cells = {}
    steps = defaultdict(set)
    for row in csv.DictReader(EARLY.open()):
        if row["method"] != "Vanilla (5TB no-GRAM)" or row["family"] not in FAMILIES:
            continue
        if int(row["budget"]) != (20 if row["family"] == "segmentation" else 0):
            continue
        model, step, family, dataset = "L no-GRAM", int(row["checkpoint"]), row["family"], row["dataset"]
        if dataset in targets[family]:
            cells[model, step, family, dataset] = (float(row["value"]), row["source"])
            steps[model].add(step)

    inventory_files = (("L no-GRAM", LATE_L), ("H+", HPLUS))
    requests = []
    for model, path in inventory_files:
        inventory = json.loads(path.read_text())
        for step_s, families in inventory["checkpoints"].items():
            step = int(step_s)
            steps[model].add(step)
            for family in FAMILIES:
                for dataset in targets[family]:
                    key = dataset if family != "segmentation" or dataset != "pannuke" else None
                    entries = ([ (dataset, families[family].get(dataset)) ] if key else
                               [("pannuke/fold" + str(fold), families[family].get("pannuke/fold" + str(fold))) for fold in (1, 2, 3)])
                    for name, entry in entries:
                        if entry and entry["status"] in STATUS and entry.get("evidence"):
                            requests.append((model, step, family, name, entry["evidence"]))
    # Duplicate evidence files (retrieval and clustering) are read independently
    # because their canonical row selection and metrics differ.
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "5090-hxw-xzj", "python3 -c " + shlex.quote(REMOTE)],
        input=json.dumps(requests), text=True, capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(f"HXW metric extraction failed: {process.stderr.strip()}")
    raw = json.loads(process.stdout)
    folds = defaultdict(list)
    for model, step, family, dataset, value, source in raw:
        if family == "segmentation" and dataset.startswith("pannuke/fold"):
            folds[model, step].append((dataset, value, source))
            continue
        key = model, step, family, dataset
        if key in cells and abs(cells[key][0] - value) > 1e-6:
            raise ValueError(f"Conflicting duplicate cell: {key}")
        cells[key] = value, source
    for (model, step), values in folds.items():
        if len(values) == 3 and {v[0] for v in values} == {"pannuke/fold1", "pannuke/fold2", "pannuke/fold3"}:
            cells[model, step, "segmentation", "pannuke"] = (
                statistics.mean(v[1] for v in values), ";".join(v[2] for v in values)
            )

    cell_rows = [dict(model=m, checkpoint=s, family=f, dataset=d, metric=METRICS[FAMILIES.index(f)],
                      value=v, source=source) for (m, s, f, d), (v, source) in sorted(cells.items())]
    save_csv(OUT / "cells.csv", cell_rows)
    curve = []
    for model in COLORS:
        for step in sorted(steps[model]):
            for family in FAMILIES:
                vals = [cells[model, step, family, d][0] for d in targets[family]
                        if (model, step, family, d) in cells]
                curve.append(dict(model=model, checkpoint=step, family=family,
                                  observed=len(vals), expected=len(targets[family]),
                                  mean=statistics.mean(vals) if len(vals) == len(targets[family]) else "",
                                  missing=";".join(d for d in targets[family]
                                                   if (model, step, family, d) not in cells)))
    save_csv(OUT / "curves_and_coverage.csv", curve)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharex=True)
    for ax, family, label, metric in zip(axes.flat, FAMILIES, LABELS, METRICS):
        for model, color in COLORS.items():
            rows = [r for r in curve if r["model"] == model and r["family"] == family]
            xx = np.array([r["checkpoint"] for r in rows]); yy = np.array([
                float(r["mean"]) if r["mean"] != "" else np.nan for r in rows
            ])
            ax.plot(xx, yy, color=color, lw=1.55, marker="o", markersize=2.7,
                    label=model, zorder=3)
            complete = np.isfinite(yy)
            ax.text(.98, .05 if model == "L no-GRAM" else .14,
                    f"{model}: {complete.sum()}/{len(rows)}", transform=ax.transAxes,
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
    fig.suptitle("5TB trajectories | v4 fixed task lists", fontsize=17, weight="bold", y=.995)
    fig.text(.5, .025, "Only complete fixed-list means are shown; missing points are gaps. "
             "LC25000 and NCT100 remain provisional. CTC / OOD excluded.",
             ha="center", fontsize=9, color="#515a60")
    fig.subplots_adjust(left=.075, right=.985, top=.89, bottom=.10, wspace=.25, hspace=.28)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"5tb_l_hplus_v4_six_families.{ext}", dpi=190)
    plt.close(fig)

    counts = defaultdict(dict)
    for model in COLORS:
        for family in FAMILIES:
            rr = [r for r in curve if r["model"] == model and r["family"] == family]
            counts[model][family] = sum(r["mean"] != "" for r in rr)
    (OUT / "README.md").write_text(
        "# 5TB L / H+ v4 six-family trajectories\n\n"
        "Fixed protocol_v4 task lists: classification 25, regression 4, retrieval 7, "
        "clustering 7, segmentation 7, detection proxy 3. Each family mean uses all "
        "required datasets, with PanNuke's three rotations averaged first. Missing "
        "cells leave a gap. Segmentation uses E20 and three seeds.\n\n"
        "L ck487–29279 is from the local v4 cell ledger; ck29767–41479 and H+ "
        "step0–34159 are extracted from the raw files referenced by HXW inventories. "
        "Classification uses balanced accuracy (chestmnist macro AUC), not accuracy. "
        "Retrieval uses Recall@1, clustering NMI, regression R², segmentation mDice, "
        "detection patch F1. No six-family overall average is calculated.\n\n"
        "LC25000 source-disjoint admission and NCT100 admission are still provisional. "
        "Thus these are v4-list observational curves, not a claim of fully admitted "
        "formal v4 scores. CTC and OOD are outside this six-family figure.\n\n"
        f"Complete checkpoints by family: {json.dumps(counts, ensure_ascii=False)}\n\n"
        "`cells.csv` contains every plotted input and its source; "
        "`curves_and_coverage.csv` contains complete-only means and missing lists.\n"
    )
    print(json.dumps({"remote_cells": len(raw), "total_cells": len(cells), "complete": counts}, indent=2))


if __name__ == "__main__":
    main()
