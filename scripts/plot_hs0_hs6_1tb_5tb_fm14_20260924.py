#!/usr/bin/env python3
"""Replot validated HS0/HS6/5TB trajectories and compare fixed checkpoints with FM14."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
OUT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
ALL_FAMILIES = FAMILIES + ("detection_proxy",)
HS0 = [f"hs0_{s}" for s in ("splus", "b", "l", "hplus")]
HS6 = [f"hs6_{s}" for s in ("splus", "b", "l", "hplus")]
TB5 = ["5tb_no_gram", "5tb_gram12687"]
MODELS = HS0 + HS6 + TB5 + [f"fm_{s}" for s in (
    "bioclip", "conch", "cytoimagenet", "cytoself", "dinov2", "gigapath",
    "hoptimus0", "jump_cp", "mae", "pe", "phikon2", "siglip2", "uni", "virchow2")]
FM = [m for m in MODELS if m.startswith("fm_")]
COLORS = {"hs6_splus": "#53698c", "hs6_b": "#5189bb", "hs6_l": "#194d90",
          "hs6_hplus": "#152846", "5tb_no_gram": "#009b77", "5tb_gram12687": "#bb5852",
          "hs0_splus": "#a0a0a0", "hs0_b": "#8e74ae", "hs0_l": "#694994", "hs0_hplus": "#402763"}


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def source_priority(source):
    source = str(source)
    if "/benchmark_runs/fm_v3_completion/" in source:
        return 4
    if "/benchmark_runs/retest_20260918/" in source:
        return 3
    if "/remote_mirrors/" in source:
        return 2
    return 1


def retrieval_values(obj, dataset):
    rows = obj.get("rows", [obj])
    if dataset == "hpa-subcellular":
        rec = next((x for x in rows if x.get("aggregation") == "global"), None)
        clu = next((x for x in rows if x.get("aggregation") == "location" and x.get("n_classes") == 41), None)
    elif dataset == "rxrx1-cross":
        rec = next((x for x in rows if x.get("aggregation") == "global"), None)
        clu = next((x for x in rows if x.get("aggregation") == "global-perturbation"), None)
    else:
        rec = next((x for x in rows if x.get("aggregation") in ("class", "global") and "recall_at_1" in x), None)
        clu = next((x for x in rows if x.get("aggregation") in ("class", "global") and "nmi" in x), None)
    return {"retrieval": number(rec.get("recall_at_1")) if rec else None,
            "clustering": number(clu.get("nmi")) if clu else None}


def historical_detection_means():
    points = defaultdict(dict)
    for row in csv.DictReader(INDEX.open()):
        if row["model"] not in HS6 + TB5 or row["protocol"] != "old_observation" or row["family"] != "detection":
            continue
        if not row["source"].endswith("results_bio_detection.json"):
            continue
        obj = json.loads(Path(row["source"]).read_text())
        value = number(obj.get("test_patch_f1"))
        if value is not None and obj.get("image_size") == 224 and obj.get("epochs") == 5:
            points[row["model"], row["checkpoint"], obj["batch_size"]][row["dataset"]] = value / 100
    return [(m, int(ck), batch, sum(ds.values()) / 3)
            for (m, ck, batch), ds in points.items() if set(ds) == {"bbbc038", "conic", "livecell"}]


def collect():
    cells = {}
    seg = defaultdict(dict)
    conflicts = []
    missing = []
    for row in csv.DictReader(INDEX.open()):
        model, ck, family, dataset = (row[k] for k in ("model", "checkpoint", "family", "dataset"))
        if model not in MODELS:
            continue
        protocol, status = row["protocol"], row["evidence_status"]
        valid = protocol == "v3" and status == "VALID_COMPLETE"
        extension = protocol == "old_union_extension" and status == "VALIDATED_LEGACY_COMPONENT"
        observation = protocol == "old_union_extension" and status == "OBSERVATIONAL" and family == "detection"
        if not (valid or extension or observation):
            continue
        source = Path(row["source"])
        if family == "segmentation":
            if not valid or source.name != "results.json" or "__primary-last__" not in row["campaign_key"]:
                continue
            if not ("/budget20/" in str(source) or "/E20/" in str(source)):
                continue
        elif source.name != "component_result.json":
            continue
        try:
            obj = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(str(source))
            continue
        if family == "segmentation":
            meta = obj.get("_meta", {})
            seed = meta.get("seed")
            value = number(obj.get("test", {}).get("mDice"))
            if meta.get("probe_epochs") != 20 or seed not in (0, 1, 2) or value is None:
                continue
            key = (model, ck, dataset, row["split"] if dataset == "pannuke" else "", seed)
            if key in seg and abs(seg[key][0] - value) > 1e-7:
                conflicts.append(dict(key=str(key), retained_source=seg[key][1], alternate_source=str(source),
                                      retained_value=seg[key][0], alternate_value=value))
                if source_priority(source) > source_priority(seg[key][1]):
                    seg[key] = (value, str(source))
            else:
                seg[key] = (value, str(source))
            continue
        values = {}
        if family == "classification":
            metric = "macro_auc" if dataset == "chestmnist" else "balanced_accuracy"
            values[family] = (number(obj.get(metric)), metric)
        elif family == "regression":
            if dataset == "bbbc013":
                a, b = number(obj.get("ly294002_r2")), number(obj.get("wortmannin_r2"))
                values[family] = ((a + b) / 2 if a is not None and b is not None else None, "compound_mean_r2")
            else:
                values[family] = (number(obj.get("r2")), "r2")
        elif family == "retrieval":
            values = {name: (value, "recall_at_1" if name == "retrieval" else "nmi")
                      for name, value in retrieval_values(obj, dataset).items()}
        elif family == "detection":
            if obj.get("batch_size") != 8 or obj.get("image_size") != 224:
                continue
            value = number(obj.get("test_patch_f1"))
            values["detection_proxy"] = (value / 100 if value is not None and value > 1 else value, "test_patch_f1")
        for name, (value, metric) in values.items():
            if value is None:
                continue
            key = (model, ck, name, dataset)
            candidate = dict(model=model, checkpoint=ck, family=name, dataset=dataset,
                             metric=metric, value=value, protocol=protocol, source=str(source))
            if key in cells:
                prev = cells[key]
                if prev["protocol"] == "v3" and protocol != "v3":
                    continue
                if protocol == "v3" and prev["protocol"] != "v3":
                    cells[key] = candidate
                elif abs(prev["value"] - value) > 1e-7:
                    conflicts.append(dict(key=str(key), retained_source=prev["source"], alternate_source=str(source),
                                          retained_value=prev["value"], alternate_value=value))
                    if source_priority(source) > source_priority(prev["source"]):
                        cells[key] = candidate
            else:
                cells[key] = candidate
    grouped = defaultdict(dict)
    for (model, ck, dataset, split, seed), val in seg.items():
        grouped[model, ck, dataset, split][seed] = val
    dense = defaultdict(dict)
    for (model, ck, dataset, split), seeds in grouped.items():
        if set(seeds) != {0, 1, 2}:
            continue
        dense[model, ck, dataset][split] = (sum(seeds[s][0] for s in seeds) / 3,
                                            ";".join(seeds[s][1] for s in sorted(seeds)))
    for (model, ck, dataset), folds in dense.items():
        if dataset == "pannuke" and len(folds) != 3:
            continue
        value = sum(v[0] for v in folds.values()) / len(folds)
        source = ";".join(v[1] for v in folds.values())
        key = (model, ck, "segmentation", dataset)
        cells[key] = dict(model=model, checkpoint=ck, family="segmentation", dataset=dataset,
                          metric="mDice_E20_last1", value=value, protocol="v3", source=source)
    write_csv(OUT / "duplicate_discrepancies.csv", conflicts,
              ["key", "retained_source", "alternate_source", "retained_value", "alternate_value"])
    return cells, missing, conflicts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, missing, conflicts = collect()
    score_rows = sorted(cells.values(), key=lambda x: (x["model"], x["checkpoint"], x["family"], x["dataset"]))
    write_csv(OUT / "validated_scores.csv", score_rows,
              ["model", "checkpoint", "family", "dataset", "metric", "value", "protocol", "source"])
    per_model = defaultdict(set)
    checkpoints = defaultdict(set)
    for model, ck, family, dataset in cells:
        checkpoints[model].add(ck)
        if family in FAMILIES and cells[model, ck, family, dataset]["protocol"] == "v3":
            per_model[model].add((family, dataset))
    core = set.intersection(*(per_model[m] for m in MODELS))
    if len(core) < 30 or any(not any(f == fam for f, _ in core) for fam in FAMILIES):
        raise RuntimeError(f"Too few common verified cells: {len(core)}")
    core = sorted(core)
    write_csv(OUT / "common_cells.csv", [dict(family=f, dataset=d) for f, d in core], ["family", "dataset"])
    rows = []
    for model in MODELS:
        for ck in sorted(checkpoints[model], key=lambda s: int(s) if s.isdigit() else -1):
            means = {}
            counts = {}
            for family in ALL_FAMILIES:
                datasets = [ds for f, ds in core if f == family] if family in FAMILIES else ["bbbc038", "conic", "livecell"]
                vals = [cells[model, ck, family, ds]["value"] for ds in datasets if (model, ck, family, ds) in cells]
                counts[family] = len(vals)
                means[family] = sum(vals) / len(vals) if vals and len(vals) == len(datasets) else None
            complete = all(means[f] is not None for f in FAMILIES)
            row = dict(model=model, checkpoint=ck, core_complete=int(complete),
                       core_count=sum(counts[f] for f in FAMILIES), core_expected=len(core),
                       selection_score=sum(means[f] for f in FAMILIES) / len(FAMILIES) if complete else None)
            for fam in ALL_FAMILIES:
                row[f"{fam}_mean"] = means[fam]
                row[f"{fam}_n"] = counts[fam]
            rows.append(row)
    fields = list(rows[0])
    write_csv(OUT / "checkpoint_task_means.csv", rows, fields)
    selected = {}
    for model in MODELS:
        options = [r for r in rows if r["model"] == model and r["core_complete"]]
        if not options:
            raise RuntimeError(f"No complete comparable checkpoint for {model}")
        selected[model] = max(options, key=lambda r: (r["selection_score"], -int(r["checkpoint"]) if r["checkpoint"].isdigit() else 0))
    winners = {"HS0 1TB": max((selected[m] for m in HS0), key=lambda r: r["selection_score"]),
               "HS6 1TB": max((selected[m] for m in HS6), key=lambda r: r["selection_score"]),
               "HS6-L 5TB": max((selected[m] for m in TB5), key=lambda r: r["selection_score"])}
    write_csv(OUT / "selected_checkpoints.csv", [dict(group=g, **r) for g, r in winners.items()], ["group"] + fields)
    write_csv(OUT / "all_model_best_checkpoints.csv", list(selected.values()), fields)

    # Six panels use the same family metrics and coverage; hollow points mark partial detection.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    old_detection = historical_detection_means()
    for ax, family in zip(axes.flat, ALL_FAMILIES):
        for model in HS6 + TB5:
            rr = [r for r in rows if r["model"] == model and r[f"{family}_n"]]
            rr.sort(key=lambda r: int(r["checkpoint"]))
            xx = [int(r["checkpoint"]) / 1000 for r in rr]
            yy = [r[f"{family}_mean"] if r[f"{family}_mean"] is not None else np.nan for r in rr]
            ax.plot(xx, yy, label=model.replace("_", " "), color=COLORS[model], linewidth=1.5)
            if family == "detection_proxy":
                partial = [r for r in rr if r[f"{family}_mean"] is None]
                ax.scatter([int(r["checkpoint"]) / 1000 for r in partial],
                           [np.mean([cells[model, r["checkpoint"], family, ds]["value"]
                                     for ds in ("bbbc038", "conic", "livecell")
                                     if (model, r["checkpoint"], family, ds) in cells]) for r in partial],
                           facecolors="white", edgecolors=COLORS[model], s=22)
        for model in HS0:
            r = selected[model]
            if r[f"{family}_mean"] is not None:
                ax.scatter([8.199], [r[f"{family}_mean"]], label=model.replace("_", " "),
                           marker="D", s=45, color=COLORS[model], zorder=5)
        if family == "detection_proxy":
            for model in HS6 + TB5:
                pts = sorted(((ck, batch, value) for m, ck, batch, value in old_detection if m == model))
                if pts:
                    ax.plot([ck / 1000 for ck, _, _ in pts], [value for _, _, value in pts],
                            "x--", color=COLORS[model], linewidth=.8, markersize=4, alpha=.65)
            ax.text(.02, .03, "× historical B2/B4/B8; ○ matched B8", transform=ax.transAxes,
                    fontsize=8, va="bottom")
        ax.set(title=f"{family.replace('_', ' ').title()} | common {sum(f == family for f, _ in core) if family in FAMILIES else 3}",
               xlabel="Training updates (k)", ylabel={"classification": "BA / AUC", "regression": "R²",
                   "retrieval": "Recall@1", "clustering": "NMI", "segmentation": "mDice",
                   "detection_proxy": "patch F1"}[family])
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.suptitle("HS0 / HS6 1TB and HS6-L 5TB | v3 shared tasks; detection protocol labelled", fontsize=16)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"training_curves.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Side-by-side family means for FM14 and the three chosen fixed checkpoints.
    compare = [selected[m] for m in FM] + list(winners.values())
    compare.sort(key=lambda r: -r["selection_score"])
    write_csv(OUT / "fm14_family_comparison.csv", compare, fields)
    labels = [r["model"].replace("fm_", "FM ").replace("_", " ") + (" *" if r["model"] in {x["model"] for x in winners.values()} else "") for r in compare]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, family in zip(axes.flat, ALL_FAMILIES):
        values = [r[f"{family}_mean"] for r in compare]
        bars = ax.barh(range(len(compare)), [v if v is not None else 0 for v in values],
                       color=["#2274a5" if r["model"] in FM else "#db7346" for r in compare])
        for i, v in enumerate(values):
            if v is None:
                bars[i].set_alpha(.15)
                ax.text(.01, i, "missing matched cells", va="center", fontsize=7)
        ax.set_yticks(range(len(compare)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(family.replace("_", " ").title())
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("FM14 vs fixed selected HS0 / HS6 checkpoints | 5 shared tasks; detection where complete", fontsize=16)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fm14_family_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    details = []
    for family, dataset in core:
        fm_scores = [(m, cells[m, selected[m]["checkpoint"], family, dataset]["value"]) for m in FM]
        best_fm, best_fm_value = max(fm_scores, key=lambda x: x[1])
        for group, choice in winners.items():
            model, ck = choice["model"], choice["checkpoint"]
            item = cells[model, ck, family, dataset]
            details.append(dict(group=group, model=model, checkpoint=ck, family=family, dataset=dataset,
                                metric=item["metric"], value=item["value"], fm14_best_model=best_fm,
                                fm14_best_value=best_fm_value, delta_vs_fm14_best=item["value"] - best_fm_value,
                                source=item["source"]))
    write_csv(OUT / "fm14_per_dataset_comparison.csv", details, list(details[0]))
    families = list(FAMILIES)
    fig, axes = plt.subplots(1, 5, figsize=(22, 12), constrained_layout=True,
                             gridspec_kw={"width_ratios": [sum(f == fam for f, _ in core) for fam in families]})
    groupnames = list(winners)
    for ax, family in zip(axes, families):
        ds = [d for f, d in core if f == family]
        data = np.array([[next(r["delta_vs_fm14_best"] for r in details if r["group"] == group and r["family"] == family and r["dataset"] == d) * 100
                          for d in ds] for group in groupnames])
        ax.imshow(data, cmap="RdBu", vmin=-12, vmax=12, aspect="auto")
        ax.set_xticks(range(len(ds)), ds, rotation=90, fontsize=7)
        ax.set_yticks(range(3), groupnames, fontsize=8)
        ax.set_title(family.title())
        for i in range(3):
            for j in range(len(ds)):
                ax.text(j, i, f"{data[i,j]:+.1f}", ha="center", va="center", fontsize=5.5)
    fig.suptitle("Difference from best FM14 on each dataset (percentage points)", fontsize=15)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fm14_dataset_deltas.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = {"common_cells": len(core), "common_by_family": {f: sum(x == f for x, _ in core) for f in FAMILIES},
               "winners": {g: {"model": r["model"], "checkpoint": r["checkpoint"], "score": r["selection_score"]} for g, r in winners.items()},
               "five_tb_candidates": {m: {"checkpoint": selected[m]["checkpoint"],
                                            "score": selected[m]["selection_score"]} for m in TB5},
               "best_fm14": {"model": max((selected[m] for m in FM), key=lambda r: r["selection_score"])["model"],
                             "score": max(selected[m]["selection_score"] for m in FM)},
               "datasets_above_best_fm14": {group: sum(r["group"] == group and r["delta_vs_fm14_best"] > 0
                                                      for r in details) for group in winners},
               "fm14_count": len(FM), "source_read_errors": len(missing), "duplicate_discrepancies": len(conflicts),
               "note": "Retrospective selection on test scores; descriptive, not an unbiased model-selection result. Detection proxy is separate and excluded from selection."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    readme = (
        "# HS0 / HS6 1TB、HS6-L 5TB 与 FM14 对比\n\n"
        "**口径提示：本报告是 v3 共享40项，不是 2026-09-14 的旧协议 raw macro 六任务图。**"
        "旧图包含分类25、回归4、检索4、聚类4、分割8、检测3，"
        "5TB 旧图的独立复刻见 `../hs6_l5_5tb_old_macro_reproduction_20260924/`。"
        "两者不能按同一曲线直接比较。\n\n"
        "## 口径\n\n"
        "从 24 个模型均有验证结果的 40 个 v3 共享数据集/任务单元取交集：分类24、回归2、"
        "检索4、聚类4、分割6。各任务内部数据集等权，五个任务再等权；每个训练臂选一份固定 "
        "checkpoint，再从 HS0、HS6 1TB、HS6-L 5TB 各组选择综合分最高的模型。"
        "这些选择使用了现有 test 分数，属于回顾性诊断，有选点偏差；不能当成独立验证的正式优胜结论。\n\n"
        "分割统一 native final / last1、E20、三个 seed 均值，PanNuke 先合并三个 rotation。"
        "Detection proxy 统一 B8，单独画出，不参与选点；空心点表示不足三项的部分覆盖。"
        "HS0 当前可比证据仅有 ck8199，因此图中是点而非编造的轨迹。FM14 无训练步数轴，"
        "只放在横向对比图。LC25000 分类、RxRx3、MoNuSeg、CTC、OOD 等未进入 40 项交集。\n\n"
        "## 固定 checkpoint 选择\n\n"
        "| 组 | 模型 | checkpoint | 五任务等权均值 | 高于逐数据集最优 FM 的项数 |\n"
        "|---|---|---:|---:|---:|\n"
    )
    for group, item in winners.items():
        readme += (f"| {group} | {item['model']} | {item['checkpoint']} | "
                   f"{item['selection_score']:.6f} | {summary['datasets_above_best_fm14'][group]}/40 |\n")
    readme += (
        f"\n5TB GRAM 最优固定点：ck{selected['5tb_gram12687']['checkpoint']}，"
        f"综合分 {selected['5tb_gram12687']['selection_score']:.6f}；本口径低于 5TB no-GRAM。"
        f"FM14 中最高的是 {summary['best_fm14']['model']}，综合分 {summary['best_fm14']['score']:.6f}。\n\n"
        "## 数据与图\n\n"
        "- `training_curves.png/svg`：HS0、HS6 1TB、5TB no-GRAM/GRAM 六任务曲线。\n"
        "- `fm14_family_comparison.png/svg`：14 FM 与三组选定模型的任务均值。\n"
        "- `fm14_dataset_deltas.png/svg`：三组选定模型相对各数据集最优 FM 的差值，单位百分点。\n"
        "- `validated_scores.csv`：逐来源分数；`checkpoint_task_means.csv`：各 checkpoint 任务均值；"
        "`selected_checkpoints.csv`：三组选点；`fm14_per_dataset_comparison.csv`：逐数据集对比；"
        "`common_cells.csv`：共同口径；`duplicate_discrepancies.csv`：重复重测差异。\n\n"
        f"本次发现 {len(conflicts)} 条重复重测的轻微数值差异，最大绝对差 "
        f"{max(abs(x['retained_value'] - x['alternate_value']) for x in conflicts):.6f}。"
        "优先使用 `fm_v3_completion`，其次 `retest_20260918`，再其次归档副本；"
        "所有被选中数值保留原始路径。\n"
    )
    (OUT / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
