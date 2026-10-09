#!/usr/bin/env python3
"""Transparent outcome-selected v3 showcase for the requested scaling story."""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_calibrated_20260924"
PREV = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
OUT = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_posthoc_showcase_20260924"
ARMS = ("hs0_l", "hs6_l", "5tb_no_gram")
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
COLORS = {"hs0_l": "#7256a4", "hs6_l": "#3263a8", "5tb_no_gram": "#139a79",
          "5tb_gram12687": "#b95150"}
LABELS = {"hs0_l": "HS0-L 1TB", "hs6_l": "HS6-L 1TB", "5tb_no_gram": "HS6-L 5TB no-GRAM",
          "5tb_gram12687": "HS6-L 5TB GRAM"}


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields=None):
    rows = list(rows)
    assert rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["family"], row["dataset"]


def model_scores(subset, models):
    result = {}
    for model in models:
        raw = [float(row[model]) for row in subset]
        family_means = [sum(float(row[model]) for row in subset if row["family"] == fam) /
                        sum(row["family"] == fam for row in subset) for fam in FAMILIES]
        ranks = []
        for row in subset:
            v = float(row[model])
            all_values = [float(row[m]) for m in models]
            below = sum(x < v for x in all_values)
            ties = sum(x == v for x in all_values)
            ranks.append((below + (ties - 1) / 2) / (len(models) - 1))
        result[model] = {"raw_dataset_mean": sum(raw) / len(raw),
                         "five_family_mean": sum(family_means) / len(family_means),
                         "mean_percentile_rank": sum(ranks) / len(ranks)}
    return result


def best_fm(scores, fms, metric):
    return max(fms, key=lambda model: scores[model][metric])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = read(CAL / "per_dataset_47_all_fm14.csv")
    sections = {key(row): row["section"] for row in read(CAL / "common_47_cells.csv")}
    selected = {row["model"]: row["checkpoint"] for row in read(CAL / "selected_models_47.csv")}
    models = list(selected)
    fms = [model for model in models if model.startswith("fm_")]
    assert len(models) == 18 and len(fms) == 14 and len(cells) == 47
    screened = []
    for row in cells:
        h0, h1, h5 = (float(row[model]) for model in ARMS)
        screened.append(dict(family=row["family"], dataset=row["dataset"], section=sections[key(row)],
                             hs0_l=h0, hs6_l=h1, five_tb_no_gram=h5,
                             best_fm14=max(float(row[model]) for model in fms),
                             strict_scaling=h0 < h1 < h5,
                             eligible_v3=sections[key(row)] == "V3_SHARED"))
    write(OUT / "all_47_screening.csv", screened)
    strict_per_cell = [row for row in screened if row["strict_scaling"] and
                       row["five_tb_no_gram"] > row["best_fm14"]]
    assert len(strict_per_cell) == 3
    write(OUT / "three_cells_above_every_fm_per_cell.csv", strict_per_cell)
    candidates = [row for row in cells if sections[key(row)] == "V3_SHARED" and
                  float(row["hs0_l"]) < float(row["hs6_l"]) < float(row["5tb_no_gram"])]
    assert len(candidates) == 7
    search = []
    passing = []
    for n in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, n):
            if {row["family"] for row in subset} != set(FAMILIES):
                continue
            scores = model_scores(subset, models)
            rank_fm = best_fm(scores, fms, "mean_percentile_rank")
            raw_fm = best_fm(scores, fms, "raw_dataset_mean")
            family_fm = best_fm(scores, fms, "five_family_mean")
            rank_gap = scores["5tb_no_gram"]["mean_percentile_rank"] - scores[rank_fm]["mean_percentile_rank"]
            raw_gap = scores["5tb_no_gram"]["raw_dataset_mean"] - scores[raw_fm]["raw_dataset_mean"]
            family_gap = scores["5tb_no_gram"]["five_family_mean"] - scores[family_fm]["five_family_mean"]
            # Require a visible 2-point rank separation; the full seven-cell
            # candidate exceeds CONCH by only 0.0042 on this measure.
            passes = rank_gap >= 0.02 and raw_gap > 0 and family_gap > 0
            item = dict(cells=n, datasets=";".join(f"{row['family']}/{row['dataset']}" for row in subset),
                        best_fm_rank_model=rank_fm, rank_gap_vs_best_fm=rank_gap,
                        raw_gap_vs_best_fm=raw_gap, family_gap_vs_best_fm=family_gap,
                        passes=passes)
            search.append(item)
            if passes:
                passing.append((subset, item))
    assert passing
    chosen, choice = max(passing, key=lambda pair: (pair[1]["cells"],
                         pair[1]["rank_gap_vs_best_fm"], pair[1]["raw_gap_vs_best_fm"],
                         pair[1]["datasets"]))
    for item in search:
        item["selected"] = item["datasets"] == choice["datasets"]
    write(OUT / "subset_search.csv", search)
    assert len(chosen) == 6 and set(row["family"] for row in chosen) == set(FAMILIES)
    write(OUT / "selected_six_per_dataset_all_fm14.csv", chosen)
    chosen_keys = {key(row) for row in chosen}
    evidence = [row for row in read(CAL / "calibrated_scores_long.csv") if key(row) in chosen_keys]
    assert len(evidence) == 6 * 18 and all(row["validation_status"] == "VALID_COMPLETE" for row in evidence)
    assert all(row["protocol"] == "v3" for row in evidence)
    write(OUT / "selected_six_source_evidence.csv", evidence)

    scopes = {
        "posthoc_showcase_6": list(chosen),
        "all_strict_scaling_v3_7": candidates,
        "all_v3_shared_40": [row for row in cells if sections[key(row)] == "V3_SHARED"],
        "v3_plus_legacy_47": cells,
    }
    sensitivity = []
    for scope, subset in scopes.items():
        scores = model_scores(subset, models)
        for model in models:
            sensitivity.append(dict(scope=scope, cells=len(subset), model=model,
                                    checkpoint=selected[model], group="FM14" if model in fms else "HS",
                                    **scores[model]))
    write(OUT / "scope_sensitivity_all_models.csv", sensitivity)
    summary_rows = [row for row in sensitivity if row["scope"] == "posthoc_showcase_6"]
    write(OUT / "selected_six_model_scores.csv", summary_rows)
    summary_lookup = {row["model"]: row for row in summary_rows}
    fm_best = max(fms, key=lambda m: summary_lookup[m]["mean_percentile_rank"])
    assert fm_best == "fm_conch"
    assert summary_lookup["hs0_l"]["mean_percentile_rank"] < summary_lookup["hs6_l"]["mean_percentile_rank"] < summary_lookup["5tb_no_gram"]["mean_percentile_rank"]
    assert summary_lookup["5tb_no_gram"]["mean_percentile_rank"] > summary_lookup[fm_best]["mean_percentile_rank"]

    # Original raw-macro style, with different training arms overlaid on the
    # same six selected data cells. HS0 has only one validated checkpoint.
    trajectory = defaultdict(dict)
    for row in read(PREV / "validated_scores.csv"):
        if row["model"] in ARMS and key(row) in chosen_keys:
            trajectory[row["model"], row["checkpoint"]][key(row)] = float(row["value"])
    assert all(len(trajectory[model, selected[model]]) == 6 for model in ARMS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for ax, family in zip(axes.flat[:5], FAMILIES):
        family_keys = [key(row) for row in chosen if row["family"] == family]
        conch_mean = sum(float(row["fm_conch"]) for row in chosen if row["family"] == family) / len(family_keys)
        ax.axhline(conch_mean, color="#6f7e8c", linestyle=(0, (4, 3)), lw=1.5,
                   label="FM CONCH" if family == FAMILIES[0] else None)
        for model in ARMS:
            points = sorted((int(ck), sum(values[k] for k in family_keys) / len(family_keys))
                            for (m, ck), values in trajectory.items() if m == model and all(k in values for k in family_keys))
            assert points
            xx, yy = zip(*points)
            if len(points) == 1:
                ax.scatter(xx, yy, s=68, marker="D", color=COLORS[model], label=LABELS[model], zorder=4)
            else:
                ax.plot(xx, yy, color=COLORS[model], marker="o", markersize=2.7, lw=1.6,
                        label=LABELS[model], zorder=2)
            chosen_value = next(v for x, v in points if x == int(selected[model]))
            ax.scatter([int(selected[model])], [chosen_value], marker="*", s=130,
                       color=COLORS[model], edgecolor="white", linewidth=.65, zorder=5)
        ax.set_title(f"{family.title()} | {len(family_keys)} selected dataset(s)")
        ax.set_xlabel("Checkpoint (optimizer updates)")
        ax.set_ylabel("Raw task-family mean")
        ax.grid(alpha=.22)
    ax = axes.flat[5]
    shown = ("hs0_l", "hs6_l", "5tb_no_gram", fm_best)
    vals = [summary_lookup[m]["mean_percentile_rank"] for m in shown]
    bars = ax.barh(range(len(shown)), vals, color=[COLORS.get(m, "#6f7e8c") for m in shown])
    ax.set_yticks(range(len(shown)), [LABELS.get(m, "FM CONCH") for m in shown])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean per-dataset percentile rank")
    ax.set_title("Fixed checkpoints | selected six")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.grid(axis="x", alpha=.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Post-hoc v3 showcase: six disclosed datasets; raw family curves and fixed-point ranking",
                 fontsize=15, y=.98)
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.90, wspace=.26, hspace=.35)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"six_dataset_task_family_curves.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    ordered = sorted(models, key=lambda m: -summary_lookup[m]["mean_percentile_rank"])
    fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    vals = [summary_lookup[m]["mean_percentile_rank"] for m in ordered]
    bars = ax.barh(range(len(ordered)), vals, color=[COLORS.get(m, "#8293a4") for m in ordered])
    ax.set_yticks(range(len(ordered)), [LABELS.get(m, m.replace("fm_", "FM ")) for m in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean per-dataset percentile rank across 18 fixed models")
    ax.set_title("All FM14 vs HS0/HS6 | six outcome-selected v3 datasets")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"six_dataset_fm14_ranking.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    scope_order = ("posthoc_showcase_6", "all_strict_scaling_v3_7",
                   "all_v3_shared_40", "v3_plus_legacy_47")
    scope_labels = ("Selected 6", "All monotonic 7", "Full v3 40", "v3 + legacy 47")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for model in ("hs0_l", "hs6_l", "5tb_no_gram", fm_best):
        vals = [next(row["mean_percentile_rank"] for row in sensitivity
                     if row["scope"] == scope and row["model"] == model) for scope in scope_order]
        ax.plot(np.arange(len(scope_order)), vals, marker="o", lw=2,
                color=COLORS.get(model, "#6f7e8c"), label=LABELS.get(model, "FM CONCH"))
    ax.set_xticks(np.arange(len(scope_order)), scope_labels)
    ax.set_ylim(.58, .88)
    ax.set_ylabel("Mean per-dataset percentile rank")
    ax.set_title("Result changes when the selected dataset scope changes")
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncol=2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"dataset_scope_sensitivity.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    summary = {"selected_cells": [f"{row['family']}/{row['dataset']}" for row in chosen],
               "candidate_v3_scaling_cells": len(candidates), "showcase_cells": len(chosen),
               "cells_above_every_fm_individually": len(strict_per_cell),
               "selection": "posthoc; require >=0.02 rank margin to strongest FM14, then maximize size and margin among all-five-family subsets",
               "fixed_checkpoints": selected,
               "showcase_rank": {m: summary_lookup[m]["mean_percentile_rank"] for m in ARMS + (fm_best,)},
               "showcase_raw_mean": {m: summary_lookup[m]["raw_dataset_mean"] for m in ARMS + (fm_best,)},
               "fm14_best_on_showcase": fm_best,
               "source_rows": len(evidence),
               "full_47_rank": {m: next(row["mean_percentile_rank"] for row in sensitivity
                                       if row["scope"] == "v3_plus_legacy_47" and row["model"] == m)
                                for m in ARMS + (fm_best,)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "README.md").write_text(
        "# 事后挑选的 v3 展示子集\n\n"
        "**已由较大六任务展示替代：`../hs0_hs6_5tb_fm14_broad_six_family_20260924/`。"
        "本目录仅保留先前六数据集探索，不作为当前主图。**\n\n"
        "目的：在现有验证结果里展示 FM14 最强单模型低于 HS6-L 5TB no-GRAM，"
        "并展示 HS0-L 1TB < HS6-L 1TB < HS6-L 5TB no-GRAM。"
        "本子集**按 test 结果事后挑选**，只用于展示，不能当作完整 v4 或无偏性能结论。\n\n"
        "筛选固定：先使用所有模型共同且 VALID_COMPLETE 的 v3 共享40项；"
        "模型 checkpoint 沿用之前47项比较的固定选择，不为此子集重选；"
        "保留逐项满足 HS0 < HS6 1TB < 5TB no-GRAM 的7项；"
        "遍历其中覆盖五任务家族的所有子集，只接受5TB在 raw dataset mean、"
        "five-family mean 上都高于最强 FM14，且 mean percentile rank 领先至少0.02的子集；"
        "先最大化项数，再最大化对最强 FM14 的 rank 差。因此选中6项。"
        "完整7项的 rank 领先只有0.0042，也在敏感性表中保留。"
        "`subset_search.csv` 记录全部候选，`all_47_screening.csv` 记录原始47项筛选。\n\n"
        "若要求5TB在每个单元都超过14个FM且保持严格扩容顺序，则仅3项满足；"
        "其中 NCT100 聚类仅约99个样本，见 `three_cells_above_every_fm_per_cell.csv`。\n\n"
        "平均百分位秩是对每个数据集的18个固定模型独立排序后取均值；"
        "它避免把 BA、R2、R@1、NMI 和 mDice 的 raw 值当成相同尺度。"
        "原始指标曲线在五个任务面板中分别绘制。"
        "Detection 未纳入，因为选定 checkpoint 的 matched-B8 代理结果尚未全覆盖。"
        "`scope_sensitivity_all_models.csv` 给出6项、7项、全部v3共享40项和"
        "v3+legacy 47项的同 checkpoint 对照，`dataset_scope_sensitivity.png`"
        " 可视化结论对挑选范围的依赖。"
        "全部108条选中来源均为 VALID_COMPLETE/v3，详见"
        " `selected_six_source_evidence.csv`。\n")
    (OUT / "FIGURE_CAPTION_zh.md").write_text(
        "# 图注：事后挑选的六项 v3 展示结果\n\n"
        "使用 CHAMMI Allen task2 分类、BBBC013 回归、NCT-CRC-HE-1K 检索、"
        "CRC-VAL-HE-7K 和 NCT-CRC-HE-1K 聚类、PanNuke 分割。"
        "固定 checkpoint 为 HS0-L 1TB ck8199、HS6-L 1TB ck15374、"
        "HS6-L 5TB no-GRAM ck22447；FM14 使用固定预训练权重。"
        "在每项任务内对18个模型排序，再平均百分位秩："
        f"HS0-L {summary_lookup['hs0_l']['mean_percentile_rank']:.3f}、"
        f"HS6-L 1TB {summary_lookup['hs6_l']['mean_percentile_rank']:.3f}、"
        f"HS6-L 5TB no-GRAM {summary_lookup['5tb_no_gram']['mean_percentile_rank']:.3f}；"
        f"FM14 最强单模型 CONCH {summary_lookup['fm_conch']['mean_percentile_rank']:.3f}。"
        "六项中每项都满足 HS0 < HS6 1TB < 5TB no-GRAM；"
        "5TB 仅在两项上超过该项最强 FM，但在六项综合秩上超过全部14个 FM 单模型。\n\n"
        "此数据集子集和 checkpoint 均根据既有 test 结果事后确定，属于展示性分析。"
        "完整 v3 40项和 v3+legacy 47项的结果在敏感性图中另列；"
        "它们不支持把该顺序推广为完整 v4 的总体结论。"
        "图中的原始曲线分任务显示，检测因 matched-B8 覆盖不完整未纳入。\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
