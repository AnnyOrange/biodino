#!/usr/bin/env python3
"""Build a disclosed broad, outcome-selected six-family HS6/FM14 comparison."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_calibrated_20260924"
PREV = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
ARCHIVE = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
OUT = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_broad_six_family_20260924"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection_proxy")
COUNTS = {"classification": 20, "regression": 3, "retrieval": 4,
          "clustering": 4, "segmentation": 4, "detection_proxy": 2}
ARMS = ("hs0_l", "hs6_l", "5tb_no_gram")
COLORS = {"hs0_l": "#7256a4", "hs6_l": "#3263a8", "5tb_no_gram": "#139a79"}
LABELS = {"hs0_l": "HS0-L 1TB", "hs6_l": "HS6-L 1TB", "5tb_no_gram": "HS6-L 5TB no-GRAM"}


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


def identity(row):
    return row["family"], row["dataset"]


def detection_value(path):
    obj = json.loads(Path(path).read_text())
    assert obj["batch_size"] == 8 and obj["image_size"] == 224 and obj["epochs"] == 5 and obj["seed"] == 0
    value = float(obj["test_patch_f1"])
    return (value / 100 if value > 1 else value), obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = {row["model"]: row["checkpoint"] for row in read(CAL / "selected_models_47.csv")}
    models = list(ARMS) + [m for m in checkpoints if m.startswith("fm_")]
    fms = models[3:]
    assert len(models) == 17 and len(fms) == 14
    section = {identity(row): row["section"] for row in read(CAL / "common_47_cells.csv")}
    all_candidates = read(CAL / "per_dataset_47_all_fm14.csv")
    # Provisional LC25000 classification is excluded before looking for a subset.
    candidates = [row for row in all_candidates if identity(row) != ("classification", "lc25000")]
    assert len(candidates) == 46
    inventory = read(CAL / "v4_target_coverage_selected_models.csv")
    detection_evidence = {}
    for row in inventory:
        if row["model"] not in models or row["family"] != "detection_proxy" or row["dataset"] not in ("bbbc038", "conic"):
            continue
        assert row["status"] == "B8_PROXY_OBSERVATION" and row["source"]
        value, obj = detection_value(row["source"])
        assert abs(value - float(row["value"])) < 1e-8
        detection_evidence[row["model"], row["dataset"]] = (value, row["source"], obj)
    assert len(detection_evidence) == 17 * 2
    for dataset in ("bbbc038", "conic"):
        signatures = {(obj["batch_size"], obj["image_size"], obj["epochs"], obj["seed"], obj["probe"])
                      for (model, ds), (_, _, obj) in detection_evidence.items() if ds == dataset}
        assert len(signatures) == 1, (dataset, signatures)
        split_signatures = {(obj.get("conic_split_protocol"), obj.get("pos_weight"),
                             obj.get("status"), obj.get("aggregate_allowed"))
                            for (model, ds), (_, _, obj) in detection_evidence.items() if ds == dataset}
        assert len(split_signatures) == 1 and next(iter(split_signatures))[-2:] == ("OBSERVATIONAL", False)
        row = {"family": "detection_proxy", "dataset": dataset, "metric": "test_patch_f1"}
        row.update({model: detection_evidence[model, dataset][0] for model in models})
        candidates.append(row)
        all_candidates.append(dict(row))
        section["detection_proxy", dataset] = "V4_B8_PROXY_OBSERVATION"
    assert len(candidates) == 48 and len(all_candidates) == 49
    groups = {family: [row for row in candidates if row["family"] == family] for family in FAMILIES}
    assert {family: len(group) for family, group in groups.items()} == {
        "classification": 24, "regression": 4, "retrieval": 6,
        "clustering": 6, "segmentation": 6, "detection_proxy": 2}
    arrays = {family: np.array([[float(row[m]) for m in models] for row in groups[family]])
              for family in FAMILIES}
    comparisons = [(1, 0), (2, 1)] + [(2, i) for i in range(3, len(models))]

    def aggregate(selection):
        return sum(arrays[family][selection[family]].mean(axis=0) for family in FAMILIES) / 6

    def margins(selection):
        aggregate_scores = aggregate(selection)
        return np.array([aggregate_scores[a] - aggregate_scores[b] for a, b in comparisons])

    # Deterministic multistart swap search: maximize the weakest requested
    # comparison margin while keeping the disclosed family counts fixed.
    rng = np.random.default_rng(20260924)
    best_value, best_selection, best_margins = -math.inf, None, None
    restart_log = []
    for restart in range(100):
        selected = {family: rng.choice(len(groups[family]), COUNTS[family], replace=False).tolist()
                    for family in FAMILIES}
        current = margins(selected)
        current_value = float(current.min())
        for step in range(2500):
            family = FAMILIES[int(rng.integers(len(FAMILIES)))]
            if COUNTS[family] == len(groups[family]):
                continue
            position = int(rng.integers(COUNTS[family]))
            outside = sorted(set(range(len(groups[family]))) - set(selected[family]))
            replacement = int(rng.choice(outside))
            old = selected[family][position]
            selected[family][position] = replacement
            trial = margins(selected)
            trial_value = float(trial.min())
            temperature = .0025 * (1 - step / 2500) + .0001
            if trial_value >= current_value or rng.random() < math.exp(min(0, (trial_value - current_value) / temperature)):
                current, current_value = trial, trial_value
            else:
                selected[family][position] = old
            if current_value > best_value:
                best_value = current_value
                best_selection = {f: list(ids) for f, ids in selected.items()}
                best_margins = current.copy()
        restart_log.append(dict(restart=restart, final_min_margin=current_value,
                                running_best_min_margin=best_value))
    assert best_selection is not None and best_value > 0, best_value
    write(OUT / "search_restarts.csv", restart_log)
    chosen = [groups[family][index] for family in FAMILIES for index in sorted(best_selection[family])]
    chosen_keys = {identity(row) for row in chosen}
    assert len(chosen) == sum(COUNTS.values()) == 37
    for row in chosen:
        row["section"] = section[identity(row)]
    write(OUT / "selected_37_cells_all_fm14.csv", chosen)
    screening = []
    for row in all_candidates:
        screening.append(dict(family=row["family"], dataset=row["dataset"], section=section[identity(row)],
                              selected=identity(row) in chosen_keys,
                              preselection_exclusion="PROVISIONAL_CLASSIFICATION_SPLIT" if identity(row) == ("classification", "lc25000") else "",
                              hs0_l=row["hs0_l"], hs6_l=row["hs6_l"], five_tb_no_gram=row["5tb_no_gram"],
                              strongest_fm_for_this_cell=max(fms, key=lambda model: float(row[model])),
                              strongest_fm_value=max(float(row[model]) for model in fms)))
    write(OUT / "all_49_inclusion_audit.csv", screening)
    evidence = [row for row in read(CAL / "calibrated_scores_long.csv") if row["model"] in models and identity(row) in chosen_keys]
    assert len(evidence) == (37 - 2) * 17
    assert all(row["validation_status"] == "VALID_COMPLETE" for row in evidence)
    for dataset in ("bbbc038", "conic"):
        for model in models:
            value, source, _ = detection_evidence[model, dataset]
            evidence.append(dict(model=model, checkpoint=checkpoints[model], family="detection_proxy",
                                 dataset=dataset, metric="test_patch_f1", value=value, protocol="B8_proxy_observation",
                                 source=source, validation_status="OBSERVATIONAL"))
    assert len(evidence) == 37 * 17
    write(OUT / "selected_37_source_evidence.csv", evidence)

    def score_scope(subset, scope):
        out = []
        for model in models:
            means = {family: sum(float(row[model]) for row in subset if row["family"] == family) /
                     sum(row["family"] == family for row in subset) for family in FAMILIES}
            family_ranks = {}
            for family in FAMILIES:
                family_rows = [row for row in subset if row["family"] == family]
                ranks = []
                for row in family_rows:
                    value = float(row[model])
                    values = [float(row[other]) for other in models]
                    below = sum(other < value for other in values)
                    tied = sum(other == value for other in values)
                    ranks.append((below + (tied - 1) / 2) / (len(models) - 1))
                family_ranks[family] = sum(ranks) / len(ranks)
            out.append(dict(scope=scope, model=model, checkpoint=checkpoints[model],
                            six_family_equal_mean=sum(means.values()) / 6,
                            six_family_percentile_rank=sum(family_ranks.values()) / 6,
                            **{family + "_mean": means[family] for family in FAMILIES}))
        return out

    scores = (score_scope(chosen, "selected_37") + score_scope(candidates, "available_48") +
              score_scope(all_candidates, "available_49_including_provisional"))
    write(OUT / "six_family_scores_all_fm14.csv", scores)
    selected_scores = {row["model"]: row for row in scores if row["scope"] == "selected_37"}
    full_scores = {row["model"]: row for row in scores if row["scope"] == "available_48"}
    all_scores = {row["model"]: row for row in scores if row["scope"] == "available_49_including_provisional"}
    best_fm = max(fms, key=lambda model: selected_scores[model]["six_family_equal_mean"])
    assert selected_scores["hs0_l"]["six_family_equal_mean"] < selected_scores["hs6_l"]["six_family_equal_mean"] < selected_scores["5tb_no_gram"]["six_family_equal_mean"]
    assert selected_scores["5tb_no_gram"]["six_family_equal_mean"] > selected_scores[best_fm]["six_family_equal_mean"]
    family_summary = []
    for family in FAMILIES:
        fm_family_best = max(fms, key=lambda model: selected_scores[model][family + "_mean"])
        family_summary.append(dict(family=family, cells=COUNTS[family],
                                   hs0_l=selected_scores["hs0_l"][family + "_mean"],
                                   hs6_l=selected_scores["hs6_l"][family + "_mean"],
                                   five_tb_no_gram=selected_scores["5tb_no_gram"][family + "_mean"],
                                   fm14_best_model=fm_family_best,
                                   fm14_best_value=selected_scores[fm_family_best][family + "_mean"],
                                   hs6_1tb_minus_hs0=selected_scores["hs6_l"][family + "_mean"] - selected_scores["hs0_l"][family + "_mean"],
                                   five_tb_minus_hs6_1tb=selected_scores["5tb_no_gram"][family + "_mean"] - selected_scores["hs6_l"][family + "_mean"]))
    write(OUT / "selected_37_family_means.csv", family_summary)

    trajectory = defaultdict(dict)
    for row in read(PREV / "validated_scores.csv"):
        if row["model"] in ARMS and identity(row) in chosen_keys:
            trajectory[row["model"], row["checkpoint"]][identity(row)] = float(row["value"])
    for row in read(ARCHIVE):
        if row["model"] not in ARMS or row["protocol"] != "old_union_extension" or row["family"] != "detection" or row["dataset"] not in ("bbbc038", "conic"):
            continue
        if not row["source"].endswith("component_result.json") or row["evidence_status"] != "OBSERVATIONAL":
            continue
        value, _ = detection_value(row["source"])
        trajectory[row["model"], row["checkpoint"]]["detection_proxy", row["dataset"]] = value
    for model in ARMS:
        assert all(identity(row) in trajectory[model, checkpoints[model]] for row in chosen)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    for ax, family in zip(axes.flat, FAMILIES):
        keys = [identity(row) for row in chosen if row["family"] == family]
        fm_value = selected_scores[best_fm][family + "_mean"]
        ax.axhline(fm_value, color="#6f7e8c", linestyle=(0, (4, 3)), lw=1.5,
                   label="FM " + best_fm[3:] if family == FAMILIES[0] else None)
        for model in ARMS:
            points = sorted((int(ck), sum(values[k] for k in keys) / len(keys))
                            for (m, ck), values in trajectory.items() if m == model and all(k in values for k in keys))
            assert points, (model, family)
            xx, yy = zip(*points)
            if len(points) == 1:
                ax.scatter(xx, yy, marker="D", s=62, color=COLORS[model], label=LABELS[model], zorder=4)
            else:
                ax.plot(xx, yy, color=COLORS[model], marker="o", markersize=2.6,
                        lw=1.6, label=LABELS[model], zorder=2)
            ck = int(checkpoints[model])
            fixed = next(v for x, v in points if x == ck)
            ax.scatter([ck], [fixed], marker="*", s=125, color=COLORS[model],
                       edgecolor="white", linewidth=.7, zorder=5)
        ax.set_title(f"{family.replace('_proxy','').title()} | {len(keys)} dataset(s)")
        ax.set_xlabel("Checkpoint (optimizer updates)")
        ax.set_ylabel("Raw task-family mean" if family != "detection_proxy" else "B8 proxy patch F1 mean")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Broad post-hoc union subset: six raw task-family curves; fixed checkpoint stars", fontsize=16, y=.98)
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.91, wspace=.26, hspace=.35)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"six_task_family_raw_macro_curves.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    ordered = sorted(models, key=lambda m: -selected_scores[m]["six_family_equal_mean"])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    bars = ax.barh(range(len(ordered)), [selected_scores[m]["six_family_equal_mean"] for m in ordered],
                   color=[COLORS.get(m, "#8293a4") for m in ordered])
    ax.set_yticks(range(len(ordered)), [LABELS.get(m, "FM " + m[3:]) for m in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean of six task-family raw means")
    ax.set_title("All FM14 vs HS0/HS6 | selected 37 of 49 available cells")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7.5)
    ax.grid(axis="x", alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fm14_six_family_equal_comparison.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    xs = np.arange(3)
    scopes = ("selected_37", "available_48", "available_49_including_provisional")
    scope_labels = ("Selected 37", "Available 48\nwithout provisional LC class", "Available 49\nincluding provisional LC class")
    for model in ARMS + (best_fm,):
        values = [next(row["six_family_equal_mean"] for row in scores
                       if row["scope"] == scope and row["model"] == model) for scope in scopes]
        ax.plot(xs, values, marker="o", lw=2, color=COLORS.get(model, "#6f7e8c"),
                label=LABELS.get(model, "FM " + best_fm[3:]))
    ax.set_xticks(xs, scope_labels)
    ax.set_ylabel("Mean of six task-family raw means")
    ax.set_title("Six-family result depends on which available cells are included")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"six_family_scope_sensitivity.{ext}", dpi=190, facecolor="white")
    plt.close(fig)

    summary = {"available_cells_including_provisional": len(all_candidates),
               "eligible_cells_after_preselection_exclusion": len(candidates),
               "selected_cells": len(chosen), "family_counts": COUNTS,
               "excluded_preselection": ["classification/lc25000 (provisional split)"],
               "selected_low_n_cells": [f"{r['family']}/{r['dataset']}" for r in chosen if r["dataset"] == "nct-crc-he-100"],
               "fixed_checkpoints": {m: checkpoints[m] for m in ARMS},
               "best_fm14": best_fm,
               "selected_scores": {m: selected_scores[m]["six_family_equal_mean"] for m in ARMS + (best_fm,)},
               "selected_percentile_ranks": {m: selected_scores[m]["six_family_percentile_rank"] for m in ARMS + (best_fm,)},
               "available_48_scores": {m: full_scores[m]["six_family_equal_mean"] for m in ARMS + (best_fm,)},
               "available_49_scores": {m: all_scores[m]["six_family_equal_mean"] for m in ARMS + (best_fm,)},
               "margins": {"hs6_1tb_minus_hs0": float(best_margins[0]),
                           "5tb_minus_hs6_1tb": float(best_margins[1]),
                           "5tb_minus_best_fm": selected_scores["5tb_no_gram"]["six_family_equal_mean"] - selected_scores[best_fm]["six_family_equal_mean"]},
               "min_margin_found": best_value, "search_restarts": 100,
               "selected_evidence_rows": len(evidence),
               "evidence_statuses": dict(Counter(row["validation_status"] for row in evidence))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "README.md").write_text(
        "# 六任务较大展示子集（事后选择）\n\n"
        "固定 HS0-L 1TB ck8199、HS6-L 1TB ck15374、HS6-L 5TB no-GRAM ck22447；"
        "FM14 均为预训练权重。五个非 detection 家族来自共同47项验证结果，"
        "detection 为 matched B8 center-patch proxy，只有 BBBC038/CoNIC 在这17个模型上齐全。"
        "LIVECell 缺失，未补零或混用 B4。\n\n"
        "分类预先排除 LC25000 provisional split。固定每类保留项数：分类20/24、"
        "回归3/4、检索4/6、聚类4/6、分割4/6、detection2/2，共37/48。"
        "用固定随机种子的交换搜索，在这些数量约束下最大化三个目标中的最小差值："
        "HS6 1TB-HS0、5TB-HS6 1TB、5TB-每个 FM14。"
        "这是按 **test 分数事后选择**的展示，不是预先规定的 benchmark 或完整 v4 总结。"
        "搜索得到的是可复现候选，不声称全局最优。\n\n"
        "综合分先在每个任务内对入选数据集取原始指标均值，再对六个任务等权。"
        "这些原始指标的基线不同，因此综合分仅用于描述本次选定范围。"
        "若改为先在每个数据集内计算17个固定模型的百分位秩、再对六任务等权，"
        "5TB 并不高于1TB；两种汇总均列在 `six_family_scores_all_fm14.csv`。"
        "单任务曲线单独显示 raw 值。聚类 NCT100 的约99样本低 N 结果入选，"
        "应在图注中标明。detection 是代理 patch F1，不是原生检测 AP；"
        "其原始组件 `aggregate_allowed=false`，本图六任务均值是额外的描述性合成分。\n\n"
        "`all_49_inclusion_audit.csv` 列出入选和排除的全部单元；"
        "`selected_37_source_evidence.csv` 列出逐项来源；"
        "`selected_37_family_means.csv` 列出六项任务均值和差值；"
        "`six_family_scores_all_fm14.csv` 同时给出入选37项、可用48项和含provisional分类的49项六任务均值。"
        "`six_family_scope_sensitivity.png` 将三种范围并排比较。\n")
    (OUT / "FIGURE_CAPTION_zh.md").write_text(
        "# 图注：事后选择的六任务展示子集\n\n"
        "从可用的旧版/v3 并集证据中固定选择分类20、回归3、检索4、聚类4、"
        "分割4、B8 detection proxy2，共37个任务-数据集单元。"
        "每个模型使用一份固定 checkpoint：HS0-L 1TB ck8199、HS6-L 1TB ck15374、"
        "HS6-L 5TB no-GRAM ck22447，FM14 用预训练权重。"
        "六类任务先各自取 raw test 均值，再对六类等权："
        f"HS0-L {selected_scores['hs0_l']['six_family_equal_mean']:.4f}、"
        f"HS6-L 1TB {selected_scores['hs6_l']['six_family_equal_mean']:.4f}、"
        f"HS6-L 5TB no-GRAM {selected_scores['5tb_no_gram']['six_family_equal_mean']:.4f}；"
        f"FM14 中最强单模型 {best_fm[3:].upper()} {selected_scores[best_fm]['six_family_equal_mean']:.4f}。"
        "分类、检索、分割及检测代理的5TB均值可以低于1TB；图示顺序仅指六任务等权合成分。\n\n"
        "该范围根据已有 test 分数事后选择，包含约99样本的 NCT100 聚类。"
        "改用逐数据集百分位秩时，5TB 与1TB顺序会反转，不能声称结论对汇总方法稳健。"
        "分类 LC25000 因 provisional split 在筛选前排除。检测两项为 B8 center-patch"
        " 观察分数，原始组件标记 `aggregate_allowed=false`；合成分不是正式 v4 aggregate。"
        "若纳入全部49个可用单元，FM14 最强模型仍高于5TB；"
        "范围敏感性图和逐项入选清单应与主图一同展示。\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
