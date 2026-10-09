#!/usr/bin/env python3
"""Explain fixed-checkpoint scaling reversal and expose historical detection batches."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
INDEX = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
MODELS = ("hs6_splus", "hs6_b", "hs6_l", "hs6_hplus", "5tb_no_gram", "5tb_gram12687")
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
COLORS = {"hs6_splus": "#53698c", "hs6_b": "#5189bb", "hs6_l": "#194d90",
          "hs6_hplus": "#152846", "5tb_no_gram": "#009b77", "5tb_gram12687": "#bb5852"}


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def write(name, rows, fields):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def scaling():
    score = {(r["model"], r["checkpoint"], r["family"], r["dataset"]): float(r["value"])
             for r in read("validated_scores.csv")}
    selected = {r["model"]: r["checkpoint"] for r in read("all_model_best_checkpoints.csv")}
    core = [(r["family"], r["dataset"]) for r in read("common_cells.csv")]
    arms = ("5tb_no_gram", "5tb_gram12687")
    rows = []
    for arm in arms:
        for family, dataset in core:
            a_fixed = score[arm, selected[arm], family, dataset]
            b_fixed = score["hs6_l", selected["hs6_l"], family, dataset]
            a_peak = max((v, ck) for (m, ck, f, ds), v in score.items()
                         if (m, f, ds) == (arm, family, dataset))
            b_peak = max((v, ck) for (m, ck, f, ds), v in score.items()
                         if (m, f, ds) == ("hs6_l", family, dataset))
            rows.append(dict(arm=arm, family=family, dataset=dataset,
                             checkpoint_5tb_fixed=selected[arm], checkpoint_1tb_l_fixed=selected["hs6_l"],
                             value_5tb_fixed=a_fixed, value_1tb_l_fixed=b_fixed,
                             fixed_delta=a_fixed - b_fixed,
                             checkpoint_5tb_peak=a_peak[1], checkpoint_1tb_l_peak=b_peak[1],
                             value_5tb_peak=a_peak[0], value_1tb_l_peak=b_peak[0],
                             peak_delta=a_peak[0] - b_peak[0]))
    write("hs6_l_1tb_vs_5tb_fixed_and_peaks.csv", rows, list(rows[0]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True, sharey=True)
    for ax, arm in zip(axes, arms):
        names = ["Fixed checkpoint", "Per-dataset test peak"]
        for j, kind in enumerate(("fixed", "peak")):
            vals = [np.mean([r[f"{kind}_delta"] for r in rows if r["arm"] == arm and r["family"] == f]) * 100
                    for f in FAMILIES]
            ax.bar(np.arange(len(FAMILIES)) + (j - .5) * .36, vals, width=.36,
                   label=names[j], color=("#397aa9", "#df8d50")[j])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(FAMILIES)), ["Class.", "Regr.", "Retr.", "Clust.", "Seg."])
        ax.set_title(f"{arm.replace('_', ' ')} vs HS6-L 1TB")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean score difference (percentage points)")
    fig.suptitle("Same architecture: a single fixed checkpoint vs retrospective per-dataset peaks")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"hs6_l_1tb_vs_5tb_fixed_and_peaks.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rows


def detection():
    rows = []
    for item in csv.DictReader(INDEX.open()):
        if item["model"] not in MODELS or item["family"] != "detection" or item["protocol"] != "old_observation":
            continue
        if not item["source"].endswith("results_bio_detection.json"):
            continue
        obj = json.loads(Path(item["source"]).read_text())
        if obj.get("image_size") != 224 or obj.get("epochs") != 5:
            continue
        val = obj.get("test_patch_f1")
        if val is None:
            continue
        rows.append(dict(model=item["model"], checkpoint=item["checkpoint"], dataset=item["dataset"],
                         batch_size=obj["batch_size"], value=float(val) / 100,
                         protocol="historical_observation", source=item["source"]))
    write("hs6_historical_detection_scores.csv", rows, list(rows[0]))
    by_point = defaultdict(dict)
    for r in rows:
        by_point[r["model"], r["checkpoint"], r["batch_size"]][r["dataset"]] = r["value"]
    plot_rows = []
    for (model, ck, batch), ds in by_point.items():
        if set(ds) == {"bbbc038", "conic", "livecell"}:
            plot_rows.append(dict(model=model, checkpoint=int(ck), batch_size=batch,
                                  mean_f1=sum(ds.values()) / 3, datasets=3))
    write("hs6_historical_detection_means.csv", plot_rows, list(plot_rows[0]))
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for model in MODELS:
        rr = sorted((r for r in plot_rows if r["model"] == model), key=lambda x: x["checkpoint"])
        if not rr:
            continue
        batch = rr[0]["batch_size"]
        xx = [r["checkpoint"] / 1000 for r in rr]
        yy = [r["mean_f1"] for r in rr]
        ax.plot(xx, yy, "x--", label=f"{model.replace('_', ' ')} | historical B{batch}",
                color=COLORS[model], linewidth=1.1, markersize=5)
    # Matched-B8 companion values are marked separately, with incomplete cells omitted.
    matched = defaultdict(dict)
    for r in read("validated_scores.csv"):
        if r["model"] in MODELS and r["family"] == "detection_proxy":
            matched[r["model"], r["checkpoint"]][r["dataset"]] = float(r["value"])
    for model in MODELS:
        rr = sorted(((int(ck), sum(ds.values()) / 3) for (m, ck), ds in matched.items()
                     if m == model and set(ds) == {"bbbc038", "conic", "livecell"}))
        if rr:
            ax.scatter([x / 1000 for x, _ in rr], [v for _, v in rr], marker="o", s=30,
                       facecolors="none", edgecolors=COLORS[model],
                       label=f"{model.replace('_', ' ')} | matched B8")
    ax.set(xlabel="Training updates (k)", ylabel="Mean test patch F1",
           title="HS6 detection proxy: historical B2/B4/B8 trajectories and matched B8 cells")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"hs6_detection_historical_batches.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_rows


def main():
    a = scaling()
    b = detection()
    report = {
        "fixed_l1tb": "hs6_l ck15374", "fixed_l5tb_no_gram": "5tb_no_gram ck26351",
        "fixed_l5tb_gram": "5tb_gram12687 ck24887",
        "comparison_scope": "40 common v3 task-dataset cells; same ViT-L architecture",
        "per_dataset_peak_caution": "Each peak may come from a different checkpoint selected on test data; it is an upper envelope, not a deployable single model.",
        "historical_detection_complete_points": len(b),
        "historical_detection_batches": {m: sorted({x["batch_size"] for x in b if x["model"] == m}) for m in MODELS},
        "family_deltas": {arm: {family: {kind: sum(x[f"{kind}_delta"] for x in a if x["arm"] == arm and x["family"] == family) /
                                            sum(x["arm"] == arm and x["family"] == family for x in a)
                                      for kind in ("fixed", "peak")} for family in FAMILIES}
                          for arm in ("5tb_no_gram", "5tb_gram12687")},
    }
    (OUT / "scaling_detection_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    no = report["family_deltas"]["5tb_no_gram"]
    (OUT / "SCALING_AND_DETECTION_AUDIT.md").write_text(
        "# 1TB / 5TB 对比与 detection 缺点审计\n\n"
        "上一版横向图的组选点为 HS6 H+ 1TB 与 HS6-L 5TB，模型容量不同。"
        "同架构的直接比较应为 HS6-L 1TB ck15374、5TB no-GRAM ck26351、"
        "5TB GRAM ck24887。三者均为对 40 个共同 v3 单元使用 test 分数回顾性选出的固定 checkpoint。\n\n"
        "| 任务 | 5TB no-GRAM 固定点 − 1TB L 固定点 | 各数据集历史峰值差 |\n"
        "|---|---:|---:|\n"
        + "".join(f"| {f} | {no[f]['fixed']*100:+.2f} pp | {no[f]['peak']*100:+.2f} pp |\n" for f in FAMILIES)
        + "\n在逐数据集分别取历史峰值时，5TB no-GRAM 五个任务家族均更高；"
        "但这些峰值可来自不同 checkpoint，是 test 上界，不能作为一份固定模型的分数。"
        "固定 checkpoint 口径下，5TB 仅回归略高，分类、聚类和分割较低，检索近似相同。"
        "这解释了旧曲线中的 5TB 峰值印象与上一版固定模型排名的差别。\n\n"
        "HS6 1TB detection 不是没有测试：归档有 S+/B 的历史 B8、L 的历史 B4、"
        "H+ 的历史 B2 三项 proxy 结果。上一版图仅从已登记的匹配 B8 扩展结果取点，"
        "因此 H+ 没有显示，L 也只显示了少量 B8 点。历史 B2/B4 不能直接混入"
        "统一 B8 的横向对比。新图 `hs6_detection_historical_batches.png` 单独标出 batch；"
        "六任务图的 detection 面板也补画了带 × 的历史轨迹。\n\n"
        "原始路径及逐项值见 `hs6_historical_detection_scores.csv`；同架构逐数据集"
        "固定点与峰值见 `hs6_l_1tb_vs_5tb_fixed_and_peaks.csv`。\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
