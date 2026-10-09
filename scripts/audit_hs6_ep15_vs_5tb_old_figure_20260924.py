#!/usr/bin/env python3
"""Recompute the exact six-family old-figure recipe at requested checkpoints."""

import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from finalize_splus_checkpoint_ab import (
    CLASSIFICATION_DATASETS, RETRIEVAL_DATASETS, SEGMENTATION_DATASETS,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
OUT = ROOT / "outputs/00_reports/hs6_l_ep15_vs_5tb_old_figure_20260924"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection")
TARGETS = (
    [("classification", dataset, "macro_f1") for dataset in CLASSIFICATION_DATASETS]
    + [("regression", "bbbc005", "r2")]
    + [("retrieval", dataset, "map_at_5") for dataset in RETRIEVAL_DATASETS]
    + [("clustering", dataset, "nmi") for dataset in RETRIEVAL_DATASETS]
    + [("segmentation", dataset, "test_mDice") for dataset in SEGMENTATION_DATASETS]
    + [("detection", "livecell", "test_patch_f1")]
)
assert len(TARGETS) == 43


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_priority(row):
    source = row["source"]
    return (0 if "hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909" in source else 1,
            source)


def metric_value(row, metric):
    data = json.loads(Path(row["source"]).read_text())
    if metric == "test_mDice":
        value = data["test"]["mDice"]
    else:
        value = data[metric]
    value = float(value)
    if metric == "test_patch_f1" and value > 1:
        value /= 100
    assert math.isfinite(value), row
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = read(ARCHIVE)
    targets = {(family, dataset): metric for family, dataset, metric in TARGETS}
    indexed = defaultdict(list)
    for row in archive:
        key = row["family"], row["dataset"]
        if row["model"] not in ("hs6_l", "5tb_no_gram") or key not in targets:
            continue
        if row["protocol"] == "old" or (row["protocol"] == "old_observation" and row["family"] == "detection"):
            indexed[row["model"], int(row["checkpoint"]), *key].append(row)

    cache = {}
    evidence = {}
    duplicates = []
    for key, source_rows in indexed.items():
        metric = targets[key[2:]]
        scored = [(row, metric_value(row, metric)) for row in source_rows]
        chosen_row, chosen_value = min(scored, key=lambda item: source_priority(item[0]))
        cache[key] = chosen_value
        evidence[key] = chosen_row
        if len(scored) > 1:
            vals = [item[1] for item in scored]
            duplicates.append(dict(model=key[0], checkpoint=key[1], family=key[2], dataset=key[3],
                                   chosen_value=chosen_value, min_value=min(vals), max_value=max(vals),
                                   spread=max(vals)-min(vals), chosen_source=chosen_row["source"],
                                   all_sources=";".join(row["source"] for row, _ in scored)))
    write(OUT / "duplicate_sources.csv", duplicates)

    def aggregate(model, checkpoint):
        family_scores = {}
        for family in FAMILIES:
            keys = [(model, checkpoint, f, dataset) for f, dataset, _ in TARGETS if f == family]
            if any(key not in cache for key in keys):
                return None
            family_scores[family] = statistics.mean(cache[key] for key in keys)
        family_scores["six_family_equal"] = statistics.mean(family_scores.values())
        return family_scores

    requested = {"hs6_l": 15374, "5tb_no_gram": 22447}
    comparison = []
    for family, dataset, metric in TARGETS:
        left = ("hs6_l", requested["hs6_l"], family, dataset)
        right = ("5tb_no_gram", requested["5tb_no_gram"], family, dataset)
        assert left in cache and right in cache, (left, right)
        a, b = cache[left], cache[right]
        comparison.append(dict(family=family, dataset=dataset, metric=metric,
                               one_tb_ep15_ck15374=a, five_tb_nogram_ck22447=b,
                               five_minus_one=b-a, difference_percentage_points=100*(b-a),
                               one_tb_source=evidence[left]["source"], five_tb_source=evidence[right]["source"],
                               one_tb_archive_status=evidence[left]["evidence_status"],
                               five_tb_archive_status=evidence[right]["evidence_status"]))
    write(OUT / "per_dataset_43.csv", comparison)

    summary = []
    for family in (*FAMILIES, "six_family_equal"):
        one = aggregate("hs6_l", 15374)[family]
        five = aggregate("5tb_no_gram", 22447)[family]
        summary.append(dict(family=family, datasets=sum(row["family"] == family for row in comparison)
                            if family != "six_family_equal" else 43,
                            one_tb_ep15_ck15374=one, five_tb_nogram_ck22447=five,
                            five_minus_one=five-one, difference_percentage_points=100*(five-one)))
    write(OUT / "task_family_summary.csv", summary)

    figure_checkpoints = []
    for model, checkpoint, svg_value in (("hs6_l", 14349, 0.837098),
                                         ("5tb_no_gram", 19031, 0.843216)):
        scores = aggregate(model, checkpoint)
        assert scores is not None and abs(scores["six_family_equal"]-svg_value) < 1e-6
        figure_checkpoints.append(dict(model=model, checkpoint=checkpoint,
                                       reproduced=scores["six_family_equal"],
                                       svg_label=svg_value,
                                       difference=scores["six_family_equal"]-svg_value, **{f: scores[f] for f in FAMILIES}))
    write(OUT / "old_best_figure_reproduction.csv", figure_checkpoints)

    group_svg = ROOT / "plot/fig2/dataset_performance_1-5-20/dataset_performance_1tb_5tb_20tb.svg"
    xml = ET.parse(group_svg).getroot()
    bars = {}
    for group in xml.findall("{http://www.w3.org/2000/svg}g"):
        fill = group.get("fill")
        if fill in ("#4E79A7", "#F28E2B"):
            heights = [float(rect.get("height")) for rect in group.findall("{http://www.w3.org/2000/svg}rect")]
            if len(heights) == 7:
                bars[fill] = [height / 328 for height in heights]
    assert set(bars) == {"#4E79A7", "#F28E2B"}
    group_audit = []
    for index, family in enumerate((*FAMILIES, "six_family_equal")):
        svg_one, svg_five = bars["#4E79A7"][index], bars["#F28E2B"][index]
        one_ep15 = aggregate("hs6_l", 15374)[family]
        five_19031 = aggregate("5tb_no_gram", 19031)[family]
        group_audit.append(dict(family=family, unlabeled_svg_1tb=svg_one,
                                actual_1tb_ep15_ck15374=one_ep15,
                                svg_minus_ep15=svg_one-one_ep15,
                                unlabeled_svg_5tb=svg_five,
                                actual_5tb_ck19031=five_19031,
                                svg_minus_ck19031=svg_five-five_19031))
    assert max(abs(row["svg_minus_ck19031"]) for row in group_audit) < 3e-6
    write(OUT / "unlabeled_group_svg_checkpoint_audit.csv", group_audit)

    curve = []
    original_49 = set(json.loads((ROOT / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914/input_manifest.json").read_text())["checkpoints"])
    for model in ("hs6_l", "5tb_no_gram"):
        for checkpoint in sorted({key[1] for key in cache if key[0] == model}):
            scores = aggregate(model, checkpoint)
            if scores is not None:
                curve.append(dict(model=model, checkpoint=checkpoint,
                                  in_original_5tb_49_curve=int(checkpoint in original_49) if model == "5tb_no_gram" else "",
                                  **scores))
    write(OUT / "complete_six_family_checkpoint_scores.csv", curve)
    best = {model: max((row for row in curve if row["model"] == model),
                       key=lambda row: row["six_family_equal"]) for model in ("hs6_l", "5tb_no_gram")}
    best_original_49 = max((row for row in curve if row["model"] == "5tb_no_gram"
                            and row["in_original_5tb_49_curve"] == 1),
                           key=lambda row: row["six_family_equal"])
    result = {"requested_checkpoints": requested,
              "requested_six_family_scores": {row["family"]: dict(one_tb=row["one_tb_ep15_ck15374"],
                                                                   five_tb=row["five_tb_nogram_ck22447"],
                                                                   delta_pp=row["difference_percentage_points"])
                                              for row in summary},
              "best_complete_old_figure_six_family": {model: dict(checkpoint=row["checkpoint"],
                                                                    score=row["six_family_equal"])
                                                      for model, row in best.items()},
              "best_in_original_5tb_49_curve": dict(checkpoint=best_original_49["checkpoint"],
                                                      score=best_original_49["six_family_equal"]),
              "reference_figure_reproduced": figure_checkpoints,
              "duplicate_source_count": len(duplicates)}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# HS6-L ep15 与 5TB 旧六任务图逐数据集对照\n\n"
        "严格复用 `plot/fig2/dataset_performance_1-5-20/best_1tb_vs_5tb.svg` 的 43 项指标配方："
        "分类25项 macro-F1，BBBC005 R²，检索4项 mAP@5，聚类4项 NMI，分割8项 test mDice，"
        "LiveCell 旧 B4 detection test patch F1。六任务等权。"
        "原图 ck14349/ck19031 的 0.837098/0.843216 已从原始 JSON 重算到 1e-6 以内。\n\n"
        "`per_dataset_43.csv` 对照用户指定的 1TB ep15 ck15374 和 5TB no-GRAM ck22447，逐项带来源。"
        "`task_family_summary.csv` 给任务族均值，`complete_six_family_checkpoint_scores.csv` 给所有完整 checkpoint。"
        "按原任务峰值图锁定的49个 5TB checkpoint，ck22447 是该配方最高分；"
        "若纳入旧库所有58个完整 checkpoint，则 ck19031 更高。"
        "`unlabeled_group_svg_checkpoint_audit.csv` 证明另一张六栏旧 SVG 的 5TB 柱等于 ck19031，"
        "但其未标 checkpoint 的 1TB 柱与 ep15 ck15374 不相同。"
        "归档标记为历史未认证结果；LiveCell detection 属于旧 B4 observation，不能当正式 v4 B8。"
        "旧分类 LC25000 split 为 provisional，NCT100 仅 99 样本。\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
