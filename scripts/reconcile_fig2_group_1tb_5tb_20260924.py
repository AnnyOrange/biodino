#!/usr/bin/env python3
"""Reconcile the hand-drawn six-family SVG with traceable old JSON values."""

import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from audit_hs6_ep15_vs_5tb_old_figure_20260924 import (
    FAMILIES, ROOT, TARGETS, metric_value, source_priority, write,
)


SVG = ROOT / "plot/fig2/dataset_performance_1-5-20/dataset_performance_1tb_5tb_20tb.svg"
ARCHIVE = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
OUT = ROOT / "outputs/00_reports/fig2_old_group_ep15_vs_5tb_ck19031_20260924"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open(newline="") as handle:
        archive = list(csv.DictReader(handle))
    targets = {(family, dataset): metric for family, dataset, metric in TARGETS}
    indexed = defaultdict(list)
    for row in archive:
        key = row["family"], row["dataset"]
        if row["model"] not in ("hs6_l", "5tb_no_gram") or key not in targets:
            continue
        if row["protocol"] == "old" or (row["protocol"] == "old_observation" and row["family"] == "detection"):
            indexed[row["model"], int(row["checkpoint"]), *key].append(row)

    values, chosen = {}, {}
    for key, source_rows in indexed.items():
        best = min(source_rows, key=source_priority)
        values[key] = metric_value(best, targets[key[2:]])
        chosen[key] = best

    def family_means(model, checkpoint):
        result = {}
        for family in FAMILIES:
            keys = [(model, checkpoint, fam, dataset)
                    for fam, dataset, _ in TARGETS if fam == family]
            if any(key not in values for key in keys):
                return None
            result[family] = statistics.mean(values[key] for key in keys)
        result["six_family_equal"] = statistics.mean(result.values())
        return result

    dataset_rows = []
    for family, dataset, metric in TARGETS:
        one_key = "hs6_l", 15374, family, dataset
        five_key = "5tb_no_gram", 19031, family, dataset
        assert one_key in values and five_key in values
        a, b = values[one_key], values[five_key]
        dataset_rows.append(dict(family=family, dataset=dataset, metric=metric,
                                 one_tb_ep15_ck15374=a, five_tb_ck19031=b,
                                 five_minus_one=b-a, delta_percentage_points=100*(b-a),
                                 one_tb_source=chosen[one_key]["source"],
                                 five_tb_source=chosen[five_key]["source"],
                                 one_tb_archive_status=chosen[one_key]["evidence_status"],
                                 five_tb_archive_status=chosen[five_key]["evidence_status"]))
    write(OUT / "per_dataset_43_ep15_vs_ck19031.csv", dataset_rows)
    one, five = family_means("hs6_l", 15374), family_means("5tb_no_gram", 19031)
    assert one and five
    family_rows = []
    for family in (*FAMILIES, "six_family_equal"):
        family_rows.append(dict(family=family,
                                datasets=sum(row["family"] == family for row in dataset_rows)
                                if family != "six_family_equal" else 43,
                                one_tb_ep15_ck15374=one[family], five_tb_ck19031=five[family],
                                delta_percentage_points=100*(five[family]-one[family])))
    write(OUT / "family_summary_ep15_vs_ck19031.csv", family_rows)

    xml = ET.parse(SVG).getroot()
    bars = {}
    for group in xml.findall("{http://www.w3.org/2000/svg}g"):
        fill = group.get("fill")
        if fill in ("#4E79A7", "#F28E2B"):
            heights = [float(rect.get("height"))
                       for rect in group.findall("{http://www.w3.org/2000/svg}rect")]
            if len(heights) == 7:
                bars[fill] = [height / 328 for height in heights]
    assert set(bars) == {"#4E79A7", "#F28E2B"}
    svg_rows = []
    for i, family in enumerate((*FAMILIES, "six_family_equal")):
        a, b = bars["#4E79A7"][i], bars["#F28E2B"][i]
        svg_rows.append(dict(family=family, svg_1tb_bar=a, svg_5tb_bar=b,
                             svg_delta_percentage_points=100*(b-a),
                             actual_1tb_ep15_ck15374=one[family],
                             svg_1tb_minus_ep15=a-one[family],
                             actual_5tb_ck19031=five[family],
                             svg_5tb_minus_ck19031=b-five[family]))
    assert abs(svg_rows[-1]["svg_delta_percentage_points"]-1.591) < .001
    assert max(abs(row["svg_5tb_minus_ck19031"]) for row in svg_rows) < 3e-6
    write(OUT / "svg_bar_arithmetic_and_source_audit.csv", svg_rows)

    checkpoint_match = []
    for checkpoint in sorted({key[1] for key in values if key[0] == "hs6_l"}):
        family_scores = family_means("hs6_l", checkpoint)
        if family_scores:
            differences = [family_scores[f] - bars["#4E79A7"][i]
                           for i, f in enumerate(FAMILIES)]
            checkpoint_match.append(dict(checkpoint=checkpoint,
                                         mean_absolute_family_difference=statistics.mean(abs(x) for x in differences),
                                         largest_absolute_family_difference=max(abs(x) for x in differences),
                                         aggregate=family_scores["six_family_equal"],
                                         aggregate_minus_svg=family_scores["six_family_equal"]-bars["#4E79A7"][6]))
    write(OUT / "svg_1tb_checkpoint_match_audit.csv", checkpoint_match)
    summary = dict(svg_group_1tb=svg_rows[-1]["svg_1tb_bar"],
                   svg_group_5tb=svg_rows[-1]["svg_5tb_bar"],
                   svg_group_delta_pp=svg_rows[-1]["svg_delta_percentage_points"],
                   svg_recomputed_from_six_family_bars_1tb=statistics.mean(bars["#4E79A7"][:6]),
                   svg_recomputed_from_six_family_bars_5tb=statistics.mean(bars["#F28E2B"][:6]),
                   svg_recomputed_delta_pp=100*(statistics.mean(bars["#F28E2B"][:6])
                                                - statistics.mean(bars["#4E79A7"][:6])),
                   actual_ep15_1tb=one["six_family_equal"],
                   actual_ck19031_5tb=five["six_family_equal"],
                   actual_delta_pp=100*(five["six_family_equal"]-one["six_family_equal"]),
                   complete_1tb_checkpoint_candidates=len(checkpoint_match),
                   closest_1tb_checkpoint=min(checkpoint_match,
                                              key=lambda row: row["mean_absolute_family_difference"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# 原六任务柱状图核对与逐数据集清单\n\n"
        "直接读取原 SVG 柱高除以328，六任务均值分别是0.8273048780和0.8432159553，"
        "差为1.5911077个百分点。5TB六个柱高逐项对应旧原始JSON的ck19031。"
        "SVG未标1TB checkpoint；旧归档有完整检测的5个1TB checkpoint中，没有一个逐栏对上该图。"
        "因此不能从这张图恢复可信的1TB逐数据集明细。\n\n"
        "`per_dataset_43_ep15_vs_ck19031.csv` 按图中的43项配方，使用用户指定的1TB ep15 ck15374"
        "与旧库最高5TB ck19031，逐项列原始JSON路径。"
        "`svg_bar_arithmetic_and_source_audit.csv` 把图中柱子和这组可追溯结果并排。"
        "旧结果为历史未认证归档，检测为旧B4 observation。\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
