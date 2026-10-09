#!/usr/bin/env python3
"""Compare HS6-L 1TB and 5TB on the exact old 5TB curve dataset set."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from audit_reproduce_hs6_l5_5tb_old_macro_20260924 import score


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
CURVE = ROOT / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914"
OUT = ROOT / "outputs/00_reports/hs6_l_1tb_vs_5tb_old_protocol_20260924"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection")
EXPECTED = dict(classification=25, regression=4, retrieval=4, clustering=4, segmentation=8, detection=3)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def priority(row):
    # Use the dedicated old-protocol sweep when the archive contains repeats.
    source = row["source"]
    return (0 if "hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909" in source else 1,
            source)


def value(row):
    if row["family"] != "detection":
        return score(row)[1]
    result = json.loads(Path(row["source"]).read_text())
    val = result.get("test_patch_f1", result.get("val_patch_f1"))
    assert isinstance(val, (int, float)), row
    return val / 100 if val > 1 else val


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archived = read_csv(ARCHIVE)
    original_support = set(json.loads((CURVE / "input_manifest.json").read_text())["checkpoints"])
    target = {(r["family"], r["dataset"]) for r in archived
              if r["model"] == "5tb_no_gram" and r["checkpoint"] == "22447" and r["protocol"] == "old"}
    target |= {(r["family"], r["dataset"]) for r in archived
               if r["model"] == "5tb_no_gram" and r["checkpoint"] == "22447"
               and r["protocol"] == "old_observation" and r["family"] == "detection"}
    assert {f: sum(family == f for family, _ in target) for f in FAMILIES} == EXPECTED
    grouped = defaultdict(list)
    for row in archived:
        if row["model"] in ("hs6_l", "5tb_no_gram") and (row["family"], row["dataset"]) in target:
            if row["protocol"] == "old" or (row["protocol"] == "old_observation" and row["family"] == "detection"):
                grouped[row["model"], int(row["checkpoint"]), row["family"], row["dataset"]].append(row)

    evidence = []
    values = {}
    duplicates = []
    for key, rows in sorted(grouped.items()):
        scored = [(row, value(row)) for row in rows]
        chosen = sorted(scored, key=lambda item: priority(item[0]))[0]
        model, checkpoint, family, dataset = key
        values[key] = chosen[1]
        evidence.append(dict(model=model, checkpoint=checkpoint, family=family, dataset=dataset,
                             value=chosen[1], source=chosen[0]["source"],
                             evidence_status=chosen[0]["evidence_status"],
                             archived_candidates=len(rows)))
        if len(rows) > 1:
            nums = [v for _, v in scored]
            duplicates.append(dict(model=model, checkpoint=checkpoint, family=family, dataset=dataset,
                                   chosen_value=chosen[1], min_value=min(nums), max_value=max(nums),
                                   spread=max(nums)-min(nums),
                                   chosen_source=chosen[0]["source"],
                                   other_sources=";".join(row["source"] for row, _ in scored
                                                          if row["source"] != chosen[0]["source"])))
    write_csv(OUT / "source_values.csv", evidence)
    write_csv(OUT / "duplicate_source_audit.csv", duplicates)
    write_csv(OUT / "old_48_dataset_inventory.csv", [dict(family=f, dataset=d) for f, d in sorted(target)])

    points = []
    for model in ("hs6_l", "5tb_no_gram"):
        for checkpoint in sorted({k[1] for k in values if k[0] == model}):
            row = dict(model=model, checkpoint=checkpoint,
                       in_original_5tb_49_curve=int(checkpoint in original_support) if model == "5tb_no_gram" else "",
                       old_45_coverage=sum((model, checkpoint, f, d) in values for f, d in target if f != "detection"),
                       old_detection_3_coverage=sum((model, checkpoint, f, d) in values for f, d in target if f == "detection"))
            for family in FAMILIES:
                observed = [values[model, checkpoint, f, d] for f, d in target
                            if f == family and (model, checkpoint, f, d) in values]
                row[family] = statistics.mean(observed) if len(observed) == EXPECTED[family] else ""
            row["five_family_equal"] = (statistics.mean(row[f] for f in FAMILIES[:5])
                                        if row["old_45_coverage"] == 45 else "")
            row["six_family_equal"] = (statistics.mean(row[f] for f in FAMILIES)
                                       if row["old_45_coverage"] == 45 and row["old_detection_3_coverage"] == 3 else "")
            points.append(row)
    write_csv(OUT / "old_fixed_checkpoint_curves.csv", points)
    fixed = {m: next(row for row in points if row["model"] == m and row["checkpoint"] == ck)
             for m, ck in (("hs6_l", 15374), ("5tb_no_gram", 22447))}
    comparisons = []
    for field in (*FAMILIES, "five_family_equal", "six_family_equal"):
        one, five = fixed["hs6_l"][field], fixed["5tb_no_gram"][field]
        comparisons.append(dict(metric=field, hs6_1tb_checkpoint=15374, hs6_1tb_value=one,
                                hs6_5tb_checkpoint=22447, hs6_5tb_value=five,
                                five_minus_one=five-one, difference_percentage_points=100*(five-one)))
    write_csv(OUT / "fixed_checkpoint_comparison.csv", comparisons)
    milestones = []
    for stage, one_ck, five_ck in (("at_about_1m_images", 1024, 975),
                                   ("later_selected_checkpoints", 15374, 22447)):
        one = next(row for row in points if row["model"] == "hs6_l" and row["checkpoint"] == one_ck)
        five = next(row for row in points if row["model"] == "5tb_no_gram" and row["checkpoint"] == five_ck)
        for field in (*FAMILIES, "five_family_equal", "six_family_equal"):
            milestones.append(dict(stage=stage, score=field, one_tb_checkpoint=one_ck,
                                   five_tb_checkpoint=five_ck, one_tb_value=one[field],
                                   five_tb_value=five[field],
                                   five_minus_one_percentage_points=100 * (five[field] - one[field])))
    write_csv(OUT / "early_vs_late_milestones.csv", milestones)
    best = []
    for model in ("hs6_l", "5tb_no_gram"):
        for field in ("five_family_equal", "six_family_equal"):
            eligible = [row for row in points if row["model"] == model and row[field] != ""]
            winner = max(eligible, key=lambda row: row[field])
            best.append(dict(model=model, score=field, complete_checkpoints=len(eligible),
                             best_checkpoint=winner["checkpoint"], best_value=winner[field]))
    write_csv(OUT / "best_fixed_checkpoints.csv", best)
    summary = {"target_counts": EXPECTED, "fixed_comparison_percentage_points": {
        row["metric"]: row["difference_percentage_points"] for row in comparisons},
        "best_fixed_checkpoints": best,
        "duplicate_count": len(duplicates),
        "largest_duplicate_spread": max(row["spread"] for row in duplicates)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# HS6-L 旧协议 1TB 与 5TB 对照\n\n"
        "数据集集合严格取自原 5TB 旧图 checkpoint 22447 的 25 分类、4 回归、4 检索、4 聚类、8 分割、3 检测，共 48 项。"
        "检测属于旧 B4 observation；它与正式 v4 的 B8 检测口径不同。"
        "每项从归档指向的原始 JSON 重算，并以六类等权计算。"
        "回归和检索的重复归档优先使用 2026-09-09 专项 sweep，冲突列在 duplicate_source_audit.csv。\n\n"
        "fixed_checkpoint_comparison.csv 对比 HS6-L 1TB checkpoint 15374 与 5TB no-GRAM checkpoint 22447。"
        "early_vs_late_milestones.csv 另把约 1M 图像的 ck1024/ck975 与后期点按同一旧版指标和数据集对照。"
        "best_fixed_checkpoints.csv 另在各模型有完整数据的 checkpoint 中选同一个 checkpoint 的六类平均最大值。"
        "旧图的每类峰值可能来自不同 checkpoint，不能当作一个模型的单个 checkpoint 分数。"
        "这些旧评测记录标记为历史未认证原始值，适合作回溯核对，不充当正式 v4 测试排名。\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
