#!/usr/bin/env python3
"""Rebuild the 2026-09-14 5TB raw task curves from old-protocol archive links."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import plot_hs6_l5_5tb_task_peak_curves as curve


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/02_eval_runs/old_v3_protocol_union/results.csv"
ORIGINAL = ROOT / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914"
OUT = ROOT / "outputs/00_reports/hs6_l5_5tb_old_macro_reproduction_20260924"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def score(row):
    obj = json.loads(Path(row["source"]).read_text())
    family = row["family"]
    if family == "classification":
        if obj.get("task") == "multilabel_classification":
            keys, metric = ("macro_auc", "macro_auroc", "auroc"), "macro_auc"
        else:
            keys, metric = ("balanced_accuracy", "accuracy", "macro_f1"), "balanced_accuracy"
        value = next((curve.finite_number(obj, key) for key in keys
                      if curve.finite_number(obj, key) is not None), None)
    elif family == "regression":
        metric, value = "r2", curve.finite_number(obj, "r2")
    elif family == "retrieval":
        metric, value = "recall_at_1", curve.finite_number(obj, "recall_at_1")
    elif family == "clustering":
        metric, value = "nmi", curve.finite_number(obj, "nmi")
    elif family == "segmentation":
        metric, value = "mDice", curve.finite_number(obj["test"], "mDice")
    else:
        raise ValueError(family)
    assert value is not None and math.isfinite(value), row
    return f"{row['dataset']}:{metric}", value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ORIGINAL / "input_manifest.json").read_text())
    support = manifest["checkpoints"]
    assert len(support) == 49
    archived = [row for row in read_csv(ARCHIVE)
                if row["model"] == "5tb_no_gram" and row["protocol"] == "old"
                and int(row["checkpoint"]) in support]
    observations = [row for row in read_csv(ARCHIVE)
                    if row["model"] == "5tb_no_gram" and row["protocol"] == "old_observation"
                    and row["family"] == "detection" and int(row["checkpoint"]) in support]
    assert len(archived) == 49 * 45
    assert len(observations) == 49 * 3
    by_checkpoint = defaultdict(list)
    for row in archived:
        by_checkpoint[int(row["checkpoint"])].append(row)
    detection_by_checkpoint = defaultdict(list)
    for row in observations:
        detection_by_checkpoint[int(row["checkpoint"])].append(row)
    collected = {}
    evidence = []
    for checkpoint in support:
        assert len(by_checkpoint[checkpoint]) == 45
        families = defaultdict(dict)
        for row in by_checkpoint[checkpoint]:
            key, value = score(row)
            assert key not in families[row["family"]], (checkpoint, row["family"], key)
            families[row["family"]][key] = value
            evidence.append(dict(checkpoint=checkpoint, family=row["family"], metric_key=key,
                                 value=value, evidence="old_archive_source", source=row["source"]))
        assert {family: len(families[family]) for family in families} == {
            "classification": 25, "regression": 4, "retrieval": 4,
            "clustering": 4, "segmentation": 8}
        # Detection is indexed separately as old_observation, not old.
        detection = {}
        for row in detection_by_checkpoint[checkpoint]:
            path = Path(row["source"])
            obj = json.loads(path.read_text())
            dataset = str(obj.get("dataset", row["dataset"]))
            value = curve.finite_number(obj, "test_patch_f1")
            if value is None:
                value = curve.finite_number(obj, "val_patch_f1")
            if value is not None and value > 1:
                value /= 100
            assert value is not None
            key = f"{dataset}:patch_f1"
            assert key not in detection
            detection[key] = value
            evidence.append(dict(checkpoint=checkpoint, family="detection", metric_key=key,
                                 value=value, evidence="old_archive_observation",
                                 source=str(path)))
        assert len(detection) == 3
        families["detection"] = detection
        collected[checkpoint] = dict(families)
    rows, peaks = curve.task_curves(support, collected)
    original = {(r["task_family"], int(r["checkpoint"])): r
                for r in read_csv(ORIGINAL / "task_family_curve.csv")}
    audit = []
    for row in rows:
        key = row["task_family"], row["checkpoint"]
        reference = original[key]
        raw_delta = row["raw_macro_mean"] - float(reference["raw_macro_mean"])
        rank_delta = row["mean_percentile_rank"] - float(reference["mean_percentile_rank"])
        assert row["metrics_covered"] == int(reference["metrics_covered"])
        assert abs(raw_delta) < 1e-12 and abs(rank_delta) < 1e-12, (key, raw_delta, rank_delta)
        audit.append(dict(task_family=key[0], checkpoint=key[1], metrics_covered=row["metrics_covered"],
                          reproduced_raw_macro_mean=row["raw_macro_mean"],
                          original_raw_macro_mean=reference["raw_macro_mean"], raw_delta=raw_delta,
                          reproduced_mean_percentile_rank=row["mean_percentile_rank"],
                          original_mean_percentile_rank=reference["mean_percentile_rank"], rank_delta=rank_delta))
    curve.write_csv(OUT / "archive_reconstructed_curve.csv", rows)
    curve.write_csv(OUT / "archive_reconstructed_peaks.csv", peaks)
    curve.write_csv(OUT / "archive_source_values.csv", evidence)
    curve.write_csv(OUT / "reproduction_audit.csv", audit)
    legacy47_path = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_calibrated_20260924/calibrated_scores_long.csv"
    legacy47 = [row for row in read_csv(legacy47_path)
                if row["model"] == "5tb_no_gram" and row["checkpoint"] == "22447"]
    assert len(legacy47) == 47
    slice_summary = []
    for family in curve.TASKS:
        old_row = next(row for row in rows if row["checkpoint"] == 22447 and row["task_family"] == family)
        slice_summary.append(dict(scope="historical_old", family=family,
                                  cells=old_row["metrics_covered"], raw_macro_mean=old_row["raw_macro_mean"],
                                  note="B4 observation" if family == "detection" else "historical recipe"))
        vals = [float(row["value"]) for row in legacy47 if row["family"] == family]
        if vals:
            slice_summary.append(dict(scope="v3_plus_legacy_47", family=family,
                                      cells=len(vals), raw_macro_mean=sum(vals) / len(vals),
                                      note="v3 recipe plus legacy extension; no detection"))
    curve.write_csv(OUT / "old_vs_v3_legacy_5tb_ck22447.csv", slice_summary)
    target = {(r["family"], r["metric_key"].split(":")[0]) for r in evidence
              if r["checkpoint"] == support[0]}
    assert len(target) == 48
    model_points = {"hs0_l": "8199", "hs6_l": "15374", "5tb_no_gram": "22447"}
    model_points.update({f"fm_{name}": "pretrained" for name in (
        "bioclip", "conch", "cytoimagenet", "cytoself", "dinov2", "gigapath",
        "hoptimus0", "jump_cp", "mae", "pe", "phikon2", "siglip2", "uni", "virchow2")})
    all_archive = read_csv(ARCHIVE)
    available = defaultdict(list)
    for item in all_archive:
        if item["protocol"] in ("old", "old_observation") and item["model"] in model_points and item["checkpoint"] == model_points[item["model"]]:
            available[item["model"], item["family"], item["dataset"]].append(item)
    coverage = []
    for model, ck in model_points.items():
        for family, dataset in sorted(target):
            source_rows = available[model, family, dataset]
            if not source_rows:
                status = "MISSING_IN_OLD_ARCHIVE"
            elif any(r["protocol"] == "old_observation" for r in source_rows):
                status = "HISTORICAL_OBSERVATION_ARCHIVED"
            elif any(r["evidence_status"] == "HISTORICAL_MISSING_OR_UNRESOLVED" for r in source_rows):
                status = "HISTORICAL_UNRESOLVED"
            elif any(r["evidence_status"] == "HISTORICAL_SUMMARY_UNCERTIFIED" for r in source_rows):
                status = "HISTORICAL_SUMMARY_UNCERTIFIED"
            else:
                status = "HISTORICAL_RAW_UNCERTIFIED"
            coverage.append(dict(model=model, checkpoint=ck, family=family, dataset=dataset,
                                 status=status, archive_sources=";".join(r["source"] for r in source_rows)))
    curve.write_csv(OUT / "old_comparison_48cell_archive_coverage.csv", coverage)
    curve.plot_raw(rows, peaks, tuple(manifest["anchor_checkpoints"]), manifest["endpoint"], OUT)
    result = {
        "reference": str(ORIGINAL / "hs6_l5_5tb_task_family_raw_macro_curves.png"),
        "old_archive_non_detection_rows": len(archived),
        "old_archive_detection_observation_rows": len(observations),
        "checkpoint_count": len(support),
        "curve_rows_checked": len(audit),
        "max_abs_raw_delta": max(abs(r["raw_delta"]) for r in audit),
        "max_abs_rank_delta": max(abs(r["rank_delta"]) for r in audit),
        "raw_peaks": {r["task_family"]: r["raw_peak_checkpoint"] for r in peaks},
        "evidence_statuses": dict(Counter(r["evidence"] for r in evidence)),
        "old_comparison_archive_coverage": {
            model: dict(Counter(r["status"] for r in coverage if r["model"] == model))
            for model in model_points},
    }
    (OUT / "audit_summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "README.md").write_text(
        "# 5TB 旧协议 raw macro 曲线复刻\n\n"
        "使用 2026-09-14 原图锁定的 49 个完整 checkpoint。"
        "旧协议归档每点含分类25、回归4、检索4、聚类4、分割8项，"
        "另有单列的 `old_observation` detection 3项。全部从归档 `source`"
        " 指向的原始 JSON 重算，逐项来源见 `archive_source_values.csv`。\n\n"
        "294 个曲线点的 raw macro 与 percentile rank 已逐点对照原图的"
        " `task_family_curve.csv`，绝对误差均小于 1e-12。"
        "`old_comparison_48cell_archive_coverage.csv` 逐项记录 HS0-L、HS6-L 1TB、"
        "5TB 和 FM14 对齐该旧版 48 项目标的归档覆盖；缺项不能靠 v3 结果填充。"
        "`old_vs_v3_legacy_5tb_ck22447.csv` 在同一5TB checkpoint 上直接展示"
        "旧版48项和先前报告的 v3+legacy 47项为何不同。"
        "这是旧版描述性 test 曲线复刻。v4 目标中的正式分割只有7项且采用 v3 recipe；"
        "v4 detection proxy 要求 B8，原图 detection 是 B4。"
        "因此并集证据可以按旧标签复刻原图，不能把原图当作正式 v4 aggregate。\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
