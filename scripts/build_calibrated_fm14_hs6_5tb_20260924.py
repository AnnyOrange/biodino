#!/usr/bin/env python3
"""Audit and plot the common validated 47-cell FM14 / HS6-L comparison."""

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
INPUT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
OUT = ROOT / "outputs/00_reports/hs0_hs6_5tb_fm14_calibrated_20260924"
L_MODELS = ("hs0_l", "hs6_l", "5tb_no_gram", "5tb_gram12687")
FM = tuple("fm_" + x for x in ("bioclip", "conch", "cytoimagenet", "cytoself", "dinov2",
                                 "gigapath", "hoptimus0", "jump_cp", "mae", "pe", "phikon2",
                                 "siglip2", "uni", "virchow2"))
MODELS = L_MODELS + FM
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
COLORS = {"hs0_l": "#7256a4", "hs6_l": "#3263a8", "5tb_no_gram": "#139a79",
          "5tb_gram12687": "#b95150"}


def rows(path):
    return list(csv.DictReader(path.open()))


def save(path, data, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def raw_score(row):
    fam, ds, metric = row["family"], row["dataset"], row["metric"]
    paths = [Path(p) for p in row["source"].split(";")]
    objects = [json.loads(p.read_text()) for p in paths]
    if fam == "segmentation":
        assert len(paths) == (9 if ds == "pannuke" else 3)
        assert all(x.get("_meta", {}).get("probe_epochs") == 20 for x in objects)
        return sum(x["test"]["mDice"] for x in objects) / len(objects)
    obj = objects[0]
    if fam == "regression" and ds == "bbbc013":
        return (obj["ly294002_r2"] + obj["wortmannin_r2"]) / 2
    if fam in ("retrieval", "clustering"):
        candidates = obj.get("rows", [obj])
        if ds == "hpa-subcellular":
            candidates = [x for x in candidates if x.get("aggregation") == ("global" if fam == "retrieval" else "location")
                          and (fam == "retrieval" or x.get("n_classes") == 41)]
        elif ds == "rxrx1-cross":
            candidates = [x for x in candidates if x.get("aggregation") == ("global" if fam == "retrieval" else "global-perturbation")]
        else:
            candidates = [x for x in candidates if x.get("aggregation") in ("class", "global") and metric in x]
        assert len(candidates) == 1, (row["model"], fam, ds, candidates)
        return candidates[0][metric]
    return obj[metric]


def source_report(path):
    parts = path.parts
    if "cells" not in parts:
        return "NO_CELL_ROOT"
    i = parts.index("cells")
    p = Path(*parts[:i + 2]) / "validation_report.json"
    if not p.exists():
        return "REPORT_NOT_FOUND"
    obj = json.loads(p.read_text())
    return obj.get("status", "STATUS_MISSING")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = [r for r in rows(INPUT / "validated_scores.csv") if r["model"] in MODELS and r["family"] in FAMILIES]
    by_model = defaultdict(set)
    by_point = defaultdict(dict)
    for r in evidence:
        key = r["family"], r["dataset"]
        by_model[r["model"]].add(key)
        by_point[r["model"], r["checkpoint"]][key] = r
    core = sorted(set.intersection(*(by_model[m] for m in MODELS)))
    assert len(core) == 47, (len(core), Counter(f for f, _ in core))
    save(OUT / "common_47_cells.csv", [dict(family=f, dataset=d,
                                            section="V4_EXTENSION_PROVISIONAL" if (f, d) == ("classification", "lc25000")
                                            else "V4_EXTENSION_LOW_N" if d == "nct-crc-he-100"
                                            else "V4_EXTENSION_PROXY" if d in ("conic-cell-count", "livecell-cell-count")
                                            else "V4_EXTENSION" if d == "lc25000" else "V3_SHARED") for f, d in core],
         ["family", "dataset", "section"])
    model_rows = []
    for model in MODELS:
        for (m, ck), point in by_point.items():
            if m != model or not set(core) <= set(point):
                continue
            means = {f: sum(float(point[f, d]["value"]) for ff, d in core if ff == f) /
                     sum(ff == f for ff, _ in core) for f in FAMILIES}
            model_rows.append(dict(model=model, checkpoint=ck, score_47=sum(means.values()) / 5,
                                   **{f + "_mean": means[f] for f in FAMILIES}))
    assert all(any(r["model"] == m for r in model_rows) for m in MODELS)
    save(OUT / "all_complete_checkpoint_scores.csv", model_rows,
         ["model", "checkpoint", "score_47"] + [f + "_mean" for f in FAMILIES])
    best = {m: max((r for r in model_rows if r["model"] == m), key=lambda r: (r["score_47"],
              -int(r["checkpoint"]) if r["checkpoint"].isdigit() else 0)) for m in MODELS}
    save(OUT / "selected_models_47.csv", [best[m] for m in MODELS],
         ["model", "checkpoint", "score_47"] + [f + "_mean" for f in FAMILIES])

    selected = []
    checked = []
    for model in MODELS:
        ck = best[model]["checkpoint"]
        for fam, ds in core:
            original = by_point[model, ck][fam, ds]
            value = float(original["value"])
            raw = raw_score(original)
            assert math.isfinite(raw) and abs(raw - value) < 1e-8, (model, ck, fam, ds, value, raw)
            statuses = sorted(set(source_report(Path(p)) for p in original["source"].split(";")))
            selected.append(dict(model=model, checkpoint=ck, family=fam, dataset=ds,
                                 metric=original["metric"], value=value, protocol=original["protocol"],
                                 source=original["source"], validation_status=";".join(statuses)))
            checked.append((model, fam, ds, statuses))
    save(OUT / "calibrated_scores_long.csv", selected,
         ["model", "checkpoint", "family", "dataset", "metric", "value", "protocol", "source", "validation_status"])
    assert len(selected) == 47 * 18
    for fam, ds in core:
        cells = [r for r in selected if r["family"] == fam and r["dataset"] == ds]
        assert len(cells) == len(MODELS)
        assert len({r["metric"] for r in cells}) == 1, (fam, ds, "metric mismatch")
        assert len({r["protocol"] for r in cells}) == 1, (fam, ds, "protocol mismatch")
    sig = defaultdict(dict)
    for row in selected:
        if row["model"] not in L_MODELS:
            continue
        source = Path(row["source"].split(";")[0])
        obj = json.loads(source.read_text())
        if row["family"] == "segmentation":
            signature = {"probe_epochs": obj.get("_meta", {}).get("probe_epochs"),
                         "probe_batch_size": obj.get("_meta", {}).get("probe_batch_size")}
        else:
            signature = {k: obj.get(k) for k in ("split", "probe", "batch_size", "seed", "image_size",
                                                    "n_train", "n_test", "n_samples", "n_query", "n_gallery") if k in obj}
        sig[row["family"], row["dataset"]][row["model"]] = signature
    audit = []
    for (fam, ds), entries in sig.items():
        fields = set().union(*(set(v) for v in entries.values()))
        for field in fields:
            vals = {m: entries[m].get(field) for m in L_MODELS}
            audit.append(dict(family=fam, dataset=ds, field=field,
                              **vals, status="MATCH" if len({json.dumps(v, sort_keys=True) for v in vals.values()}) == 1 else "DIFFERENT_OR_MISSING"))
    save(OUT / "protocol_signature_audit.csv", audit,
         ["family", "dataset", "field"] + list(L_MODELS) + ["status"])

    # Show strict v3 and the comparable v3 plus legacy-extension subset.
    shared40 = {(r["family"], r["dataset"]) for r in rows(INPUT / "common_cells.csv")}
    assert len(shared40) == 40 and shared40 <= set(core)
    scopes = []
    for model in MODELS:
        for label, keys in (("v3_shared_40", shared40), ("v3_plus_legacy_47", set(core))):
            options = []
            for (m, ck), point in by_point.items():
                if m != model or not keys <= set(point):
                    continue
                means = {f: sum(float(point[f, d]["value"]) for ff, d in keys if ff == f) /
                         sum(ff == f for ff, _ in keys) for f in FAMILIES}
                options.append(dict(model=model, checkpoint=ck, scope=label,
                                    score=sum(means.values()) / 5,
                                    **{f + "_mean": means[f] for f in FAMILIES}))
            scopes.append(max(options, key=lambda r: (r["score"],
                          -int(r["checkpoint"]) if r["checkpoint"].isdigit() else 0)))
    save(OUT / "two_scope_comparison.csv", scopes,
         ["model", "checkpoint", "scope", "score"] + [f + "_mean" for f in FAMILIES])
    sensitivity = []
    for model in MODELS:
        options = []
        for (m, ck), point in by_point.items():
            if m == model and set(core) <= set(point):
                options.append(dict(model=model, checkpoint=ck,
                                    score=sum(float(point[key]["value"]) for key in core) / len(core)))
        dataset_best = max(options, key=lambda r: (r["score"],
                           -int(r["checkpoint"]) if r["checkpoint"].isdigit() else 0))
        for scope in ("v3_shared_40", "v3_plus_legacy_47"):
            r = next(x for x in scopes if x["model"] == model and x["scope"] == scope)
            sensitivity.append(dict(model=model, weighting="five_family_equal", scope=scope,
                                    checkpoint=r["checkpoint"], score=r["score"]))
        sensitivity.append(dict(model=model, weighting="dataset_equal", scope="v3_plus_legacy_47",
                                checkpoint=dataset_best["checkpoint"], score=dataset_best["score"]))
    save(OUT / "aggregation_sensitivity.csv", sensitivity,
         ["model", "weighting", "scope", "checkpoint", "score"])
    wide = []
    for fam, ds in core:
        line = dict(family=fam, dataset=ds, metric=by_point["hs0_l", best["hs0_l"]["checkpoint"]][fam, ds]["metric"])
        for model in MODELS:
            line[model] = float(by_point[model, best[model]["checkpoint"]][fam, ds]["value"])
        line["fm14_best"] = max(line[m] for m in FM)
        line["fm14_best_model"] = max(FM, key=lambda m: line[m])
        line["delta_5tb_no_gram_vs_1tb_l"] = line["5tb_no_gram"] - line["hs6_l"]
        wide.append(line)
    save(OUT / "per_dataset_47_all_fm14.csv", wide,
         ["family", "dataset", "metric"] + list(MODELS) +
         ["fm14_best", "fm14_best_model", "delta_5tb_no_gram_vs_1tb_l"])
    pairwise = []
    for newer, older in (("hs6_l", "hs0_l"), ("5tb_no_gram", "hs6_l"),
                         ("5tb_gram12687", "hs6_l")):
        for family in list(FAMILIES) + ["all"]:
            subset = [r for r in wide if family == "all" or r["family"] == family]
            deltas = [r[newer] - r[older] for r in subset]
            pairwise.append(dict(newer=newer, older=older, family=family, cells=len(deltas),
                                 wins=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas),
                                 losses=sum(d < 0 for d in deltas), mean_delta=sum(deltas) / len(deltas)))
    save(OUT / "pairwise_family_47.csv", pairwise,
         ["newer", "older", "family", "cells", "wins", "ties", "losses", "mean_delta"])

    registry = json.loads((ROOT / "Evaluation Rules/protocol_v4.json").read_text())
    targets = {
        "classification": registry["tier_a"]["classification"] + registry["tier_b"]["classification"] + registry["union_extension"]["classification"],
        "regression": registry["tier_a"]["regression"] + registry["tier_b"]["regression"] + registry["union_extension"]["regression"],
        "retrieval": registry["tier_a"]["retrieval"] + registry["tier_b"]["retrieval"] + registry["union_extension"]["retrieval"],
        "clustering": registry["tier_a"]["retrieval"] + registry["tier_b"]["retrieval"] + registry["union_extension"]["clustering"],
        "segmentation": registry["tier_a"]["segmentation"] + registry["tier_b"]["segmentation"],
        "detection_proxy": registry["union_extension"]["detection_proxy"],
        "cell_tracking": ["ctc"], "ood": ["xray", "cryo"]}
    assert sum(map(len, targets.values())) == 56
    all_scores = [r for r in rows(INPUT / "validated_scores.csv") if r["model"] in MODELS]
    score_lookup = {(r["model"], r["checkpoint"], r["family"], r["dataset"]): r for r in all_scores}
    inventory = []
    for model in MODELS:
        ck = best[model]["checkpoint"]
        for family, datasets in targets.items():
            for dataset in datasets:
                item = score_lookup.get((model, ck, family, dataset))
                status = ("BLOCKED_PROTOCOL" if family in ("cell_tracking", "ood") else
                          "MISSING_MATCHED_RESULT" if item is None else
                          "B8_PROXY_OBSERVATION" if family == "detection_proxy" else
                          "VALIDATED_V3" if item["protocol"] == "v3" else
                          "VALIDATED_LEGACY_EXTENSION")
                inventory.append(dict(model=model, checkpoint=ck, family=family, dataset=dataset,
                                      status=status, value=item["value"] if item else "",
                                      source=item["source"] if item else ""))
    save(OUT / "v4_target_coverage_selected_models.csv", inventory,
         ["model", "checkpoint", "family", "dataset", "status", "value", "source"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ordered = list(L_MODELS) + sorted(FM, key=lambda m: -best[m]["score_47"])
    labels = [m.replace("fm_", "FM ").replace("_", " ") for m in ordered]
    for ax, name in zip(axes.flat, list(FAMILIES) + ["score_47"]):
        vals = [best[m][name if name == "score_47" else name + "_mean"] for m in ordered]
        ax.barh(range(len(ordered)), vals, color=[COLORS.get(m, "#70879d") for m in ordered])
        ax.set_yticks(range(len(ordered)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title("Five-family mean" if name == "score_47" else name.title())
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("47 comparable v3 + legacy cells: one fixed checkpoint per model; FM14 shown in full")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fm14_hs6_l_47cell_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    xx = np.arange(len(L_MODELS))
    definitions = (("v3 shared 40, task equal", "five_family_equal", "v3_shared_40", "#52799f"),
                   ("v3 + legacy 47, task equal", "five_family_equal", "v3_plus_legacy_47", "#169977"),
                   ("v3 + legacy 47, dataset equal", "dataset_equal", "v3_plus_legacy_47", "#d89046"))
    for i, (label, weight, scope, color) in enumerate(definitions):
        vals = [next(r["score"] for r in sensitivity if r["model"] == m and
                     r["weighting"] == weight and r["scope"] == scope) for m in L_MODELS]
        bars = ax.bar(xx + (i - 1) * .26, vals, width=.26, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=3)
    ax.set_xticks(xx, ["HS0-L 1TB", "HS6-L 1TB", "HS6-L 5TB\nno-GRAM", "HS6-L 5TB\nGRAM"])
    ax.set_ylabel("Mean score (weighting shown in legend)")
    ax.set_ylim(0, .80)
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    ax.set_title("The 1TB/5TB ordering changes with inventory and weighting")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"v3_40_vs_union_47_scope.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = {"models": len(MODELS), "fm14": len(FM), "common_cells": len(core),
               "common_by_family": dict(Counter(f for f, _ in core)),
               "selected": {m: {"checkpoint": best[m]["checkpoint"], "score_47": best[m]["score_47"]} for m in MODELS},
               "v3_shared_cells": 40, "source_rows_verified_against_raw": len(selected),
               "validation_statuses": dict(Counter(x[3][0] if len(x[3]) == 1 else ";".join(x[3]) for x in checked)),
               "protocol_signature_differences": sum(r["status"] != "MATCH" for r in audit),
               "selected_v4_inventory_statuses": {m: dict(Counter(r["status"] for r in inventory if r["model"] == m)) for m in MODELS},
               "aggregation_sensitivity": {m: {r["scope"] + "/" + r["weighting"]:
                     {"checkpoint": r["checkpoint"], "score": r["score"]}
                     for r in sensitivity if r["model"] == m} for m in L_MODELS},
               "pairwise_47": {r["newer"] + "_vs_" + r["older"]: r for r in pairwise if r["family"] == "all"},
               "fm14_best": max(FM, key=lambda m: best[m]["score_47"])}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "README.md").write_text(
        "# 校准后的 FM14 / HS0-L / HS6-L 1TB、5TB 比较\n\n"
        "**口径提示：此报告的47项是可用的 v3/legacy 并集，不是 2026-09-14 的"
        "旧协议 raw macro 六任务曲线。**旧图复刻和来源审计见"
        " `../hs6_l5_5tb_old_macro_reproduction_20260924/`。\n\n"
        "从每个模型均有结果的 47 个非 detection cell 取交集：分类25、回归4、检索6、"
        "聚类6、分割6。它是 v3 共享40项加7项 legacy extension，**不是完整 v4，也不是旧版 raw macro 图的口径**："
        "RxRx3、MoNuSeg、CTC、OOD 尚未进入本表，detection B8 覆盖不齐而单列。"
        "LC25000 分类为 provisional split；NCT100 为 99 样本低 N。\n\n"
        "每个模型先在 47 个 cell 上选一份固定 checkpoint，先对每个任务内数据集等权，"
        "再对五个任务等权。选点使用 test 分数，是回顾性诊断。FM14 无轨迹。"
        "`per_dataset_47_all_fm14.csv` 列出 FM14 全部逐项结果与 HS0/HS6 结果；"
        "`calibrated_scores_long.csv` 保留指标、协议、来源和验证状态。"
        "`aggregation_sensitivity.csv` 将 v3 共享40项、并集47项及数据集等权结果分开列出，避免混用。"
        "`pairwise_family_47.csv` 列出逐任务胜负和均值差。"
        "`protocol_signature_audit.csv` 和 summary.json 记录结果源文件与协议字段复核。"
        "`v4_target_coverage_selected_models.csv` 将56项完整目标逐项列出，明确缺失和正式阻断。\n\n"
        "47项五任务等权的固定点：HS0-L ck8199、HS6-L 1TB ck15374、HS6-L 5TB no-GRAM ck22447、"
        "5TB GRAM ck26839。该口径下前三者综合分依次上升；若47项数据集直接等权，"
        "HS6-L 1TB 仍高于5TB no-GRAM，因此严格单调不是稳健结论。"
        "5TB no-GRAM 对1TB仅在47项中的18项领先，29项落后。"
        "FM14 中 CONCH 的五任务等权均值仍高于 5TB no-GRAM；不能声称 HS6 在所有任务或所有 FM 上全面领先。\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
