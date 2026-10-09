#!/usr/bin/env python3
"""Plot best complete five- and six-family diagnostics for HS6-L data scales."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plot/fig2/hs6_l_1tb_5tb_20tb_best_means_20261008"
SELECTED = ROOT / "plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv"
ONE_TB = ROOT / "outputs/00_reports/v4_subset_scaling_fm14_search_20260924/search_input_evidence.csv"
FIVE_TB = ROOT / "plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/v2_update_20261008/selected29_curve.csv"
TWENTY_TB = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/family_curve.csv"
V4_MODELS = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_models"
RXRX3_ONE_TB = ROOT / "outputs/02_eval_runs/rxrx3_core_formal_all_hs6_ckpts_20260911/models"
TWENTY_PAIRED = ROOT / "outputs/02_eval_runs/v2_20tb_paired_20261006"
TWENTY_FULL = ROOT / "outputs/02_eval_runs/v2_full_v4_20261007"
PROTOCOL = ROOT / "Evaluation Rules/protocol_v4.json"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation")
SIX_FAMILIES = (*FAMILIES, "detection_proxy")
MODELS = ("1TB", "5TB", "20TB")
COLORS = {"1TB": "#D58166", "5TB": "#DCA879", "20TB": "#F5E3B1"}
FIVE_METRICS = {"classification": "macro_f1", "regression": "spearman", "retrieval": "map_at_5",
                "clustering": "nmi", "segmentation": "mDice_E20_primary_last"}
SIX_METRICS = {"classification": {"balanced_accuracy", "macro_auc"},
               "regression": {"r2", "compound_mean_r2"}, "retrieval": {"recall_at_1"},
               "clustering": {"nmi"}, "segmentation": {"mDice_E20", "mDice_E20_last1", "mDice"},
               "detection_proxy": {"patch_f1", "test_patch_f1"}}


def records(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_datasets():
    result = defaultdict(set)
    for row in records(SELECTED):
        result[row["family"]].add(row["dataset"])
    assert {family: len(result[family]) for family in FAMILIES} == {
        "classification": 16, "regression": 2, "retrieval": 4, "clustering": 2, "segmentation": 5}
    return result


def five_family_candidates(selected):
    points = defaultdict(dict)
    evidence = []
    for row in records(ONE_TB):
        if row["model"] != "hs6_l":
            continue
        family, dataset = row["family"], row["dataset"]
        if dataset not in selected[family]:
            continue
        if row["metric"] != FIVE_METRICS[family]:
            raise ValueError(f"1TB metric changed: {row}")
        key = ("1TB", int(row["checkpoint"]))
        cell = (family, dataset)
        if cell in points[key]:
            raise ValueError(f"Duplicate 1TB cell: {row}")
        points[key][cell] = float(row["value"])
        evidence.append({"scope": "selected29", "model": "1TB", "checkpoint": key[1],
                         "family": family, "dataset": dataset, "metric": row["metric"],
                         "value": row["value"], "source": row["source"]})
    five_curve = records(FIVE_TB)
    twenty_curve = records(TWENTY_TB)
    candidates = []
    for (model, step), cells in sorted(points.items()):
        if set(cells) != {(family, dataset) for family in FAMILIES for dataset in selected[family]}:
            raise ValueError(f"Incomplete 1TB selected-29 checkpoint: {step}")
        means = [statistics.mean(cells[family, dataset] for dataset in selected[family]) for family in FAMILIES]
        candidates.append({"scope": "selected29_five_family", "model": model, "checkpoint": step,
                           "mean": statistics.mean(means), "complete_families": 5, "required_families": 5})
    five_tb_methods = {"noGRAM", "GRAM", "global_cls", "global_cls_slow2", "global_cls_w3"}
    for model, source in (("5TB", five_curve), ("20TB", twenty_curve)):
        for row in source:
            if model == "5TB" and (row["model"] not in five_tb_methods or row["family"] != "overall"):
                continue
            if model == "20TB" and row["family"] != "overall":
                continue
            value = float(row["value"]) if row["value"] else math.nan
            complete = (int(row["n"]) == int(row["required"]) if model == "5TB"
                        else row["complete"] == "True")
            if not complete or not math.isfinite(value):
                continue
            candidates.append({"scope": "selected29_five_family", "model": model,
                               "checkpoint": int(row["checkpoint"]), "mean": value,
                               "complete_families": 5, "required_families": 5,
                               "method": row["model"] if model == "5TB" else "route2"})
    counts = {model: sum(row["model"] == model for row in candidates) for model in MODELS}
    if counts != {"1TB": 15, "5TB": 160, "20TB": 34}:
        raise ValueError(f"Unexpected five-family coverage: {counts}")
    return candidates, evidence


def strict_six_datasets():
    protocol = json.loads(PROTOCOL.read_text())
    result = {}
    for family in ("classification", "regression", "retrieval", "segmentation"):
        result[family] = set().union(*(set(protocol[part].get(family, []))
                                       for part in ("tier_a", "tier_b", "union_extension")))
    result["classification"].discard("lc25000")
    result["retrieval"].discard("nct-crc-he-100")
    result["clustering"] = result["retrieval"].copy()
    result["detection_proxy"] = set(protocol["union_extension"]["detection_proxy"])
    expected = {"classification": 24, "regression": 4, "retrieval": 6,
                "clustering": 6, "segmentation": 7, "detection_proxy": 3}
    if {family: len(result[family]) for family in SIX_FAMILIES} != expected:
        raise ValueError("Strict v4 ID dataset list changed")
    return result


def one_tb_rxrx3(step, family):
    path = RXRX3_ONE_TB / f"hs6_l_1tb_ck{step}" / "results.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    test = result.get("tests", {}).get("rxrx3", {})
    checkpoint = Path(result.get("checkpoint", ""))
    expected_checkpoint = ROOT / ("outputs/01_training_runs/"
                                  "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/"
                                  f"ckpt/{step}/checkpoint.pth")
    if (result.get("status") != "VALID_COMPLETE" or result.get("model") != f"hs6_l_1tb_ck{step}"
            or checkpoint.resolve() != expected_checkpoint.resolve() or not checkpoint.is_file()
            or test.get("status") != "FORMAL" or test.get("proxy") is not False
            or test.get("protocol_id") != "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
            or (test.get("n_gallery"), test.get("n_query")) != (734, 734)):
        raise ValueError(f"Invalid formal 1TB RxRx3 result: {path}")
    metric = "recall_at_1" if family == "retrieval" else "nmi"
    return {"value": test[metric], "metric": metric, "provenance": "formal_rxrx3_existing_result",
            "source": str(path)}


def six_family_candidates(expected):
    candidates, evidence = [], []
    for model, filename in (("1TB", "hs6_l.json"), ("5TB", "hs6_l_5tb_no_gram.json")):
        document = json.loads((V4_MODELS / filename).read_text())
        for step_text, checkpoint in document["checkpoints"].items():
            step = int(step_text)
            family_means = {}
            for family in SIX_FAMILIES:
                cells = checkpoint["cells"].get(family, {})
                values = []
                for dataset in sorted(expected[family]):
                    cell_record = cells.get(dataset, {})
                    preferred = cell_record.get("preferred")
                    if preferred is None and model == "1TB" and dataset == "rxrx3-core" and family in ("retrieval", "clustering"):
                        preferred = one_tb_rxrx3(step, family)
                    if preferred is None:
                        continue
                    if preferred["metric"] not in SIX_METRICS[family]:
                        raise ValueError(f"Unexpected six-family metric: {model} ck{step} {family}/{dataset}: {preferred['metric']}")
                    value = float(preferred["value"])
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite score: {model} ck{step} {family}/{dataset}")
                    if family == "segmentation" and preferred["metric"] == "mDice":
                        e20_matches = [observation for observation in cell_record.get("observations", [])
                                       if observation["metric"] == "mDice_E20"
                                       and math.isclose(float(observation["value"]), value, abs_tol=1e-10)]
                        if not e20_matches:
                            raise ValueError(f"Unverified E20 mDice: {model} ck{step} {dataset}")
                    values.append(value)
                    evidence.append({"scope": "v4_list_strict_id50", "model": model, "checkpoint": step,
                                     "family": family, "dataset": dataset, "metric": preferred["metric"],
                                     "value": value, "provenance": preferred["provenance"],
                                     "source": preferred["source"]})
                if len(values) == len(expected[family]):
                    family_means[family] = statistics.mean(values)
            if len(family_means) == 6:
                candidates.append({"scope": "v4_list_strict_id50_six_family", "model": model,
                                   "checkpoint": step, "mean": statistics.mean(family_means.values()),
                                   "complete_families": 6, "required_families": 6})
    if not candidates or {row["model"] for row in candidates} != {"1TB", "5TB"}:
        raise ValueError("No complete six-family checkpoints for both 1TB and 5TB")
    return candidates, evidence


def twenty_six_family_candidates(expected):
    paired = json.loads((TWENTY_PAIRED / "campaign_manifest.json").read_text())
    paired_tasks = defaultdict(list)
    for task in paired["tasks"]:
        if task["asset"]["arm"] == "noGRAM20tb":
            key = (int(task["asset"]["checkpoint_id"]), task["dataset"]["task"],
                   task["dataset"]["dataset"])
            paired_tasks[key].append(task)
    candidates, evidence = [], []

    def paired_value(step, family, dataset):
        task_family = "retrieval" if family == "clustering" else family
        tasks = paired_tasks.get((step, task_family, dataset), [])
        if not tasks:
            return None
        values, paths = [], []
        for task in tasks:
            marker = TWENTY_PAIRED / "_state/done" / f"{task['key']}.json"
            if not marker.is_file() or json.loads(marker.read_text()).get("status") != "VALID_COMPLETE":
                return None
            cell = TWENTY_PAIRED / "cells" / task["key"]
            if family == "segmentation":
                results = sorted(cell.glob(f"results/*/budget20/seed*/{dataset}/{step}/results.json"))
                if len(results) != 3:
                    return None
                values.append(statistics.mean(float(json.loads(p.read_text())["test"]["mDice"])
                                              for p in results))
                paths.extend(results)
            else:
                path = cell / "component_result.json"
                result = json.loads(path.read_text())
                metric = ("macro_auc" if dataset == "chestmnist" else "balanced_accuracy") if family == "classification" else (
                    "r2" if family == "regression" else "recall_at_1" if family == "retrieval" else "nmi")
                if family in ("retrieval", "clustering") and "rows" in result:
                    aggregation = ("global" if family == "retrieval" else
                                   "location" if dataset == "hpa-subcellular" else "global-perturbation")
                    rows = [row for row in result["rows"] if row.get("aggregation") == aggregation]
                    if dataset == "hpa-subcellular" and family == "clustering":
                        rows = [row for row in rows if row.get("n_classes") == 41]
                    if len(rows) != 1:
                        raise ValueError(f"Unexpected retrieval aggregation: ck{step} {dataset}/{family}")
                    result = rows[0]
                values.append(float(result[metric]))
                paths.append(path)
        if family == "segmentation" and dataset == "pannuke" and len(tasks) != 3:
            return None
        return statistics.mean(values), "mDice_E20", paths

    def extension_value(step, family, dataset):
        arm = f"noGRAM20tb_ck{step}"
        if family == "regression":
            key = f"cls__{dataset}__{arm}"
        elif family in ("retrieval", "clustering"):
            key = f"ret__{dataset}__{arm}"
        elif family == "detection_proxy":
            key = f"det__{dataset}__{arm}"
        else:
            key = f"{arm}__segmentation__monuseg__primary-last__formal-static-v1"
        task_path = TWENTY_FULL / "tasks" / f"{key}.json"
        status_path = TWENTY_FULL / "claims" / key / "status.json"
        if not task_path.is_file() or not status_path.is_file() or json.loads(status_path.read_text()).get("state") != "VALID_COMPLETE":
            return None
        task = json.loads(task_path.read_text())
        if family == "segmentation":
            paths = sorted((TWENTY_FULL / "cells" / key).glob(
                f"results/*/budget20/seed*/monuseg/{step}/results.json"))
            if len(paths) != 3:
                return None
            return statistics.mean(float(json.loads(p.read_text())["test"]["mDice"]) for p in paths), "mDice_E20", paths
        path = Path(task["done"]["path"])
        if family == "detection_proxy":
            result = json.loads(path.read_text())
            if dataset == "conic" and result.get("conic_split_protocol") != "official-baseline-fold0-nested-v1":
                raise ValueError(f"Invalid CoNIC split: {path}")
            return float(result["test_patch_f1"]) / 100, "test_patch_f1", [path]
        rows = records(path)
        if len(rows) != 1 or rows[0].get("error"):
            raise ValueError(f"Invalid extension result: {path}")
        if dataset == "rxrx3-core":
            result_path = Path(rows[0]["result_file"])
            result = json.loads(result_path.read_text())
            test = result["tests"]["rxrx3"]
            if test.get("protocol_id") != "crispr-query-guide-plate-disjoint-all-eligible-genes-v1" or test.get("status") != "FORMAL":
                raise ValueError(f"Invalid formal RxRx3 result: {result_path}")
            metric = "recall_at_1" if family == "retrieval" else "nmi"
            return float(test[metric]), metric, [path, result_path]
        metric = "r2" if family == "regression" else "recall_at_1" if family == "retrieval" else "nmi"
        return float(rows[0][metric]), metric, [path]

    for step in sorted({int(row["checkpoint"]) for row in records(TWENTY_TB)
                        if row["family"] == "overall"}):
        means = {}
        for family in SIX_FAMILIES:
            values = []
            for dataset in sorted(expected[family]):
                value = paired_value(step, family, dataset)
                if value is None:
                    value = extension_value(step, family, dataset)
                if value is None:
                    break
                score, metric, paths = value
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite 20TB score: ck{step} {family}/{dataset}")
                values.append(score)
                evidence.append({"scope": "v4_list_strict_id50", "model": "20TB", "checkpoint": step,
                                 "family": family, "dataset": dataset, "metric": metric, "value": score,
                                 "provenance": "validated_v4_20261006_07", "source": ";".join(map(str, paths))})
            if len(values) == len(expected[family]):
                means[family] = statistics.mean(values)
        if len(means) == 6:
            candidates.append({"scope": "v4_list_strict_id50_six_family", "model": "20TB",
                               "checkpoint": step, "mean": statistics.mean(means.values()),
                               "complete_families": 6, "required_families": 6})
    if len(candidates) != 15:
        raise ValueError(f"Expected 15 complete recent 20TB points, got {len(candidates)}")
    return candidates, evidence


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = selected_datasets()
    five, five_evidence = five_family_candidates(selected)
    six_expected = strict_six_datasets()
    six, six_evidence = six_family_candidates(six_expected)
    twenty_six, twenty_evidence = twenty_six_family_candidates(six_expected)
    six += twenty_six
    six_evidence += twenty_evidence
    candidates = five + six
    for row in candidates:
        row.setdefault("method", "noGRAM" if row["model"] == "5TB" else "baseline")
    candidate_counts = {scope: {model: sum(row["scope"] == scope and row["model"] == model
                                      for row in candidates) for model in MODELS}
                        for scope in ("selected29_five_family", "v4_list_strict_id50_six_family")}
    best = []
    for scope in ("selected29_five_family", "v4_list_strict_id50_six_family"):
        for model in MODELS:
            options = [row for row in candidates if row["scope"] == scope and row["model"] == model]
            if options:
                top = max(options, key=lambda row: (row["mean"], -row["checkpoint"]))
                best.append({**top, "status": "OBSERVED"})
            else:
                best.append({"scope": scope, "model": model, "checkpoint": "", "mean": "",
                             "complete_families": 0, "required_families": 6,
                             "status": "NO_COMPLETE_CHECKPOINT"})
    write_csv(OUT / "candidate_checkpoint_means.csv", candidates)
    write_csv(OUT / "best_means.csv", best)
    write_csv(OUT / "five_family_source_evidence.csv", five_evidence)
    write_csv(OUT / "six_family_source_evidence.csv", six_evidence)
    make_plot(best)
    sources = [SELECTED, ONE_TB, FIVE_TB, TWENTY_TB, PROTOCOL,
               V4_MODELS / "hs6_l.json", V4_MODELS / "hs6_l_5tb_no_gram.json"]
    sources.extend(sorted(RXRX3_ONE_TB.glob("hs6_l_1tb_ck*/results.json")))
    sources.extend((TWENTY_PAIRED / "campaign_manifest.json", TWENTY_FULL / "campaign_manifest.json"))
    manifest = {
        "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "five_family_dataset_counts": {family: len(selected[family]) for family in FAMILIES},
        "six_family_dataset_counts": {family: len(six_expected[family]) for family in SIX_FAMILIES},
        "best_rule": "Highest complete equal-family mean at one checkpoint; ties use earliest checkpoint",
        "six_family_scope": "Descriptive v4-list strict ID50, not an official v4 aggregate",
        "six_family_provenance": "Preferred report observations have mixed validated-v3, v4-list-raw and extension provenance; 1TB RxRx3 formal results omitted by the older report are added from their original files",
        "twenty_tb_six_family": "15 recent main-route2 checkpoints have complete v4-list strict ID50 coverage; 19 earlier points still lack extension-family coverage",
        "candidate_checkpoint_counts": candidate_counts,
        "posthoc_test_selection": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    five_scores = {row["model"]: row for row in best if row["scope"] == "selected29_five_family"}
    six_scores = {row["model"]: row for row in best if row["scope"] == "v4_list_strict_id50_six_family"}
    (OUT / "README.md").write_text(
        "# HS6-L 1TB / 5TB / 20TB best complete means\n\n"
        "`best_means.csv` gives the highest complete mean at a single checkpoint; "
        "it does not average separate family peaks. The five-family figure uses the same "
        "post hoc selected 29 datasets as the earlier 5TB/20TB curve (16/2/4/2/5). "
        "The 1TB source is the validated v3 cell ledger; the five-family 5TB winner "
        "is selected across no-GRAM, GRAM and Adaptive-v2 variants, including hxw no-GRAM "
        "continuation; 20TB is the route2 v4 shared-component campaign.\n\n"
        "The six-family figure uses the v4-list strict ID50 subset (24/4/6/6/7/3), "
        "excluding provisional LC25000 classification and low-N NCT100 retrieval/clustering. "
        "Metrics differ from the five-family figure: BA/chest AUC, R2, Recall@1, NMI, "
        "E20 mDice and detection-proxy patch F1. The source report mixes validated-v3, "
        "v4-list-raw and extension observations, so this is a descriptive diagnostic, "
        "not an official full-v4 aggregate. The 20TB values combine the validated "
        "paired shared components from October 6 with the October 7 v4 extensions. "
        "Only the 15 later route2 checkpoints have complete strict-ID50 coverage; "
        "19 earlier checkpoints currently have detection retests but still lack other extensions.\n\n"
        f"Five-family best: 1TB ck{five_scores['1TB']['checkpoint']} {five_scores['1TB']['mean']:.6f}; "
        f"5TB ck{five_scores['5TB']['checkpoint']} {five_scores['5TB']['mean']:.6f}; "
        f"20TB ck{five_scores['20TB']['checkpoint']} {five_scores['20TB']['mean']:.6f}.\n"
        f"Six-family diagnostic best: 1TB ck{six_scores['1TB']['checkpoint']} {six_scores['1TB']['mean']:.6f}; "
        f"5TB ck{six_scores['5TB']['checkpoint']} {six_scores['5TB']['mean']:.6f}; "
        f"20TB ck{six_scores['20TB']['checkpoint']} {six_scores['20TB']['mean']:.6f}.\n\n"
        f"Complete candidate checkpoints: five-family 1TB 15, 5TB {candidate_counts['selected29_five_family']['5TB']}, 20TB 34; "
        f"six-family 1TB {candidate_counts['v4_list_strict_id50_six_family']['1TB']}, "
        f"5TB {candidate_counts['v4_list_strict_id50_six_family']['5TB']}, "
        f"20TB {candidate_counts['v4_list_strict_id50_six_family']['20TB']}. "
        "The old 1TB report omitted four existing formal RxRx3 results, now included here.\n\n"
        "The two figures use different dataset lists and metrics. Comparing their "
        "heights to each other does not measure a model improvement. Source cells, "
        "candidate checkpoints and SHA-256 hashes are recorded beside the figures. "
        "The figures have percentage axes and omit checkpoint labels; checkpoint "
        "identifiers and exact means remain in `best_means.csv`.\n"
    )
    print(OUT)
    for row in best:
        print(row)


def make_plot(best):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    plots = (
        ("selected29_five_family", "best_five_family", "Best five-family mean",
         "Selected 29 datasets", 74.0, 76.0, (74.0, 74.5, 75.0, 75.5, 76.0)),
        ("v4_list_strict_id50_six_family", "best_six_family", "Best six-family mean",
         "v4-list strict ID50 diagnostic", 71.5, 72.8, (71.5, 72.0, 72.5)),
    )
    for scope, filename, title, subtitle, lower, upper, ticks in plots:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        for index, model in enumerate(MODELS):
            row = next(row for row in best if row["scope"] == scope and row["model"] == model)
            if row["status"] != "OBSERVED":
                ax.text(index, (lower + upper) / 2, "N/A\nincomplete\nevaluation", ha="center", va="center",
                        fontsize=10, color="#6B7379")
                continue
            mean = 100 * float(row["mean"])
            bar = ax.bar(index, mean - lower, bottom=lower, width=0.48,
                         color=COLORS[model], edgecolor="#8A6B5C", linewidth=0.8, zorder=3)
            ax.bar_label(bar, labels=[f"{mean:.2f}%"], padding=5, fontsize=11, fontweight="bold")
        ax.set_xticks(range(3), MODELS)
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(lower, upper)
        ax.set_yticks(ticks, [f"{tick:g}%" for tick in ticks])
        ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=16)
        ax.set_ylabel("Mean score (%)")
        ax.grid(axis="y", color="#DFE4E7", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if scope == "selected29_five_family":
            fig.text(0.5, 0.035, "5TB: Adaptive-v2 fixed CLS, w=3", ha="center",
                     fontsize=9, color="#493D36")
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.82)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"{filename}.{extension}", dpi=200, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    build()
