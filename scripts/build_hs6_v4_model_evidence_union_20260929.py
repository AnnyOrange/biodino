#!/usr/bin/env python3
"""Consolidate HS0/HS6, 5TB L/H+, and FM14 v4-list evidence without relabeling it."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/02_eval_runs"
REPORT = ROOT / "outputs/00_reports"
OUT = REPORT / "hs0_hs6_1tb_5tb_fm14_20260924"
PROTOCOL = ROOT / "Evaluation Rules/protocol_v4.json"
VALIDATED = REPORT / "hs0_hs6_1tb_5tb_fm14_20260924/validated_scores.csv"
SELECTED = REPORT / "hs0_hs6_5tb_fm14_calibrated_20260924/v4_target_coverage_selected_models.csv"
FOUR = EVAL / "hs6_l_1tb_5tb_1pb_100tb_v4_completion_20260924/V4_COVERAGE.csv"
METHOD = REPORT / "deepcad_method_20260927/v4_six_tasks_20260929/CELLS.csv"
METHOD_CARRIED = REPORT / "hs0_hs6_1tb_5tb_fm14_20260924/v4_selected_carried_from_20260929_build.csv"
STRICT = REPORT / "hs6_5tb_l_hplus_v4_curves_20260929/cells.csv"
RELAXED = REPORT / "hs6_5tb_l_hplus_relaxed_curves_20260929/cells_with_provenance.csv"
TRACKS = REPORT / "hs6_5tb_trajectory_union_20260929/checkpoints.csv"
SELECTIVE = EVAL / "hs6_l5_selective_retention_v4_20260923/V4_INVENTORY.json"
L_TAIL = EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260924/l_checkpoint_task_inventory.json"
HPLUS = EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260929/hplus_checkpoint_task_inventory.json"
OLD_INDEX = EVAL / "old_v3_protocol_union/results.csv"
# MoNuSeg official train30 / extra7 val / test14 (user instruction 2026-09-29). Only these results
# may be preferred for MoNuSeg; the 24/6 seed split and legacy 37-pool val7 results are kept as
# superseded alternatives.  Remote sites are mirrored by scripts/sync_monuseg_remote_sites_20260930.sh.
MONUSEG_SPLIT = "monuseg2018-train30-extra7val-test14-v1"
MONUSEG_CAMPAIGNS = [EVAL / "monuseg_train30val7_test14_retest_20260929",
                     EVAL / "monuseg_train30val7_test14_fm14_20260930",
                     *sorted((EVAL / "monuseg_train30val7_test14_remote_sites_20260930").glob("*/campaign"))]
MONUSEG_PROVENANCE = "monuseg_30_7_14_validated"
# CoNIC detection written before commit 8529307 (2026-09-09) used the legacy random patch-level
# 80/10/10 split: every test patch shares its source image with training patches.  Its training
# pos_weight (1.3915689) differs from the official baseline fold-0 split (1.3762682), which is how
# results without a recorded split field are classified.  Such values are kept but never preferred.
CONIC_OFFICIAL = "official-baseline-fold0-nested-v1"
CONIC_POS_WEIGHT = {"official": 1.3762682128, "legacy-random": 1.3915689403}
DETECTION_FILL = [EVAL / "detection_b8_official_conic_fill_20260930/nfs",
                  *sorted((EVAL / "detection_b8_official_conic_fill_20260930/sites").glob("*"))]
DETECTION_FILL_PROVENANCE = "v4_detection_b8_fill_validated"
FAMILIES = ("classification", "regression", "retrieval", "clustering", "segmentation", "detection_proxy", "cell_tracking", "ood")
METRIC = {"classification": "balanced_accuracy", "regression": "r2", "retrieval": "recall_at_1",
          "clustering": "nmi", "segmentation": "mDice_E20", "detection_proxy": "test_patch_f1"}
ALIASES = {"5tb_no_gram": "hs6_l_5tb_no_gram", "5tb_gram12687": "hs6_l_5tb_gram12687",
           "L no-GRAM": "hs6_l_5tb_no_gram", "H+": "hs6_hplus_5tb",
           "Vanilla (5TB no-GRAM)": "hs6_l_5tb_no_gram", "GRAM (5TB)": "hs6_l_5tb_gram12687"}
PRIORITY = {MONUSEG_PROVENANCE: 110, DETECTION_FILL_PROVENANCE: 100, "v4_completion_validated": 100, "v4_list_raw": 90, "v4_selected": 85,
            "validated_v3": 75, "validated_legacy_extension": 70, "legacy_accepted": 30}


def rows(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def load(path):
    return json.loads(path.read_text())


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def step_key(value):
    return str(int(value)) if str(value).isdigit() else str(value)


def canonical_metric(family, dataset):
    if family == "classification" and dataset == "chestmnist":
        return "macro_auc"
    if family == "regression" and dataset == "bbbc013":
        return "compound_mean_r2"
    return METRIC.get(family, "")


_CONIC_CACHE = {}


def conic_detection_split(observation):
    """official / legacy-random / unverified for one CoNIC detection observation."""
    path = observation["source"]
    if path.startswith("{"):  # selective-retention ledger: a map of cell -> evidence path
        path = next((v for k, v in json.loads(path).items() if k.startswith("det__conic")), "")
    path = path.split(";")[0]
    if path not in _CONIC_CACHE:
        split = "unverified"
        if Path(path).is_file():
            obj = load(Path(path))
            weight = number(obj.get("pos_weight"))
            if obj.get("conic_split_protocol"):
                split = "official" if obj["conic_split_protocol"] == CONIC_OFFICIAL else obj["conic_split_protocol"]
            elif weight is not None:
                split = next((k for k, v in CONIC_POS_WEIGHT.items() if abs(weight - v) < 1e-5), "unverified")
        elif path.startswith("/data/hs6_") and "/v4/detection_b8/" in path:
            split = "official"  # HXW v4 B8 files, audited 2026-09-30: all record the official split
        _CONIC_CACHE[path] = split
    return _CONIC_CACHE[path]


def result_from_validation(row):
    report = Path(row["evidence_paths"].split(";")[0])
    parent, family, dataset = report.parent, row["family"], row["dataset"]
    if family == "segmentation":
        paths = sorted(parent.glob("results/**/budget20/seed*/monuseg/*/results.json"))
        values = [number(load(path).get("test", {}).get("mDice")) for path in paths]
        if len(values) == 3 and all(v is not None for v in values):
            return statistics.mean(values), "mDice_E20", [str(p) for p in paths]
        return None, "mDice_E20", []
    if family == "detection_proxy":
        path = parent / "results_bio_detection.json"
        value = number(load(path).get("test_patch_f1")) if path.is_file() else None
        return (value / 100 if value is not None else None), "test_patch_f1", [str(path)] if path.is_file() else []
    if dataset == "rxrx3-core":
        paths = sorted(parent.glob("models/*/results.json"))
        if paths:
            rx = load(paths[0]).get("tests", {}).get("rxrx3", {})
            return number(rx.get("recall_at_1" if family == "retrieval" else "nmi")), METRIC[family], [str(paths[0])]
        return None, METRIC[family], []
    path = parent / "summary.csv"
    if not path.is_file():
        return None, METRIC[family], []
    entries = list(rows(path))
    field = canonical_metric(family, dataset)
    if family in ("retrieval", "clustering"):
        entries = [r for r in entries if r.get("task") in (family, "retrieval_clustering")]
    value = next((number(r.get(field)) for r in entries if number(r.get(field)) is not None), None)
    return value, field, [str(path)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v4_models").mkdir(exist_ok=True)
    spec = load(PROTOCOL)
    expected = {}
    for family in ("classification", "regression", "retrieval", "segmentation"):
        expected[family] = list(dict.fromkeys(ds for part in ("tier_a", "tier_b", "union_extension")
                                               for ds in spec[part].get(family, [])))
    expected["clustering"] = expected["retrieval"][:]
    expected["detection_proxy"] = spec["union_extension"]["detection_proxy"]
    expected["cell_tracking"] = ["ctc"]
    expected["ood"] = ["xray", "cryo"]
    assert sum(map(len, expected.values())) == 56
    models = {}

    def checkpoint(model, step):
        model = ALIASES.get(model, model)
        group = ("fm14" if model.startswith("fm_") else "hs0_1tb" if model.startswith("hs0_") else
                 "hs6_1tb" if model in ("hs6_splus", "hs6_b", "hs6_l", "hs6_hplus") else "hs6_5tb")
        document = models.setdefault(model, {"model": model, "group": group, "checkpoints": {}})
        step = step_key(step)
        return document["checkpoints"].setdefault(step, {"step": step, "cells": {}, "training": {}})

    def cell(model, step, family, dataset):
        if family not in expected or dataset not in expected[family]:
            return None
        point = checkpoint(model, step)
        return point["cells"].setdefault(family, {}).setdefault(dataset, {"observations": [], "audits": []})

    def observe(model, step, family, dataset, value, metric, provenance, source, *, note="", budget=None):
        target = cell(model, step, family, dataset)
        if target is None or value is None:
            return
        observation = {"value": float(value), "metric": metric, "provenance": provenance,
                       "source": source, "priority": PRIORITY[provenance]}
        if budget is not None:
            observation["budget_epochs"] = budget
        if note:
            observation["note"] = note
        signature = (metric, source, budget)
        if any((x["metric"], x["source"], x.get("budget_epochs")) == signature for x in target["observations"]):
            return
        target["observations"].append(observation)

    def audit(model, step, family, dataset, status, source, *, note=""):
        target = cell(model, step, family, dataset)
        if target is None:
            return
        record = {"status": status, "source": source}
        if note:
            record["note"] = note
        if record not in target["audits"]:
            target["audits"].append(record)

    # Validated v3 common results and validated union extensions. These hold
    # all HS0/HS6 1TB steps, FM14, and early 5TB steps in one scalar table.
    for row in rows(VALIDATED):
        provenance = "validated_v3" if row["protocol"] == "v3" else "validated_legacy_extension"
        observe(row["model"], row["checkpoint"], row["family"], row["dataset"],
                number(row["value"]), row["metric"], provenance, row["source"],
                budget=20 if row["family"] == "segmentation" else None)

    # The dated 56-cell coverage table is valuable even where it recorded no
    # score. Its statuses remain an audit snapshot, never an invented metric.
    for row in rows(SELECTED):
        audit(row["model"], row["checkpoint"], row["family"], row["dataset"],
              row["status"], str(SELECTED))
        if row["status"] in ("VALIDATED_V3", "VALIDATED_LEGACY_EXTENSION", "B8_PROXY_OBSERVATION"):
            observe(row["model"], row["checkpoint"], row["family"], row["dataset"],
                    number(row["value"]), canonical_metric(row["family"], row["dataset"]),
                    "validated_v3" if row["status"] == "VALIDATED_V3" else "validated_legacy_extension",
                    row["source"], budget=20 if row["family"] == "segmentation" else None)

    for row in rows(FOUR):
        if row["model_label"] != "1TB":
            continue
        model, step = "hs6_l", 1024
        audit(model, step, row["family"], row["dataset"], row["cell_status"],
              row["evidence_paths"] or str(FOUR), note=row["admission_status_or_note"])
        if row["provenance"] == "v4_completion_campaign" and row["cell_status"].startswith("VALID_COMPLETE"):
            value, metric, paths = result_from_validation(row)
            observe(model, step, row["family"], row["dataset"], value, metric,
                    "v4_completion_validated", ";".join(paths) or row["evidence_paths"],
                    note=row["admission_status_or_note"], budget=20 if row["family"] == "segmentation" else None)

    # The shared-workspace v4 collector covers sparse original N/G checkpoints.  Its output
    # directory was removed on 2026-09-30; the observations recorded by the 2026-09-29 build are
    # carried forward unchanged (same values and original source paths).
    if not METHOD.is_file():
        for row in rows(METHOD_CARRIED):
            observe(row["model"], row["checkpoint"], row["family"], row["dataset"],
                    number(row["value"]), row["metric"], "v4_selected", row["source"],
                    note="carried_from_20260929_build", budget=int(row["budget"]) if row["budget"] else None)
    for row in (rows(METHOD) if METHOD.is_file() else ()):
        if row["method"] not in ("Vanilla (5TB no-GRAM)", "GRAM (5TB)"):
            continue
        budget = int(row["budget"])
        if row["family"] == "segmentation" and budget != 20:
            continue
        observe(row["method"], row["checkpoint"], row["family"], row["dataset"],
                number(row["value"]), row["metric"], "v4_selected", row["source"],
                budget=budget if row["family"] == "segmentation" else None)

    # Exact raw metrics for L no-GRAM and H+; remote paths remain in the JSON,
    # while values are stored locally and can be queried without SSH.
    for row in rows(STRICT):
        observe(row["model"], row["checkpoint"], row["family"], row["dataset"],
                number(row["value"]), canonical_metric(row["family"], row["dataset"]),
                "v4_list_raw", row["source"],
                budget=20 if row["family"] == "segmentation" else None)
    for row in rows(RELAXED):
        if row["provenance"] not in ("accepted_legacy_b4", "legacy_monuseg_e50_b16_seed0_unverified_identity"):
            continue
        observe(row["model"], row["checkpoint"], row["family"], row["dataset"],
                number(row["value"]), "mDice_E50" if row["family"] == "segmentation" else "test_patch_f1",
                "legacy_accepted", row["source"], note=row["provenance"],
                budget=50 if row["family"] == "segmentation" else None)

    # Preserve older B4 detection and MoNuSeg observations across the other
    # models too. They remain alternatives, even when all three E20 seeds exist.
    monuseg_paths = defaultdict(set)
    for row in rows(OLD_INDEX):
        model, step = row["model"], row["checkpoint"]
        canonical = ALIASES.get(model, model)
        if canonical not in models or step_key(step) not in models[canonical]["checkpoints"]:
            continue
        source = Path(row["source"])
        if row["family"] == "detection" and row["protocol"] == "old_observation":
            if not source.is_file():
                continue
            obj = load(source)
            value = number(obj.get("test_patch_f1"))
            if value is not None and obj.get("batch_size") == 4:
                observe(model, step, "detection_proxy", row["dataset"], value / 100,
                        "test_patch_f1", "legacy_accepted", str(source), note="historical_B4_detection")
        if row["family"] != "segmentation" or row["dataset"] != "monuseg" or not source.is_file():
            continue
        if row["protocol"] == "old":
            obj = load(source)
            value = number(obj.get("test", {}).get("mDice"))
            if value is not None:
                observe(model, step, "segmentation", "monuseg", value, "mDice_E50",
                        "legacy_accepted", str(source),
                        note="historical_MoNuSeg_single_seed_identity_unverified", budget=50)
        elif row["protocol"] == "old_union_extension" and row["evidence_status"] == "OBSERVATIONAL":
            if source.name == "validation_report.json":
                for path in load(source).get("result_sha256", {}):
                    if "/E20/" in path or "/budget20/" in path:
                        monuseg_paths[model, step].add(Path(path) if Path(path).is_absolute() else source.parent / path)
            elif source.name == "results.json" and ("/E20/" in str(source) or "/budget20/" in str(source)):
                monuseg_paths[model, step].add(source)
    for (model, step), paths in monuseg_paths.items():
        values = []
        for path in sorted(paths):
            if not path.is_file():
                continue
            obj = load(path)
            seed = obj.get("_meta", {}).get("seed")
            value = number(obj.get("test", {}).get("mDice"))
            if seed in (0, 1, 2) and value is not None:
                values.append((seed, value, str(path)))
        if len(values) == 3 and {v[0] for v in values} == {0, 1, 2}:
            observe(model, step, "segmentation", "monuseg",
                    statistics.mean(v[1] for v in values), "mDice_E20", "legacy_accepted",
                    ";".join(v[2] for v in values),
                    note="legacy_MoNuSeg_three_seed_identity_unverified", budget=20)

    # Dedicated selected-point v4 ledger records formal state independently
    # from the later scalar collectors.
    selected = load(SELECTIVE)
    arm_steps = {"baseline_E": 12687, "baseline_M": 20007, "baseline_L": 29279}
    for item in selected["cells"]:
        arm = item["arm"]
        if arm.startswith("N") and arm[1:].isdigit():
            model, step = "hs6_l_5tb_no_gram", int(arm[1:])
        elif arm.startswith("G") and arm[1:].isdigit():
            model, step = "hs6_l_5tb_gram12687", int(arm[1:])
        elif arm in arm_steps:
            model, step = "hs6_l_5tb_no_gram", arm_steps[arm]
        else:
            continue
        evidence = item.get("evidence", {})
        source = evidence.get("path") or evidence.get("root") or str(SELECTIVE)
        audit(model, step, item["family"], item["dataset"], item["state"],
              source, note=item.get("admission", ""))

    # HXW inventories cover L's 25 continuation points and all 71 H+ points.
    for model, path in (("hs6_l_5tb_no_gram", L_TAIL), ("hs6_hplus_5tb", HPLUS)):
        inventory = load(path)
        for step, families in inventory["checkpoints"].items():
            checkpoint(model, step)
            for family, datasets in families.items():
                for dataset, entry in datasets.items():
                    name = "pannuke" if family == "segmentation" and dataset.startswith("pannuke/fold") else dataset
                    audit(model, step, family, name, entry["status"],
                          entry.get("evidence", str(path)), note=entry.get("protocol_note", ""))

    monuseg_inputs = []
    for campaign in MONUSEG_CAMPAIGNS:
        manifest_path = campaign / "campaign_manifest.json"
        if not manifest_path.is_file():
            continue
        monuseg_inputs.append(manifest_path)
        tasks = {task["key"]: task for task in load(manifest_path)["tasks"]}
        for done in sorted((campaign / "_state/done").glob("*.json")):
            task = tasks.get(done.stem)
            if (task is None or load(done).get("status") != "VALID_COMPLETE"
                    or task["dataset"].get("split_protocol_id") != MONUSEG_SPLIT):
                continue
            model = ALIASES.get(task["asset"]["arm"], task["asset"]["arm"])
            step = step_key(task["asset"]["checkpoint_id"])
            if model not in models or step not in models[model]["checkpoints"]:
                continue  # 20TB arms and points outside this report's index
            seeds = {}
            for path in (campaign / "cells" / done.stem).rglob("results.json"):
                obj = load(path)
                meta = obj.get("_meta", {})
                if meta.get("probe_epochs") == 20 and meta.get("full_train_samples") == 30:
                    seeds[meta.get("seed")] = (number(obj.get("test", {}).get("mDice")), str(path))
            if set(seeds) == {0, 1, 2} and all(v[0] is not None for v in seeds.values()):
                observe(model, step, "segmentation", "monuseg",
                        statistics.mean(seeds[s][0] for s in (0, 1, 2)), "mDice_E20", MONUSEG_PROVENANCE,
                        ";".join(seeds[s][1] for s in (0, 1, 2)),
                        note=f"{MONUSEG_SPLIT}; campaign {campaign.relative_to(EVAL)}", budget=20)

    detection_inputs = []
    for campaign in DETECTION_FILL:
        manifest_path = campaign / "campaign_manifest.json"
        if not manifest_path.is_file():
            continue
        detection_inputs.append(manifest_path)
        tasks = {task["key"]: task for task in load(manifest_path)["tasks"]}
        for done in sorted((campaign / "done").glob("*.json")):
            task = tasks.get(done.stem)
            result = campaign / "cells" / done.stem / "results_bio_detection.json"
            if task is None or load(done).get("status") != "VALID_COMPLETE" or not result.is_file():
                continue
            model = ALIASES.get(task["asset"]["arm"], task["asset"]["arm"])
            step = step_key(task["asset"]["checkpoint_id"])
            if model not in models or step not in models[model]["checkpoints"]:
                continue
            value = number(load(result).get("test_patch_f1"))
            observe(model, step, "detection_proxy", task["dataset"], value / 100 if value is not None else None,
                    "test_patch_f1", DETECTION_FILL_PROVENANCE, str(result),
                    note=f"B8; CoNIC {CONIC_OFFICIAL}; site {load(manifest_path)['site']}")

    for row in rows(TRACKS):
        point = checkpoint(row["model"], row["step"])
        point["training"] = {key: row[key] for key in ("segment", "evidence_host", "checkpoint",
                         "checkpoint_available", "evaluation_root", "formal_root", "evaluation_scope")}
    gram = checkpoint("hs6_l_5tb_gram12687", 12687)
    gram["training"]["branch_anchor_shared_with"] = "hs6_l_5tb_no_gram/12687"
    models["hs6_l_5tb_gram12687"]["shared_history"] = {
        "model": "hs6_l_5tb_no_gram", "through_checkpoint": 12687,
        "reason": "GRAM branches from the no-GRAM ck12687 weights; earlier points are not independent GRAM evaluations"}

    summary = []
    for model, document in sorted(models.items()):
        for step, point in document["checkpoints"].items():
            counts = Counter()
            strict_counts = Counter()
            missing_any = {}
            missing_without_legacy = {}
            for family in FAMILIES:
                family_cells = point["cells"].setdefault(family, {})
                absent_any = []
                absent_without_legacy = []
                for dataset in expected[family]:
                    entry = family_cells.setdefault(dataset, {"observations": [], "audits": []})
                    entry["observations"].sort(key=lambda x: (-x["priority"], x["source"]))
                    eligible = entry["observations"]
                    if family == "detection_proxy" and dataset == "conic":
                        eligible = []
                        for x in entry["observations"]:
                            x["conic_split"] = conic_detection_split(x)
                            if x["conic_split"] == "official":
                                eligible.append(x)
                            else:
                                x["excluded_reason"] = "CoNIC split is not the official source-disjoint split"
                    if dataset == "monuseg":
                        eligible = [x for x in eligible if x["provenance"] == MONUSEG_PROVENANCE]
                        for x in entry["observations"]:
                            if x["provenance"] != MONUSEG_PROVENANCE:
                                x["superseded_by_split"] = MONUSEG_SPLIT
                    entry["preferred"] = eligible[0] if eligible else None
                    entry["status"] = ("OBSERVED" if entry["preferred"] else
                                       "PENDING_CONIC_OFFICIAL_SPLIT" if family == "detection_proxy" and dataset == "conic" and entry["observations"] else
                                       "PENDING_MONUSEG_30_7_14" if dataset == "monuseg" else
                                       "BLOCKED" if any(a["status"] in ("BLOCKED_NOT_TESTED", "BLOCKED_PROTOCOL", "PROTOCOL_IMPLEMENTATION_PENDING", "SKIPPED_BY_USER") for a in entry["audits"]) else
                                       "MISSING_OR_UNVERIFIED")
                    if entry["preferred"]:
                        counts[family] += 1
                    else:
                        absent_any.append(dataset)
                    if any(x["provenance"] != "legacy_accepted" for x in eligible):
                        strict_counts[family] += 1
                    else:
                        absent_without_legacy.append(dataset)
                missing_any[family] = absent_any
                missing_without_legacy[family] = absent_without_legacy
            point["coverage"] = {family: {"observed_any": counts[family],
                                          "observed_without_user_accepted_legacy": strict_counts[family],
                                          "expected": len(expected[family]),
                                          "missing_any": missing_any[family],
                                          "missing_without_user_accepted_legacy": missing_without_legacy[family]}
                                 for family in FAMILIES}
            summary.append({"model": model, "group": document["group"], "checkpoint": step,
                            **{f"{family}_observed_any": counts[family] for family in FAMILIES},
                            **{f"{family}_observed_without_legacy": strict_counts[family]
                               for family in FAMILIES}})
        (OUT / "v4_models" / f"{model}.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")

    summary.sort(key=lambda x: (x["group"], x["model"],
                                int(x["checkpoint"]) if x["checkpoint"].isdigit() else -1))
    with (OUT / "v4_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    monuseg_rows = []
    for model, document in sorted(models.items()):
        for step, point in document["checkpoints"].items():
            entry = point["cells"]["segmentation"]["monuseg"]
            older = [x for x in entry["observations"] if x["provenance"] != MONUSEG_PROVENANCE]
            monuseg_rows.append({"model": model, "group": document["group"], "checkpoint": step,
                                 "status": entry["status"],
                                 "mDice_E20_train30_val7_test14": entry["preferred"]["value"] if entry["preferred"] else "",
                                 "source": entry["preferred"]["source"].split(";")[0] if entry["preferred"] else "",
                                 "superseded_value": older[0]["value"] if older else "",
                                 "superseded_metric": older[0]["metric"] if older else "",
                                 "superseded_provenance": (older[0].get("note") or older[0]["provenance"]) if older else ""})
    monuseg_rows.sort(key=lambda x: (x["group"], x["model"], int(x["checkpoint"]) if x["checkpoint"].isdigit() else -1))
    with (OUT / "monuseg_train30_val7_test14.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(monuseg_rows[0]))
        writer.writeheader()
        writer.writerows(monuseg_rows)
    (OUT / "v4_all_models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n")
    expected_points = defaultdict(set)
    for row in rows(VALIDATED):
        expected_points[ALIASES.get(row["model"], row["model"])].add(step_key(row["checkpoint"]))
    for row in rows(TRACKS):
        expected_points[ALIASES.get(row["model"], row["model"])].add(step_key(row["step"]))
    point_audit = {}
    for model, document in sorted(models.items()):
        actual = set(document["checkpoints"])
        planned = expected_points.get(model, actual)
        point_audit[model] = {
            "expected_checkpoints": len(planned),
            "indexed_checkpoints": len(actual),
            "missing_checkpoints": sorted(planned - actual, key=lambda s: int(s) if s.isdigit() else -1),
            "extra_checkpoints": sorted(actual - planned, key=lambda s: int(s) if s.isdigit() else -1),
            "points_without_any_numeric_cell": sorted(
                (step for step, point in document["checkpoints"].items()
                 if not any(cell["preferred"] for family in point["cells"].values() for cell in family.values())),
                key=lambda s: int(s) if s.isdigit() else -1),
        }
    (OUT / "v4_checkpoint_audit.json").write_text(json.dumps(point_audit, indent=2) + "\n")
    inputs = (PROTOCOL, VALIDATED, SELECTED, FOUR, METHOD if METHOD.is_file() else METHOD_CARRIED, STRICT, RELAXED,
              TRACKS, SELECTIVE, L_TAIL, HPLUS, OLD_INDEX, *monuseg_inputs, *detection_inputs)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": spec["protocol_id"], "expected": expected,
                "all_models_path": "v4_all_models.json", "coverage_path": "v4_coverage.csv",
                "models": {m: {"group": x["group"], "checkpoints": len(x["checkpoints"]),
                               "path": f"v4_models/{m}.json"} for m, x in sorted(models.items())},
                "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in inputs],
                "monuseg_policy": f"MoNuSeg preferred values come only from {MONUSEG_SPLIT} campaigns (E20, seeds 0/1/2 mean test mDice); older MoNuSeg observations are kept with superseded_by_split and never preferred.",
                "conic_detection_policy": "CoNIC detection is preferred only when the result is on official-baseline-fold0-nested-v1 (recorded field or official training pos_weight). Legacy random patch-level results carry conic_split=legacy-random and excluded_reason.",
                "interpretation": "Each observation keeps its own protocol and source. Preferred is a lookup convenience, not a formal all-v4 admission claim. Legacy accepted B4/MoNuSeg values are retained as low-priority alternatives. Full v4 also needs CTC/OOD and dataset admission."}
    (OUT / "v4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model_count": len(models), "checkpoint_count": len(summary),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
