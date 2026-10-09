#!/usr/bin/env python3
"""Audit H+ and L 5TB v4 result files on hxw by checkpoint and task."""

import datetime as dt
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "Evaluation Rules/protocol_v4.json").read_text())
OUTPUT = ROOT / "outputs/02_eval_runs/hs6_5tb_v4_checkpoint_task_inventory_20260924"

REMOTE = r'''
import json
from pathlib import Path

base = {
 "l": Path("/data/hs6_l_5tb_nogram_eval_20260921"),
 "hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
}
roots = {
 "l": [base["l"] / "results"],
 "hplus": [base["hplus"] / "old", base["hplus"] / "continuation_v4/nonseg"],
}
items = []
for model, scan_roots in roots.items():
 for root in scan_roots:
  for path in root.glob("point_*/*/bio_*/*/*/last_result.json"):
   try:
    obj = json.loads(path.read_text())
    parts = path.relative_to(root).parts
    items.append({"model": model, "point": int(parts[0][6:]), "family": parts[1].split("_")[0], "dataset": parts[3], "path": str(path), "value": obj})
   except Exception as exc:
    items.append({"model": model, "path": str(path), "read_error": str(exc)})
  for path in root.glob("point_*/*/bio_*/*/*/failed_result.json"):
   try:
    obj = json.loads(path.read_text())
    parts = path.relative_to(root).parts
    items.append({"model": model, "point": int(parts[0][6:]), "family": parts[1].split("_")[0], "dataset": parts[3], "path": str(path), "failure": obj.get("error", "unknown")})
   except Exception as exc:
    items.append({"model": model, "path": str(path), "read_error": str(exc)})
for model, root in base.items():
 for path in (root / "v3/cells").glob("point_*/validation_report.json"):
  try:
   obj = json.loads(path.read_text())
   name = path.parent.name.split("__")
   items.append({"model": model, "point": int(name[0][6:]), "family": "segmentation", "dataset": name[1], "cell": name[2], "path": str(path), "value": obj})
  except Exception as exc:
   items.append({"model": model, "path": str(path), "read_error": str(exc)})
 for path in (root / "v4/detection_b8").glob("point_*/*/results_bio_detection.json"):
  try:
   parts = path.relative_to(root / "v4/detection_b8").parts
   items.append({"model": model, "point": int(parts[0][6:]), "family": "detection_proxy", "dataset": parts[1], "path": str(path), "value": json.loads(path.read_text())})
  except Exception as exc:
   items.append({"model": model, "path": str(path), "read_error": str(exc)})
 for path in Path("/data/hs6_5tb_v4_rxrx3_20260924").glob(model + "/point_*/models/*/results.json"):
  try:
   point = int(path.parts[-4][6:])
   items.append({"model": model, "point": point, "family": "rxrx3", "dataset": "rxrx3-core", "path": str(path), "value": json.loads(path.read_text())})
  except Exception as exc:
   items.append({"model": model, "path": str(path), "read_error": str(exc)})
 for path in Path("/data/hs6_5tb_v4_ood_20260924/results").glob(model + "_*_xray/*/last_result.json"):
  try:
   items.append({"model": model, "point": int(path.parent.name), "family": "ood", "dataset": "xray", "path": str(path), "value": json.loads(path.read_text())})
  except Exception as exc:
   items.append({"model": model, "path": str(path), "read_error": str(exc)})
print(json.dumps(items, separators=(",", ":")))
'''


def collect():
    p = subprocess.run(
        ["ssh", "5090-hxw-xzj", "python3 -"],
        input=REMOTE,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(p.stdout)


def expected():
    tasks = {}
    for family in ("classification", "regression", "retrieval"):
        tasks[family] = list(dict.fromkeys(
            ds for part in ("tier_a", "tier_b", "union_extension")
            for ds in PROTOCOL[part].get(family, [])
        ))
    tasks["clustering"] = tasks["retrieval"][:]
    tasks["segmentation"] = [
        "cellpose", "conic", "livecell", "monuseg", "multimodal_cellseg",
        "pannuke/fold1", "pannuke/fold2", "pannuke/fold3", "tissuenet",
    ]
    tasks["detection_proxy"] = PROTOCOL["union_extension"]["detection_proxy"]
    tasks["cell_tracking"] = ["ctc"]
    tasks["ood"] = ["xray", "cryo"]
    return tasks


def result_entry(item, family):
    value = item["value"]
    entry = {"status": "RESULT_PRESENT", "evidence": item["path"]}
    if family in ("retrieval", "clustering"):
        rows = value.get("rows", [value])
        field = "recall_at_1" if family == "retrieval" else "nmi"
        matches = [row for row in rows if row.get(field) is not None]
        if not matches:
            entry["status"] = "METRIC_MISSING"
        else:
            entry["metrics"] = {field: matches[0][field]}
            entry["row_count"] = len(matches)
    elif family == "segmentation":
        entry["status"] = value.get("status", "UNKNOWN")
        entry["validated_fits"] = len(value.get("result_sha256", {}))
    elif family == "rxrx3":
        test = value.get("tests", {}).get("rxrx3", {})
        entry["status"] = value.get("status", "UNKNOWN")
        entry["metrics"] = {k: test[k] for k in ("recall_at_1", "nmi") if k in test}
        if entry["status"] == "VALID_COMPLETE" and (test.get("status") != "FORMAL" or test.get("proxy") is not False):
            entry["status"] = "NOT_FORMAL"
        if value.get("error"):
            entry["error"] = str(value["error"])[:240]
            if entry["status"] == "FAILED" and ("ALLOC_FAILED" in entry["error"] or "OutOfMemory" in entry["error"]):
                entry["status"] = "FAILED_OOM"
    else:
        keys = {
            "classification": ("macro_auc", "accuracy", "macro_f1"),
            "regression": ("mae", "r2", "spearman"),
            "detection_proxy": ("test_patch_f1",),
            "ood": ("macro_auc", "accuracy", "macro_f1"),
        }.get(family, ())
        entry["metrics"] = {k: value[k] for k in keys if value.get(k) is not None}
        if not entry["metrics"]:
            entry["status"] = "METRIC_MISSING"
    return entry


def build(items, model, points, tasks):
    result = {}
    for point in points:
        result[str(point)] = {family: {task: {"status": "MISSING"} for task in names}
                              for family, names in tasks.items()}
        for task in ("xray", "cryo"):
            result[str(point)]["ood"][task] = {
                "status": "SKIPPED_BY_USER", "scope": "excluded_by_user"
            }
        result[str(point)]["cell_tracking"]["ctc"] = {
            "status": "PROTOCOL_IMPLEMENTATION_PENDING",
            "scope": "formal v4 cell_tracking",
        }
    for item in items:
        if item.get("model") != model or "read_error" in item:
            continue
        point = str(item["point"])
        if point not in result:
            continue
        family = item["family"]
        dataset = item["dataset"]
        if "failure" in item:
            if family == "retrieval":
                targets = [("retrieval", dataset), ("clustering", dataset)]
            else:
                targets = [(family, dataset)]
            for target, name in targets:
                if target in result[point] and name in result[point][target]:
                    current = result[point][target][name]
                    if current["status"] == "MISSING":
                        current.update(status="FAILED_OOM" if "OutOfMemory" in item["failure"] else "FAILED",
                                       failure_evidence=item["path"], error=item["failure"][:240])
            continue
        if family == "rxrx3":
            entry = result_entry(item, "rxrx3")
            for target in ("retrieval", "clustering"):
                result[point][target][dataset] = dict(entry)
            continue
        if family == "retrieval":
            for target in ("retrieval", "clustering"):
                if dataset in result[point][target]:
                    result[point][target][dataset] = result_entry(item, target)
            continue
        if family == "segmentation":
            dataset = dataset if dataset != "pannuke" else "pannuke/fold" + item["cell"].split("fold")[1][0]
        if family not in result[point] or dataset not in result[point][family]:
            continue
        result[point][family][dataset] = result_entry(item, family)
    for point in result:
        for family in ("classification", "retrieval", "clustering"):
            result[point][family]["lc25000"]["protocol_note"] = "source-disjoint admission not proven; provisional"
        for family in ("retrieval", "clustering"):
            result[point][family]["nct-crc-he-100"]["protocol_note"] = "admission gate requires separate review"
    return result


def summary(inventory, tasks):
    counts = {}
    for family, names in tasks.items():
        if family in ("cell_tracking", "ood"):
            continue
        states = [inventory[p][family][name]["status"] for p in inventory for name in names]
        counts[family] = {
            "total": len(states),
            "present": sum(x in ("RESULT_PRESENT", "VALID_COMPLETE") for x in states),
            "missing_or_invalid": sum(x not in ("RESULT_PRESENT", "VALID_COMPLETE") for x in states),
        }
    counts["id_total_excluding_ctc"] = sum(x["total"] for x in counts.values() if isinstance(x, dict))
    counts["id_present_excluding_ctc"] = sum(x["present"] for x in counts.values() if isinstance(x, dict))
    counts["id_missing_excluding_ctc"] = sum(x["missing_or_invalid"] for x in counts.values() if isinstance(x, dict))
    return counts


def main():
    items = collect()
    tasks = expected()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    # The L training source contains 25 checkpoints; H+ contains 28 including step 0.
    l_points = list(range(29767, 41480, 488))
    h_points = [0] + list(range(487, 6832, 488)) + list(range(7319, 9760, 488)) + list(range(10247, 13176, 488))
    for model, points in (("l", l_points), ("hplus", h_points)):
        inventory = build(items, model, points, tasks)
        doc = {
            "protocol": PROTOCOL["protocol_id"],
            "model": model,
            "generated_utc": stamp,
            "source_host": "5090-hxw-xzj",
            "source_checkpoint_host": "5090-lyx-xr",
            "point_count": len(points),
            "checkpoint_steps": points,
            "task_counts_per_checkpoint": {k: len(v) for k, v in tasks.items()},
            "status_meaning": {
                "RESULT_PRESENT": "result file contains a metric; protocol admission is separate",
                "VALID_COMPLETE": "segmentation validator or RxRx3 formal result marked complete",
                "MISSING": "no result metric found at audit time",
                "FAILED_OOM": "evaluation result records an out-of-memory failure",
                "FAILED": "evaluation result records another failure",
                "PROTOCOL_IMPLEMENTATION_PENDING": "formal CTC protocol is not launch-ready",
                "SKIPPED_BY_USER": "OOD excluded from requested ID-only scope",
            },
            "summary": summary(inventory, tasks),
            "checkpoints": inventory,
        }
        path = OUTPUT / f"{model}_checkpoint_task_inventory.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        print(model, path, json.dumps(doc["summary"], ensure_ascii=False))
    errors = [{k: v for k, v in item.items() if k != "value"} for item in items if "read_error" in item]
    print("read_errors", len(errors), errors[:5])


if __name__ == "__main__":
    main()
