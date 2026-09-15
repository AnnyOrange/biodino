#!/usr/bin/env python3
"""Validate and summarize external-4 per-model JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT = Path("outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910")
METRICS = {
    "ctc": ("r2", "mae", "spearman"),
    "hest": ("gene_wise_pearson", "gene_wise_spearman", "r2", "mae"),
    "rxrx3": ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "nmi"),
    "midogpp": ("f1", "ap", "recall"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    args = parser.parse_args()
    rows, validation = [], {}
    for path in sorted((args.campaign / "models").glob("*/results.json")):
        data = json.loads(path.read_text())
        errors = []
        if data.get("status") != "VALID_COMPLETE": errors.append(f"status={data.get('status')}")
        for task, keys in METRICS.items():
            test = data.get("tests", {}).get(task)
            if not test:
                errors.append(f"missing {task}"); continue
            for key in keys:
                value = test.get(key)
                if value is None or not math.isfinite(float(value)): errors.append(f"invalid {task}.{key}")
            row = {"model": data["model"], "task": task, "protocol_id": test.get("protocol_id"), "status": data.get("status")}
            row.update({key: test.get(key) for key in keys})
            rows.append(row)
        validation[data["model"]] = {"valid": not errors, "errors": errors, "elapsed_seconds": data.get("elapsed_seconds")}
    fields = ["model", "task", "protocol_id", "status"] + sorted({k for r in rows for k in r if k not in {"model","task","protocol_id","status"}})
    with (args.campaign / "summary_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.campaign / "validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    manifest_path = args.campaign / "campaign_manifest.json"
    if manifest_path.exists() and len(validation) == 15 and all(x["valid"] for x in validation.values()):
        manifest = json.loads(manifest_path.read_text())
        manifest["initial_launcher_status"] = manifest.get("status")
        manifest["status"] = "VALID_COMPLETE"
        manifest["validated_models"] = sorted(validation)
        manifest["validated_test_cells"] = len(rows)
        manifest["retry_recoveries"] = {
            "bioclip": "Initial load lacked ftfy; retried successfully using the existing pure-Python ftfy installation."
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    lines = ["# External-4 fixed-budget results", "", "These results are observational; CTC and MIDOG++ are proxies.", ""]
    for task, keys in METRICS.items():
        lines += [f"## {task}", "", "| model | " + " | ".join(keys) + " |", "|---|" + "---:|"*len(keys)]
        task_rows = [r for r in rows if r["task"] == task]
        primary = keys[0]
        reverse = primary != "mae"
        task_rows.sort(key=lambda r: float(r.get(primary) or (-1e99 if reverse else 1e99)), reverse=reverse)
        for row in task_rows:
            vals = ["" if row.get(k) is None else f"{float(row[k]):.4f}" for k in keys]
            lines.append(f"| {row['model']} | " + " | ".join(vals) + " |")
        lines.append("")
    (args.campaign / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"models_seen": len(validation), "valid_models": sum(x["valid"] for x in validation.values()), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
