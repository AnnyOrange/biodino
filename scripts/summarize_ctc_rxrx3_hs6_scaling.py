#!/usr/bin/env python3
"""Validate and summarize the HS6 CTC/RxRx3 scaling quick screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/02_eval_runs/ctc_rxrx3_hs6_scaling_quickscreen_3090qi_20260911"
SOURCE = ROOT / "outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910"
ORDER = (
    "hs6_splus_1tb_ck15374",
    "hs6_b_1tb_ck13324",
    "hs6_l_1tb_ck15374",
    "hs6_hplus_1tb_ck15374",
    "hs6_l_5tb_ck23911",
)
NEW = set(ORDER) - {"hs6_hplus_1tb_ck15374"}
METRICS = (
    ("ctc_r2", True), ("ctc_mae", False), ("ctc_spearman", True),
    ("rxrx3_recall_at_1", True), ("rxrx3_recall_at_5", True),
    ("rxrx3_recall_at_10", True), ("rxrx3_mrr_at_10", True),
    ("rxrx3_nmi", True),
)


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def flatten(model: str, result: dict, family: str, provenance: str) -> dict:
    ctc, rx = result["tests"]["ctc"], result["tests"]["rxrx3"]
    return {
        "model": model, "family": family, "provenance": provenance,
        "ctc_r2": ctc["r2"], "ctc_mae": ctc["mae"],
        "ctc_spearman": ctc["spearman"],
        "rxrx3_recall_at_1": rx["recall_at_1"],
        "rxrx3_recall_at_5": rx["recall_at_5"],
        "rxrx3_recall_at_10": rx["recall_at_10"],
        "rxrx3_mrr_at_10": rx["mrr_at_10"], "rxrx3_nmi": rx["nmi"],
    }


def rank(rows: list[dict]) -> None:
    for key, higher in METRICS:
        ordered = sorted(rows, key=lambda x: x[key], reverse=higher)
        previous = None
        competition_rank = 0
        for position, row in enumerate(ordered, 1):
            if previous is None or row[key] != previous:
                competition_rank = position
            row[f"{key}_rank"] = competition_rank
            previous = row[key]


def main() -> None:
    errors = []
    hs6_rows = []
    new_results = []
    for model in ORDER:
        if model in NEW:
            path = CAMPAIGN / "models" / model / "results.json"
            result = load_result(path)
            new_results.append(result)
            provenance = "computed_20260911"
        else:
            path = SOURCE / "models/hs6_hplus/results.json"
            result = load_result(path)
            provenance = "exact_reuse_external4_20260910"
        if result.get("status") != "VALID_COMPLETE":
            errors.append(f"{model}: status={result.get('status')}")
        if set(result.get("tests", {})) < {"ctc", "rxrx3"}:
            errors.append(f"{model}: missing CTC or RxRx3")
        if result["tests"]["ctc"].get("protocol_id") != "ctc_2d_count_proxy_v1":
            errors.append(f"{model}: unexpected CTC protocol")
        if result["tests"]["rxrx3"].get("protocol_id") != "rxrx3_plate_disjoint_128_v1":
            errors.append(f"{model}: unexpected RxRx3 protocol")
        hs6_rows.append(flatten(model, result, "HS6", provenance))

    all_rows = list(hs6_rows)
    for path in sorted((SOURCE / "models").glob("*/results.json")):
        result = load_result(path)
        if result["model"] == "hs6_hplus":
            continue
        if result.get("status") != "VALID_COMPLETE":
            errors.append(f"FM {result['model']}: status={result.get('status')}")
            continue
        all_rows.append(flatten(result["model"], result, "external_FM", "external4_20260910"))
    rank(all_rows)

    fields = list(all_rows[0]) + [f"{key}_rank" for key, _ in METRICS]
    with (CAMPAIGN / "all_models_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    with (CAMPAIGN / "hs6_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hs6_rows[0]))
        writer.writeheader(); writer.writerows(hs6_rows)

    hs6_by_name = {row["model"]: row for row in hs6_rows}
    lines = [
        "# CTC + RxRx3 HS6 scaling quick-screen results", "",
        "Status: VALID_COMPLETE" if not errors else "Status: INVALID", "",
        "These are observational frozen-feature results. CTC is a 2D count proxy, not native tracking; RxRx3 is the fixed 128-query subset, not the full formal eligible-gene protocol.", "",
        "## HS6 results", "",
        "| Model | CTC R2 ↑ | CTC MAE ↓ | CTC Spearman ↑ | Rx R@1 ↑ | Rx R@5 ↑ | Rx R@10 ↑ | Rx MRR@10 ↑ | Rx NMI ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ORDER:
        r = hs6_by_name[model]
        lines.append(
            f"| {model} | {r['ctc_r2']:.4f} | {r['ctc_mae']:.2f} | {r['ctc_spearman']:.4f} | "
            f"{r['rxrx3_recall_at_1']:.4f} | {r['rxrx3_recall_at_5']:.4f} | {r['rxrx3_recall_at_10']:.4f} | "
            f"{r['rxrx3_mrr_at_10']:.4f} | {r['rxrx3_nmi']:.4f} |"
        )
    one = hs6_by_name["hs6_l_1tb_ck15374"]
    five = hs6_by_name["hs6_l_5tb_ck23911"]
    lines += ["", "## 5TB L minus 1TB L", ""]
    for key, _ in METRICS:
        lines.append(f"- {key}: {five[key] - one[key]:+.6f}")
    lines += ["", "## Best external-FM reference for each metric", ""]
    fms = [row for row in all_rows if row["family"] == "external_FM"]
    for key, higher in METRICS:
        best = sorted(fms, key=lambda x: x[key], reverse=higher)[0]
        lines.append(f"- {key}: {best['model']} = {best[key]:.6f}")
    (CAMPAIGN / "summary.md").write_text("\n".join(lines) + "\n")

    validation = {
        "status": "VALID_COMPLETE" if not errors else "INVALID",
        "errors": errors, "hs6_models": len(hs6_rows), "external_fms": len(all_rows) - len(hs6_rows),
        "new_test_cells": len(new_results) * 2, "reused_test_cells": 2,
        "total_hs6_test_cells": len(hs6_rows) * 2,
        "protocols": ["ctc_2d_count_proxy_v1", "rxrx3_plate_disjoint_128_v1"],
    }
    (CAMPAIGN / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    manifest_path = CAMPAIGN / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(validation)
    manifest["status"] = validation["status"]
    manifest["started_unix"] = min(r["started_unix"] for r in new_results)
    manifest["finished_unix"] = max(r["finished_unix"] for r in new_results)
    manifest["wall_seconds"] = manifest["finished_unix"] - manifest["started_unix"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
