#!/usr/bin/env python3
"""Summarize the >=6k HS6 checkpoint sweep on the CTC/RxRx3 quick screen."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs/02_eval_runs/ctc_rxrx3_hs6_ckpt_sweep_3090qi_20260911"
HPLUS = ROOT / "outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910/models/hs6_hplus/results.json"
METRICS = (("ctc_r2", True), ("ctc_mae", False), ("ctc_spearman", True),
           ("rx_r1", True), ("rx_r5", True), ("rx_r10", True),
           ("rx_mrr10", True), ("rx_nmi", True))
EXPECTED = {"hs6_splus_1tb": 10, "hs6_b_1tb": 10, "hs6_l_1tb": 4,
            "hs6_hplus_1tb": 1, "hs6_l_5tb": 37}


def row(model: str, result: dict, provenance: str) -> dict:
    family, ckpt = model.rsplit("_ck", 1)
    ctc, rx = result["tests"]["ctc"], result["tests"]["rxrx3"]
    return {"family": family, "checkpoint": int(ckpt), "model": model,
            "host": result["host"], "provenance": provenance,
            "ctc_r2": ctc["r2"], "ctc_mae": ctc["mae"], "ctc_spearman": ctc["spearman"],
            "rx_r1": rx["recall_at_1"], "rx_r5": rx["recall_at_5"],
            "rx_r10": rx["recall_at_10"], "rx_mrr10": rx["mrr_at_10"], "rx_nmi": rx["nmi"]}


def main() -> None:
    rows = []
    for path in sorted((SWEEP / "models").glob("*/results.json")):
        result = json.loads(path.read_text())
        if result["model"].endswith("_smoke"):
            continue
        rows.append(row(result["model"], result,
                        "5090-source-checkpoint" if str(result.get("checkpoint", "")).startswith("/mnt/data/") else "shared"))
    hplus = json.loads(HPLUS.read_text())
    rows.append(row("hs6_hplus_1tb_ck15374", hplus, "exact-reuse-20260910"))

    grouped = defaultdict(list)
    for item in rows:
        grouped[item["family"]].append(item)
    errors = []
    for family, count in EXPECTED.items():
        if len(grouped[family]) != count:
            errors.append(f"{family}: expected {count}, found {len(grouped[family])}")

    rankings = []
    for family, members in grouped.items():
        for member in members:
            member["mean_rank_score"] = 0.0
            member["metric_wins"] = 0
        for metric, higher in METRICS:
            ordered = sorted(members, key=lambda x: x[metric], reverse=higher)
            index = 0
            while index < len(ordered):
                end = index + 1
                while end < len(ordered) and ordered[end][metric] == ordered[index][metric]:
                    end += 1
                average_rank = ((index + 1) + end) / 2.0
                score = 1.0 if len(members) == 1 else 1.0 - (average_rank - 1.0) / (len(members) - 1)
                for member in ordered[index:end]:
                    member["mean_rank_score"] += score
                    member["metric_wins"] += int(average_rank == 1.0)
                index = end
        for member in members:
            member["mean_rank_score"] /= len(METRICS)
        rankings.extend(sorted(members, key=lambda x: (-x["mean_rank_score"], x["checkpoint"])))

    fields = list(rows[0])
    with (SWEEP / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: (x["family"], x["checkpoint"])))
    with (SWEEP / "family_rankings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rankings)

    lines = ["# HS6 CTC/RxRx3 checkpoint quick-screen sweep", "",
             "Status: VALID_COMPLETE" if not errors else "Status: INVALID", "",
             "This table is observational checkpoint screening, not the formal CTC/RxRx3 result.", "",
             "| family | checkpoints | best composite checkpoint | score | next candidates |", "|---|---:|---:|---:|---|"]
    for family in EXPECTED:
        picked = sorted(grouped[family], key=lambda x: (-x["mean_rank_score"], x["checkpoint"]))
        lines.append(f"| {family} | {len(picked)} | {picked[0]['checkpoint']} | {picked[0]['mean_rank_score']:.4f} | " + ", ".join(str(x["checkpoint"]) for x in picked[1:5]) + " |")
    lines += ["", "Independent pre-existing 5TB full-registry (59 metrics) peak: checkpoint 20007.",
              "The quick-screen 5TB top region is 15615--20007; checkpoint 23911 is not the selected peak."]
    (SWEEP / "summary.md").write_text("\n".join(lines) + "\n")
    validation = {"status": "VALID_COMPLETE" if not errors else "INVALID", "errors": errors,
                  "families": {key: len(value) for key, value in grouped.items()},
                  "checkpoints": len(rows), "test_cells": len(rows) * 2}
    (SWEEP / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
