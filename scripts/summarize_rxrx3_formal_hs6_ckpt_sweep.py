#!/usr/bin/env python3
"""Validate and summarize the formal RxRx3-core HS6 checkpoint sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


PROTOCOL = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
EXPECTED = {"splus_1tb": 10, "b_1tb": 10, "l_1tb": 4, "hplus_1tb": 1, "l_5tb": 37}
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "nmi")


def family(name: str) -> str:
    for value in EXPECTED:
        if f"hs6_{value}_ck" in name:
            return value
    raise ValueError(f"unknown model name: {name}")


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    for result_path in sorted(args.campaign.glob("models/hs6_*_ck*/results.json")):
        result = json.loads(result_path.read_text())
        name = result_path.parent.name
        test = result.get("tests", {}).get("rxrx3", {})
        if result.get("status") != "VALID_COMPLETE":
            errors.append(f"{name}: status={result.get('status')}")
            continue
        if test.get("status") != "FORMAL" or test.get("protocol_id") != PROTOCOL or test.get("proxy") is not False:
            errors.append(f"{name}: not a formal {PROTOCOL} result")
            continue
        match = re.search(r"_ck(\d+)$", name)
        row = {"model": name, "family": family(name), "checkpoint": int(match.group(1))}
        row.update({key: float(test[key]) for key in METRICS})
        row.update({"map": float(test["map"]), "n_query": int(test["n_query"]), "n_gallery": int(test["n_gallery"]),
                    "host": result.get("host"), "checkpoint_path": result.get("checkpoint")})
        rows.append(row)

    counts = defaultdict(int)
    for row in rows:
        counts[row["family"]] += 1
    for key, value in EXPECTED.items():
        if counts[key] != value:
            errors.append(f"{key}: expected {value}, found {counts[key]}")

    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    ranked = []
    for fam, group in by_family.items():
        per_metric = {metric: average_ranks([row[metric] for row in group]) for metric in METRICS}
        for i, row in enumerate(group):
            out = dict(row)
            out["mean_rank"] = sum(per_metric[m][i] for m in METRICS) / len(METRICS)
            ranked.append(out)
    ranked.sort(key=lambda row: (list(EXPECTED).index(row["family"]), row["mean_rank"], row["checkpoint"]))

    fields = ["family", "model", "checkpoint", *METRICS, "map", "mean_rank", "n_query", "n_gallery", "host", "checkpoint_path"]
    with (args.campaign / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in ranked])
    best = [min(by, key=lambda row: (row["mean_rank"], row["checkpoint"])) for fam in EXPECTED
            if (by := [row for row in ranked if row["family"] == fam])]
    with (args.campaign / "family_best.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in best])

    lines = ["# HS6 formal RxRx3-core checkpoint sweep", "",
             f"Protocol: `{PROTOCOL}`; 734 query and 734 gallery wells; same-well and same-plate positive pairs excluded.", "",
             "| Family | Best checkpoint | R@1 | R@5 | R@10 | MRR | NMI | Mean rank |", "",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in best:
        lines.append(f"| {row['family']} | {row['checkpoint']} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['nmi']:.4f} | {row['mean_rank']:.2f} |")
    (args.campaign / "summary.md").write_text("\n".join(lines) + "\n")
    validation = {"status": "VALID_COMPLETE" if not errors else "INCOMPLETE", "protocol_id": PROTOCOL,
                  "expected": EXPECTED, "counts": dict(counts), "rows": len(rows), "errors": errors,
                  "checkpoint_metrics_sha256": hashlib.sha256((args.campaign / "checkpoint_metrics.csv").read_bytes()).hexdigest()}
    (args.campaign / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
