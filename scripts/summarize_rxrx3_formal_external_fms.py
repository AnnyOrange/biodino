#!/usr/bin/env python3
"""Summarize full-protocol RxRx3-core external-FM results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROTOCOL = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
EXPECTED = {"bioclip", "conch", "cytoimagenet", "cytoself", "dinov2", "gigapath", "hoptimus0",
            "jump_cp", "mae", "pe", "phikon2", "siglip2", "uni", "virchow2"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("campaign", type=Path); args = parser.parse_args()
    rows, errors = [], []
    for p in sorted(args.campaign.glob("models/*/results.json")):
        d = json.loads(p.read_text()); name = p.parent.name; t = d.get("tests", {}).get("rxrx3", {})
        if d.get("status") != "VALID_COMPLETE" or t.get("status") != "FORMAL" or t.get("protocol_id") != PROTOCOL:
            errors.append(f"{name}: incomplete/non-formal")
            continue
        rows.append({"model": name, **{k: t[k] for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "map", "nmi")},
                     "logical_batch_size": d.get("batch_size"), "inference_microbatch": d.get("inference_microbatch"), "host": d.get("host")})
    found = {r["model"] for r in rows}
    for name in sorted(EXPECTED - found): errors.append(f"{name}: missing")
    rows.sort(key=lambda r: (-r["mrr"], -r["recall_at_10"], r["model"]))
    fields = ["model", "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "map", "nmi", "logical_batch_size", "inference_microbatch", "host"]
    with (args.campaign / "model_metrics.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = ["# External FMs on formal RxRx3-core", "", "| Model | R@1 | R@5 | R@10 | MRR | NMI |", "", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['recall_at_1']:.4f} | {r['recall_at_5']:.4f} | {r['recall_at_10']:.4f} | {r['mrr']:.4f} | {r['nmi']:.4f} |" for r in rows]
    (args.campaign / "summary.md").write_text("\n".join(lines) + "\n")
    report = {"status": "VALID_COMPLETE" if not errors and found == EXPECTED else "INCOMPLETE", "expected": len(EXPECTED), "complete": len(rows), "errors": errors}
    (args.campaign / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); raise SystemExit(0 if report["status"] == "VALID_COMPLETE" else 1)


if __name__ == "__main__": main()
