#!/usr/bin/env python3
"""Estimate DoReMi mixture weights from audited per-domain loss records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from dinov3.data.domain_reweighting import DoReMiDomainWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domains",
        required=True,
        help="Comma-separated domain order used by the final mix",
    )
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Leave absent domains unchanged on a step instead of failing",
    )
    return parser.parse_args()


def _load_records(path: Path):
    records = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "domain", "proxy_loss", "reference_loss"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for row in reader:
            step = int(row["step"])
            domain = row["domain"].strip()
            count = float(row.get("count") or 1.0)
            if count <= 0:
                raise ValueError(f"count must be positive, got {count} at step {step}")
            aggregate = records[step][domain]
            aggregate[0] += float(row["proxy_loss"]) * count
            aggregate[1] += float(row["reference_loss"]) * count
            aggregate[2] += count
    return records


def main() -> None:
    args = parse_args()
    domains = tuple(domain.strip() for domain in args.domains.split(",") if domain.strip())
    optimizer = DoReMiDomainWeights(
        domains,
        step_size=args.step_size,
        smoothing=args.smoothing,
    )
    records = _load_records(args.input)
    history = []
    for step in sorted(records):
        proxy_losses = []
        reference_losses = []
        present = []
        for domain in domains:
            aggregate = records[step].get(domain)
            if aggregate is None:
                if not args.allow_missing:
                    raise ValueError(f"step {step} has no record for domain {domain!r}")
                proxy_losses.append(0.0)
                reference_losses.append(0.0)
                present.append(False)
                continue
            proxy_total, reference_total, count = aggregate
            proxy_losses.append(proxy_total / count)
            reference_losses.append(reference_total / count)
            present.append(True)
        result = optimizer.update(
            proxy_losses,
            reference_losses,
            present=present,
        )
        history.append(
            {
                "input_step": step,
                "excess_losses": dict(zip(domains, result.excess_losses)),
                "weights": dict(zip(domains, result.weights)),
                "averaged_weights": dict(zip(domains, result.averaged_weights)),
            }
        )

    payload = {
        "method": "DoReMi Algorithm 1 exponentiated-gradient averaging",
        "input": str(args.input),
        "step_size": args.step_size,
        "smoothing": args.smoothing,
        "num_steps": optimizer.num_steps,
        "final_weights": dict(zip(domains, optimizer.weights.tolist())),
        "averaged_weights": dict(zip(domains, optimizer.averaged_weights.tolist())),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
