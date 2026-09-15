#!/usr/bin/env python3
"""Paired stratified bootstrap for NCT-to-CRC C-scale endpoints."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def endpoint(path: Path) -> int:
    ckpt = int(path.parents[1].name)
    if ckpt == 0:
        return 0
    match = re.search(r"_e([1248])_", str(path))
    if match is None:
        raise ValueError(f"cannot infer endpoint from {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for run in sorted(args.run_root.glob(args.run_glob)):
        for path in run.glob(
            "eval/nct_cscale_audit_20260908/bio_classification/nct-crc-he/"
            "*/readout_audit/tuned_c_predictions.npz"
        ):
            ep = endpoint(path)
            with np.load(path, allow_pickle=False) as pack:
                found[ep] = (np.asarray(pack["y_true"]).reshape(-1), np.asarray(pack["y_pred"]).reshape(-1))
    if 0 not in found:
        raise RuntimeError("ep0 predictions are missing")

    y_true, base_pred = found[0]
    classes = np.unique(y_true)
    class_indices = [np.flatnonzero(y_true == label) for label in classes]
    rng = np.random.default_rng(args.seed)
    rows = []
    for ep, (other_true, pred) in sorted(found.items()):
        if ep == 0:
            continue
        if not np.array_equal(y_true, other_true):
            raise ValueError(f"test labels/order differ at ep{ep}")
        observed = float(
            f1_score(y_true, pred, average="macro", zero_division=0)
            - f1_score(y_true, base_pred, average="macro", zero_division=0)
        )
        deltas = np.empty(args.samples, dtype=np.float64)
        for sample in range(args.samples):
            idx = np.concatenate([rng.choice(part, size=len(part), replace=True) for part in class_indices])
            deltas[sample] = (
                f1_score(y_true[idx], pred[idx], average="macro", zero_division=0)
                - f1_score(y_true[idx], base_pred[idx], average="macro", zero_division=0)
            )
        rows.append(
            {
                "endpoint": ep,
                "baseline_endpoint": 0,
                "delta_macro_f1": observed,
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_samples": args.samples,
                "seed": args.seed,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} endpoints={sorted(found)}", flush=True)


if __name__ == "__main__":
    main()
