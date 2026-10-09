#!/usr/bin/env python3
"""Collect the per-metric x per-checkpoint trajectory of the HS6-L 5TB no-Gram run.

Reads the audited full-registry curve campaign and emits a tidy long CSV plus a
wide matrix.  No metric is invented, imputed, or replaced by zero: a missing
(metric, checkpoint) cell stays missing and is reported as such.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

REPO = Path("/mnt/huawei_deepcad/dinov3")
sys.path.insert(0, str(REPO / "scripts"))

from watch_hs6_l_6m_full_peak import collect_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=REPO / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs/00_reports/hs6_l5_capability_trajectory_20260923",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = sorted(
        (int(d.name.split("_", 1)[1]), d)
        for d in args.eval_root.glob("point_*")
        if d.is_dir() and d.name.split("_", 1)[1].isdigit()
    )
    print(f"[collect] {len(points)} checkpoint directories under {args.eval_root}")

    table: dict[int, dict[str, float]] = {}
    for idx, (ck, root) in enumerate(points, start=1):
        m = collect_metrics(root, ck)
        table[ck] = m
        print(f"[collect] {idx:02d}/{len(points)} ck{ck}: {len(m)} metrics", flush=True)

    all_keys = sorted(set().union(*(set(v) for v in table.values())))
    cks = [ck for ck, _ in points]

    long_path = args.out_dir / "trajectory_long.csv"
    with long_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric_key", "family", "dataset", "metric", "checkpoint", "value"])
        for key in all_keys:
            parts = key.split(":")
            fam = parts[0]
            ds = parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
            met = parts[-1]
            for ck in cks:
                v = table[ck].get(key)
                if v is None:
                    continue
                w.writerow([key, fam, ds, met, ck, f"{float(v):.10g}"])

    wide_path = args.out_dir / "trajectory_wide.csv"
    with wide_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric_key"] + [str(c) for c in cks])
        for key in all_keys:
            w.writerow([key] + [
                (f"{float(table[ck][key]):.10g}" if key in table[ck] else "") for ck in cks
            ])

    cov = {k: sum(1 for ck in cks if k in table[ck]) for k in all_keys}
    full = [k for k, c in cov.items() if c == len(cks)]
    meta = {
        "eval_root": str(args.eval_root),
        "n_checkpoints": len(cks),
        "checkpoints": cks,
        "n_metric_keys": len(all_keys),
        "n_full_coverage_keys": len(full),
        "coverage": cov,
        "families": sorted({k.split(":")[0] for k in all_keys}),
    }
    (args.out_dir / "trajectory_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {len(all_keys)} metric keys, {len(full)} with full {len(cks)}-checkpoint coverage")
    print(f"[done] wrote {long_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
