#!/usr/bin/env python3
"""Patch 10-shot cells in dscale lattice in-place. Keeps user notes."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
CSV_LATTICE = ROOT / "plot/fig2/data/dscale_e8_data_lattice_20260904.csv"
KSHOT = ROOT / "outputs/auto_eval_logs/hs6_dscale_k10_fill_20260904/kshot.csv"
LAST = {"0.1M": 823, "0.2M": 1639, "0.5M": 4103, "1M": 8199}
REV = {v: k for k, v in LAST.items()}
MODELS = {"S+", "B", "L", "H+"}


def load_k10() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    if not KSHOT.is_file():
        return {}
    with KSHOT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("k")) != "10" or row.get("axis") != "data":
                continue
            if row.get("error_macro_f1") in (None, ""):
                continue
            try:
                ckpt = int(row["ckpt"])
            except Exception:
                continue
            pool = row.get("pool") or REV.get(ckpt)
            if pool not in LAST:
                continue
            out[(row["model"], row["dataset"], pool)][str(row.get("seed", ""))] = 100.0 * float(
                row["error_macro_f1"]
            )
    mean = {}
    for key, seeds in out.items():
        vals = [seeds[s] for s in ("0", "1", "2") if s in seeds]
        if len(vals) == 3:
            mean[key] = sum(vals) / 3.0
    return mean


def patch() -> int:
    k10 = load_k10()
    if not k10:
        print("no complete k10 triples yet")
        return 0
    text = CSV_LATTICE.read_text()
    lines = text.splitlines()
    ds = None
    n = 0
    new_lines = []
    for line in lines:
        if line.startswith("# ") and "  [" in line:
            ds = line.split()[1]
        model = line.split(",", 1)[0]
        if ds and model in MODELS and "," in line:
                parts = line.split(",")
                # 连线,0.1M,0.2M,0.5M,1M,host,checkpoint_abs
                if len(parts) >= 5:
                    changed = False
                    for i, pool in enumerate(("0.1M", "0.2M", "0.5M", "1M"), start=1):
                        cell = parts[i]
                        if " / —" not in cell and " /—" not in cell:
                            continue
                        val = k10.get((model, ds, pool))
                        if val is None:
                            continue
                        full = cell.split("/")[0].strip()
                        parts[i] = f"{full} / {val:.1f}"
                        changed = True
                    if changed:
                        line = ",".join(parts)
                        n += 1
        new_lines.append(line)
    if n:
        CSV_LATTICE.write_text("\n".join(new_lines) + "\n")
    print(f"patched rows={n} complete_cells={len(k10)}")
    return n


if __name__ == "__main__":
    raise SystemExit(patch() and 0)
