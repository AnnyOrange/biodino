#!/usr/bin/env python3
"""Merge complete BloodMNIST 10-shot K10 checkpoint results and summarize them."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/02_eval_runs/bloodmnist_k10_cache_20261012"
OLD = OUT / "k10_seed_results.csv"
FULL_RAW = OUT / "k10_seed_results_complete.csv"
CHECKPOINTS = OUT / "k10_checkpoint_summary_complete.csv"
BEST = OUT / "k10_best_by_arm_complete.csv"
CLS_ARMS = (
    "global_cls",
    "global_cls_adaptive_w1to3",
    "global_cls_slow",
    "global_cls_slow2",
    "global_cls_slow2_w03",
    "global_cls_slow2_w3",
    "global_cls_w03",
    "global_cls_w3",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_old_row(row: dict[str, str], arm: str) -> dict[str, str]:
    return {"arm": arm, **{key: value for key, value in row.items() if key != "arm"}}


def main() -> None:
    rows: list[dict[str, str]] = []
    for row in read_rows(OLD):
        # Replace the earlier sparse 20TB samples with the full checkpoint sweep.
        if row["arm"] not in {"20TB_noGRAM", "20TB_CLS_slow2"}:
            rows.append(row)

    sources: list[tuple[Path, str, set[str] | None]] = [
        (OUT / "5tb_noGRAM_lyxxr_k10_seed_raw.csv", "5TB_noGRAM", None),
        (
            OUT / "noGRAM20tb_k10_seed_raw_full.csv",
            "20TB_noGRAM",
            {
                p.name
                for p in (OUT / "noGRAM20tb/bio_classification/bloodmnist").iterdir()
                if p.is_dir() and (p / "summary.csv").is_file()
            },
        ),
        (
            OUT / "cls_slow2_20tb_k10_seed_raw_full.csv",
            "20TB_CLS_slow2",
            {
                p.name
                for p in (OUT / "cls_slow2_20tb/bio_classification/bloodmnist").iterdir()
                if p.is_dir() and (p / "summary.csv").is_file()
            },
        ),
    ]
    sources.extend(
        (OUT / f"5tb_cls_{arm}_k10_raw.csv", f"5TB_{arm}", None) for arm in CLS_ARMS
    )
    for path, arm, allowed_ckpts in sources:
        for row in read_rows(path):
            if allowed_ckpts is None or row["ckpt"] in allowed_ckpts:
                rows.append(as_old_row(row, arm))

    keys = [(r["arm"], r["ckpt"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate arm/checkpoint/seed rows in merged results")
    rows.sort(key=lambda r: (r["arm"], int(r["ckpt"]), int(r["seed"])))

    fields = ["arm"] + [key for key in rows[0] if key != "arm"]
    with FULL_RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] == "bloodmnist" and row["k"] == "10":
            grouped[(row["arm"], row["ckpt"])].append(row)

    summaries = []
    for (arm, ckpt), group in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if len(group) != 3:
            raise ValueError(f"Expected 3 seeds for {arm} ck{ckpt}, got {len(group)}")
        errors = {int(r["seed"]): float(r["error_balanced"]) for r in group}
        bas = [float(r["balanced_accuracy"]) for r in group]
        summaries.append(
            {
                "arm": arm,
                "ckpt": ckpt,
                "n_seeds": len(group),
                "seed0_error_pct": f"{100 * errors[0]:.6f}",
                "seed1_error_pct": f"{100 * errors[1]:.6f}",
                "seed2_error_pct": f"{100 * errors[2]:.6f}",
                "mean_balanced_accuracy": f"{sum(bas) / 3:.9f}",
                "mean_balanced_error_pct": f"{100 * sum(errors.values()) / 3:.6f}",
            }
        )

    summary_fields = list(summaries[0])
    with CHECKPOINTS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    best_by_arm: dict[str, dict[str, str]] = {}
    for row in summaries:
        arm = row["arm"]
        if arm not in best_by_arm or float(row["mean_balanced_error_pct"]) < float(
            best_by_arm[arm]["mean_balanced_error_pct"]
        ):
            best_by_arm[arm] = row
    best_rows = sorted(best_by_arm.values(), key=lambda row: row["arm"])
    with BEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(best_rows)

    counts: dict[str, int] = defaultdict(int)
    for row in summaries:
        counts[row["arm"]] += 1
    print(f"raw_rows={len(rows)} checkpoints={len(summaries)}")
    for arm, count in sorted(counts.items()):
        print(f"{arm},{count}")
    print(f"raw={FULL_RAW}")
    print(f"per_checkpoint={CHECKPOINTS}")
    print(f"best={BEST}")


if __name__ == "__main__":
    main()
