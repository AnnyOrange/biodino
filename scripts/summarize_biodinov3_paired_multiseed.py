#!/usr/bin/env python3
"""Summarize BioDINO 20ep screening and matched 50ep paired adaptation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


BASE = Path("/mnt/huawei_deepcad/dinov3/outputs/instance_seg_tuning")
ROOT = BASE / "biodinov3_paired_multiseed"
OUT_CSV = ROOT / "paired_seed_results.csv"
OUT_MD = ROOT / "paired_seed_summary.md"

CONFIG = {
    "cellpose": {
        "metric": "CellposeStyleAP",
        "expected_steps": 4000,
        "vitl16_fullft": 0.277973,
        "screen20": BASE / "biodinov3_seven_dataset_results_run/cellpose/no_trick_seed0/results.json",
        "full0": BASE / "biodinov3_validated_run/cellpose_fullft_50ep/results.json",
    },
    "livecell": {
        "metric": "SEG",
        "expected_steps": 20350,
        "vitl16_fullft": 0.626160,
        "screen20": BASE / "biodinov3_seven_dataset_results_run/livecell/no_trick_seed0/results.json",
        "full0": BASE / "biodinov3_validated_run/livecell_fullft_50ep/results.json",
    },
}


def read_complete(path: Path, expected_steps: int, seed: int) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        meta = data["_meta"]
        if meta["seed"] != seed:
            return None
        if meta["global_steps"] != expected_steps or meta["total_steps"] != expected_steps:
            return None
        exit_path = path.parent / "exit_code.txt"
        if seed in (1, 2) and (not exit_path.exists() or exit_path.read_text().strip() != "0"):
            return None
        return data
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12f}"


def main() -> int:
    rows = []
    sections = []
    for dataset, cfg in CONFIG.items():
        metric = cfg["metric"]
        expected_steps = cfg["expected_steps"]
        screen = json.loads(cfg["screen20"].read_text())
        full0 = read_complete(cfg["full0"], expected_steps, 0)
        screen_value = screen["val"][metric]
        full0_value = full0["val"][metric] if full0 else None
        screening_gain = full0_value - screen_value if full0_value is not None else None

        frozen_values = []
        full_values = []
        gains = []
        table_rows = []
        for seed in (0, 1, 2):
            frozen_path = ROOT / dataset / f"frozen50ep_seed{seed}/results.json"
            full_path = cfg["full0"] if seed == 0 else ROOT / dataset / f"finetune_seed{seed}_retry1/results.json"
            frozen = read_complete(frozen_path, expected_steps, seed)
            full = read_complete(full_path, expected_steps, seed)
            frozen_value = frozen["val"][metric] if frozen else None
            full_value = full["val"][metric] if full else None
            gain = full_value - frozen_value if frozen_value is not None and full_value is not None else None
            if frozen_value is not None:
                frozen_values.append(frozen_value)
            if full_value is not None:
                full_values.append(full_value)
            if gain is not None:
                gains.append(gain)
            frozen_meta = frozen.get("_meta", {}) if frozen else {}
            full_meta = full.get("_meta", {}) if full else {}
            row = {
                "dataset": dataset,
                "metric": metric,
                "seed": seed,
                "standardized_frozen_probe_20ep_seed0": fmt(screen_value) if seed == 0 else "",
                "fullft50_minus_probe20_epoch_confounded": fmt(screening_gain) if seed == 0 else "",
                "frozen50ep": fmt(frozen_value),
                "fullft50ep": fmt(full_value),
                "paired_gain": fmt(gain),
                "fullft50_minus_vitl16_fullft": fmt(full_value - cfg["vitl16_fullft"] if full_value is not None else None),
                "frozen_peak_gib": fmt(frozen_meta.get("peak_cuda_memory_gib")),
                "frozen_wall_seconds": fmt(frozen_meta.get("run_wall_seconds")),
                "fullft_peak_gib": fmt(full_meta.get("peak_cuda_memory_gib")),
                "fullft_wall_seconds": fmt(full_meta.get("run_wall_seconds")),
                "status": "paired_complete" if gain is not None else "pending",
            }
            rows.append(row)
            table_rows.append((seed, frozen_value, full_value, gain, row["status"]))

        complete = len(frozen_values) == len(full_values) == len(gains) == 3
        if complete:
            frozen_mean = statistics.mean(frozen_values)
            frozen_std = statistics.stdev(frozen_values)
            full_mean = statistics.mean(full_values)
            full_std = statistics.stdev(full_values)
            gain_mean = statistics.mean(gains)
            gain_std = statistics.stdev(gains)
            positive = sum(value > 0 for value in gains)
            stable = positive == 3 and gain_mean >= 0.005
            stability = "stable_effective" if stable else "not_stable_effective"
            stats_text = (
                f"Frozen 50ep `{frozen_mean:.6f} +/- {frozen_std:.6f}`; Full FT 50ep "
                f"`{full_mean:.6f} +/- {full_std:.6f}`; paired gain `{gain_mean:+.6f} +/- {gain_std:.6f}`; "
                f"positive seeds `{positive}/3`; status `{stability}`; Full FT mean - ViT-L/16 Full FT "
                f"`{full_mean - cfg['vitl16_fullft']:+.6f}`."
            )
        else:
            stats_text = "Matched three-seed statistics are pending until all Frozen 50ep and Full FT 50ep results pass step and exit-code validation."

        table = [
            f"## {dataset}",
            "",
            f"Historical screening only: Full FT 50ep - standardized_frozen_probe_20ep = `{screening_gain:+.12f}` (epoch-confounded).",
            "",
            "| Seed | Frozen 50ep | Full FT 50ep | Paired gain | Status |",
            "|---:|---:|---:|---:|---|",
        ]
        for seed, frozen_value, full_value, gain, status in table_rows:
            table.append(
                f"| {seed} | {fmt(frozen_value) or 'pending'} | {fmt(full_value) or 'pending'} | "
                f"{fmt(gain) or 'pending'} | {status} |"
            )
        table.extend(["", stats_text])
        sections.append("\n".join(table))

    fields = list(rows[0])
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    OUT_MD.write_text(
        "# BioDINO Matched 50-Epoch Paired Adaptation\n\n"
        "Only Full FT 50ep - Frozen 50ep at the same seed is a formal paired adaptation gain. "
        "The historical Full FT 50ep - standardized_frozen_probe_20ep difference is reported separately "
        "and is explicitly epoch-confounded. Sample standard deviation is used across seeds.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
