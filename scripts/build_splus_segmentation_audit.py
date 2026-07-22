#!/usr/bin/env python3
"""Build protocol-matched S+ segmentation tables from deterministic probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from run_splus_segmentation_probe_audit import DATASETS, DEFAULT_OUT, MODELS, ROOT


COMMON7 = tuple(dataset for dataset in DATASETS if dataset != "multimodal_cellseg")
OUT = ROOT / "outputs/00_reports/splus_segmentation_audit_20260716"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.5f}"


def fmt_delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.5f}"


def spread(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def macro_seed_map(
    raw: dict[tuple[str, int, str], dict], model_key: str, seeds: list[int], datasets: tuple[str, ...], metric: str
) -> dict[int, float]:
    values = {}
    for seed in seeds:
        per_dataset = [raw.get((model_key, seed, dataset), {}).get(metric) for dataset in datasets]
        if all(value is not None for value in per_dataset):
            values[seed] = mean(per_dataset)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    raw = {}
    provenance = []
    for model_key, model in MODELS.items():
        for seed in args.seeds:
            for dataset in DATASETS:
                path = args.input_root / model_key / f"seed{seed}" / dataset / "results.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                test = result.get("test", {})
                raw[(model_key, seed, dataset)] = {
                    "mDice": test.get("mDice"),
                    "mIoU": test.get("mIoU"),
                }
                meta = result.get("_meta", {})
                provenance.append(
                    {
                        "model_key": model_key,
                        "model": model.label,
                        "checkpoint": model.checkpoint,
                        "dataset": dataset,
                        "seed": seed,
                        "mDice": test.get("mDice"),
                        "mIoU": test.get("mIoU"),
                        "probe_rng_seeded": meta.get("probe_rng_seeded"),
                        "probe_batch_size": meta.get("probe_batch_size"),
                        "probe_epochs": meta.get("probe_epochs"),
                        "result_path": path.relative_to(ROOT),
                    }
                )

    per_dataset = []
    grouped = defaultdict(lambda: {"mDice": [], "mIoU": []})
    for (model_key, _seed, dataset), values in raw.items():
        for metric in ("mDice", "mIoU"):
            if values[metric] is not None:
                grouped[(model_key, dataset)][metric].append(values[metric])
    for model_key, model in MODELS.items():
        for dataset in DATASETS:
            values = grouped[(model_key, dataset)]
            per_dataset.append(
                {
                    "model_key": model_key,
                    "model": model.label,
                    "checkpoint": model.checkpoint,
                    "dataset": dataset,
                    "seeds_complete": len(values["mDice"]),
                    "mDice_mean": mean(values["mDice"]) if values["mDice"] else None,
                    "mDice_std": spread(values["mDice"]),
                    "mIoU_mean": mean(values["mIoU"]) if values["mIoU"] else None,
                    "mIoU_std": spread(values["mIoU"]),
                }
            )

    per_dataset_map = {(row["model_key"], row["dataset"]): row for row in per_dataset}
    for row in per_dataset:
        official = per_dataset_map.get(("official", row["dataset"]), {})
        row["mDice_delta_vs_official"] = (
            row["mDice_mean"] - official.get("mDice_mean")
            if row["mDice_mean"] is not None and official.get("mDice_mean") is not None
            else None
        )
        row["mIoU_delta_vs_official"] = (
            row["mIoU_mean"] - official.get("mIoU_mean")
            if row["mIoU_mean"] is not None and official.get("mIoU_mean") is not None
            else None
        )

    def build_summary(datasets: tuple[str, ...], exclude_h: bool) -> list[dict]:
        summary = []
        official_dice_by_seed = macro_seed_map(raw, "official", args.seeds, datasets, "mDice")
        official_iou_by_seed = macro_seed_map(raw, "official", args.seeds, datasets, "mIoU")
        for model_key, model in MODELS.items():
            if exclude_h and model_key == "h_s0":
                continue
            dice_by_seed = macro_seed_map(raw, model_key, args.seeds, datasets, "mDice")
            iou_by_seed = macro_seed_map(raw, model_key, args.seeds, datasets, "mIoU")
            dice_macros = list(dice_by_seed.values())
            iou_macros = list(iou_by_seed.values())
            paired_dice = [
                dice_by_seed[seed] - official_dice_by_seed[seed]
                for seed in args.seeds
                if seed in dice_by_seed and seed in official_dice_by_seed
            ]
            paired_iou = [
                iou_by_seed[seed] - official_iou_by_seed[seed]
                for seed in args.seeds
                if seed in iou_by_seed and seed in official_iou_by_seed
            ]
            summary.append(
                {
                    "model_key": model_key,
                    "model": model.label,
                    "checkpoint": model.checkpoint,
                    "passes": model.passes,
                    "datasets": len(datasets),
                    "seeds_complete": len(dice_macros),
                    "macro_mDice_mean": mean(dice_macros) if dice_macros else None,
                    "macro_mDice_std": spread(dice_macros),
                    "macro_mIoU_mean": mean(iou_macros) if iou_macros else None,
                    "macro_mIoU_std": spread(iou_macros),
                    "mDice_paired_delta_mean": mean(paired_dice) if paired_dice else None,
                    "mDice_paired_delta_std": spread(paired_dice),
                    "mDice_seeds_above_official": sum(value > 0 for value in paired_dice),
                    "mIoU_paired_delta_mean": mean(paired_iou) if paired_iou else None,
                    "mIoU_paired_delta_std": spread(paired_iou),
                    "mIoU_seeds_above_official": sum(value > 0 for value in paired_iou),
                }
            )
        for row in summary:
            row["mDice_delta_vs_official"] = (
                row["mDice_paired_delta_mean"]
            )
            row["mIoU_delta_vs_official"] = (
                row["mIoU_paired_delta_mean"]
            )
            row["mDice_wins_vs_official"] = sum(
                per_dataset_map[(row["model_key"], dataset)]["mDice_delta_vs_official"] > 0
                for dataset in datasets
                if per_dataset_map[(row["model_key"], dataset)]["mDice_delta_vs_official"] is not None
            )
            row["mIoU_wins_vs_official"] = sum(
                per_dataset_map[(row["model_key"], dataset)]["mIoU_delta_vs_official"] > 0
                for dataset in datasets
                if per_dataset_map[(row["model_key"], dataset)]["mIoU_delta_vs_official"] is not None
            )
        return summary

    common7 = build_summary(COMMON7, exclude_h=False)
    dense8 = build_summary(DATASETS, exclude_h=True)

    contrast_specs = [
        ("H-S0 minus Official", "h_s0", "official", COMMON7, "Common-7"),
        ("R-S0 minus Official", "r_s0", "official", DATASETS, "Dense-8"),
        ("robust decoder: S1 minus R-S0", "s1", "r_s0", DATASETS, "Dense-8"),
        ("BioAug under robust: S2-wu2 minus S1", "s2_wu2", "s1", DATASETS, "Dense-8"),
        ("warmup 5 minus warmup 2", "s2_wu5", "s2_wu2", DATASETS, "Dense-8"),
        ("crop 224 minus crop 256", "s3", "s2_wu2", DATASETS, "Dense-8"),
        ("30-pass horizon at pass 15: S6-3899 minus S2-wu2", "s6_ck3899", "s2_wu2", DATASETS, "Dense-8"),
        ("extend pass 15 to pass 20: S6-5199 minus S6-3899", "s6_ck5199", "s6_ck3899", DATASETS, "Dense-8"),
    ]
    contrasts = []
    for contrast, candidate, reference, datasets, protocol in contrast_specs:
        candidate_dice = macro_seed_map(raw, candidate, args.seeds, datasets, "mDice")
        reference_dice = macro_seed_map(raw, reference, args.seeds, datasets, "mDice")
        candidate_iou = macro_seed_map(raw, candidate, args.seeds, datasets, "mIoU")
        reference_iou = macro_seed_map(raw, reference, args.seeds, datasets, "mIoU")
        dice_deltas = [
            candidate_dice[seed] - reference_dice[seed]
            for seed in args.seeds
            if seed in candidate_dice and seed in reference_dice
        ]
        iou_deltas = [
            candidate_iou[seed] - reference_iou[seed]
            for seed in args.seeds
            if seed in candidate_iou and seed in reference_iou
        ]
        contrasts.append(
            {
                "contrast": contrast,
                "candidate": candidate,
                "reference": reference,
                "protocol": protocol,
                "seeds": len(dice_deltas),
                "mDice_delta_mean": mean(dice_deltas) if dice_deltas else None,
                "mDice_delta_std": spread(dice_deltas),
                "mDice_positive_seeds": sum(value > 0 for value in dice_deltas),
                "mIoU_delta_mean": mean(iou_deltas) if iou_deltas else None,
                "mIoU_delta_std": spread(iou_deltas),
                "mIoU_positive_seeds": sum(value > 0 for value in iou_deltas),
            }
        )

    write_csv(args.output_root / "segmentation_common7.csv", common7)
    write_csv(args.output_root / "segmentation_dense8.csv", dense8)
    write_csv(args.output_root / "segmentation_per_dataset.csv", per_dataset)
    write_csv(args.output_root / "segmentation_contrasts.csv", contrasts)
    write_csv(args.output_root / "segmentation_provenance.csv", provenance)

    lines = [
        "# S+ segmentation protocol audit",
        "",
        "All headline rows refit the same cached frozen features with the corrected deterministic probe: 50 epochs, batch 32, seed 0 (or the listed seed set), AdamW LR 1e-3, weight decay 1e-4, dataset-specific frozen-feature preset, and `first3` semantics for RGB backbones. No backbone features are re-extracted.",
        "",
        "## Common-7 headline",
        "",
        "This is the strict common set including H-S0: BBBC038, Cellpose, LiveCell, TissueNet, MoNuSeg, PanNuke, and CoNIC.",
        "",
        "| Backbone | Ckpt | Passes | Seeds | Macro mDice mean+-sd | Paired delta mean+-sd | Delta signs | Dataset wins | Macro mIoU mean+-sd | Paired delta mean+-sd | Delta signs | Dataset wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in common7:
        lines.append(
            f"| {row['model']} | {row['checkpoint']} | {row['passes']} | {row['seeds_complete']}/{len(args.seeds)} | "
            f"{fmt(row['macro_mDice_mean'])}+-{fmt(row['macro_mDice_std'])} | "
            f"{fmt_delta(row['mDice_delta_vs_official'])}+-{fmt(row['mDice_paired_delta_std'])} | "
            f"{row['mDice_seeds_above_official']}/{row['seeds_complete']} | {row['mDice_wins_vs_official']}/7 | "
            f"{fmt(row['macro_mIoU_mean'])}+-{fmt(row['macro_mIoU_std'])} | "
            f"{fmt_delta(row['mIoU_delta_vs_official'])}+-{fmt(row['mIoU_paired_delta_std'])} | "
            f"{row['mIoU_seeds_above_official']}/{row['seeds_complete']} | {row['mIoU_wins_vs_official']}/7 |"
        )

    lines.extend(
        [
            "",
            "## Dense-8 headline",
            "",
            "This adds Multimodal CellSeg and excludes H-S0, whose historical cache does not cover that dataset.",
            "",
            "| Backbone | Ckpt | Passes | Seeds | Macro mDice mean+-sd | Paired delta mean+-sd | Delta signs | Dataset wins | Macro mIoU mean+-sd | Paired delta mean+-sd | Delta signs | Dataset wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dense8:
        lines.append(
            f"| {row['model']} | {row['checkpoint']} | {row['passes']} | {row['seeds_complete']}/{len(args.seeds)} | "
            f"{fmt(row['macro_mDice_mean'])}+-{fmt(row['macro_mDice_std'])} | "
            f"{fmt_delta(row['mDice_delta_vs_official'])}+-{fmt(row['mDice_paired_delta_std'])} | "
            f"{row['mDice_seeds_above_official']}/{row['seeds_complete']} | {row['mDice_wins_vs_official']}/8 | "
            f"{fmt(row['macro_mIoU_mean'])}+-{fmt(row['macro_mIoU_std'])} | "
            f"{fmt_delta(row['mIoU_delta_vs_official'])}+-{fmt(row['mIoU_paired_delta_std'])} | "
            f"{row['mIoU_seeds_above_official']}/{row['seeds_complete']} | {row['mIoU_wins_vs_official']}/8 |"
        )

    lines.extend(
        [
            "",
            "## Paired element contrasts",
            "",
            "These contrasts compare the same probe seed before averaging. The S6-at-pass-15 contrast changes the cosine schedule horizon, not image exposure at that checkpoint.",
            "",
            "| Contrast | Set | mDice delta mean+-sd | Signs | mIoU delta mean+-sd | Signs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['contrast']} | {row['protocol']} | {fmt_delta(row['mDice_delta_mean'])}+-{fmt(row['mDice_delta_std'])} | "
            f"{row['mDice_positive_seeds']}/{row['seeds']} | {fmt_delta(row['mIoU_delta_mean'])}+-{fmt(row['mIoU_delta_std'])} | "
            f"{row['mIoU_positive_seeds']}/{row['seeds']} |"
        )

    lines.extend(
        [
            "",
            "## Result reading",
            "",
            "- Plain packwds continuation is effectively tied with Official: H-S0 and R-S0 have small positive mDice point estimates, but paired deltas change sign across seeds and their mIoU means are slightly lower.",
            "- Replacing packwds with the robust decoder alone hurts segmentation consistently. Adding BioAug under robust recovers more than that loss on all three seeds, so the useful element is the robust+BioAug interaction rather than robust alone.",
            "- Warmup 2 versus 5 and crop 224 versus 256 remain too close for a universal winner. Their differences are much smaller than the robust/BioAug contrast and depend on whether Common-7 or Dense-8 is used.",
            "- Among fixed 15-pass rows, S6 checkpoint 3899 has the highest macro mDice. Its advantage over S2-wu2 reflects a longer cosine schedule horizon at the same 15-pass exposure.",
            "- Extending S6 from pass 15 to pass 20 adds about 0.0009 macro mDice on all three seeds. This is consistent but small; checkpoint 5199 belongs to an extended-training row, not the fixed-15-pass ablation.",
            "- With only three probe seeds, sign stability and mean+-sd are descriptive uncertainty checks, not formal statistical significance claims.",
        ]
    )

    lines.extend(["", "## Per-dataset mDice", "", "| Backbone | " + " | ".join(COMMON7) + " |", "|---|" + "---:|" * len(COMMON7)])
    for model_key, model in MODELS.items():
        values = [per_dataset_map[(model_key, dataset)]["mDice_mean"] for dataset in COMMON7]
        lines.append("| " + model.label + " | " + " | ".join(fmt(value) for value in values) + " |")

    lines.extend(["", "## Per-dataset mIoU", "", "| Backbone | " + " | ".join(COMMON7) + " |", "|---|" + "---:|" * len(COMMON7)])
    for model_key, model in MODELS.items():
        values = [per_dataset_map[(model_key, dataset)]["mIoU_mean"] for dataset in COMMON7]
        lines.append("| " + model.label + " | " + " | ".join(fmt(value) for value in values) + " |")

    lines.extend(
        [
            "",
            "## Historical-number reconciliation",
            "",
            "- `0.5882` (Official) is the historical eight-dataset macro **mIoU**, while `0.7012` is macro **mDice** from those same eight datasets. They are not two segmentation protocols or interchangeable metrics.",
            "- The historical fixed-15-pass recipe table (`R-S0=0.59072`, `S1=0.58666`, `S2-wu2=0.59227`, `S2-wu5=0.59161`, `S3=0.59316`) is eight-dataset macro mIoU, but those heads used batch 16 and an unseeded RNG. Its ranking is diagnostic only.",
            "- The historical scaling pair Official `0.701201` versus S6 ck3899 `0.700944` used matched batch-32 heads and eight datasets, but the old evaluator did not seed head initialization/dropout/shuffling. This report supersedes that single-run comparison.",
            "- S6 ck5199 is shown as a separate approximately 20-pass row; it must not be presented as part of the fixed 15-pass ablation.",
            "- `Paired delta` subtracts Official at the same probe seed before averaging. `Delta signs` counts seeds with a positive paired difference; with only three seeds these are descriptive stability checks, not formal significance tests.",
            "",
            "## Correctness finding",
            "",
            "The previous evaluator's `--seed` controlled only optional train-subset sampling. It did not seed PyTorch head initialization, dropout, or DataLoader shuffling. Duplicate S6 ck5199 MoNuSeg runs therefore produced mDice 0.72713 and 0.69905 despite both reporting seed 0. The evaluator now seeds NumPy, PyTorch/CUDA, cuDNN behavior, and the shuffle generator; an exact-repeat smoke test produced byte-equivalent metric dictionaries.",
        ]
    )
    (args.output_root / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output_root / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
