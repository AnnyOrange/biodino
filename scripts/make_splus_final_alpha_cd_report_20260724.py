#!/usr/bin/env python3
"""Build the final alpha=0.75 ViT-S compute and fixed-compute data report."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/00_reports/splus_final_alpha_cd_seed0_20260724"

C_RAW = ROOT / "outputs/01_training_runs/S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716"
C_TRAIN = ROOT / "outputs/01_training_runs/SplusFinal_s6_sigreg005_a075_fixed1m_compute_seed0_20260724"
C_EVAL = ROOT / "outputs/02_eval_runs/SplusFinal_s6_sigreg005_a075_fixed1m_compute_full_seed0_20260724"

D_EVAL = ROOT / "outputs/02_eval_runs/SplusFinal_s6_sigreg005_a075_fixedC8199_data_full_seed0_20260724"
D_SPECS = (
    (
        "0.1M",
        104_877,
        "random_10",
        ROOT / "outputs/01_training_runs/DscaleFinal_splus_sigreg005_random10_fixed15M_b1024_seed0_qi4gbs64acc4",
        ROOT / "outputs/01_training_runs/SplusFinal_s6_sigreg005_a075_random10_fixedC8199_seed0_20260724",
    ),
    (
        "0.2M",
        209_754,
        "random_20",
        ROOT / "outputs/01_training_runs/DscaleFinal_splus_sigreg005_random20_fixed15M_b1024_seed0_qi4gbs64acc4",
        ROOT / "outputs/01_training_runs/SplusFinal_s6_sigreg005_a075_random20_fixedC8199_seed0_20260724",
    ),
    (
        "0.5M",
        524_385,
        "random_50",
        ROOT / "outputs/01_training_runs/DscaleFinal_splus_sigreg005_random50_fixed15M_b1024_seed0_local8gbs64acc2",
        ROOT / "outputs/01_training_runs/SplusFinal_s6_sigreg005_a075_random50_fixedC8199_seed0_20260724",
    ),
    ("1.0M", 1_048_771, "random_100", C_RAW, C_TRAIN),
)

INK = "#26343B"
TEAL = "#176B6B"
DATA_COLOR = "#D06C3B"
GRID = "#D9DEE2"
MUTED = "#69767C"
ANCHOR = "#C83E36"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compute = load_module(
    "splus_compute_report_base",
    ROOT / "scripts/make_splus_fixed1m_full_compute_scaling_20260723.py",
)
data = load_module(
    "splus_data_report_base",
    ROOT / "scripts/make_splus_data_scaling_full_current_20260723.py",
)


def training_rows(raw_root: Path, checkpoints: set[int]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    path = raw_root / "raw_loss_metrics.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        checkpoint = int(row.get("optimizer_update", -1))
        if checkpoint in checkpoints:
            if int(row["effective_global_batch_size"]) != 1024:
                raise ValueError(f"Unexpected GBS at {raw_root}/ck{checkpoint}")
            found[checkpoint] = row
    if set(found) != checkpoints:
        raise ValueError(f"Missing training rows in {path}: {sorted(checkpoints - set(found))}")
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_compute() -> None:
    compute.TRAIN_ROOT = C_TRAIN
    compute.FULL_ROOT = C_EVAL
    compute.CK8199_ROOT = C_EVAL / "ckpt_8199"
    compute.CK8199_OOD_ROOT = C_EVAL / "ckpt_8199"
    compute.OUT = OUT / "compute"
    compute.load_training = lambda: training_rows(C_RAW, set(compute.CHECKPOINTS))

    def write_readme(current: list[dict], comparison: list[dict]) -> None:
        lines = [
            "# Final S+ fixed-1M compute scaling (seed0)",
            "",
            "Every point is `0.25 * official + 0.75 * EMA teacher` from the same",
            "S6 + SIGReg 0.05 trajectory. The data pool, architecture, optimizer",
            "schedule, evaluation suite, and alpha are fixed; only training compute changes.",
            "",
            "Evaluation is C25 + Reg-2 + Ret-4 + Seg-8 + X-ray/cryo OOD using bf16,",
            "batch 64, auto-channel TTA8, and seed0. Proxy-9 is not used.",
            "",
            "| ckpt | passes | visits (M) | C25 | Reg-2 | Ret-4 | Seg-8 | OOD | ID-4 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in current:
            lines.append(
                f"| {row['checkpoint']} | {float(row['passes']):.0f} | "
                f"{float(row['image_visits_m']):.4f} | "
                f"{float(row['classification25_macro_f1']):.6f} | "
                f"{float(row['regression2_spearman']):.6f} | "
                f"{float(row['retrieval4_map_at_5']):.6f} | "
                f"{float(row['segmentation8_mdice']):.6f} | "
                f"{float(row['ood_composite']):.6f} | {float(row['id4_overall']):.6f} |"
            )
        lines.extend(
            [
                "",
                "The dashed S6 curve is historical context only; it is not included in the",
                "final S+ fit because its recipe, GBS, and evaluation provenance differ.",
                "",
                "## Outputs",
                "",
                "- `splus_compute_scaling_tasks.{png,pdf,svg}`",
                "- `splus_compute_scaling_overall.{png,pdf,svg}`",
                "- `splus_full_compute_summary.csv`",
                "- `splus_full_compute_per_dataset.csv`",
                "- `matched_15pass_vs_s6.csv`",
            ]
        )
        (compute.OUT / "README.md").write_text("\n".join(lines) + "\n")

    compute.write_readme = write_readme


def configure_data() -> None:
    pools = []
    raw_by_label: dict[str, Path] = {}
    for label, samples, slug, raw_root, interp_root in D_SPECS:
        result_root = C_EVAL / "ckpt_8199" if label == "1.0M" else D_EVAL / slug / "ckpt_8199"
        pools.append(data.Pool(label, samples, result_root, result_root, interp_root))
        raw_by_label[label] = raw_root
    data.POOLS = tuple(pools)
    data.OUT = OUT / "data"

    def training_metadata(pool) -> dict:
        return training_rows(raw_by_label[pool.label], {data.CHECKPOINT})[data.CHECKPOINT]

    def checkpoint_integrity() -> list[dict]:
        rows = []
        for pool in data.POOLS:
            path = data.expected_checkpoint(pool)
            manifest_path = pool.train_dir / "interpolation_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if float(manifest["alpha"]) != 0.75:
                raise ValueError(f"Wrong alpha in {manifest_path}")
            raw_path = Path(manifest["checkpoints_dir"]) / str(data.CHECKPOINT) / "checkpoint.pth"
            if not raw_path.is_absolute():
                raw_path = ROOT / raw_path
            dataset_path = next(
                line.split("dataset_path:", 1)[1].strip()
                for line in (pool.train_dir / "config.yaml").read_text().splitlines()
                if "dataset_path:" in line
            )
            rows.append(
                {
                    "pool": pool.label,
                    "checkpoint": data.CHECKPOINT,
                    "checkpoint_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "alpha": manifest["alpha"],
                    "definition": manifest["definition"],
                    "official_checkpoint": manifest["official_checkpoint"],
                    "raw_checkpoint": str(raw_path),
                    "dataset_path": dataset_path,
                    "checkpoint_path": str(path),
                }
            )
        return rows

    def plot_tasks(current: list[dict]) -> None:
        samples = np.array([float(row["samples"]) for row in current])
        panels = (
            ("classification", "(a) Classification (25 datasets)", "macro-F1"),
            ("regression", "(b) Regression (2 datasets)", r"Spearman $\rho$"),
            ("retrieval_clustering", "(c) Retrieval (4 datasets)", "mAP@5"),
            ("segmentation", "(d) Segmentation (8 datasets)", "mDice"),
            ("ood", "(e) OOD (X-ray + cryo-EM)", "non-saturated composite"),
        )
        figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.5))
        for axis, (key, title, ylabel) in zip(axes.flat, panels):
            values = np.array([float(row[key]) for row in current])
            spread = float(values.max() - values.min())
            pad = max(spread * 0.35, 0.004)
            data.style_axis(axis, samples)
            axis.set_ylim(float(values.min()) - pad, float(values.max()) + pad * 1.35)
            axis.plot(samples, values, color=data.COLORS[key], linewidth=2.2, marker="o", markersize=8)
            for x, value in zip(samples, values):
                axis.annotate(f"{value:.4f}", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_ylabel(ylabel)
        axes.flat[5].axis("off")
        axes.flat[5].text(0.04, 0.84, "STRICT FIXED-COMPUTE VIEW", color=data.COLORS["id4"], fontsize=13, fontweight="bold", va="top")
        axes.flat[5].text(
            0.04,
            0.68,
            "S6 + SIGReg 0.05 + EMA\nalpha=0.75 official interpolation\nck8199 / 8.397M visits per point\nbf16 / auto-channel TTA8 / batch 64",
            color=data.INK,
            fontsize=10.5,
            va="top",
            linespacing=1.3,
        )
        pass_text = "  ".join(f"{row['pool']}: {float(row['dataset_equivalent_passes']):.1f}x" for row in current)
        axes.flat[5].text(0.04, 0.27, "actual pool passes", color=data.MUTED, fontsize=9, va="top")
        axes.flat[5].text(0.04, 0.20, pass_text, color=data.MUTED, fontsize=9, va="top", wrap=True)
        figure.suptitle(r"$\mathrm{S^+}$ full-suite unique-data scaling at fixed compute", fontsize=16, fontweight="bold")
        figure.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.075, wspace=0.28, hspace=0.38)
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(data.OUT / f"splus_data_scaling_tasks.{suffix}", dpi=220 if suffix == "png" else None, facecolor="white")
        plt.close(figure)

    def write_readme(current: list[dict], protocol: list[dict]) -> None:
        values = [float(row["id4"]) for row in current]
        lines = [
            "# Final S+ fixed-compute unique-data scaling (seed0)",
            "",
            "All four points use exactly 8,200 optimizer updates, 8,396,800 image",
            "visits, effective GBS 1024, and the same update-based scheduler. The only",
            "training variable is the nested unique-image pool size.",
            "",
            "Every evaluated checkpoint is `0.25 * official + 0.75 * EMA teacher`.",
            "The evaluation is C25 + Reg-2 + Ret-4 + Seg-8 + X-ray/cryo OOD using",
            "bf16, batch 64, auto-channel TTA8, and seed0. Proxy-9 is not used.",
            "",
            "| pool | visits | pool passes | C25 | Reg-2 | Ret-4 | Seg-8 | OOD | ID-4 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in current:
            lines.append(
                f"| {row['pool']} | {int(row['image_visits']):,} | "
                f"{float(row['dataset_equivalent_passes']):.3f} | "
                f"{float(row['classification']):.6f} | {float(row['regression']):.6f} | "
                f"{float(row['retrieval_clustering']):.6f} | {float(row['segmentation']):.6f} | "
                f"{float(row['ood']):.6f} | {float(row['id4']):.6f} |"
            )
        lines.extend(
            [
                "",
                f"Seed0 endpoint delta (1.0M - 0.1M) is {values[-1] - values[0]:+.6f}; "
                f"the max-min range is {max(values) - min(values):.6f}.",
                "Training/probe seeds 1 and 2 are required before interpreting small adjacent differences.",
                "",
                "The historical S6 curve remains a separate fixed-pass diagnostic; it is not",
                "overlaid as a matched-data comparison because its compute budget changes with pool size.",
                "",
                "## Outputs",
                "",
                "- `splus_data_scaling_tasks.{png,pdf,svg}`",
                "- `splus_data_scaling_overall.{png,pdf,svg}`",
                "- `splus_data_scaling_protocol_side_by_side.{png,pdf,svg}`",
                "- `current_full_summary.csv`",
                "- `current_full_per_dataset.csv`",
                "- `checkpoint_integrity.csv`",
            ]
        )
        (data.OUT / "README.md").write_text("\n".join(lines) + "\n")

    data.training_metadata = training_metadata
    data.build_checkpoint_integrity = checkpoint_integrity
    data.plot_tasks = plot_tasks
    data.write_readme = write_readme


def log_fit(curve: str, x: np.ndarray, y: np.ndarray) -> dict:
    log_x = np.log10(x)
    slope, intercept = np.polyfit(log_x, y, 1)
    prediction = intercept + slope * log_x
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum((y - prediction) ** 2)) / denominator
    return {
        "curve": curve,
        "n_points": len(x),
        "slope_score_per_log10_x": slope,
        "intercept": intercept,
        "r2": r2,
        "endpoint_delta": y[-1] - y[0],
        "monotonic_increases": int(np.sum(np.diff(y) > 0)),
        "adjacent_steps": len(y) - 1,
    }


def combined_outputs() -> None:
    with (compute.OUT / "splus_full_compute_summary.csv").open(newline="") as handle:
        c_rows = list(csv.DictReader(handle))
    with (data.OUT / "current_full_summary.csv").open(newline="") as handle:
        d_rows = list(csv.DictReader(handle))

    c_x = np.array([float(row["image_visits_m"]) for row in c_rows])
    c_y = np.array([float(row["id4_overall"]) for row in c_rows])
    d_x = np.array([float(row["samples"]) for row in d_rows])
    d_y = np.array([float(row["id4"]) for row in d_rows])
    fits = [log_fit("compute_C", c_x, c_y), log_fit("unique_data_Du", d_x, d_y)]
    write_csv(OUT / "scaling_fits_seed0.csv", fits)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)
    low = min(float(c_y.min()), float(d_y.min()))
    high = max(float(c_y.max()), float(d_y.max()))
    pad = max((high - low) * 0.22, 0.003)
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0, colors=INK)
        axis.set_ylim(low - pad, high + pad)

    axes[0].set_xscale("log")
    axes[0].plot(c_x, c_y, color=TEAL, linewidth=2.7, marker="o", markersize=8)
    axes[0].scatter([8.3968], [c_y[4]], marker="*", s=210, color=ANCHOR, edgecolors="white", linewidths=0.8, zorder=5)
    axes[0].set_xticks(c_x, ["1", "2", "4", "6", "8", "10", "12", "15"])
    axes[0].set_xlabel("Compute C (1M-reference passes, log)")
    axes[0].set_ylabel("ID-4 family-balanced transfer")
    axes[0].set_title("(a) Compute scaling | fixed ViT-S + 1M", fontweight="bold")

    axes[1].set_xscale("log")
    axes[1].plot(d_x, d_y, color=DATA_COLOR, linewidth=2.7, marker="o", markersize=8)
    axes[1].scatter([d_x[-1]], [d_y[-1]], marker="*", s=210, color=ANCHOR, edgecolors="white", linewidths=0.8, zorder=5)
    axes[1].set_xticks(d_x, ["0.1M", "0.2M", "0.5M", "1M"])
    axes[1].set_xlabel(r"Unique microscopy images $D_u$ (log)")
    axes[1].set_title("(b) Unique-data scaling | fixed 8.397M visits", fontweight="bold")
    figure.suptitle(r"Final $\mathrm{S^+}$ scaling slices (seed0)", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.018,
        "S6 + SIGReg 0.05 + EMA teacher; theta = 0.25 official + 0.75 EMA; full suite, no Proxy-9.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.12)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"splus_cd_scaling_main.{suffix}", dpi=240 if suffix == "png" else None, facecolor="white")
    plt.close(figure)

    manifest_rows = []
    c_raw_rows = training_rows(C_RAW, set(compute.CHECKPOINTS))
    for checkpoint in compute.CHECKPOINTS:
        path = C_TRAIN / "ckpt" / str(checkpoint) / "checkpoint.pth"
        manifest_rows.append(
            {
                "curve": "C",
                "point": f"ck{checkpoint}",
                "unique_images": 1_048_771,
                "optimizer_updates": checkpoint + 1,
                "image_visits": int(c_raw_rows[checkpoint]["image_visits"]),
                "alpha": 0.75,
                "checkpoint_sha256": sha256(path),
                "checkpoint_path": str(path),
                "evaluation_root": str(C_EVAL / f"ckpt_{checkpoint}"),
            }
        )
    for pool, row in zip(data.POOLS, d_rows):
        path = pool.train_dir / "ckpt/8199/checkpoint.pth"
        manifest_rows.append(
            {
                "curve": "Du",
                "point": pool.label,
                "unique_images": pool.samples,
                "optimizer_updates": 8_200,
                "image_visits": int(row["image_visits"]),
                "alpha": 0.75,
                "checkpoint_sha256": sha256(path),
                "checkpoint_path": str(path),
                "evaluation_root": str(pool.result_root),
            }
        )
    write_csv(OUT / "experiment_manifest.csv", manifest_rows)

    lines = [
        "# Final S+ C/D scaling report (seed0)",
        "",
        "This directory contains the final-alpha ViT-S compute and fixed-compute",
        "unique-data slices. Model-size N scaling is intentionally excluded.",
        "",
        "- S+ definition: S6 + SIGReg 0.05 + EMA teacher + fixed alpha 0.75.",
        "- Interpolation: `theta = 0.25 * official + 0.75 * EMA teacher`.",
        "- Evaluation: C25, Reg-2, Ret-4, Seg-8, and X-ray/cryo OOD.",
        "- Protocol: bf16, batch 64, auto-channel TTA8, seed0; no Proxy-9.",
        "- Shared anchor: ViT-S, 1,048,771 images, 8,200 updates, 8,396,800 visits.",
        "",
        "## Seed0 trend diagnostics",
        "",
        "| curve | endpoint delta | log10 slope | R2 | increasing steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fits:
        lines.append(
            f"| {row['curve']} | {float(row['endpoint_delta']):+.6f} | "
            f"{float(row['slope_score_per_log10_x']):+.6f} | {float(row['r2']):.4f} | "
            f"{row['monotonic_increases']}/{row['adjacent_steps']} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive seed0 fits, not uncertainty-aware scaling exponents.",
            "Training/probe seeds 1 and 2 should be added before the final paper claim.",
            "",
            "## Main files",
            "",
            "- `splus_cd_scaling_main.{png,pdf,svg}`",
            "- `compute/splus_compute_scaling_{overall,tasks}.{png,pdf,svg}`",
            "- `data/splus_data_scaling_{overall,tasks}.{png,pdf,svg}`",
            "- `loss/splus_loss_cd_scaling_main.{png,pdf,svg}`",
            "- `experiment_manifest.csv`",
            "- `scaling_fits_seed0.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_compute()
    configure_data()
    compute.main()
    data.main()
    combined_outputs()
    print(f"wrote final alpha C/D report to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
