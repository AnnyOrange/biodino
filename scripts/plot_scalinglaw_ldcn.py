#!/usr/bin/env python3
"""Build diagnostic scaling-law style plots from existing training logs.

The plots intentionally use only local artifacts:
  - outputs/03_comparisons/scaling_manifest_20260707/run_summary.csv
  - outputs/01_training_runs/*/training_metrics.json

This is a Chinchilla-style diagnostic, not a rigorous biological SSL scaling
law: DINO/SSL total_loss is not a calibrated held-out likelihood and is not
guaranteed to be comparable across architectures.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_SUMMARY = ROOT / "outputs/03_comparisons/scaling_manifest_20260707/run_summary.csv"
OUT = ROOT / "outputs/00_reports/scalinglaw_ldcn_20260707"

# Approximate parameter counts for plotting. These are intentionally rounded.
PARAMS = {
    "vit_small": 22e6,   # S+
    "vit_base": 86e6,   # B
    "vit_large": 300e6, # L
    "vit_huge2": 840e6, # H+
}

ARCH_LABEL = {
    "vit_small": "S+",
    "vit_base": "B",
    "vit_large": "L",
    "vit_huge2": "H+",
}

ARCH_COLOR = {
    "vit_small": "#2A9D8F",
    "vit_base": "#457B9D",
    "vit_large": "#E76F51",
    "vit_huge2": "#7B2CBF",
}

D_STYLE = {
    "1TB": "-",
    "5TB": "--",
    "10TB": "-.",
    "100TB": ":",
}

EXCLUDE_RUN_SUBSTRINGS = ("uint8",)
FIT_MIN_IMAGE_VISITS = 3e6


@dataclass
class RunMeta:
    run_id: str
    label: str
    arch: str
    arch_label: str
    decoder: str
    d_label: str
    d_pool: float
    params: float
    effective_global_batch: float
    crop: str
    lr: str
    warmup_epochs: str
    role: str


@dataclass
class Point:
    run_id: str
    run_label: str
    arch: str
    arch_label: str
    d_label: str
    d_pool: float
    params: float
    iteration: int
    image_visits: float
    data_passes: float
    compute_proxy: float
    total_loss: float
    lr: float


def friendly_label(run_id: str, arch: str, d_label: str) -> str:
    name = Path(run_id).name
    mapping = {
        "bio_continue_vits16_ep15_1025": "S+ 1TB 15ep",
        "bio_chinchilla_vitsplus16_packwds_b1024_b64acc2": "S+ 1TB 30ep",
        "bio_continue_1025_a100_grad_acc_2_base": "B 1TB",
        "bio_5tb_rgb1024_vitb16_ep30": "B 5TB partial",
        "bio_continue_vitL16_OEP1025_ep15_b1024_1025": "L 1TB",
        "bio_5tb_mixed_ori_slfm_rgb1024_vitl16_ep30_clean_trainstats": "L 5TB",
        "10tb_lossless_uint16_vitl16_b1024_ep30_20260610_074015": "L 10TB",
        "bio_continue_rgb3_vith16plus": "H+ 1TB",
        "5tb_hplus_packwds_ep15_b1024": "H+ 5TB",
    }
    return mapping.get(name, f"{ARCH_LABEL.get(arch, arch)} {d_label}")


def ffloat(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_run_meta() -> list[RunMeta]:
    runs: list[RunMeta] = []
    with RUN_SUMMARY.open(newline="") as f:
        for row in csv.DictReader(f):
            run_id = row["run_id"]
            if row["role"] != "mainline_candidate":
                continue
            if row["decoder"] != "packwds":
                continue
            if any(s in run_id for s in EXCLUDE_RUN_SUBSTRINGS):
                continue
            arch = row["arch"]
            if arch not in PARAMS:
                continue
            metrics_path = ROOT / run_id / "training_metrics.json"
            if not metrics_path.exists():
                continue
            runs.append(
                RunMeta(
                    run_id=run_id,
                    label=friendly_label(run_id, arch, row["D_label"]),
                    arch=arch,
                    arch_label=ARCH_LABEL[arch],
                    decoder=row["decoder"],
                    d_label=row["D_label"],
                    d_pool=ffloat(row["D_unique_images_est"]),
                    params=PARAMS[arch],
                    effective_global_batch=ffloat(row["effective_global_batch"]),
                    crop=row["crop"],
                    lr=row["lr"],
                    warmup_epochs=row["warmup_epochs"],
                    role=row["role"],
                )
            )
    return runs


def parse_training_points(run: RunMeta) -> list[Point]:
    path = ROOT / run.run_id / "training_metrics.json"
    # Logs can contain restarts/concatenated segments. Use image_visits as the
    # x-axis key; when the same point appears repeatedly, keep the last line.
    best_by_visit: dict[int, Point] = {}
    with path.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            loss = item.get("total_loss")
            iteration = item.get("iteration")
            real_gbs = item.get("real_global_batch_size")
            if loss is None or iteration is None or real_gbs is None:
                continue
            try:
                iteration = int(iteration)
                loss = float(loss)
                real_gbs = float(real_gbs)
            except (TypeError, ValueError):
                continue
            if iteration <= 0 or not math.isfinite(loss) or not math.isfinite(real_gbs) or real_gbs <= 0:
                continue
            image_visits = iteration * real_gbs
            lr = ffloat(str(item.get("lr", "")))
            point = Point(
                run_id=run.run_id,
                run_label=run.label,
                arch=run.arch,
                arch_label=run.arch_label,
                d_label=run.d_label,
                d_pool=run.d_pool,
                params=run.params,
                iteration=iteration,
                image_visits=image_visits,
                data_passes=image_visits / run.d_pool if run.d_pool else math.nan,
                compute_proxy=run.params * image_visits,
                total_loss=loss,
                lr=lr,
            )
            best_by_visit[int(round(image_visits))] = point
    return [best_by_visit[k] for k in sorted(best_by_visit)]


def thin(points: list[Point], max_points: int) -> list[Point]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return [points[int(i)] for i in idx]


def write_points_csv(points: list[Point], path: Path) -> None:
    fields = [
        "run_id",
        "run_label",
        "arch",
        "arch_label",
        "d_label",
        "d_pool",
        "params",
        "params_M",
        "iteration",
        "image_visits",
        "image_visits_M",
        "data_passes",
        "compute_proxy_param_images",
        "compute_proxy_1e15_param_images",
        "total_loss",
        "lr",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "run_id": p.run_id,
                    "run_label": p.run_label,
                    "arch": p.arch,
                    "arch_label": p.arch_label,
                    "d_label": p.d_label,
                    "d_pool": p.d_pool,
                    "params": p.params,
                    "params_M": p.params / 1e6,
                    "iteration": p.iteration,
                    "image_visits": p.image_visits,
                    "image_visits_M": p.image_visits / 1e6,
                    "data_passes": p.data_passes,
                    "compute_proxy_param_images": p.compute_proxy,
                    "compute_proxy_1e15_param_images": p.compute_proxy / 1e15,
                    "total_loss": p.total_loss,
                    "lr": p.lr,
                }
            )


def summarize_runs(runs: list[RunMeta], all_points: dict[str, list[Point]]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for run in runs:
        pts = all_points.get(run.run_id, [])
        if not pts:
            continue
        losses = np.array([p.total_loss for p in pts])
        min_i = int(np.argmin(losses))
        final = pts[-1]
        best = pts[min_i]
        rows.append(
            {
                "run_id": run.run_id,
                "label": run.label,
                "arch": run.arch,
                "arch_label": run.arch_label,
                "D_label": run.d_label,
                "D_pool": run.d_pool,
                "params_M": run.params / 1e6,
                "n_log_points": len(pts),
                "final_image_visits_M": final.image_visits / 1e6,
                "final_data_passes": final.data_passes,
                "final_compute_1e15": final.compute_proxy / 1e15,
                "first_loss": pts[0].total_loss,
                "final_loss": final.total_loss,
                "min_loss": best.total_loss,
                "min_loss_image_visits_M": best.image_visits / 1e6,
                "min_loss_compute_1e15": best.compute_proxy / 1e15,
                "crop": run.crop,
                "lr": run.lr,
                "warmup_epochs": run.warmup_epochs,
            }
        )
    return rows


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arrays(points: list[Point]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.array([p.params / 1e6 for p in points], dtype=float)
    t = np.array([p.image_visits / 1e6 for p in points], dtype=float)
    d = np.array([p.d_pool / 1e6 for p in points], dtype=float)
    c = n * t
    y = np.array([p.total_loss for p in points], dtype=float)
    return n, t, d, c, y


def fit_classic(points: list[Point]) -> dict:
    n, t, _d, _c, y = arrays(points)
    best = None
    # Allow alpha_N < 0 because SSL train loss may not decrease with larger N.
    for alpha in np.linspace(-0.6, 0.8, 71):
        n_term = n ** (-alpha)
        for beta in np.linspace(0.02, 1.2, 60):
            x = np.column_stack([np.ones_like(y), n_term, t ** (-beta)])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ coef
            mse = float(np.mean((y - pred) ** 2))
            if best is None or mse < best["mse"]:
                best = {
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "E": float(coef[0]),
                    "A": float(coef[1]),
                    "B": float(coef[2]),
                    "pred": pred,
                }
    assert best is not None
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - best["pred"]) ** 2))
    best["r2"] = 1.0 - sse / sst if sst else math.nan

    positive = None
    for alpha in np.linspace(0.02, 0.8, 40):
        n_term = n ** (-alpha)
        for beta in np.linspace(0.02, 1.2, 60):
            x = np.column_stack([np.ones_like(y), n_term, t ** (-beta)])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            if coef[1] < 0 or coef[2] < 0:
                continue
            pred = x @ coef
            mse = float(np.mean((y - pred) ** 2))
            if positive is None or mse < positive["mse"]:
                positive = {
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "E": float(coef[0]),
                    "A": float(coef[1]),
                    "B": float(coef[2]),
                }
    best["positive_fit_available"] = positive is not None
    best["positive_fit"] = positive
    return best


def predict_classic(fit: dict, n_m: np.ndarray | float, t_m: np.ndarray | float) -> np.ndarray:
    return fit["E"] + fit["A"] * np.asarray(n_m) ** (-fit["alpha"]) + fit["B"] * np.asarray(t_m) ** (-fit["beta"])


def fit_ldcn(points: list[Point]) -> dict:
    n, _t, d, c, y = arrays(points)
    best = None
    for alpha in np.linspace(-0.4, 0.8, 31):
        n_term = n ** (-alpha)
        for beta in np.linspace(-0.4, 0.8, 31):
            d_term = d ** (-beta)
            for gamma in np.linspace(0.02, 1.2, 40):
                x = np.column_stack([np.ones_like(y), n_term, d_term, c ** (-gamma)])
                coef, *_ = np.linalg.lstsq(x, y, rcond=None)
                pred = x @ coef
                mse = float(np.mean((y - pred) ** 2))
                if best is None or mse < best["mse"]:
                    best = {
                        "mse": mse,
                        "rmse": math.sqrt(mse),
                        "alpha_N": float(alpha),
                        "beta_Dpool": float(beta),
                        "gamma_C": float(gamma),
                        "E": float(coef[0]),
                        "A_N": float(coef[1]),
                        "B_Dpool": float(coef[2]),
                        "G_C": float(coef[3]),
                        "pred": pred,
                    }
    assert best is not None
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - best["pred"]) ** 2))
    best["r2"] = 1.0 - sse / sst if sst else math.nan
    return best


def predict_ldcn(fit: dict, n_m: np.ndarray | float, dpool_m: np.ndarray | float, c_mparam_mimg: np.ndarray | float) -> np.ndarray:
    return (
        fit["E"]
        + fit["A_N"] * np.asarray(n_m) ** (-fit["alpha_N"])
        + fit["B_Dpool"] * np.asarray(dpool_m) ** (-fit["beta_Dpool"])
        + fit["G_C"] * np.asarray(c_mparam_mimg) ** (-fit["gamma_C"])
    )


def fit_per_run(points: list[Point]) -> dict:
    fit_pts = [p for p in points if p.image_visits >= FIT_MIN_IMAGE_VISITS]
    if len(fit_pts) < 10:
        return {}
    _n, t, _d, _c, y = arrays(fit_pts)
    best = None
    for gamma in np.linspace(0.02, 1.5, 100):
        x = np.column_stack([np.ones_like(y), t ** (-gamma)])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ coef
        mse = float(np.mean((y - pred) ** 2))
        if best is None or mse < best["mse"]:
            best = {"gamma": float(gamma), "L_inf": float(coef[0]), "A": float(coef[1]), "mse": mse, "rmse": math.sqrt(mse)}
    return best or {}


def setup_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_loss_vs_visits(ax, runs: list[RunMeta], points_by_run: dict[str, list[Point]]) -> None:
    setup_ax(ax, "Loss vs seen images D_seen", "D_seen / image visits (M, log)", "SSL total_loss")
    for run in runs:
        pts = thin(points_by_run[run.run_id], 800)
        if not pts:
            continue
        x = np.array([p.image_visits / 1e6 for p in pts])
        y = np.array([p.total_loss for p in pts])
        ax.plot(x, y, color=ARCH_COLOR[run.arch], linestyle=D_STYLE.get(run.d_label, "-"), lw=1.6, label=run.label)
        ax.scatter([x[-1]], [y[-1]], color=ARCH_COLOR[run.arch], s=18, zorder=5)
    ax.set_xscale("log")
    ax.legend(fontsize=6.5, ncols=2, frameon=False)


def plot_loss_vs_compute(ax, runs: list[RunMeta], points_by_run: dict[str, list[Point]]) -> None:
    setup_ax(ax, "Loss vs compute proxy C", "C = N_params x D_seen (1e15 param-images, log)", "SSL total_loss")
    for run in runs:
        pts = thin(points_by_run[run.run_id], 800)
        if not pts:
            continue
        x = np.array([p.compute_proxy / 1e15 for p in pts])
        y = np.array([p.total_loss for p in pts])
        ax.plot(x, y, color=ARCH_COLOR[run.arch], linestyle=D_STYLE.get(run.d_label, "-"), lw=1.6, label=run.label)
    ax.set_xscale("log")


def plot_fixed_model_data_scaling(ax, runs: list[RunMeta], points_by_run: dict[str, list[Point]], arch: str) -> None:
    title = f"Data scaling at fixed N: {ARCH_LABEL[arch]}"
    setup_ax(ax, title, "D_seen / image visits (M)", "SSL total_loss")
    selected = [r for r in runs if r.arch == arch]
    selected.sort(key=lambda r: (ffloat(r.d_label.replace("TB", "")), r.label))
    for run in selected:
        pts = thin(points_by_run[run.run_id], 800)
        if not pts:
            continue
        x = np.array([p.image_visits / 1e6 for p in pts])
        y = np.array([p.total_loss for p in pts])
        ax.plot(x, y, color=ARCH_COLOR[run.arch], linestyle=D_STYLE.get(run.d_label, "-"), lw=1.8, label=run.label)
    ax.legend(fontsize=7, frameon=False)


def plot_fit_parity(ax, points: list[Point], fit: dict, title: str, pred_kind: str = "classic") -> None:
    setup_ax(ax, title, "actual total_loss", "predicted total_loss")
    n, t, d, c, y = arrays(points)
    if pred_kind == "classic":
        pred = predict_classic(fit, n, t)
    else:
        pred = predict_ldcn(fit, n, d, c)
    colors = [ARCH_COLOR[p.arch] for p in points]
    ax.scatter(y, pred, c=colors, s=12, alpha=0.55, edgecolors="none")
    lo = min(float(y.min()), float(pred.min()))
    hi = max(float(y.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#1f2937", lw=1.0, alpha=0.75)
    ax.text(
        0.05,
        0.95,
        f"RMSE={fit['rmse']:.3f}\\nR2={fit['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75),
    )


def plot_isocompute(ax, fit: dict, points: list[Point]) -> None:
    setup_ax(ax, "Chinchilla-style iso-C profiles", "N params (M, log)", "predicted SSL loss")
    n_obs, t_obs, _d, c_obs, _y = arrays(points)
    n_grid = np.logspace(math.log10(max(10, n_obs.min() * 0.8)), math.log10(n_obs.max() * 1.25), 250)
    budgets = np.geomspace(np.percentile(c_obs, 10), np.percentile(c_obs, 95), 5)
    cmap = plt.get_cmap("viridis")
    for i, c_budget in enumerate(budgets):
        t_grid = c_budget / n_grid
        valid = (t_grid >= max(1.0, t_obs.min() * 0.5)) & (t_grid <= t_obs.max() * 1.5)
        y = predict_classic(fit, n_grid[valid], t_grid[valid])
        color = cmap(i / max(1, len(budgets) - 1))
        ax.plot(n_grid[valid], y, color=color, lw=2, label=f"C={c_budget:.0f}")
        if len(y):
            j = int(np.argmin(y))
            ax.scatter([n_grid[valid][j]], [y[j]], color=color, s=22, zorder=5)
    ax.scatter(n_obs, [p.total_loss for p in points], c=[ARCH_COLOR[p.arch] for p in points], s=8, alpha=0.15, edgecolors="none")
    ax.set_xscale("log")
    ax.legend(title="Mparam x Mimg", fontsize=7, title_fontsize=7, frameon=False)
    ax.text(
        0.02,
        0.03,
        f"fit alpha_N={fit['alpha']:.2f}; negative means SSL loss favors smaller N",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#475569",
    )


def plot_optimal_allocation(ax, fit: dict, points: list[Point]) -> None:
    setup_ax(ax, "Predicted compute allocation", "C proxy (Mparam x Mimg, log)", "N_opt M / D_seen_opt M")
    n_obs, t_obs, _d, c_obs, _y = arrays(points)
    n_grid = np.logspace(math.log10(max(10, n_obs.min() * 0.8)), math.log10(n_obs.max() * 1.25), 500)
    c_grid = np.geomspace(c_obs.min(), c_obs.max(), 120)
    n_opts = []
    t_opts = []
    boundary = []
    for c_budget in c_grid:
        t_grid = c_budget / n_grid
        valid = (t_grid >= max(0.5, t_obs.min() * 0.5)) & (t_grid <= t_obs.max() * 1.5)
        n_valid = n_grid[valid]
        t_valid = t_grid[valid]
        pred = predict_classic(fit, n_valid, t_valid)
        j = int(np.argmin(pred))
        n_opts.append(n_valid[j])
        t_opts.append(t_valid[j])
        boundary.append(j == 0 or j == len(pred) - 1)
    ax.plot(c_grid, n_opts, color="#7B2CBF", lw=2, label="N_opt (M params)")
    ax.plot(c_grid, t_opts, color="#2A9D8F", lw=2, label="D_seen_opt (M images)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    if any(boundary):
        ax.text(
            0.02,
            0.04,
            "Many optima sit on the grid boundary; treat as diagnostic only.",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#475569",
        )


def plot_ldcn_slices(path: Path, fit: dict, points: list[Point]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=180, sharey=True)
    fig.patch.set_facecolor("#FBFAF6")
    d_values = [1.0, 5.0, 10.0]
    n_lines = [22.0, 86.0, 300.0, 840.0]
    c_grid = np.geomspace(0.1, 60.0, 220)
    for ax, d_m in zip(axes, d_values):
        ax.set_facecolor("#FBFAF6")
        setup_ax(ax, f"D_pool={d_m:.0f}M images", "C proxy (Mparam x Mimg, log)", "predicted total_loss")
        for n_m in n_lines:
            y = predict_ldcn(fit, n_m, d_m, c_grid)
            arch = min(PARAMS, key=lambda a: abs(PARAMS[a] / 1e6 - n_m))
            ax.plot(c_grid, y, lw=2, color=ARCH_COLOR[arch], label=f"{n_m:.0f}M")
        ax.set_xscale("log")
        ax.legend(title="N", fontsize=7, title_fontsize=7, frameon=False)
    fig.suptitle("Diagnostic L(D_pool, C, N) surface slices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_composite(path: Path, runs: list[RunMeta], points_by_run: dict[str, list[Point]], fit_points: list[Point], classic: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), dpi=180)
    fig.patch.set_facecolor("#FBFAF6")
    for ax in axes.ravel():
        ax.set_facecolor("#FBFAF6")
    plot_loss_vs_visits(axes[0, 0], runs, points_by_run)
    plot_loss_vs_compute(axes[0, 1], runs, points_by_run)
    plot_fixed_model_data_scaling(axes[0, 2], runs, points_by_run, "vit_large")
    plot_fit_parity(axes[1, 0], fit_points, classic, "Classic L(N, D_seen) fit parity", "classic")
    plot_isocompute(axes[1, 1], classic, fit_points)
    plot_optimal_allocation(axes[1, 2], classic, fit_points)
    fig.suptitle("BioDINOv3 scaling-law diagnostics from existing logs", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_panel_note(ax, text: str, *, loc: str = "top") -> None:
    if loc == "bottom":
        y = 0.03
        va = "bottom"
    else:
        y = 0.97
        va = "top"
    ax.text(
        0.03,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha="left",
        fontsize=7.2,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1", lw=0.7, alpha=0.92),
    )


def make_ppt_composite(path: Path, runs: list[RunMeta], points_by_run: dict[str, list[Point]], fit_points: list[Point], classic: dict) -> None:
    """White-background, annotation-heavy version for slide decks."""
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=220)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
    plot_loss_vs_visits(axes[0, 0], runs, points_by_run)
    add_panel_note(
        axes[0, 0],
        "Within-run optimization.\nX = images seen by the model.\nLower curve = lower SSL training loss.",
        loc="bottom",
    )
    plot_loss_vs_compute(axes[0, 1], runs, points_by_run)
    add_panel_note(
        axes[0, 1],
        "Compute-normalized view.\nC = model parameters x image visits.\nUseful for cost/efficiency comparison.",
        loc="bottom",
    )
    plot_fixed_model_data_scaling(axes[0, 2], runs, points_by_run, "vit_large")
    add_panel_note(
        axes[0, 2],
        "Fixed-N data scaling.\nFor ViT-L, larger data pools reduce SSL loss,\nbut downstream quality must be checked separately.",
        loc="bottom",
    )
    plot_fit_parity(axes[1, 0], fit_points, classic, "Classic L(N, D_seen) fit parity", "classic")
    add_panel_note(
        axes[1, 0],
        "Fit check for L = E + A N^-alpha + B D_seen^-beta.\nCloser to diagonal = better fit.",
        loc="bottom",
    )
    plot_isocompute(axes[1, 1], classic, fit_points)
    add_panel_note(
        axes[1, 1],
        "Iso-compute slices from the fitted surface.\nShows predicted loss when total C is fixed\nand N vs D_seen changes.",
        loc="top",
    )
    plot_optimal_allocation(axes[1, 2], classic, fit_points)
    add_panel_note(
        axes[1, 2],
        "Implied N/D_seen allocation under the fit.\nDiagnostic only: current SSL loss is not a\nvalidated Chinchilla law.",
        loc="top",
    )
    fig.suptitle("BioDINOv3 Scaling-Law Diagnostics From Existing Logs", fontsize=22, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.018,
        "Caveat: total_loss is DINO/SSL training loss, not held-out likelihood. Use these plots as diagnostics, not as final N/D/C allocation rules.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_fit_csv(path: Path, classic: dict, ldcn: dict, per_run: dict[str, dict]) -> None:
    rows = [
        {
            "fit_name": "classic_L_N_Dseen_free",
            "formula": "L = E + A*N_M^-alpha + B*D_seen_M^-beta",
            "E": classic["E"],
            "A_or_A_N": classic["A"],
            "B_or_B_D": classic["B"],
            "G_C": "",
            "alpha_N": classic["alpha"],
            "beta_D_or_Dseen": classic["beta"],
            "gamma_C": "",
            "rmse": classic["rmse"],
            "r2": classic["r2"],
            "notes": "alpha_N is allowed to be negative; positive Chinchilla-style N term was not required.",
        },
        {
            "fit_name": "ldcn_free",
            "formula": "L = E + A*N_M^-alpha + B*D_pool_M^-beta + G*C_proxy^-gamma",
            "E": ldcn["E"],
            "A_or_A_N": ldcn["A_N"],
            "B_or_B_D": ldcn["B_Dpool"],
            "G_C": ldcn["G_C"],
            "alpha_N": ldcn["alpha_N"],
            "beta_D_or_Dseen": ldcn["beta_Dpool"],
            "gamma_C": ldcn["gamma_C"],
            "rmse": ldcn["rmse"],
            "r2": ldcn["r2"],
            "notes": "C_proxy is N_M * D_seen_M; free signs/exponents, diagnostic only.",
        },
    ]
    for run_id, fit in per_run.items():
        if not fit:
            continue
        rows.append(
            {
                "fit_name": f"per_run_compute::{Path(run_id).name}",
                "formula": "L = L_inf + A*D_seen_M^-gamma",
                "E": fit["L_inf"],
                "A_or_A_N": fit["A"],
                "B_or_B_D": "",
                "G_C": "",
                "alpha_N": "",
                "beta_D_or_Dseen": "",
                "gamma_C": fit["gamma"],
                "rmse": fit["rmse"],
                "r2": "",
                "notes": "Within-run compute curve fit after warmup cutoff.",
            }
        )
    write_dict_csv(path, rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def write_readme(path: Path, runs: list[RunMeta], run_rows: list[dict], classic: dict, ldcn: dict) -> None:
    run_table = []
    for row in run_rows:
        run_table.append(
            [
                str(row["label"]),
                str(row["arch_label"]),
                str(row["D_label"]),
                f"{float(row['params_M']):.0f}",
                f"{float(row['final_image_visits_M']):.2f}",
                f"{float(row['final_compute_1e15']):.2f}",
                f"{float(row['final_loss']):.3f}",
                f"{float(row['min_loss']):.3f}",
            ]
        )
    positive_note = "available" if classic["positive_fit_available"] else "not available"
    lines = [
        "# Scaling-law diagnostics: L(D, C, N)",
        "",
        "This folder contains Chinchilla-style diagnostic plots built from existing training logs.",
        "",
        "Important caveat: `total_loss` is the DINO/SSL training objective, not held-out cross-entropy. It is useful for optimization diagnostics, but it is not reliably comparable across model sizes as an LM loss. In the fitted classic form the best free exponent for N is negative, which means this SSL loss currently prefers smaller N; therefore the Chinchilla-like optimum plots are diagnostic rather than prescriptive.",
        "",
        "## Variables",
        "",
        "- `N`: approximate model parameters, rounded to S+=22M, B=86M, L=300M, H+=840M.",
        "- `D_seen`: processed images / image visits, computed from `iteration * real_global_batch_size` in `training_metrics.json`.",
        "- `D_pool`: estimated unique image pool from `run_summary.csv`.",
        "- `C`: compute proxy, `N * D_seen`; units shown as `1e15 parameter-images` or `Mparam x Mimg`.",
        "- `L`: logged `total_loss`.",
        "",
        "## Outputs",
        "",
        "- `scalinglaw_chinchilla_style_panel.png`: main 2x3 figure.",
        "- `scalinglaw_chinchilla_style_panel_ppt_white.png`: white-background annotated version for slides.",
        "- `loss_vs_image_visits.png`: loss curves against seen data.",
        "- `loss_vs_compute_proxy.png`: loss curves against compute proxy.",
        "- `data_scaling_fixed_model.png`: fixed-model data scaling slices.",
        "- `classic_fit_parity.png`: parity for `L(N, D_seen)`.",
        "- `ldcn_fit_parity.png`: parity for `L(D_pool, C, N)`.",
        "- `ldcn_surface_slices.png`: surface slices from the diagnostic LDCN fit.",
        "- `scaling_points.csv`, `run_loss_summary.csv`, `fit_parameters.csv`.",
        "",
        "## Runs included",
        "",
        md_table(["run", "arch", "D", "N M", "final D_seen M", "final C 1e15", "final L", "min L"], run_table),
        "",
        "## Fits",
        "",
        f"Classic free fit: `L = E + A*N_M^-alpha + B*D_seen_M^-beta`, with `E={classic['E']:.4f}`, `A={classic['A']:.4g}`, `B={classic['B']:.4g}`, `alpha={classic['alpha']:.3f}`, `beta={classic['beta']:.3f}`, RMSE `{classic['rmse']:.3f}`, R2 `{classic['r2']:.3f}`. Positive-coefficient Chinchilla-style fit: `{positive_note}`.",
        "",
        f"LDCN free fit: `L = E + A*N_M^-alpha + B*D_pool_M^-beta + G*C^-gamma`, with `alpha={ldcn['alpha_N']:.3f}`, `beta={ldcn['beta_Dpool']:.3f}`, `gamma={ldcn['gamma_C']:.3f}`, RMSE `{ldcn['rmse']:.3f}`, R2 `{ldcn['r2']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "- The loss-vs-compute curves are the most trustworthy part of this report.",
        "- Cross-N fitted optima should not be used to choose model size until we have a held-out SSL loss or a calibrated downstream quality target.",
        "- Existing downstream evidence should be treated as a separate representation-scaling line; lower SSL loss alone did not always mean better downstream quality.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_run_meta()
    points_by_run = {run.run_id: parse_training_points(run) for run in runs}
    all_plot_points = [p for run in runs for p in points_by_run[run.run_id]]
    fit_points: list[Point] = []
    for run in runs:
        eligible = [p for p in points_by_run[run.run_id] if p.image_visits >= FIT_MIN_IMAGE_VISITS]
        fit_points.extend(thin(eligible, 450))
    if len(fit_points) < 30:
        raise RuntimeError("Not enough fit points found.")

    run_rows = summarize_runs(runs, points_by_run)
    classic = fit_classic(fit_points)
    ldcn = fit_ldcn(fit_points)
    per_run = {run.run_id: fit_per_run(points_by_run[run.run_id]) for run in runs}

    write_points_csv(all_plot_points, OUT / "scaling_points.csv")
    write_dict_csv(OUT / "run_loss_summary.csv", run_rows)
    write_fit_csv(OUT / "fit_parameters.csv", classic, ldcn, per_run)

    # Single-purpose figures.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    fig.patch.set_facecolor("#FBFAF6")
    ax.set_facecolor("#FBFAF6")
    plot_loss_vs_visits(ax, runs, points_by_run)
    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_image_visits.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    fig.patch.set_facecolor("#FBFAF6")
    ax.set_facecolor("#FBFAF6")
    plot_loss_vs_compute(ax, runs, points_by_run)
    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_compute_proxy.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180, sharey=True)
    fig.patch.set_facecolor("#FBFAF6")
    for ax, arch in zip(axes, ["vit_large", "vit_huge2"]):
        ax.set_facecolor("#FBFAF6")
        plot_fixed_model_data_scaling(ax, runs, points_by_run, arch)
    fig.tight_layout()
    fig.savefig(OUT / "data_scaling_fixed_model.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    fig.patch.set_facecolor("#FBFAF6")
    ax.set_facecolor("#FBFAF6")
    plot_fit_parity(ax, fit_points, classic, "Classic L(N, D_seen) parity", "classic")
    fig.tight_layout()
    fig.savefig(OUT / "classic_fit_parity.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    fig.patch.set_facecolor("#FBFAF6")
    ax.set_facecolor("#FBFAF6")
    plot_fit_parity(ax, fit_points, ldcn, "L(D_pool, C, N) parity", "ldcn")
    fig.tight_layout()
    fig.savefig(OUT / "ldcn_fit_parity.png", bbox_inches="tight")
    plt.close(fig)

    plot_ldcn_slices(OUT / "ldcn_surface_slices.png", ldcn, fit_points)
    make_composite(OUT / "scalinglaw_chinchilla_style_panel.png", runs, points_by_run, fit_points, classic)
    make_ppt_composite(OUT / "scalinglaw_chinchilla_style_panel_ppt_white.png", runs, points_by_run, fit_points, classic)
    write_readme(OUT / "README.md", runs, run_rows, classic, ldcn)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
