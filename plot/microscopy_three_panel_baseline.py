#!/usr/bin/env python3
"""
For each dataset: one figure with 3 subplots (ViT-S / ViT-B / ViT-L).
  - x=0: baseline test mIoU from outputs/linear_probe/<candidate>/results.json
  - x>0: test mIoU from outputs/outputs_microscopy_continual_* / <dataset>/linear_probe/<step>/results.json

Run from repo root:
  python plot/microscopy_three_panel_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_PROBE = REPO_ROOT / "outputs" / "linear_probe"

DATASETS: List[str] = ["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"]

CONTINUAL: Dict[str, Path] = {
    "s": REPO_ROOT / "outputs/outputs_microscopy_continual_vits16",
    "b": REPO_ROOT / "outputs/outputs_microscopy_continual_vitb16",
    "l": REPO_ROOT / "outputs/outputs_microscopy_continual_vitl16",
}

# (dataset, arch) -> ordered folder names under outputs/linear_probe (first hit wins)
BASELINE_CANDIDATES: Dict[Tuple[str, str], List[str]] = {
    ("bbbc038", "s"): ["bbbc038_vitS"],
    ("bbbc038", "b"): ["bbbc038_vitB", "bbbc038_vitb_new"],
    ("bbbc038", "l"): ["bbbc038_vitl", "bbbc038_vitL", "bbbc038_vitl_new"],
    ("conic", "s"): ["conic_vits"],
    ("conic", "b"): ["conic_vitb"],
    ("conic", "l"): ["conic_vitl"],
    ("monuseg", "s"): ["monuseg_vits", "monuseg_vitS"],
    ("monuseg", "b"): ["monuseg_vitB", "monuseg_vitb_new"],
    ("monuseg", "l"): ["monuseg_vitl", "monuseg_vitl_new"],
    ("pannuke", "s"): ["pannuke_vits", "pannuke_vitS"],
    ("pannuke", "b"): ["pannuke_vitb", "pannuke_vitB", "pannuke_vitb_new"],
    ("pannuke", "l"): ["pannuke_vitl", "pannuke_vitl_new"],
    ("tissuenet", "s"): ["tissuenet_vits", "tissuenet_vitS"],
    ("tissuenet", "b"): ["tissuenet_vitb", "tissuenet_vitB", "tissuenet_vitb_new"],
    ("tissuenet", "l"): ["tissuenet_vitl", "tissuenet_vitl_new"],
}

SUBPLOT_TITLES = {"s": "ViT-S", "b": "ViT-B", "l": "ViT-L"}
LINE_COLOR = "#2a5a9e"
MARKER_COLOR = "#2a5a9e"


def read_test_miou(results_path: Path) -> float | None:
    if not results_path.is_file():
        return None
    try:
        with results_path.open() as f:
            data = json.load(f)
        return float(data["test"]["mIoU"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def resolve_baseline_results(dataset: str, arch: str) -> Tuple[Path | None, float | None]:
    """Return (results.json path used, mIoU) or (None, None) if missing."""
    for name in BASELINE_CANDIDATES.get((dataset, arch), []):
        p = LINEAR_PROBE / name / "results.json"
        m = read_test_miou(p)
        if m is not None:
            return p, m
    return None, None


def load_continual_curve(continual_root: Path, dataset: str) -> List[Tuple[int, float]]:
    """Sorted list of (training_step, test mIoU) from continual linear_probe runs."""
    probe = continual_root / dataset / "linear_probe"
    out: List[Tuple[int, float]] = []
    if not probe.is_dir():
        return out
    for step_dir in probe.iterdir():
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            continue
        if step <= 0:
            continue
        m = read_test_miou(step_dir / "results.json")
        if m is None:
            continue
        out.append((step, m))
    out.sort(key=lambda t: t[0])
    return out


def build_xy(
    baseline_miou: float | None, continual: Sequence[Tuple[int, float]]
) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    if baseline_miou is not None:
        xs.append(0)
        ys.append(baseline_miou)
    for step, m in continual:
        xs.append(step)
        ys.append(m)
    return xs, ys


def plot_one_dataset(
    dataset: str,
    out_path: Path,
    *,
    annotate_baseline_path: bool = False,
) -> bool:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    all_y: List[float] = []

    for ax, arch in zip(axes, ("s", "b", "l")):
        res_path, base_m = resolve_baseline_results(dataset, arch)
        cont = load_continual_curve(CONTINUAL[arch], dataset)
        xs, ys = build_xy(base_m, cont)
        if not ys:
            ax.set_title(SUBPLOT_TITLES[arch])
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(xs, ys, color=LINE_COLOR, marker="o", ms=4, lw=1.5, clip_on=False)
        all_y.extend(ys)
        title = SUBPLOT_TITLES[arch]
        if annotate_baseline_path and res_path is not None:
            title += f"\n{res_path.parent.name}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step (0 = baseline)")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("test mIoU")
    fig.suptitle(f"{dataset} — linear probe (test mIoU) [baseline + continual]", y=1.05, fontsize=11)
    if all_y:
        y0, y1 = float(np.min(all_y)), float(np.max(all_y))
        pad = max(0.008, (y1 - y0) * 0.12)
        axes[0].set_ylim(y0 - pad, y1 + pad)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return bool(all_y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plot" / "figures" / "three_panel_baseline",
        help="Output directory for PNGs.",
    )
    ap.add_argument(
        "--annotate-baseline-dirs",
        action="store_true",
        help="Put chosen baseline folder name in subplot title (debug).",
    )
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for ds in DATASETS:
        path = out_dir / f"{ds}_vit_s_b_l_miou.png"
        if plot_one_dataset(ds, path, annotate_baseline_path=args.annotate_baseline_dirs):
            ok += 1
            print(f"Wrote {path}")
        else:
            print(f"[WARN] no curve data for dataset={ds}", file=sys.stderr)

    print(f"Done: {ok}/{len(DATASETS)} figures under {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
