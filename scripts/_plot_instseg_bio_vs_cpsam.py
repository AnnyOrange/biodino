#!/usr/bin/env python3
"""Plot frozen Bio-DINOv3 vs zero-shot Cellpose-SAM instance-seg results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = ["pannuke", "tissuenet", "conic", "bbbc038", "livecell", "monuseg"]
DISPLAY = {
    "pannuke": "PanNuke",
    "tissuenet": "TissueNet",
    "conic": "CoNIC",
    "bbbc038": "BBBC038",
    "livecell": "LiveCell",
    "monuseg": "MoNuSeg",
}


def load_rows(summary_path: Path, datasets: list[str]):
    data = json.loads(summary_path.read_text())
    rows = []
    for ds in datasets:
        rec = data["datasets"][ds]
        bio = rec["frozen_bio"]
        cps = rec["cpsam_zero_shot"]
        if bio is None or cps is None:
            raise SystemExit(f"Missing frozen_bio or cpsam_zero_shot for {ds} in {summary_path}")
        bm, cm = bio["metrics"], cps["metrics"]
        rows.append((DISPLAY.get(ds, ds), bm["AJI"], bm["bPQ"], cm["AJI"], cm["bPQ"]))
    return rows


def write_note(path: Path, rows):
    lines = [
        "# Frozen Bio-DINOv3 vs Zero-shot Cellpose-SAM",
        "",
        "Official test splits; metrics are AJI / bPQ. Bio-DINOv3 uses the frozen ViT-H checkpoint `ckpt/12299/checkpoint.pth` with the HoVerNet decoder (`feature_size=64`, `embed_proj=512`, layers `[7, 15, 23, 31]`).",
        "",
        "| Dataset | Frozen Bio-DINOv3 | Cellpose-SAM zero-shot | Delta AJI | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    wins = 0
    for name, ba, bb, ca, cb in rows:
        verdict = "WIN" if ba >= ca else "lose"
        wins += int(ba >= ca)
        lines.append(f"| {name} | {ba:.3f} / {bb:.3f} | {ca:.3f} / {cb:.3f} | {ba-ca:+.3f} | {verdict} |")
    lines += [
        "",
        f"Summary: frozen Bio-DINOv3 beats zero-shot Cellpose-SAM on {wins}/{len(rows)} datasets, specifically the tissue-nuclei datasets PanNuke, TissueNet, and CoNIC. It loses on BBBC038, LiveCell, and MoNuSeg, which are mixed-modality, phase-contrast, or data-scarce settings.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="outputs/instance_seg/final_test_summary.json")
    p.add_argument("--out-dir", default="outputs/instance_seg/figures")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = p.parse_args()

    rows = load_rows(Path(args.summary), args.datasets)
    names = [r[0] for r in rows]
    bio_aji = np.array([r[1] for r in rows])
    bio_bpq = np.array([r[2] for r in rows])
    cps_aji = np.array([r[3] for r in rows])
    cps_bpq = np.array([r[4] for r in rows])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    bio_color = "#0B6E69"
    cps_color = "#D9822B"
    win_color = "#0B6E69"
    lose_color = "#9E2A2B"

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    x = np.arange(len(names))
    width = 0.36
    panels = [
        (axes[0], bio_aji, cps_aji, "AJI", "Aggregated Jaccard Index"),
        (axes[1], bio_bpq, cps_bpq, "bPQ", "Binary Panoptic Quality"),
    ]
    for ax, bio, cps, metric, ylabel in panels:
        ax.bar(x - width / 2, bio, width=width, color=bio_color, label="Frozen Bio-DINOv3 + HoVerNet")
        ax.bar(x + width / 2, cps, width=width, color=cps_color, label="Cellpose-SAM zero-shot")
        ax.set_ylim(0.45, 0.86)
        ax.set_ylabel(ylabel)
        ax.set_title(metric, loc="left", pad=6)
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for i, (b, c) in enumerate(zip(bio, cps)):
            delta = b - c
            color = win_color if delta >= 0 else lose_color
            ax.text(i, max(b, c) + 0.014, f"{delta:+.3f}", ha="center", va="bottom", color=color, fontsize=9, fontweight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    fig.suptitle("Frozen Bio-DINOv3 vs Zero-shot Cellpose-SAM", fontsize=14, fontweight="bold")
    fig.supxlabel("Official test splits. Bar labels show Bio-DINOv3 minus Cellpose-SAM.", fontsize=9, color="#4D5656")

    png = out_dir / "frozen_biodino_vs_cpsam_zeroshot_aji_bpq.png"
    pdf = out_dir / "frozen_biodino_vs_cpsam_zeroshot_aji_bpq.pdf"
    md = out_dir / "frozen_biodino_vs_cpsam_zeroshot_summary.md"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    write_note(md, rows)
    print(png)
    print(pdf)
    print(md)


if __name__ == "__main__":
    main()
