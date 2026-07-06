#!/usr/bin/env python3
"""Draw the DINOHoVerNet decoder/head structure used in Line-2 instance seg."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path("outputs/instance_seg/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "input": "#E9F2F0",
    "backbone": "#D8E7FA",
    "tap": "#EAF1FB",
    "fusion": "#F4E3C1",
    "decoder": "#DDEFE8",
    "head": "#F7D6D0",
    "post": "#ECE4F6",
    "note": "#F7F9F9",
    "edge": "#34495E",
}


def box(ax, x, y, w, h, text, fc, ec="#34495E", lw=1.4, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)
    return patch


def arrow(ax, x1, y1, x2, y2, text=None, color="#34495E", style="->", lw=1.6, dashed=False, fontsize=8):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle="--" if dashed else "-",
            shrinkA=3,
            shrinkB=3,
        ),
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fontsize, color=color)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fig.suptitle(
        "DINOHoVerNet Instance-Segmentation Decoder and Heads",
        fontsize=19,
        fontweight="bold",
        y=0.97,
    )

    # High-level pipeline.
    box(ax, 0.45, 6.55, 1.55, 1.05, "Input image\n3 x H x W", COLORS["input"], fontsize=10, weight="bold")
    box(ax, 2.45, 6.35, 2.0, 1.45, "Frozen Bio-DINOv3\nViT-H/16 backbone\n(config.yaml + ckpt/12299)", COLORS["backbone"], fontsize=9, weight="bold")
    arrow(ax, 2.0, 7.08, 2.45, 7.08)

    tap_y = [7.7, 7.22, 6.74, 6.26]
    tap_labels = ["Layer 7", "Layer 15", "Layer 23", "Layer 31"]
    for y, lab in zip(tap_y, tap_labels):
        box(ax, 4.85, y, 1.2, 0.36, f"{lab}\nD x H/16 x W/16", COLORS["tap"], fontsize=7.2)
    arrow(ax, 4.45, 7.08, 4.85, 7.88, style="->", lw=1.1)
    arrow(ax, 4.45, 7.08, 4.85, 7.40, style="->", lw=1.1)
    arrow(ax, 4.45, 7.08, 4.85, 6.92, style="->", lw=1.1)
    arrow(ax, 4.45, 7.08, 4.85, 6.44, style="->", lw=1.1)

    box(ax, 6.55, 6.28, 2.0, 1.55, "Tap-bucket fusion\n4 buckets, shallow->deep\nconcat taps + 1x1 conv\nD=1280 -> embed_proj=512", COLORS["fusion"], fontsize=8.7, weight="bold")
    for y in tap_y:
        arrow(ax, 6.05, y + 0.18, 6.55, 7.05, lw=1.0)

    z_y = [7.72, 7.25, 6.78, 6.31]
    z_labels = ["z0", "z1", "z2", "z3"]
    for y, lab in zip(z_y, z_labels):
        box(ax, 8.9, y, 0.72, 0.32, f"{lab}", COLORS["fusion"], fontsize=8, weight="bold")
        arrow(ax, 8.55, 7.05, 8.9, y + 0.16, lw=1.0)

    # Decoder heads.
    branch_x = 10.15
    branch_w = 2.45
    branches = [
        (7.42, "NP branch\nUNETR decoder + 1x1 head", "NP logits\n2 x H x W\nforeground/background"),
        (6.74, "HV branch\nUNETR decoder + 1x1 head", "HV regression\n2 x H x W\nhorizontal/vertical"),
        (6.06, "TP branch (optional)\nUNETR decoder + 1x1 head", "Type logits\nC x H x W\nPanNuke/CoNIC"),
    ]
    for y, left, right in branches:
        box(ax, branch_x, y, branch_w, 0.5, left, COLORS["decoder"], fontsize=8.4, weight="bold")
        box(ax, branch_x + 2.85, y, 1.85, 0.5, right, COLORS["head"], fontsize=7.8)
        arrow(ax, branch_x + branch_w, y + 0.25, branch_x + 2.85, y + 0.25)
        arrow(ax, 9.62, 7.0, branch_x, y + 0.25, lw=1.0)
        arrow(ax, 1.23, 6.55, branch_x, y + 0.14, dashed=True, lw=1.0, text="raw-image skip", fontsize=7)

    box(ax, 14.0, 6.48, 1.65, 1.52, "HoVerNet\npostprocess\nNP softmax + HV ridges\nmarker watershed\nTP majority vote", COLORS["post"], fontsize=8.3, weight="bold")
    for y in [7.67, 6.99, 6.31]:
        arrow(ax, 12.6 + 1.85, y, 14.0, 7.24, lw=1.0)
    box(ax, 14.0, 5.78, 1.65, 0.43, "instance map + type map", COLORS["post"], fontsize=8, weight="bold")
    arrow(ax, 14.82, 6.48, 14.82, 6.21)

    # Detailed branch inset.
    box(ax, 0.45, 0.55, 15.1, 4.65, "", "#FFFFFF", ec="#D0D3D4", lw=1.2)
    ax.text(0.72, 4.85, "Inside one UNETR decoder branch (repeated independently for NP / HV / TP)", fontsize=13, fontweight="bold", ha="left")

    y = 3.35
    box(ax, 0.8, y, 1.35, 0.55, "z3\nH/16", COLORS["fusion"], fontsize=8, weight="bold")
    box(ax, 2.55, y, 1.45, 0.55, "UpBlock\nup x2", COLORS["decoder"], fontsize=8)
    box(ax, 4.45, y, 1.55, 0.55, "concat e4\nfrom z2 -> H/8", COLORS["tap"], fontsize=8)
    box(ax, 6.45, y, 1.45, 0.55, "UpBlock\nup x2", COLORS["decoder"], fontsize=8)
    box(ax, 8.35, y, 1.55, 0.55, "concat e3\nfrom z1 -> H/4", COLORS["tap"], fontsize=8)
    box(ax, 10.35, y, 1.45, 0.55, "UpBlock\nup x2", COLORS["decoder"], fontsize=8)
    box(ax, 12.25, y, 1.55, 0.55, "concat e2\nfrom z0 -> H/2", COLORS["tap"], fontsize=8)
    box(ax, 14.15, y, 1.0, 0.55, "up x2", COLORS["decoder"], fontsize=8)

    for x1, x2 in [(2.15, 2.55), (4.0, 4.45), (6.0, 6.45), (7.9, 8.35), (9.9, 10.35), (11.8, 12.25), (13.8, 14.15)]:
        arrow(ax, x1, y + 0.275, x2, y + 0.275, lw=1.2)

    box(ax, 0.8, 2.0, 2.0, 0.62, "image stem e1\n3x3 conv block\nfull-res H x W", COLORS["input"], fontsize=8)
    box(ax, 3.35, 2.0, 1.6, 0.62, "concat e1\nH x W", COLORS["tap"], fontsize=8)
    box(ax, 5.55, 2.0, 1.75, 0.62, "BasicBlock\n2x 3x3 conv", COLORS["decoder"], fontsize=8)
    box(ax, 7.9, 2.0, 1.4, 0.62, "1x1 conv\ndecoder head", COLORS["head"], fontsize=8, weight="bold")
    box(ax, 10.05, 2.0, 2.0, 0.62, "branch output\nNP / HV / TP", COLORS["head"], fontsize=8, weight="bold")
    arrow(ax, 14.65, y, 14.65, 2.62, lw=1.2)
    arrow(ax, 2.8, 2.31, 3.35, 2.31, lw=1.2)
    arrow(ax, 4.95, 2.31, 5.55, 2.31, lw=1.2)
    arrow(ax, 7.3, 2.31, 7.9, 2.31, lw=1.2)
    arrow(ax, 9.3, 2.31, 10.05, 2.31, lw=1.2)

    box(
        ax,
        12.75,
        1.15,
        2.45,
        1.0,
        "Decoder-head channels\nNP: 2 logits\nHV: 2 regression maps\nTP: num_types logits",
        COLORS["note"],
        ec="#ABB2B9",
        fontsize=8.3,
    )

    fig.text(
        0.02,
        0.02,
        "K tapped ViT layers are split into 4 buckets. Each decoder branch has its own UNETR path and final 1x1 head; the fusion front-end is shared.",
        fontsize=9,
        color="#4D5656",
    )

    png = OUT_DIR / "dino_hovernet_decoder_structure.png"
    pdf = OUT_DIR / "dino_hovernet_decoder_structure.pdf"
    svg = OUT_DIR / "dino_hovernet_decoder_structure.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    md = OUT_DIR / "dino_hovernet_decoder_structure.md"
    md.write_text(
        "# DINOHoVerNet decoder structure\n\n"
        "The added decoder is `HoVerNetDecoder` in `dinov3/eval/bio_segmentation/instance_seg/decoder.py`. "
        "It receives ViT intermediate feature maps from the frozen Bio-DINOv3 backbone and the raw image. "
        "The tapped layers are split into four buckets and projected by shared 1x1 fusion convolutions to z0-z3. "
        "Three independent UNETR-style branches then produce NP, HV, and optional TP outputs.\n\n"
        "- NP head: 2-channel nucleus-pixel logits.\n"
        "- HV head: 2-channel horizontal/vertical regression maps for separating touching nuclei.\n"
        "- TP head: `num_types` type logits for multi-class datasets such as PanNuke and CoNIC.\n"
        "- Post-processing: NP foreground + HV gradient ridges -> marker-controlled watershed; TP is majority-voted per instance.\n\n"
        f"![DINOHoVerNet decoder structure]({png.name})\n",
        encoding="utf-8",
    )

    print(png)
    print(pdf)
    print(svg)
    print(md)


if __name__ == "__main__":
    main()
