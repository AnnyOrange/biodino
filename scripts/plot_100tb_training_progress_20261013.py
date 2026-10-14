#!/usr/bin/env python3
"""Plot current 100TB training loss and mark the latest completed evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/huawei_deepcad/dinov3")
RUN = ROOT / "outputs/01_training_runs/HS6_L_quality_long_100tb_fullpool_ddp_gb1024_mb128_ga2_40k_4xa100_deepcad_20261010"
METRICS = RUN / "training_metrics.json"
OUT = ROOT / "plot/fig2/data_quality/training/100tb"
OUT.mkdir(parents=True, exist_ok=True)

updates = []
losses = []
with METRICS.open() as handle:
    for line in handle:
        row = json.loads(line)
        if "total_loss" not in row:
            continue
        updates.append(row["iteration"] / 2)
        losses.append(row["total_loss"])

x = np.asarray(updates, dtype=float)
y = np.asarray(losses, dtype=float)
window = min(41, len(y) if len(y) % 2 else len(y) - 1)
smooth = np.array([np.median(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
last_update = int(x[-1])
last_micro_step = int(x[-1] * 2)
fraction = last_update / 40000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(10.5, 5.4), layout="constrained")
ax.plot(x, y, color="#7C9E9A", linewidth=0.65, alpha=0.45, label="Logged loss")
ax.plot(x, smooth, color="#176B65", linewidth=1.8, label=f"Trailing median ({window} points)")
ax.axvline(12294, color="#C9683C", linestyle="--", linewidth=1.2, label="Warmup end (12,294 updates)")
ax.axvline(8000, color="#545A60", linestyle=":", linewidth=1.2, label="Latest evaluated checkpoint (8,000 updates)")
ax.scatter([last_update], [y[-1]], s=36, color="#176B65", zorder=5)
ax.annotate(f"current: {last_update:,} / 40,000 ({fraction:.1%})\nloss {y[-1]:.3f}",
            (last_update, y[-1]), xytext=(12, 42), textcoords="offset points",
            ha="left", va="bottom", fontsize=9)
ax.set(xlim=(0, 40000), xlabel="Optimizer updates", ylabel="Training total loss",
       title="100TB full-pool HS6-L training progress")
ax.grid(axis="y", color="#D8DEDC", linewidth=0.7, alpha=0.8)
ax.legend(frameon=False, ncols=2, loc="upper right")
fig.text(0.01, -0.015,
         "DDP, effective batch 1,024. v4 ID at checkpoint 7999 (8,000 updates): 38/38 components and 9/9 extensions complete. "
         "No later checkpoint evaluation is present yet; loss is not downstream performance.",
         ha="left", va="top", fontsize=8, color="#444A4E")

stem = OUT / "100tb_training_progress_20261013"
fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
plt.close(fig)

manifest = {
    "source_metrics": str(METRICS),
    "source_log_iteration": last_micro_step,
    "micro_steps_completed": last_micro_step,
    "gradient_accumulation_steps": 2,
    "optimizer_updates_completed": last_update,
    "optimizer_updates_target": 40000,
    "fraction_complete": fraction,
    "last_logged_total_loss": float(y[-1]),
    "smoothing": f"trailing median over {window} logged points; raw values also shown",
    "warmup_optimizer_updates": 12294,
    "latest_evaluated_checkpoint_optimizer_updates": 8000,
    "latest_evaluated_checkpoint_status": "v4 ID 38/38; extensions 9/9; no newer checkpoint evaluation found",
    "outputs": [str(stem.with_suffix(ext)) for ext in (".png", ".svg", ".pdf")],
}
(OUT / "100tb_training_progress_20261013_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
