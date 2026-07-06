#!/usr/bin/env python3
"""Plot dual-route SSL training curves (loss components, lr, grad norm) to check health.
Usage: python _plot_dualroute_curves.py <metrics.jsonl> <out.png> [title]
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
out = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "dual-route training"

rows = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass

def series(key):
    xs, ys = [], []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)) and "iteration" in r:
            xs.append(r["iteration"]); ys.append(r[key])
    return xs, ys

panels = [
    ("total_loss", ["total_loss"]),
    ("DINO loss (global / local)", ["dino_global_crops_loss", "dino_local_crops_loss"]),
    ("iBOT loss", ["ibot_loss"]),
    ("KoLeo loss", ["koleo_loss"]),
    ("learning rate", ["lr"]),
    ("backbone grad norm", ["backbone_grad_norm"]),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
for ax, (ttl, keys) in zip(axes, panels):
    plotted = False
    for k in keys:
        xs, ys = series(k)
        if xs:
            ax.plot(xs, ys, lw=0.8, label=k)
            plotted = True
    ax.set_title(ttl)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
    if len(keys) > 1 and plotted:
        ax.legend(fontsize=8)
    if not plotted:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)

last = rows[-1] if rows else {}
fig.suptitle(f"{title}   (last iter={last.get('iteration','?')}, total_loss={last.get('total_loss','?'):.3f}, n={len(rows)})"
             if isinstance(last.get("total_loss"), (int, float)) else f"{title}  (n={len(rows)})",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out, dpi=110)
print("saved", out, "from", len(rows), "rows; last iter", last.get("iteration"))
