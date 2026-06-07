#!/usr/bin/env python3
"""Qualitative OOD figures from cached eval_ood features.

  6_umap_xray.png   - UMAP of X-ray slice features, colored by variant/sample/dose/resolution
  7_umap_cryo.png   - UMAP of cryo particle features, colored by project/quality/ncc
  8_xray_pair_retrieval.png - query X-ray slice + top-k cosine neighbors (mark rigid/nonrigid twin)
  9_cryo_retrieval.png      - query cryo particle + top-k neighbors (mark same 2D-class)

Run (features are on the shared NFS, code needs the dinov3 env for umap):
    conda run -n dinov3 python dinov3/eval/eval_ood/plot_qualitative.py \
        --features-dir benchmark_runs/eval_ood_remote_fullcryo_top3_20260602_212923/results/vitl_oep1025_nlb4_cls_three_slices_raw/8199/features \
        --out benchmark_runs/eval_ood_analysis
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, path, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight")
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    plt.close(fig)
    print(f"  wrote {path} ({len(data)/1e3:.0f} KB)")


def l2n(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def umap_embed(feats, seed=0, n_neighbors=30, min_dist=0.1):
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed)
    return reducer.fit_transform(l2n(feats))


# ---------- thumbnail loaders (replicate datasets.py readers) ----------
def _pct_norm(arr, lo=0.5, hi=99.5):
    arr = np.asarray(arr, dtype=np.float32)
    a, b = np.percentile(arr, [lo, hi])
    if b <= a:
        b = a + 1.0
    return np.clip((arr - a) / (b - a), 0, 1)


_XRAY_SHAPE_CACHE = {}


def xray_thumb(meta, ood_root):
    vid = meta["volume_id"]
    raw_path = meta["path"]
    z = int(meta["z_index"])
    if vid not in _XRAY_SHAPE_CACHE:
        jp = Path(ood_root) / "xray_brain_ultrastructure" / "webknossos" / f"{vid}.json"
        sx, sy, sz = (int(v) for v in json.load(open(jp))["rawShapeXYZ"])
        _XRAY_SHAPE_CACHE[vid] = (sx, sy, sz)
    sx, sy, sz = _XRAY_SHAPE_CACHE[vid]
    mm = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=(sz, sy, sx))
    return _pct_norm(mm[min(z, sz - 1)])


class MRCStack:
    _DT = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}

    def __init__(self, path):
        with open(path, "rb") as f:
            h = f.read(1024)
        self.nx, self.ny, self.nz, self.mode = struct.unpack("<4i", h[:16])
        nsymbt = struct.unpack("<i", h[92:96])[0]
        off = 1024 + max(0, int(nsymbt))
        self.data = np.memmap(path, dtype=np.dtype(self._DT[self.mode]), mode="r", offset=off,
                              shape=(self.nz, self.ny, self.nx))

    def read(self, i):
        return np.asarray(self.data[int(i)], dtype=np.float32)


_MRC_CACHE = {}


def cryo_thumb(meta):
    p = meta["path"]
    if p not in _MRC_CACHE:
        _MRC_CACHE[p] = MRCStack(p)
    return _pct_norm(_MRC_CACHE[p].read(int(meta["particle_index"])))


# ---------- figures ----------
def fig_umap_xray(emb, metas, out):
    dose = np.array([float(m.get("dose", "nan") or "nan") for m in metas])
    res = np.array([float(m.get("resolution", "nan") or "nan") for m in metas])
    variant = np.array([str(m.get("variant", "")) for m in metas])
    sample = np.array([str(m.get("sample_id", "")) for m in metas])
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    # variant
    for v, c in [("rigid", "#1f77b4"), ("nonrigid", "#d62728")]:
        s = variant == v
        ax[0].scatter(emb[s, 0], emb[s, 1], s=10, c=c, label=v, alpha=0.7)
    ax[0].legend(); ax[0].set_title("variant (rigid vs nonrigid)")
    # sample (categorical, many)
    cats = sorted(set(sample))
    cmap = plt.get_cmap("tab20")
    for i, cval in enumerate(cats):
        s = sample == cval
        ax[1].scatter(emb[s, 0], emb[s, 1], s=10, color=cmap(i % 20), alpha=0.7)
    ax[1].set_title(f"sample_id ({len(cats)} samples)")
    # dose (log)
    sc = ax[2].scatter(emb[:, 0], emb[:, 1], s=12, c=np.log10(dose + 1), cmap="viridis")
    fig.colorbar(sc, ax=ax[2]); ax[2].set_title("log10 dose")
    # resolution
    sc = ax[3].scatter(emb[:, 0], emb[:, 1], s=12, c=res, cmap="plasma")
    fig.colorbar(sc, ax=ax[3]); ax[3].set_title("resolution (nm)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("X-ray slice feature UMAP — ViT-L OEP1025 @8199 (992 slices, 124 volumes)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, out)


def fig_umap_cryo(emb, metas, out):
    proj = np.array([str(m.get("project_id", "")) for m in metas])
    qual = np.array([float(m.get("quality_score", "nan") or "nan") for m in metas])
    ncc = np.array([float(m.get("ncc_score", "nan") or "nan") for m in metas])
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    cats = sorted(set(proj))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        s = proj == c
        ax[0].scatter(emb[s, 0], emb[s, 1], s=8, color=cmap(i), label=c, alpha=0.6)
    ax[0].legend(title="project"); ax[0].set_title(f"project_id ({len(cats)} datasets)")
    for lab, c in [(1.0, "#2ca02c"), (0.0, "#d62728")]:
        s = qual == lab
        ax[1].scatter(emb[s, 0], emb[s, 1], s=8, c=c, label=("good" if lab else "junk"), alpha=0.5)
    ax[1].legend(); ax[1].set_title("quality_score (good vs junk)")
    sc = ax[2].scatter(emb[:, 0], emb[:, 1], s=8, c=ncc, cmap="viridis")
    fig.colorbar(sc, ax=ax[2]); ax[2].set_title("ncc_score")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Cryo particle feature UMAP — ViT-L OEP1025 @8199 ({len(metas)} particles sampled)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, out)


def retrieval_panel(feats, metas, thumb_fn, out, *, title, key, exclude_key=None,
                    n_queries=6, topk=5, seed=0):
    """Grid: each row = a query (col0) + its top-k cosine neighbors. Green border = same `key`."""
    fn = l2n(feats)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(len(metas), size=n_queries, replace=False)
    fig, axes = plt.subplots(n_queries, topk + 1, figsize=((topk + 1) * 2.0, n_queries * 2.0))
    for r, qi in enumerate(qidx):
        sims = fn @ fn[qi]
        sims[qi] = -2
        if exclude_key is not None:  # drop neighbors from the same volume (find the *other* version)
            same = np.array([metas[j].get(exclude_key) == metas[qi].get(exclude_key) for j in range(len(metas))])
            sims[same] = -2
        nn = np.argsort(-sims)[:topk]
        cells = [qi] + list(nn)
        for c, idx in enumerate(cells):
            ax = axes[r, c] if n_queries > 1 else axes[c]
            try:
                ax.imshow(thumb_fn(metas[idx]), cmap="gray")
            except Exception as e:
                ax.text(0.5, 0.5, "load err", ha="center", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{key}={metas[qi].get(key)}", fontsize=8)
                for sp in ax.spines.values():
                    sp.set_color("black"); sp.set_linewidth(2)
                ax.set_title("query", fontsize=8)
            else:
                match = metas[idx].get(key) == metas[qi].get(key)
                col = "#2ca02c" if match else "#d62728"
                for sp in ax.spines.values():
                    sp.set_color(col); sp.set_linewidth(2.5)
                ax.set_title(f"{metas[idx].get(key)}{' ✓' if match else ''}", fontsize=8, color=col)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--ood-root", default="/mnt/huawei_deepcad/benchmark/ood")
    ap.add_argument("--out", default="benchmark_runs/eval_ood_analysis")
    ap.add_argument("--cryo-sample", type=int, default=6000)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    outdir = Path(args.out) / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    xr = np.load(fdir / "xray_three_slices_spv8.npz", allow_pickle=True)
    cr = np.load(list(fdir.glob("cryo_*.npz"))[0], allow_pickle=True)
    xr_f, xr_m = xr["features"], list(xr["metas"])
    cr_f, cr_m = cr["features"], list(cr["metas"])
    print(f"xray {xr_f.shape}  cryo {cr_f.shape}")

    print("UMAP x-ray...")
    save_fig_xray_emb = umap_embed(xr_f)
    fig_umap_xray(save_fig_xray_emb, xr_m, outdir / "6_umap_xray.png")

    print("UMAP cryo (subsample)...")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(cr_m), size=min(args.cryo_sample, len(cr_m)), replace=False)
    cr_emb = umap_embed(cr_f[sel])
    fig_umap_cryo(cr_emb, [cr_m[i] for i in sel], outdir / "7_umap_cryo.png")

    print("X-ray pair retrieval panel...")
    retrieval_panel(
        xr_f, xr_m, lambda m: xray_thumb(m, args.ood_root), outdir / "8_xray_pair_retrieval.png",
        title="X-ray retrieval: query slice + top-5 neighbors (exclude same volume) — green=same tomogram (rigid/nonrigid twin)",
        key="tomo_id", exclude_key="volume_id", n_queries=6, topk=5,
    )

    print("Cryo retrieval panel...")
    # subsample cryo for the NN search to keep it fast & thumbnails loadable
    retrieval_panel(
        cr_f[sel], [cr_m[i] for i in sel], cryo_thumb, outdir / "9_cryo_retrieval.png",
        title="Cryo retrieval: query particle + top-5 neighbors — green=same 2D class_id",
        key="class_id", exclude_key=None, n_queries=6, topk=5,
    )
    print("done")


if __name__ == "__main__":
    main()
