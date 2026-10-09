#!/usr/bin/env python3
"""Where, in the anchor's own eigenbasis, does the late model lose decodability?

Global reconstruction R2 is dominated by the few highest-variance directions, so it
can look healthy while every direction a task actually uses has been destroyed.  Here
we resolve decodability per direction.

  W             ridge map base -> anchor, fit on TRAIN
  U             eigenbasis of the anchor's TRAIN covariance, ordered by variance
  R2_j          1 - ||res_test u_j||^2 / ||(A_test - mean_train) u_j||^2

and, for the direction that actually carries the label,

  beta          coefficients of the protocol linear probe fit on the ANCHOR's train
                features (one vector per class)
  R2_task       decodability of the anchor's own decision scores A beta from the base

R2_task is the quantity a retention objective must hold near 1.  Comparing it with the
variance-weighted global R2 shows whether a variance-weighted retention loss would even
look at the right subspace.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
REG = {"conic-cell-count", "livecell-cell-count"}


def load(ds, role, split):
    fs = sorted(glob.glob(f"{FROZEN}/{ds}/{role}/features/{ds}/*_{split}.npz"))
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{role}/{split}")
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]), np.asarray(d["paths"])


def fit_map(src, dst, ridge_frac=1e-3):
    s = src.astype(np.float64); t = dst.astype(np.float64)
    ms, mt = s.mean(0), t.mean(0)
    sc = s - ms
    g = sc.T @ sc
    g.flat[:: g.shape[0] + 1] += ridge_frac * np.trace(g) / g.shape[0]
    return ms, mt, np.linalg.solve(g, sc.T @ (t - mt))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--anchor", default="E")
    ap.add_argument("--base", default="L")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_directional_decodability_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    A, B = a.anchor, a.base
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        task = "regression" if ds in REG else "classification"
        atr, ytr, ptr = load(ds, A, "train"); ate, yte, pte = load(ds, A, "test")
        btr, _, ptr2 = load(ds, B, "train"); bte, _, pte2 = load(ds, B, "test")
        assert np.array_equal(ptr, ptr2) and np.array_equal(pte, pte2)
        n = len(ytr); idx = np.arange(n)
        if n > a.max_train:
            idx = np.random.default_rng(0).choice(n, a.max_train, replace=False); idx.sort()
        ms, mt, W = fit_map(btr[idx], atr[idx])
        pred_te = (bte.astype(np.float64) - ms) @ W + mt
        res_te = ate.astype(np.float64) - pred_te
        cen_te = ate.astype(np.float64) - mt

        ac = atr[idx].astype(np.float64); ac = ac - ac.mean(0)
        ev, U = np.linalg.eigh(ac.T @ ac / max(1, len(ac) - 1))
        o = np.argsort(ev)[::-1]; ev, U = ev[o], U[:, o]
        num = ((res_te @ U) ** 2).sum(0); den = ((cen_te @ U) ** 2).sum(0)
        r2_dir = 1.0 - num / np.maximum(den, 1e-30)
        global_r2 = 1.0 - num.sum() / den.sum()

        # decodability of the anchor's own decision scores
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if task == "regression":
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(atr[idx], ytr[idx].astype(float))
            beta = (m[-1].coef_ / m[0].scale_).reshape(-1, 1)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000,
                              class_weight="balanced", C=1.0, random_state=0))
            m.fit(atr[idx], ytr[idx].astype(int))
            beta = (m[-1].coef_ / m[0].scale_).T
        beta = beta / np.linalg.norm(beta, axis=0, keepdims=True)
        num_t = ((res_te @ beta) ** 2).sum(0); den_t = ((cen_te @ beta) ** 2).sum(0)
        r2_task_per = 1.0 - num_t / np.maximum(den_t, 1e-30)
        r2_task = float(np.mean(r2_task_per))

        # where does the probe put its weight, in anchor-variance rank?
        w_in_U = (beta.T @ U) ** 2                       # [C, d]
        w_in_U = w_in_U / w_in_U.sum(1, keepdims=True)
        cum = np.cumsum(w_in_U.mean(0))
        rank_50 = int(np.searchsorted(cum, 0.50) + 1)
        rank_90 = int(np.searchsorted(cum, 0.90) + 1)

        bands = {}
        for lo, hi in [(0, 8), (8, 32), (32, 128), (128, 512), (512, 2048)]:
            bands[f"{lo}-{hi}"] = float(1.0 - num[lo:hi].sum() / max(den[lo:hi].sum(), 1e-30))
        print(f"\n=== {ds} [{task}]  n_probe={len(idx)}  {A}<-{B} ===")
        print(f"  global variance-weighted R2 = {global_r2:.4f}")
        print(f"  R2 by anchor-eigen band     : " + "  ".join(f"{k}={v:.3f}" for k, v in bands.items()))
        print(f"  R2 of the anchor's own probe direction(s) = {r2_task:.4f}")
        print(f"  probe weight sits in anchor-variance ranks: 50% by rank {rank_50}, 90% by rank {rank_90}")
        rep[ds] = {"task": task, "n_probe": int(len(idx)), "anchor": A, "base": B,
                   "global_r2": float(global_r2), "bands": bands, "r2_task": r2_task,
                   "r2_task_per_class": [float(x) for x in r2_task_per],
                   "probe_weight_rank50": rank_50, "probe_weight_rank90": rank_90,
                   "r2_dir_head": [float(x) for x in r2_dir[:64]],
                   "eigen_frac_head": [float(x) for x in (ev[:64] / ev.sum())]}
        (a.out / f"{A}_from_{B}.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / f"{A}_from_{B}.json")


if __name__ == "__main__":
    main()
