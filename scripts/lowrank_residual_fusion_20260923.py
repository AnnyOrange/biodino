#!/usr/bin/env python3
"""Is the early/late capability conflict confined to a low-dimensional subspace?

For each dataset, all maps are fit on TRAIN features only and applied to test.

  W        ridge map  L -> E                 (how much of E is linearly inside L)
  Eres     E - L W                           (the part of E that L cannot express)
  V_k      top-k PCA of the TRAIN residual   (the 'erosion subspace')
  L(+)k    concat(L, Eres @ V_k)             (d = 2048 + k, only +k dims)

Controls at the same k: random k-dim Gaussian projection of E, and top-k PCA of E
itself (not residualised).  Because the probe standardises every dimension and k <=
256 against d = 2048, these arms carry no meaningful conditioning penalty, unlike the
protocol's d = 4096 balanced concat.

Recovery of the early capability at rank k:
    rho_k = (S(L(+)k) - S(L)) / (S(E) - S(L))
A small k with rho_k near 1 on early-matured capabilities, together with no loss on
late-growing ones, means a rank-k consolidation objective is sufficient.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"

REG = {"conic-cell-count", "livecell-cell-count"}


def load(ds: str, role: str, split: str):
    fs = sorted(glob.glob(f"{FROZEN}/{ds}/{role}/features/{ds}/*_{split}.npz"))
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{role}/{split}: {fs}")
    d = np.load(fs[0], allow_pickle=True)
    return (np.asarray(d["features"], dtype=np.float32),
            np.asarray(d["labels"]), np.asarray(d["paths"]))


def ridge_map(src: np.ndarray, dst: np.ndarray, ridge_frac: float = 1e-3):
    """Least-squares map src -> dst with a trace-scaled ridge, fit on these rows only."""
    s64 = src.astype(np.float64)
    mu_s = s64.mean(0)
    mu_d = dst.astype(np.float64).mean(0)
    sc = s64 - mu_s
    dc = dst.astype(np.float64) - mu_d
    g = sc.T @ sc
    lam = ridge_frac * np.trace(g) / g.shape[0]
    g.flat[:: g.shape[0] + 1] += lam
    w = np.linalg.solve(g, sc.T @ dc)
    return mu_s, mu_d, w, float(lam)


def apply_map(x, mu_s, mu_d, w):
    return (x.astype(np.float64) - mu_s) @ w + mu_d


def probe(task, xtr, ytr, xte, yte):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, r2_score
    if task == "regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte)))
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=10000, class_weight="balanced", C=1.0,
                                         random_state=0, n_jobs=1))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def balanced_concat(a, b):
    return unit_rows(np.concatenate((unit_rows(a), unit_rows(b)), axis=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--ks", default="8,16,32,64,128,256")
    ap.add_argument("--donor", default="E", choices=["E", "M"])
    ap.add_argument("--base", default="L", choices=["L", "M"])
    ap.add_argument("--with-controls", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_lowrank_residual_fusion_20260923")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",")]
    DON, BASE = args.donor, args.base

    report = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        t0 = time.time()
        task = "regression" if ds in REG else "classification"
        b = {}
        for role in (DON, BASE):
            xtr, ytr, ptr = load(ds, role, "train")
            xte, yte, pte = load(ds, role, "test")
            b[role] = dict(xtr=xtr, xte=xte, ytr=ytr, yte=yte, ptr=ptr, pte=pte)
        assert np.array_equal(b[DON]["ptr"], b[BASE]["ptr"]) and np.array_equal(b[DON]["pte"], b[BASE]["pte"])
        assert np.array_equal(b[DON]["ytr"], b[BASE]["ytr"]) and np.array_equal(b[DON]["yte"], b[BASE]["yte"])
        ytr, yte = b[DON]["ytr"], b[DON]["yte"]
        n_tr, d = b[BASE]["xtr"].shape
        print(f"\n=== {ds} [{task}] n_train={n_tr} n_test={len(yte)} d={d} "
              f"donor={DON} base={BASE} ===", flush=True)

        # how much of the donor is linearly inside the base
        mu_s, mu_d, W, lam = ridge_map(b[BASE]["xtr"], b[DON]["xtr"])
        pred_te = apply_map(b[BASE]["xte"], mu_s, mu_d, W)
        num = float(((b[DON]["xte"].astype(np.float64) - pred_te) ** 2).sum())
        den = float(((b[DON]["xte"].astype(np.float64) - b[DON]["xtr"].astype(np.float64).mean(0)) ** 2).sum())
        lin_r2 = 1.0 - num / den
        pred_tr = apply_map(b[BASE]["xtr"], mu_s, mu_d, W)
        res_tr = b[DON]["xtr"].astype(np.float64) - pred_tr
        res_te = b[DON]["xte"].astype(np.float64) - pred_te
        rc = res_tr - res_tr.mean(0)
        cov = rc.T @ rc / max(1, len(rc) - 1)
        ev, evec = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        ev, evec = ev[order], evec[:, order]
        tot = float(ev.sum())
        print(f"  donor variance linearly explained by base (test): R2={lin_r2:.4f}", flush=True)
        print(f"  residual spectrum: top8 frac={ev[:8].sum()/tot:.3f}  top32={ev[:32].sum()/tot:.3f}  "
              f"top128={ev[:128].sum()/tot:.3f}", flush=True)

        scores = {}
        scores[BASE] = probe(task, b[BASE]["xtr"], ytr, b[BASE]["xte"], yte)
        scores[DON] = probe(task, b[DON]["xtr"], ytr, b[DON]["xte"], yte)
        print(f"  {BASE}={scores[BASE]:.4f}   {DON}={scores[DON]:.4f}   "
              f"gap({DON}-{BASE})={scores[DON]-scores[BASE]:+.4f}", flush=True)

        gap = scores[DON] - scores[BASE]
        rows = {}
        for k in ks:
            V = evec[:, :k]
            xtr = np.concatenate((b[BASE]["xtr"], (res_tr @ V).astype(np.float32)), axis=1)
            xte = np.concatenate((b[BASE]["xte"], (res_te @ V).astype(np.float32)), axis=1)
            s = probe(task, xtr, ytr, xte, yte)
            rho = (s - scores[BASE]) / gap if abs(gap) > 1e-9 else float("nan")
            rows[f"{BASE}+res{k}"] = {"score": s, "dim": int(xtr.shape[1]), "rho": rho,
                                      "resid_var_frac": float(ev[:k].sum() / tot)}
            print(f"  {BASE}+res{k:<4d} d={xtr.shape[1]:5d}  score={s:.4f}  "
                  f"delta={s-scores[BASE]:+.4f}  rho={rho:+.2f}", flush=True)

        if args.with_controls:
            rng = np.random.default_rng(0)
            for k in ks:
                P = rng.standard_normal((d, k)) / np.sqrt(d)
                xtr = np.concatenate((b[BASE]["xtr"], (b[DON]["xtr"] @ P).astype(np.float32)), axis=1)
                xte = np.concatenate((b[BASE]["xte"], (b[DON]["xte"] @ P).astype(np.float32)), axis=1)
                s = probe(task, xtr, ytr, xte, yte)
                rows[f"{BASE}+rand{k}"] = {"score": s, "dim": int(xtr.shape[1]),
                                           "rho": (s - scores[BASE]) / gap if abs(gap) > 1e-9 else float("nan")}
                print(f"  {BASE}+rand{k:<3d} d={xtr.shape[1]:5d}  score={s:.4f}  "
                      f"delta={s-scores[BASE]:+.4f}   [control]", flush=True)

        s = probe(task, balanced_concat(b[DON]["xtr"], b[BASE]["xtr"]), ytr,
                  balanced_concat(b[DON]["xte"], b[BASE]["xte"]), yte)
        rows["protocol_balanced_concat"] = {"score": s, "dim": 2 * d,
                                            "rho": (s - scores[BASE]) / gap if abs(gap) > 1e-9 else float("nan")}
        print(f"  protocol {DON}+{BASE} d={2*d}  score={s:.4f}  delta={s-scores[BASE]:+.4f}", flush=True)

        report[ds] = {"task": task, "n_train": int(n_tr), "n_test": int(len(yte)), "dim": int(d),
                      "donor": DON, "base": BASE,
                      "donor_linear_r2_from_base": lin_r2, "ridge_lambda": lam,
                      "residual_var_frac": {str(k): float(ev[:k].sum() / tot) for k in ks},
                      "baseline_scores": scores, "gap_donor_minus_base": gap, "arms": rows,
                      "seconds": time.time() - t0}
        (args.out / f"{DON}_into_{BASE}.json").write_text(json.dumps(report, indent=1))
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    print(f"\nwrote {args.out / f'{DON}_into_{BASE}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
