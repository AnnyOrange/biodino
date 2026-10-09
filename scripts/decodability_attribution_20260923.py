#!/usr/bin/env python3
"""Attribute the late model's probe deficit to the anchor directions it cannot decode.

Per dataset, fit the ridge map base -> anchor on TRAIN, resolve decodability per
anchor eigen-direction, then probe the anchor restricted to the directions the base
decodes above a threshold:

  E_keep(rho)   anchor projected onto { j : R2_j >= rho }
  E_drop(rho)   anchor projected onto the complement

If score(E_keep) is close to score(base), the base's whole deficit is explained by the
loss of the poorly decodable directions, and a retention objective only has to protect
those.  If score(E_keep) stays near score(anchor), the deficit lies elsewhere.

Contribution share of each direction to the anchor's own decision is also reported as
lambda_j * beta_j^2 (variance times squared coefficient), which is the quantity that
actually matters, unlike the bare coefficient mass.
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
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]), np.asarray(d["paths"])


def probe(task, xtr, ytr, xte, yte):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, r2_score
    if task == "regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte)))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000,
                      class_weight="balanced", C=1.0, random_state=0))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--anchor", default="E"); ap.add_argument("--base", default="L")
    ap.add_argument("--rhos", default="0.25,0.5,0.75")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_decodability_attribution_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    A, B = a.anchor, a.base
    rhos = [float(x) for x in a.rhos.split(",")]
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        task = "regression" if ds in REG else "classification"
        atr, ytr, ptr = load(ds, A, "train"); ate, yte, pte = load(ds, A, "test")
        btr, _, p2 = load(ds, B, "train"); bte, _, p3 = load(ds, B, "test")
        assert np.array_equal(ptr, p2) and np.array_equal(pte, p3)
        n = len(ytr); idx = np.arange(n)
        if n > a.max_train:
            idx = np.random.default_rng(0).choice(n, a.max_train, replace=False); idx.sort()
        s = btr[idx].astype(np.float64); t = atr[idx].astype(np.float64)
        ms, mt = s.mean(0), t.mean(0); sc = s - ms
        g = sc.T @ sc; g.flat[:: g.shape[0] + 1] += 1e-3 * np.trace(g) / g.shape[0]
        W = np.linalg.solve(g, sc.T @ (t - mt))
        # decodability per anchor eigen-direction, all estimated on TRAIN
        tc = t - mt
        ev, U = np.linalg.eigh(tc.T @ tc / max(1, len(tc) - 1))
        o = np.argsort(ev)[::-1]; ev, U = ev[o], U[:, o]
        res_tr = tc - sc @ W
        num = ((res_tr @ U) ** 2).sum(0); den = ((tc @ U) ** 2).sum(0)
        r2_dir = 1.0 - num / np.maximum(den, 1e-30)

        sa = probe(task, atr[idx], ytr[idx], ate, yte)
        sb = probe(task, btr[idx], ytr[idx], bte, yte)
        print(f"\n=== {ds} [{task}] n_probe={len(idx)}  {A}={sa:.4f}  {B}={sb:.4f}  "
              f"gap={sa-sb:+.4f} ===", flush=True)
        rows = {}
        for rho in rhos:
            keep = np.where(r2_dir >= rho)[0]
            drop = np.where(r2_dir < rho)[0]
            if len(keep) < 2 or len(drop) < 2:
                rows[str(rho)] = {"n_keep": int(len(keep)), "status": "DEGENERATE"}
                print(f"  rho={rho}: degenerate split (keep={len(keep)})", flush=True)
                continue
            Vk, Vd = U[:, keep], U[:, drop]
            sk = probe(task, (atr[idx].astype(np.float64) - mt) @ Vk, ytr[idx],
                       (ate.astype(np.float64) - mt) @ Vk, yte)
            sd = probe(task, (atr[idx].astype(np.float64) - mt) @ Vd, ytr[idx],
                       (ate.astype(np.float64) - mt) @ Vd, yte)
            expl = (sa - sk) / (sa - sb) if abs(sa - sb) > 1e-9 else float("nan")
            rows[str(rho)] = {"n_keep": int(len(keep)), "n_drop": int(len(drop)),
                              "keep_var_frac": float(ev[keep].sum() / ev.sum()),
                              "score_keep": sk, "score_drop": sd,
                              "explained_fraction_of_gap": expl}
            print(f"  rho={rho}: keep {len(keep):4d} dirs ({ev[keep].sum()/ev.sum():.3f} of var) "
                  f"-> {A}_keep={sk:.4f}  {A}_drop={sd:.4f}   explains {expl*100:5.1f}% of the gap",
                  flush=True)
        # contribution share of each direction to the anchor's own decision
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
        contrib = (ev[:, None] * (U.T @ beta) ** 2).sum(1)
        contrib = contrib / contrib.sum()
        cum = np.cumsum(contrib)
        c50, c90 = int(np.searchsorted(cum, .5) + 1), int(np.searchsorted(cum, .9) + 1)
        # decodability weighted by decision contribution
        r2_contrib = float((np.clip(r2_dir, 0, 1) * contrib).sum())
        print(f"  decision contribution: 50% by anchor-variance rank {c50}, 90% by rank {c90}")
        print(f"  contribution-weighted decodability = {r2_contrib:.4f} "
              f"(variance-weighted = {1 - num.sum()/den.sum():.4f})", flush=True)
        rep[ds] = {"task": task, "score_anchor": sa, "score_base": sb, "gap": sa - sb,
                   "splits": rows, "contrib_rank50": c50, "contrib_rank90": c90,
                   "contribution_weighted_decodability": r2_contrib,
                   "variance_weighted_decodability": float(1 - num.sum() / den.sum())}
        (a.out / f"{A}_from_{B}.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / f"{A}_from_{B}.json")


if __name__ == "__main__":
    main()
